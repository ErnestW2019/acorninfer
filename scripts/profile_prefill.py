"""prefill 批量前向 profiler 拆账(M5 Task 0;M9 Task 0 扩展归因表)。

跑法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_prefill.py
对象:Qwen3ForCausalLM.forward_prefill_batch(8B,bf16,cuda),[8,128] 随机 ids
(seed=20261027,与 bench/profile_decode 同源)× 10 步。每步新建 views
(forward_prefill_batch 经 view.update 追加 KV,复用会续写 —— 步间零状态:
每步各序列分 8 页(128 tok / page 16),步末归还),前向本体不含调度/采样,
即 M4.1 批量 prefill 的纯模型侧账。prefill 路径不接 attention_backend /
fused_ops 选择器(qwen3.py forward_prefill_batch 契约),保持默认不置位。

输出:
- 每步墙钟、GPU 自用时间(kernel+memcpy)、gap(墙钟 − GPU 端)、
  有效 TFLOPS(14.2 TFLOP/次 ÷ 实测;总 FLOP 按 36 层静态 GEMM 估算:
  qkv+o 2*D*(D+2KV*D)+MLP 2*3*D*F,×36 层 ×1024 token,attention 与
  lm_head 不计 —— 标注"按 36 层静态 FLOP 估算",非实测 FLOP)、top-15
  kernel 表(按自用设备时间,归类进 docs)。
- M9 Task 0 追加 **归因表**(行 = 面,列 = ms/步 + 占比):
  ① 按 kernel 名归类(GEMM / SDPA / memcpy / elementwise+other);
  ② record_function 插桩实测注意力循环内部诸面 —— view.update(gather
  +scatter 写入)、_repeat_kv(物化拷贝)、SDPA 本体、qkv/o 投影、norm+rope
  段。插桩 = 脚本内 monkeypatch Qwen3Attention.forward_prefill_batch 的
  逐算子同构副本(算子序列与 qwen3.py forward_prefill_batch 逐行一致,
  仅插入 record_function 标注;产线文件零改动);面级 GPU 时间取 chrome
  trace 的 gpu_user_annotation 事件(Kineto 把 CPU 标注域投影到 GPU 流,
  只含域内 kernel 时间、不含域内 GPU 空闲 —— 实测先踩过 cuda Event 对
  把域内 host 间隙记进 GPU 账的坑,已弃);host 侧取 CPU user_annotation
  时长(含 record_function 本身开销,量级参考)。

profiler 口径沿用 scripts/profile_decode.py 实测结论(torch 2.13):
GPU 忙时只按 device_type=CUDA 事件累加,属性名 self_device_time_total
(旧名 self_cuda_time_total 在 FunctionEventAvg 上不存在);表排序键同名。
"""

from __future__ import annotations

import collections
import json
import math

import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from acorninfer.bench import DEFAULT_MODEL, _prompts
from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.models.qwen3 import (
    Qwen3Attention,
    _repeat_kv,
    _rotate_half,
    dispatch_linear,
    from_pretrained,
)

BATCH, IN_TOKENS, SEED = 8, 128, 20261027

# M9 归因表插桩面(注意力循环内部 + 投影段)
FACETS = (
    "attn_qkv_proj",
    "attn_norm_rope",
    "attn_view_update",
    "attn_repeat_kv",
    "attn_sdpa",
    "attn_o_proj",
)

# kernel 名归类模式(小写子串;顺序即优先级)
GEMM_PAT = ("gemm", "nvjet", "cutlass", "matmul")
SDPA_PAT = ("flash", "sdpa", "fmha", "attention", "attn")


def _is_gpu(evt) -> bool:
    """GPU 侧事件(kernel / device Memcpy)判定,CPU 侧算子事件返回 False。"""
    return str(evt.device_type).endswith("CUDA")


def _classify(name: str) -> str:
    n = name.lower()
    if any(p in n for p in SDPA_PAT):
        return "sdpa"
    if any(p in n for p in GEMM_PAT):
        return "gemm"
    if "memcpy" in n:
        return "memcpy"
    return "elementwise+other"


_USE_RF = True       # 遍 2(True)= record_function 进 profiler trace;遍 3(False)= host perf_counter
_HOST_ACC: dict | None = None
_t0 = 0.0


class _HostSpan:
    def __init__(self, facet: str):
        self._facet = facet

    def __enter__(self):
        global _t0
        _t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _HOST_ACC[self._facet] += (time.perf_counter() - _t0) * 1e3
        return False


def _instrumented_prefill_batch(
    self: Qwen3Attention, hidden, cos, sin, views, layer_idx, *, real_lens
):
    """qwen3.py forward_prefill_batch 逐算子同构副本(仅插入插桩标注)。

    同步契约:qwen3.py 该方法体改动时本副本必须逐行跟随(M9 施工后此副本
    仅服务 spike 测量,数值不进产线)。"""
    N, Lmax, _ = hidden.shape
    rf = torch.autograd.profiler.record_function if _USE_RF else _HostSpan
    with rf("attn_qkv_proj"):
        q = dispatch_linear(self.q_proj, hidden).view(
            N, Lmax, self.num_heads, self.head_dim).transpose(1, 2)
        k = dispatch_linear(self.k_proj, hidden).view(
            N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = dispatch_linear(self.v_proj, hidden).view(
            N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)
    with rf("attn_norm_rope"):
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
    rep = self.num_heads // self.num_kv_heads
    outs = torch.zeros(N, self.num_heads, Lmax, self.head_dim,
                       dtype=hidden.dtype, device=hidden.device)
    for b, view in enumerate(views):
        L = real_lens[b]
        with rf("attn_view_update"):
            k_full, v_full = view.update(layer_idx, k[b: b + 1, :, :L],
                                         v[b: b + 1, :, :L])
        with rf("attn_repeat_kv"):
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
        with rf("attn_sdpa"):
            outs[b, :, :L] = F.scaled_dot_product_attention(
                q[b: b + 1, :, :L], k_full, v_full, is_causal=(L > 1))[0]
    with rf("attn_o_proj"):
        out = outs.transpose(1, 2).reshape(N, Lmax, -1)
        return dispatch_linear(self.o_proj, out)


def _facet_sums(prof_obj, n_steps: int):
    """从 chrome trace 做逐 kernel 归因:kernel 的 External id = 发射它的
    cpu_op 的 External id;cpu_op 按 CPU 时间戳区间落入最内层插桩标注域
    (user_annotation,同 tid)→ 面名。面级 GPU 时间 = Σ 域内 kernel dur
    (纯 kernel 时间,不含域内 GPU 空闲);面外 kernel 按 kernel 名归类。"""
    path = "/tmp/opencode/m9t0/m9_prefill_facets.json"
    prof_obj.export_chrome_trace(path)
    with open(path) as fh:
        tr = json.load(fh)

    anns = []  # (name, ts0, ts1, tid)
    ops = {}   # ext_id -> (ts, tid)
    for e in tr["traceEvents"]:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        name = e.get("name", "")
        if cat == "user_annotation" and name in FACETS:
            anns.append((name, e["ts"], e["ts"] + e["dur"], e.get("tid")))
        elif cat == "cpu_op":
            ext = e.get("args", {}).get("External id")
            if ext is not None:
                ops[ext] = (e["ts"], e.get("tid"))

    # 同 tid 内标注域互不重叠(同构副本内 6 面为顺序兄弟域):按 ts0 排序,
    # 二分取 ts0 ≤ op.ts 的最近域,ts 落在其区间内即为归属域
    by_tid = collections.defaultdict(list)
    for name, ts0, ts1, tid in anns:
        by_tid[tid].append((ts0, ts1, name))
    for tid in by_tid:
        by_tid[tid].sort()

    import bisect

    starts = {tid: [a[0] for a in lst] for tid, lst in by_tid.items()}

    def _facet_of(ts, tid):
        lst = by_tid.get(tid)
        if not lst:
            return None
        i = bisect.bisect_right(starts[tid], ts) - 1
        if i < 0:
            return None
        ts0, ts1, name = lst[i]
        return name if ts <= ts1 else None

    facet_gpu = collections.defaultdict(float)
    facet_cnt = collections.Counter()
    bucket_gpu = collections.defaultdict(float)
    for e in tr["traceEvents"]:
        cat = e.get("cat")
        if cat not in ("kernel", "gpu_memcpy"):
            continue
        dur = e.get("dur", 0.0)
        ext = e.get("args", {}).get("External id")
        facet = None
        op = ops.get(ext)
        if op is not None:
            facet = _facet_of(op[0], op[1])
        if facet is not None:
            facet_gpu[facet] += dur
            facet_cnt[facet] += 1
        else:
            bucket_gpu[_classify(e.get("name", ""))] += dur
    return {
        "facet_gpu": {f: facet_gpu.get(f, 0.0) / n_steps / 1000.0 for f in FACETS},
        "facet_cnt": {f: facet_cnt.get(f, 0) / n_steps for f in FACETS},
        "bucket_gpu": {k: v / n_steps / 1000.0 for k, v in bucket_gpu.items()},
    }

def main() -> None:
    device = "cuda:0"
    model = from_pretrained(DEFAULT_MODEL, device, torch.bfloat16)
    cfg = model.cfg
    allocator = BlockAllocator(1024)
    cache = PagedKVCache(
        cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
        16, allocator, device, torch.bfloat16,
    )
    pages_per_seq = math.ceil(IN_TOKENS / 16)

    ids = torch.tensor(
        _prompts(BATCH, IN_TOKENS, cfg.vocab_size, SEED),
        dtype=torch.long, device=device,
    )
    lens = [IN_TOKENS] * BATCH

    def one_step() -> torch.Tensor:
        # engine.step 同款 inference_mode:无 backward 图,激活即用即弃
        # (缺此包装时 autograd 保留整段前向中间量,24GB 卡 OOM,实测)
        with torch.inference_mode():
            views = []
            for _ in range(BATCH):
                table = [allocator.allocate() for _ in range(pages_per_seq)]
                views.append(PagedCacheView(cache, table))
            logits = model.forward_prefill_batch(ids, views, real_lens=lens)
            for v in views:
                for b in list(v.table):
                    allocator.free(b)
                v.table.clear()
            return logits

    # 形状 sanity + 一次性初始化(CUDA context / cuBLAS handle)先出清
    logits = one_step()
    assert tuple(logits.shape) == (BATCH, cfg.vocab_size), logits.shape
    allocator.assert_balanced()

    # profiler 预热窗:CUPTI 首次启用/缓冲分配的一次性开销落在独立窗口,
    # 不进计时也不进账目(同 profile_decode)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        one_step()
    torch.cuda.synchronize()

    n = 10
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            one_step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    evs = prof.key_averages()
    gpu = [e for e in evs if _is_gpu(e)]
    gpu_us = sum(e.self_device_time_total for e in gpu)
    wall_ms = wall / n * 1000
    gpu_ms = gpu_us / n / 1000
    print(f"steps={n} wall/step={wall_ms:.2f}ms "
          f"gpu_kernel+memcpy/step={gpu_ms:.2f}ms "
          f"gap/step={(wall - gpu_us / 1e6) / n * 1000:.2f}ms")

    # M9 归因表·遍 1:GPU kernel 按名归类(ms/步 + 占 GPU 忙)
    buckets = {"gemm": 0.0, "sdpa": 0.0, "memcpy": 0.0, "elementwise+other": 0.0}
    for e in gpu:
        buckets[_classify(e.key)] += e.self_device_time_total
    print("[pass1 kernel-name buckets] (self device time / step)")
    for k, us in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {us / n / 1000:8.3f} ms  {us / gpu_us * 100:5.1f}% of gpu")

    # 有效 TFLOPS:14.2 TFLOP/次 ÷ 实测(墙钟与 GPU 侧各一行);
    # 总 FLOP 按 36 层静态 GEMM 估算(attention QK/AV 与 lm_head 不计)
    d, f = cfg.hidden_size, cfg.intermediate_size
    qkv_o = d * (d + 2 * cfg.num_key_value_heads * cfg.head_dim + d)
    flop = 2.0 * cfg.num_hidden_layers * (qkv_o + 3 * d * f) * BATCH * IN_TOKENS
    tflop = flop / 1e12
    eff_wall = flop / (wall / n) / 1e12
    eff_gpu = flop / (gpu_us / n / 1e6) / 1e12
    print(f"flops={tflop:.2f}TFLOP/次(按 36 层静态 FLOP 估算) "
          f"eff_wall={eff_wall:.2f}TFLOPS eff_gpu={eff_gpu:.2f}TFLOPS")

    # M9 归因表·遍 2:注意力循环内部 record_function 插桩(monkeypatch 副本)
    m = 6
    orig = Qwen3Attention.forward_prefill_batch
    Qwen3Attention.forward_prefill_batch = _instrumented_prefill_batch
    try:
        one_step()  # 插桩路径预热
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof2:
            for _ in range(m):
                one_step()
            torch.cuda.synchronize()
    finally:
        Qwen3Attention.forward_prefill_batch = orig
    agg = _facet_sums(prof2, m)
    facet_gpu, facet_cnt, bucket2 = (
        agg["facet_gpu"], agg["facet_cnt"], agg["bucket_gpu"])

    print(f"[pass2 instrumented facets] ({m} steps; gpu = 面内 kernel 自用时间"
          f"和(chrome trace 逐 kernel External-id 归属,不含面内 GPU 空闲),"
          f"cnt = kernel 数/步)")
    for facet in FACETS:
        print(f"  {facet:<18} gpu {facet_gpu[facet]:8.3f} ms  "
              f"cnt {facet_cnt[facet]:7.1f}")
    attn_nonproj = (facet_gpu["attn_view_update"] + facet_gpu["attn_repeat_kv"]
                    + facet_gpu["attn_sdpa"] + facet_gpu["attn_norm_rope"])
    attn_all = attn_nonproj + facet_gpu["attn_qkv_proj"] + facet_gpu["attn_o_proj"]
    print(f"  attn 面小计(不含投影){attn_nonproj:8.3f} ms;含投影 "
          f"{attn_all:8.3f} ms")
    print("[pass2 面外 kernel 桶](= GPU 总账 − 插桩面)")
    tot_buckets = sum(bucket2.values()) + attn_all
    for k, v in sorted(bucket2.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v:8.3f} ms")
    print(f"  核对: 面内+面外 = {tot_buckets:.2f} ms vs GPU busy {gpu_ms:.2f} ms")

    # M9 归因表·遍 3:无 profiler 纯 host 插桩(perf_counter,无探针自身开销
    # 除一次函数调用),量 host gap 在注意力诸面的落点
    m3 = 10
    global _USE_RF, _HOST_ACC
    _USE_RF = False
    _HOST_ACC = {f: 0.0 for f in FACETS}
    Qwen3Attention.forward_prefill_batch = _instrumented_prefill_batch
    try:
        one_step()  # host 插桩路径预热
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(m3):
            one_step()
        torch.cuda.synchronize()
        wall3 = time.perf_counter() - t0
    finally:
        Qwen3Attention.forward_prefill_batch = orig
    host_acc, _USE_RF, _HOST_ACC = _HOST_ACC, True, None
    print(f"[pass3 host facets] ({m3} steps, 无 profiler; wall/step="
          f"{wall3 / m3 * 1000:.2f}ms)")
    host_sum = 0.0
    for facet in FACETS:
        hms = host_acc[facet] / m3
        host_sum += hms
        print(f"  {facet:<18} host {hms:8.3f} ms")
    print(f"  host 面内合计 {host_sum:.2f} ms;面外 host ≈ wall − 面内 "
          f"{wall3 / m3 * 1000 - host_sum:.2f} ms(含 embed/pos 构造/残差/"
          f"executor 侧与调度无关的纯前向 host 段)")

    print(evs.table(sort_by="self_device_time_total", row_limit=15))

    # 步间零状态核对:所有页归还,无泄漏
    allocator.assert_balanced()
    print(f"sanity: blocks live={allocator.num_live} "
          f"(每步 8 序列 × {pages_per_seq} 页步末全归还)")
    try:
        model.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
