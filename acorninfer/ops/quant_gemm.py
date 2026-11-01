"""M7 Task 1:W8 dequant-GEMM 定档 kernel + 装载期 autotune + bucket 分派。

段 0 裁决定档(docs/benchmarks.md M7 段 0 控制器裁决,2026-11-01):
- 主形态 **(BLOCK_N=64, BLOCK_K=128, warps=4, stages=4, SPLIT_K=1)** 五档全胜
  (kv 14.96µs / qo 20.41µs / gate_up 56.94µs / lm_head 661.37µs,η 0.28–0.93);
- **down (4096,12288) 加 sk2**(57.83µs / 870.4GB/s;N=4096 grid 64 CTA 半波,
  ×2 = 128 programs 整 1 波,同 decode_gemm down 先例);
- **qo 的 autotune 候选开放 sk∈{1,2}**(同 down 先例,装载期实测定);
- **kv (1024,4096) sk1 定死**(段 0 实证 split-K 全输:sk2 19.22 vs sk1 14.96µs,
  reduce 流量吞掉并行收益);gate_up / lm_head 同证(段 0 段2 只对 N-不足类开
  split 候选,此处五档主形态即 sk1)。

核内 dequant(定档形态,spike `_w8_kernel` 逐字进产线):int8 load → fp32 ×
scale(行广播 per-channel,循环不变提出)→ bf16 → tl.dot fp32 累加;
BLOCK_M=16(M≤16 全 pad 16,tensor-core dot)。bf16 in/out。

骨架复用(`acorninfer/ops/decode_gemm.py` 按现状导入,**该文件零改动**,既有
bf16 kernel 与 17 测试行为零变化):`_bench_rot`(冷口径轮转计时)/
`_cfg_shmem_ok`(shmem ≤96KB 预过滤;bf16 w 瓦片 2B/elt 的保守式对 int8 只会
更严,零假阳性放行)/ `_workspace` + `_splitk_reduce_kernel`(fp32 partials
常驻 + 确定性 second-pass 归约,零 atomic;partial 布局 [SPLIT_K,16,N] 同款,
键 (N, split_k) 跨模块共享安全 —— 写后读同 launch 内完成)。`_bucket_for`
归桶法**不适用本模块**(如实偏离):decode_gemm 的表按 (cls, bucket) 记
backend 归桶,本模块 cfg 按类一档(段 0 实证 config M-不变:M∈{1,8} 逐类
同胜出),M 守门用 M ≤ BLOCK_M=16 直判。

装载期 autotune(非全网格,段 0 胜出点 ± 邻域小集;装载/捕获前一次,Engine
构造前):数值网格 = {(64,128,4,4) 冠军, (64,128,4,3) stages 邻,
(64,64,4,4) BK 邻, (128,64,4,4) BN 邻} × shmem/整除过滤;split 段 = down/qo
top-2 × sk2(段 0 先例:top-2 × sk,整除过滤)。8 套轮转(decode_gemm
autotune 同款;权重为随机 int8+fp16 scale —— 带宽只看字节流,免 bf16 瞬态)。
warm(表内逐类零值发射一次,编译/工作区就位)预算 <1s,configure() 报告实测。

CUDA-graph 兼容硬约束(decode_gemm 同款):launch 路径零 host 同步(无
.item()/synchronize;M 为 runtime 参数零重编译);autotune 全部发生在装载期
(configure(),graph 捕获前);捕获/重放路径只查表 + launch。

M regime:decode M≤16 本任务范围(BLOCK_M=16 单瓦片);**prefill 大 M 桶
(M>16)是 Task 2 接线范围,本任务只留接口注记** —— `_launch` 对 M>16 显式
NotImplementedError(防误用静默错慢),Task 2 按独立 autotune 桶落地(spec §2
段 A 两 regime)。
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from acorninfer.ops.decode_gemm import (
    _BLOCK_M,
    _REDUCE_BLOCK,
    _bench_rot,
    _cfg_shmem_ok,
    _splitk_reduce_kernel,
    _workspace,
)

# 生产五档类形状(独立模块口径;与 quantize.py docstring / spike CLASSES 同源)
CLASSES = {
    "qo": (4096, 4096),         # q_proj / o_proj
    "kv": (1024, 4096),         # k_proj / v_proj
    "gate_up": (12288, 4096),   # gate_proj / up_proj 各一次
    "down": (4096, 12288),
    "lm_head": (151936, 4096),
}

_AUTO_SETS = 8                 # autotune 轮转套(decode_gemm _autotune_config 同款)
_AUTO_SPLIT = {"down", "qo"}   # 开 sk2 候选的类(段 0 裁决;kv 定 sk1)
_AUTO_SKS = (2,)               # 段 0:down sk2 胜出,qo 评估 sk∈{1,2}(sk1 在数值网格)

_state: dict = {
    "configs": {},             # cls -> (BLOCK_N, BLOCK_K, warps, stages, SPLIT_K)
    "shapes": {},              # cls -> (N, K)(warmup 用)
    "device": None,
    "report": None,
}


# ---------- kernel(定档形态:spike _w8_kernel 逐字,探索变体不入产线) ----------

@triton.jit
def _quant_store(acc, out_ptr, part_ptr, pid_k,
                 BM: tl.constexpr, N, SPLIT_K: tl.constexpr,
                 offs_m, offs_n, m_mask):
    """SPLIT_K==1 直接 bf16 写 out;否则 fp32 写 partials[pid_k](确定性归约)。"""
    if SPLIT_K == 1:
        out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        p_ptrs = part_ptr + pid_k * BM * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(p_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def _w8_gemm_kernel(x_ptr, q_ptr, s_ptr, out_ptr, part_ptr,
                    M, N, K, K_CHUNK,
                    SPLIT_K: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    """C[M,N] = x[M,K] @ dequant(q,s)[N,K]^T;bf16 in/out,fp32 累加。

    q int8 [N,K] 行主;scale fp16 [N] per-channel 行广播(循环不变提出);
    核内 dequant:int8 → fp32 ×scale → bf16 → tl.dot。N % BN == 0 与
    K % (SPLIT_K·BK) == 0 由 host 断言,W 侧免 mask;M 侧 mask(BM=16 pad)。
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_lo = pid_k * K_CHUNK
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    s = tl.load(s_ptr + offs_n).to(tl.float32)          # [BN] 循环不变
    x_ptrs = x_ptr + offs_m[:, None] * K + k_lo + offs_k[None, :]
    q_ptrs = q_ptr + offs_n[:, None] * K + k_lo + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K_CHUNK, BK):
        a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        q = tl.load(q_ptrs)
        w = (q.to(tl.float32) * s[:, None]).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(w), acc)
        x_ptrs += BK
        q_ptrs += BK
    _quant_store(acc, out_ptr, part_ptr, pid_k, BM, N, SPLIT_K, offs_m, offs_n, m_mask)


# ---------- launch(捕获/重放路径:只查表 + launch,零 host 同步) ----------

def _launch(x2: torch.Tensor, q: torch.Tensor, s: torch.Tensor, cls: str,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """x2 [M, K] 连续 bf16、q [N, K] int8、s [N] fp16 → [M, N] bf16。

    cfg = _state["configs"][cls](装载期 autotune 产物);split workspace 装载期
    已分配(重入 dict 命中,零分配,地址冻结重放安全);grid 全 host 侧 int,
    零 host 同步。
    """
    cfg = _state["configs"][cls]
    bn, bk, warps, stages, split_k = cfg
    M, K = x2.shape
    N = q.shape[0]
    assert N % bn == 0 and K % (split_k * bk) == 0, \
        f"quant_gemm 形状不整除: N={N} K={K} cfg={cfg}"
    if M > _BLOCK_M:
        # M>16(prefill 大 M 桶)= Task 2 范围(接口注记;BLOCK_M=16 单瓦片
        # 只覆盖 decode)。显式报错防误用静默错慢,不回落(cuBLAS 不吃 int8 权重)。
        raise NotImplementedError(
            f"quant_gemm M={M} > 16:prefill 大 M 桶为 M7 Task 2 范围(接口注记)")
    if out is None:
        out = torch.empty(M, N, dtype=x2.dtype, device=x2.device)
    grid = (N // bn, split_k)
    if split_k == 1:
        _w8_gemm_kernel[grid](
            x2, q, s, out, out,         # partial_ptr 占位(SPLIT_K=1 编译期剪除)
            M, N, K, K,
            SPLIT_K=1, BM=_BLOCK_M, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages,
        )
    else:
        part = _workspace(N, split_k, x2.device)
        _w8_gemm_kernel[grid](
            x2, q, s, out, part,
            M, N, K, K // split_k,
            SPLIT_K=split_k, BM=_BLOCK_M, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages,
        )
        total = M * N
        _splitk_reduce_kernel[(triton.cdiv(total, _REDUCE_BLOCK),)](
            part, out, total, _BLOCK_M * N, split_k, BLOCK=_REDUCE_BLOCK,
        )
    return out


def quant_gemm(x, q, scale, cls, cfg=None, out=None):
    """W8 dequant-GEMM 入口(Task 2 调用点形态):x [..., K]、q int8 [N,K]、
    scale fp16 [N]、cls ∈ CLASSES → [..., N] bf16。

    cfg None = 装载期 tuned config(须先 configure());显式 cfg = 绕表
    (单测对拍 / 装载期实测仪;decode_gemm.triton_gemm 同款存取恢复)。
    """
    K = x.shape[-1]
    x2 = x.reshape(x.numel() // K, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    saved = _state["configs"].get(cls)
    if cfg is not None:
        _state["configs"][cls] = tuple(cfg)
    try:
        return _launch(x2, q, scale, cls, out)
    finally:
        if cfg is not None:
            if saved is None:
                _state["configs"].pop(cls, None)
            else:
                _state["configs"][cls] = saved


# ---------- 装载期:autotune(胜出点 ± 邻域小集)+ warm ----------

def _qbytes(n: int, k: int) -> int:
    """W8 量化权重字节(含元数据):int8 [N,K] + fp16 scale [N](spike 同式)。"""
    return n * k + 2 * n


def _rot_sets(n, k, n_sets, device):
    """autotune 轮转套:(q int8 随机, s fp16 随机)。带宽只看字节流,免 spike
    的 bf16→quantize 瞬态(lm_head 8 套 5GB 进得了显存;36 套为冷口径专用)。"""
    return [(torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device),
             torch.randn(n, device=device).half())
            for _ in range(n_sets)]


def _autotune_config(cls, n, k, device, verbose=True):
    """装载期 autotune(段 0 胜出点 ± 邻域小集,非全网格;8 套轮转热测)。

    数值网格 = 冠军 (64,128,4,4) ± {stages 3, BK 64, BN 128} —— 段 0 全网格
    数据已把 warps 钉在 4(w8 全档不敌)、BN/BK 钉在 64/128(次优差 ≥10%);
    shmem 预过滤照 _cfg_shmem_ok(decode_gemm 同款,会把 (128,128,·) 滤掉)。
    split 段 = down/qo top-2 × sk2(段 0 先例;kv/gate_up/lm_head sk1 定死)。
    返回 (BLOCK_N, BLOCK_K, warps, stages, SPLIT_K)。
    """
    sets = _rot_sets(n, k, _AUTO_SETS, device)
    x = torch.randn(_BLOCK_M, k, dtype=torch.bfloat16, device=device)
    grid_cfgs = [c for c in ((64, 128, 4, 4, 1),   # 段 0 冠军
                             (64, 128, 4, 3, 1),   # stages 邻
                             (64, 64, 4, 4, 1),    # BK 邻
                             (128, 64, 4, 4, 1))   # BN 邻
                 if _cfg_shmem_ok(c) and n % c[0] == 0 and k % c[1] == 0]
    results, skipped = [], 0

    def _try(cfg):
        nonlocal skipped
        try:
            us = _bench_rot(
                lambda t, c=cfg: quant_gemm(x, t[0], t[1], cls, cfg=c), sets)
            results.append((us, cfg))
            if verbose:
                gb = _qbytes(n, k) / (us * 1e-6) / 1e9
                print(f"W8 AUTOTUNE {cls} cfg={cfg} us={us:.2f} gb={gb:.0f}",
                      flush=True)
        except Exception as e:   # 编译期资源失败:跳过并记账
            skipped += 1
            if verbose:
                print(f"W8 AUTOTUNE {cls} cfg={cfg} SKIPPED {type(e).__name__}",
                      flush=True)

    for cfg in grid_cfgs:
        _try(cfg)
    results.sort()
    if not results:
        raise RuntimeError(f"quant_gemm autotune 全配置失败: {cls}")
    if cls in _AUTO_SPLIT:
        for base in [c for _, c in results[:2]]:
            for sk in _AUTO_SKS:
                if k % (sk * base[1]) == 0:
                    _try((*base[:4], sk))
    results.sort()
    best_us, best_cfg = results[0]
    gb = _qbytes(n, k) / (best_us * 1e-6) / 1e9
    print(f"W8 AUTOTUNE_BEST {cls} cfg={best_cfg} us={best_us:.2f} gb={gb:.0f} "
          f"skipped={skipped}", flush=True)
    del sets, x
    torch.cuda.empty_cache()
    return best_cfg


def _warmup_table() -> float:
    """表内逐类零值发射一次:编译(命中 autotune 期缓存)/ workspace / 表查
    全链路就位(捕获前)。返回耗时秒(warm 预算 <1s,configure 报告)。"""
    t0 = time.perf_counter()
    dev = _state["device"]
    for cls, (n, k) in sorted(_state["shapes"].items()):
        q = torch.zeros(n, k, dtype=torch.int8, device=dev)
        s = torch.ones(n, dtype=torch.float16, device=dev)
        x = torch.zeros(8, k, dtype=torch.bfloat16, device=dev)
        out = quant_gemm(x, q, s, cls)          # 零值:数值无关,链路就位即可
        del q, s, x, out
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def configure(classes=None, *, device="cuda:0", verbose=True) -> dict:
    """装载期入口(Engine 构造 / graph 捕获之前调用一次;Task 2 接线)。

    classes: {类名: (N, K)},None = 生产五档 CLASSES。逐类 autotune → warm
    (预算 <1s,实测入 report)。返回 report(docs 账目直接引用)。
    """
    reset()
    classes = CLASSES if classes is None else dict(classes)
    _state["device"] = device
    _state["shapes"] = classes
    t0 = time.perf_counter()
    report = {"classes": {}}
    for cls, (n, k) in classes.items():
        t1 = time.perf_counter()
        cfg = _autotune_config(cls, n, k, device, verbose=verbose)
        _state["configs"][cls] = cfg
        report["classes"][cls] = {"config": cfg,
                                  "autotune_s": time.perf_counter() - t1}
    warm_s = _warmup_table()
    total_s = time.perf_counter() - t0
    report["warm_s"] = warm_s
    report["total_s"] = total_s
    _state["report"] = report
    print(f"W8 CONFIGURE total_s={total_s:.2f} warm_s={warm_s:.3f} "
          f"(warm 预算 <1s;autotune 含 JIT 编译,装载期一次)"
          f" cfgs={ {c: r['config'] for c, r in report['classes'].items()} }",
          flush=True)
    return report


def reset() -> None:
    """清空全部状态(测试间隔离;workspace 归 decode_gemm 管理,不动)。"""
    _state.clear()
    _state.update({"configs": {}, "shapes": {}, "device": None, "report": None})


# _AUTO_SETS=8 套热测轮转(decode_gemm autotune 同款);36 套冷口径轮转
# (_bench_rot + _ROT_SETS 属 decode_gemm 实测仪口径)是 A/B 实测仪需求,
# 生产装载不需要 —— Task 2 的 m6_ab --quant w8a16 侧按需复用 decode_gemm 面。
