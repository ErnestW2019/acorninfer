"""M6 Task 5:decode GEMM triton kernel + 逐 (类, M-bucket) dispatch 表。

**结论(2026-11-01 实测):两类均未过 ≥5% 接入阀 —— 未接入生产,cuBLAS
TN 继续站岗。**证据(docs/benchmarks.md M6 段 5):triton 冷口径最优
gate_up 925–933GB/s、down 902–911GB/s,对各自 cuBLAS 最优变体仅
+1.3~+2.5%(m∈{2,4,8};M=1 输 gemvx);阀要求 ≥5%(即 ~945–956GB/s)。
判别实验:同瓦片纯读上限 gate_up 920–934 / down ≤904(模式天花板);
triton 在 fused 24576×4096(768 CTA)也只到 937–944(cuBLAS fused 947)
—— per-call N=12288/4096 的 CTA 并行度下,triton 无 5% 空间。且 cuBLAS
nn 变体逐轮漂移 ±5%(110.9→116.4µs),+2% 级优势在统计上不可靠,阀值
本身即是对此类边缘收益的过滤。本模块保留 = 实测仪 + kernel + 阀机制
存档(测试全绿),生产零接线:qwen3/config/server 均不动,linear_dispatch
无生产调用点,缺省 layout="none" 直通 F.linear。

范围(裁决):仅两类 —— gate_up(引擎真实 per-call 形状 N=12288, K=4096;
gate_proj / up_proj 各一次,spike 的 N=24576 为融合口径)与 down(N=4096,
K=12288)。qkv/o 冷口径已 88–92% roofline、lm_head 93.5% 峰值,均不接。

物理口径:decode M≤16 权重流主导;bf16 in/out、fp32 累加;BLOCK_M=16
(M≤16 全部 pad 到 16,tensor-core dot 路径)。目标 ≥940GB/s(4090 峰值
1.008TB/s 的 93%+)。

CUDA-graph 兼容硬约束:
- launch 路径零 host 同步(无 .item()/synchronize;grid 全部由 host 侧
  python int 计算,M 是 runtime 标量参数不触发重编译);
- autotune 与 dispatch 表实测全部发生在装载期(configure(),Engine 构造
  前 = graph 捕获前),捕获/重放路径只做表查 + launch;
- split-K workspace(fp32 partials)装载期预分配、常驻引用(地址冻结,
  重放安全);输出张量与其它 aten 算子一样走捕获期 graph pool(既有先例:
  paged_attention 的 per-call workspace 同机制)。

split-K(naive tile grid 常 < 128 programs,SM89 有 128 SM):采用 **fp32
partial buffer + 确定性 second-pass 归约**(非 atomic_add)。选择理由:①
逐位确定性 —— graph 捕获重放与两次独立 launch 输出逐位同,parity margin
账目稳定(atomic 非确定性要靠 TOL 覆盖,能免则免);②无 atomic 争用,
second-pass 读 SPLIT_K×M×N fp32(≤2MB)≈2–4µs,相对 ~110µs 主 kernel 可
忽略;③kernel 数相同(主 + 归约 = atomic 方案的预清零 + 主 + cast)。
split 候选两类均开放(整除过滤后 autotune 择优):down(N=4096,K=12288)
救 grid 欠填充(64 n-tiles × split 2 = 128 programs = 整 1 波);gate_up
(N=12288,K=4096)救波形尾 —— BN=64 时 192 programs = 1.5 波、尾波半空,
×2 → 384 = 整 3 波,代价仅 ~3% partial+reduce 流量,由 autotune 实测裁决。

dispatch 表:逐 (类, M-bucket) 装载期实测择优(triton vs cuBLAS TN;M
buckets = graph_runner.BUCKETS(禁改文件,值 (1,2,4,8),测试钉住一致),
查表按 min(bucket ≥ M) 归桶,超最大桶回 cuBLAS)。实测仪 = 冷口径协议
(m6_spike layout 腿 / phase0 --gemm-cold 同款:36 套独立权重轮转 > 72MB
L2 强制 HBM,warmup 一整轮 + 计时数轮 ÷ 轮×36;两类单套权重 100.7MB 本就
超 L2,hot≈cold 双重成立)。接入阀:某类 triton 冷口径超该类 cuBLAS 最优
变体(tn/nn/pad16 取最小)≥5% 才接入;类内逐桶再择优(输桶留 cuBLAS TN,
生产可达形态;nn 需 [K,N] 存储变换 / pad16 需 padding 拷贝,均只作阀参照)。

缺省 layout="none":linear_dispatch 直通 F.linear,逐字节硬门(与 T1b 同款)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# M buckets = graph_runner.BUCKETS(禁改文件;测试钉住两处一致)。本模块
# 不 import engine 包(防 import 环;executor.py → models.qwen3 链)。
M_BUCKETS = (1, 2, 4, 8)

_BLOCK_M = 16                # tl.dot 张量核路径;M≤16 全部 pad 到 16
_SHMEM_BUDGET = 96 * 1024    # sm89 每 block shared memory 上限内留余量
_REDUCE_BLOCK = 1024
_ROT_SETS = 36               # 冷口径轮转套数 = 36 层(m6_spike layout 腿同款)
_ROT_SETS_NN = 8             # nn 变体需 [K,N] 拷贝,8 套轮转(>L2 亦充分冷)
_WARM_ROUNDS = 1
_TIME_ROUNDS = 5

_state: dict = {
    "layout": "none",        # "none" | "triton"
    "active": set(),         # 过阀接入的类(set_active_classes 可 A/B 覆写)
    "table": {},             # (cls, bucket) -> "triton" | "cublas"
    "configs": {},           # cls -> (BLOCK_N, BLOCK_K, warps, stages, SPLIT_K)
    "variant": {},           # cls -> (TRANS_B, CACHE) 访存变体(缺省 (0,0))
    "workspaces": {},        # (N, SPLIT_K) -> fp32 [SPLIT_K, 16, N] 常驻
    "shapes": {},            # cls -> (N, K)(warmup 用)
    "adopted": set(),        # 过 ≥5% 阀的类(set_active_classes 的硬边界)
    "device": None,
    "report": None,
}


# ---------- kernels ----------

@triton.jit
def _decode_gemm_kernel(
    x_ptr, w_ptr, out_ptr, partial_ptr,
    M, N, K, K_CHUNK,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TRANS_B: tl.constexpr, CACHE: tl.constexpr,
):
    """C[M, N] = x[M, K] @ W[N, K]^T;bf16 in/out,fp32 累加。

    grid = (cdiv(N, BLOCK_N), SPLIT_K)。W 行主 [N, K] 连续(nn.Linear 布局,
    零存储变换),行内 K 维连续 → 每 program 流水读 BLOCK_N×BLOCK_K 瓦片,
    权重字节恰读一次(纯权重流)。N % BLOCK_N == 0 与 K % (SPLIT_K·BLOCK_K)
    == 0 由 wrapper 断言(两类生产形状均整除),故 W 侧免 mask,仅 M 侧
    mask(BLOCK_M=16 pad 行)。
    TRANS_B:0 = 载 [BLOCK_N, BLOCK_K] 后 tl.trans 进 dot(经典形态);
    1 = 直接按 [BLOCK_K, BLOCK_N] 布局寻址载入(K 维连续,转置折进
    shared swizzle,免显式 trans)——同一批地址,访存变体候选。
    CACHE:W 瓦片访存修饰:0 = plain;1 = .cg(穿 L1 直 L2,流式权重
    无 L1 复用);2 = eviction_policy evict_first(流式早逐出,少挤 L2
    里被全体 program 复用的 x);3 = .cg + evict_first。constexpr 分支
    编译期剪枝,缺省 (0, 0) 与初版逐位同。
    SPLIT_K == 1:直接 bf16 写 out;否则 fp32 写 partials[pid_k, 0:16, tile]
    (行距 BLOCK_M·N,确定性 second-pass 归约交 _splitk_reduce_kernel)。
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_lo = pid_k * K_CHUNK
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    x_ptrs = x_ptr + offs_m[:, None] * K + k_lo + offs_k[None, :]
    if TRANS_B == 0:
        w_ptrs = w_ptr + offs_n[:, None] * K + k_lo + offs_k[None, :]
    else:
        w_ptrs = w_ptr + (k_lo + offs_k)[:, None] + offs_n[None, :] * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_CHUNK, BLOCK_K):
        a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        if TRANS_B == 0:
            if CACHE == 0:
                b = tl.load(w_ptrs)
            elif CACHE == 1:
                b = tl.load(w_ptrs, cache_modifier=".cg")
            elif CACHE == 2:
                b = tl.load(w_ptrs, eviction_policy="evict_first")
            else:
                b = tl.load(w_ptrs, cache_modifier=".cg",
                            eviction_policy="evict_first")
            acc = tl.dot(a, tl.trans(b), acc)
        else:
            if CACHE == 0:
                bt = tl.load(w_ptrs)
            elif CACHE == 1:
                bt = tl.load(w_ptrs, cache_modifier=".cg")
            elif CACHE == 2:
                bt = tl.load(w_ptrs, eviction_policy="evict_first")
            else:
                bt = tl.load(w_ptrs, cache_modifier=".cg",
                             eviction_policy="evict_first")
            acc = tl.dot(a, bt, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    if SPLIT_K == 1:
        out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        p_ptrs = (partial_ptr + pid_k * BLOCK_M * N
                  + offs_m[:, None] * N + offs_n[None, :])
        tl.store(p_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def _splitk_reduce_kernel(
    partial_ptr, out_ptr,
    total, pad_total, split_k,
    BLOCK: tl.constexpr,
):
    """确定性归约:out[m, n] = Σ_s partial[s, m, n](fp32 定序求和)→ bf16。

    partial 布局 [SPLIT_K, BLOCK_M, N](行距 pad_total = BLOCK_M·N;M<16 的
    pad 行永不写入也不读);out 展平 [M·N]。split_k 为 runtime 参数 → 单一
    编译服务全部 SPLIT_K 档。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(split_k):
        acc += tl.load(partial_ptr + s * pad_total + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


# ---------- launch(捕获/重放路径:只查表 + launch,零 host 同步) ----------

def _workspace(N: int, split_k: int, device) -> torch.Tensor:
    key = (N, split_k)
    ws = _state["workspaces"].get(key)
    if ws is None:
        ws = torch.empty(split_k, _BLOCK_M, N, dtype=torch.float32, device=device)
        _state["workspaces"][key] = ws
    return ws


def _launch(x2: torch.Tensor, w: torch.Tensor, cls: str,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """triton 路径:x2 [M, K] 连续、w [N, K] → [M, N] bf16。

    cfg = _state["configs"][cls](装载期 autotune 产物);workspace 装载期
    已分配(重入时 dict 命中,零分配);grid 全 host 侧 int,零 host 同步。
    """
    cfg = _state["configs"][cls]
    block_n, block_k, warps, stages, split_k = cfg
    trans_b, cache = _state["variant"].get(cls, (0, 0))
    M, K = x2.shape
    N = w.shape[0]
    assert K % block_k == 0 and N % block_n == 0 and K % (split_k * block_k) == 0, \
        f"decode_gemm 形状不整除: N={N} K={K} cfg={cfg}"
    if out is None:
        out = torch.empty(M, N, dtype=x2.dtype, device=x2.device)
    grid = (triton.cdiv(N, block_n), split_k)
    if split_k == 1:
        _decode_gemm_kernel[grid](
            x2, w, out, out,          # partial_ptr 占位(SPLIT_K=1 编译期剪除)
            M, N, K, K,
            SPLIT_K=1, BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k,
            TRANS_B=trans_b, CACHE=cache,
            num_warps=warps, num_stages=stages,
        )
    else:
        part = _workspace(N, split_k, x2.device)
        _decode_gemm_kernel[grid](
            x2, w, out, part,
            M, N, K, K // split_k,
            SPLIT_K=split_k, BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k,
            TRANS_B=trans_b, CACHE=cache,
            num_warps=warps, num_stages=stages,
        )
        total = M * N
        _splitk_reduce_kernel[(triton.cdiv(total, _REDUCE_BLOCK),)](
            part, out, total, _BLOCK_M * N, split_k, BLOCK=_REDUCE_BLOCK,
        )
    return out


def triton_gemm(x, w, cls, cfg=None, out=None):
    """强制 triton 路径(单测对拍 / 装载期实测仪;绕过 dispatch 表)。

    x: [..., K](内部展平 [M, K]);w: [N, K];cfg: (BLOCK_N, BLOCK_K,
    warps, stages, SPLIT_K),None = 装载期 tuned config;out: None 或 [M, N]
    (graph 捕获测试用静态缓冲)。返回 [M, N] bf16。
    """
    K = x.shape[-1]
    x2 = x.reshape(x.numel() // K, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    saved = _state["configs"].get(cls)
    if cfg is not None:
        _state["configs"][cls] = tuple(cfg)
    try:
        return _launch(x2, w, cls, out)
    finally:
        if cfg is not None:
            if saved is None:
                _state["configs"].pop(cls, None)
            else:
                _state["configs"][cls] = saved


def linear_dispatch(x, w, cls):
    """调用点形态(qwen3 MLP 三处投影):查 dispatch 表分派。

    layout="none"(缺省)或类未接入 / M 超桶 / 该桶判 cuBLAS → F.linear
    原径(逐字节硬门)。triton 路径仅查表 + launch(零 host 同步)。
    """
    if _state["layout"] != "triton" or cls not in _state["active"]:
        return F.linear(x, w)
    K = w.shape[1]
    M = x.numel() // K
    b = _bucket_for(M)
    if b is None or _state["table"].get((cls, b)) != "triton":
        return F.linear(x, w)
    x2 = x.reshape(M, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = _launch(x2, w, cls)
    return out if x.dim() == 2 else out.view(*x.shape[:-1], w.shape[0])


def _bucket_for(m: int):
    for b in M_BUCKETS:
        if m <= b:
            return b
    return None


# ---------- 装载期:autotune + 冷口径实测 + 接入阀 ----------

def _bench_rot(call, ws, rounds=_TIME_ROUNDS) -> float:
    """轮转计时(m6_spike layout 腿同款):warmup 一整轮 + 计 rounds 轮。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(_WARM_ROUNDS):
        for w in ws:
            call(w)
    torch.cuda.synchronize()
    start.record()
    for _ in range(rounds):
        for w in ws:
            call(w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (rounds * len(ws))   # µs


def _cfg_shmem_ok(cfg) -> bool:
    return (_BLOCK_M + cfg[0]) * cfg[1] * 2 * cfg[3] <= _SHMEM_BUDGET


def _autotune_config(cls, N, K, device, dtype, verbose=True):
    """两段 autotune(装载期;两类单套权重 100.7MB > L2,hot≈cold)。

    段1 数值网格:BLOCK_N ∈ {64,128,256} × BLOCK_K ∈ {64,128,256} ×
    warps ∈ {4,8} × stages ∈ {2,3,4}(shmem 预过滤 ≤96KB,~24 活配置);
    段2 split:段1 top-2 × SPLIT_K {2,4}(整除过滤;两类均开放)。
    迭代账目(iter3-6 探针,见 docs M6 段 5):CACHE/TRANS_B 访存变体、
    PAD_X 免 mask、stages 5/6 深流水、BN=32、SWAP 操作数序 —— 全部
    噪声级(±0.5%)或更差,不入网格(装载期预算 <2s);split>2 对 down
    亦更差(reduce 流量随 SPLIT_K 线性涨)。返回 (BLOCK_N, BLOCK_K,
    warps, stages, SPLIT_K)。
    """
    n_sets = 8
    ws = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(n_sets)]
    x = torch.randn(_BLOCK_M, K, dtype=dtype, device=device)
    grid_cfgs = [c for c in (
        (bn, bk, w, s, 1)
        for bn in (64, 128, 256)
        for bk in (64, 128, 256)
        for w in (4, 8)
        for s in (2, 3, 4)
    ) if _cfg_shmem_ok(c)]
    results, skipped = [], 0

    def _try(cfg):
        nonlocal skipped
        try:
            us = _bench_rot(lambda w_, c=cfg: triton_gemm(x, w_, cls, cfg=c), ws)
            results.append((us, cfg))
            if verbose:
                gb = N * K * 2 / (us * 1e-6) / 1e9
                print(f"T5 AUTOTUNE {cls} cfg={cfg} us={us:.2f} gb={gb:.0f}",
                      flush=True)
        except Exception as e:   # shmem 超限等编译期资源失败:跳过并记账
            skipped += 1
            if verbose:
                print(f"T5 AUTOTUNE {cls} cfg={cfg} SKIPPED {type(e).__name__}",
                      flush=True)

    for cfg in grid_cfgs:
        _try(cfg)
    results.sort()
    if not results:
        raise RuntimeError(f"decode_gemm autotune 全配置失败: {cls}")
    for base in [c for _, c in results[:2]]:
        for sk in (2, 4):
            if K % (sk * base[1]) == 0:
                _try((*base[:4], sk))
    if not results:
        raise RuntimeError(f"decode_gemm autotune 全配置失败: {cls}")
    results.sort()
    best_us, best_cfg = results[0]
    gb = N * K * 2 / (best_us * 1e-6) / 1e9
    print(f"T5 AUTOTUNE_BEST {cls} cfg={best_cfg} us={best_us:.2f} gb={gb:.0f} "
          f"skipped={skipped}", flush=True)
    del ws, x
    torch.cuda.empty_cache()
    return best_cfg


def _measure_class(cls, N, K, buckets, device, dtype, verbose=True):
    """冷口径逐桶实测:triton(tuned cfg)vs cuBLAS 三变体(tn/nn/pad16)。

    tn/pad16 用 36 套轮转(_ROT_SETS,= 36 层;spike 同款),nn 用 8 套
    ([K,N] 拷贝贵;8 套 = 805MB > L2 亦充分冷)。返回 (tri, cub) 逐桶字典。
    """
    torch.manual_seed(0)
    ws = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(_ROT_SETS)]
    tri, tn, pad16 = {}, {}, {}
    for m in buckets:
        x = torch.randn(m, K, dtype=dtype, device=device)
        tri[m] = _bench_rot(lambda w: triton_gemm(x, w, cls), ws)
        tn[m] = _bench_rot(lambda w: F.linear(x, w), ws)
        if m != _BLOCK_M:
            xp = torch.zeros(_BLOCK_M, K, dtype=dtype, device=device)
            xp[:m] = x
            pad16[m] = _bench_rot(lambda w: F.linear(xp, w)[:m], ws)
        del x
    del ws
    torch.cuda.empty_cache()
    nn = {}
    ws_t = [w.t().contiguous() for w in (
        torch.randn(N, K, dtype=dtype, device=device)
        for _ in range(_ROT_SETS_NN))]
    for m in buckets:
        x = torch.randn(m, K, dtype=dtype, device=device)
        nn[m] = _bench_rot(lambda w_t: torch.mm(x, w_t), ws_t)
        del x
    del ws_t
    torch.cuda.empty_cache()
    for m in buckets:
        variants = [tn[m], nn.get(m), pad16.get(m)]
        best = min(v for v in variants if v is not None)
        win = (best / tri[m] - 1) * 100
        gb = N * K * 2 / (tri[m] * 1e-6) / 1e9
        if verbose:
            print(f"T5 COLD {cls} m={m} triton={tri[m]:.2f}us/{gb:.0f}GBps "
                  f"tn={tn[m]:.2f} nn={nn.get(m, float('nan')):.2f} "
                  f"pad16={pad16.get(m, float('nan')):.2f} "
                  f"best_cublas={best:.2f} win_vs_best={win:+.1f}%", flush=True)
    return tri, {"tn": tn, "nn": nn, "pad16": pad16}


def _warmup_table() -> None:
    """表内 triton 桶逐个发射一次:编译/工作区/表查全链路就位(捕获前)。"""
    ws = {cls: torch.randn(N, K, dtype=torch.bfloat16, device=_state["device"])
          for cls, (N, K) in _state["shapes"].items()}
    for (cls, b), choice in sorted(_state["table"].items()):
        if choice != "triton" or cls not in _state["active"]:
            continue
        N, K = _state["shapes"][cls]
        x = torch.zeros(b, K, dtype=torch.bfloat16, device=_state["device"])
        out = _launch(x, ws[cls], cls)
        del x, out
    del ws
    torch.cuda.synchronize()


def configure(layout, classes, *, device="cuda:0", dtype=torch.bfloat16,
              buckets=M_BUCKETS, valve=0.05, verbose=True) -> dict:
    """装载期入口(Engine 构造 / graph 捕获之前调用一次)。

    classes: {类名: (N, K)};layout="none"(缺省)时零 GPU 工作(逐字节
    硬门)。triton:autotune → 冷口径逐桶实测 → 接入阀(≥5% 类级,对照
    cuBLAS 最优变体)→ 表内逐桶择优(triton vs 生产可达的 cuBLAS TN)
    → warmup。返回 report(docs 账目直接引用)。
    """
    reset()
    _state["layout"] = layout
    if layout != "triton":
        return {"layout": "none"}
    _state["device"] = device
    _state["shapes"] = dict(classes)
    report = {"layout": "triton", "valve": valve, "classes": {}}
    for cls, (N, K) in classes.items():
        cfg = _autotune_config(cls, N, K, device, dtype, verbose=verbose)
        _state["configs"][cls] = cfg
        tri, cub = _measure_class(cls, N=N, K=K, buckets=buckets,
                                  device=device, dtype=dtype, verbose=verbose)
        rows = {}
        adopted = False
        for m in buckets:
            variants = [cub["tn"][m], cub["nn"].get(m), cub["pad16"].get(m)]
            best = min(v for v in variants if v is not None)
            adopted_m = tri[m] <= best * (1 - valve)
            adopted = adopted or adopted_m
            pick = "triton" if tri[m] < cub["tn"][m] else "cublas"
            rows[m] = {
                "triton_us": tri[m], "tn_us": cub["tn"][m],
                "nn_us": cub["nn"].get(m), "pad16_us": cub["pad16"].get(m),
                "best_cublas_us": best,
                "win_vs_best_pct": (best / tri[m] - 1) * 100,
                "adopted_m": adopted_m, "table": pick,
            }
            # 全桶记录(输桶 "cublas" 行可查;接入与否由 active 门控)
            _state["table"][(cls, m)] = pick
        if adopted:
            _state["active"].add(cls)
            _state["adopted"].add(cls)
        report["classes"][cls] = {"config": cfg, "adopted": adopted, "rows": rows}
        if verbose:
            print(f"T5 VALVE {cls} adopted={adopted} "
                  f"active={sorted(_state['active'])}", flush=True)
    _warmup_table()
    _state["report"] = report
    return report


def set_active_classes(classes) -> None:
    """A/B 逐类开关(configure 之后、Engine 构造之前;生产路径不用)。

    只能在已接入(active)的类里收窄/选腿;传入未接入类不生效。
    """
    _state["active"] = set(classes) & _state["adopted"] if classes else set()


def reset() -> None:
    """清空全部状态(测试间隔离;常驻 workspace 一并释放)。"""
    _state.clear()
    _state.update({
        "layout": "none", "active": set(), "table": {}, "configs": {},
        "variant": {}, "workspaces": {}, "shapes": {}, "adopted": set(),
        "device": None, "report": None,
    })
