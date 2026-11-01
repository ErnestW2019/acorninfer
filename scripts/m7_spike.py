"""M7 段 0 spike:W8/W4 dequant-GEMM 变体孤立微基准(锚修正复核 + 段 A GO 数据)。

用法(单卡,CUDA_VISIBLE_DEVICES=1,跑前 nvidia-smi 应 ~4MiB;前台 timeout ≥ 1800000):
  .venv/bin/python scripts/m7_spike.py w8 [--classes kv,qo,gate_up,down,lm_head]
  .venv/bin/python scripts/m7_spike.py w4 [--classes ...]

生产源码零触碰:探索级 triton kernel 直接写在本文件(_w8_kernel/_w4_kernel);
`ops/decode_gemm.py` 导入复用(非 protected,本任务零改动):`_bench_rot`(冷
口径轮转计时:warmup 一整轮 + 计 5 轮 ÷ 轮×套)、`_ROT_SETS=36`、`_cfg_shmem_ok`
(shmem ≤96KB 预过滤)、`_workspace`(fp32 partials 常驻,装载期分配地址冻结)
与 `_splitk_reduce_kernel`(确定性 second-pass 归约,零 atomic);`_autotune_config`
为 bf16 特化(gemm/bf16 权重/常量网格),此处按其 **两段式模式**(段1 数值网格
→ 段2 split 候选)重实现量化版。

变体面(plan Task 0 Step 4):
- W8:BLOCK_N{64,128,256} × BLOCK_K{64,128} × warps{4,8} × stages{2,3,4};
  核内 dequant:int8 load → fp32 ×scale(行广播,循环不变提出)→ bf16 →
  tl.dot fp32 累加;BLOCK_M=16(M≤16 pad;M∈{1,8} 两档,M runtime 参数零重编译)。
- W4:同网格子集 × unpack 两法(u0 = 位运算移位掩码:偶/奇 nibble 拆半、x 偶/奇
  列跨步加载、两次半-K tl.dot,零 shuffle;u1 = tl.interleave 织回 [BN,BK] 单
  dot)× 组 scale 两形态(g0 = 组对齐:k0 整 128,BLOCK 内组号标量、scale 行广播;
  g1 = gather 索引 (k0+i)//128 逐元素,允许任意 BLOCK_K)。
- 五档类形状(生产独立模块口径,M6 的 gate_up 24576 为聚合口径不用):
  qo (4096,4096) / kv (1024,4096) / gate_up (12288,4096) / down (4096,12288) /
  lm_head (151936,4096);每步调用数 72/72/72/36/1(q,o,k,v,gate,up 各 36 次/步)。
- SPLIT_K:N-不足类 kv(N=1024)与 down(N=4096)照 decode_gemm 先例评估
  (段2 top-2 × sk{2,4[,8]},fp32 partials + 确定性 reduce,atomic 能免则免)。
- 冷口径:_bench_rot 36 套轮转(>72MB L2 强制 HBM);lm_head 单套 622MB
  (W8)/311MB(W4),36 套 = 22.4GB 超显存 → 8 套轮转(5GB,单套 > L2 70×,
  充分冷,如实注记)。
- gb 口径 = 量化权重字节(含元数据)/µs:W8 = N·K + 2N(int8 + fp16 scale);
  W4 = N·K/2 + 4·N·(K/128)(uint8 打包 + fp16 scale/zp)。roofline = 4090
  峰值 1.008TB/s,η = gb/1008。
- 汇总:逐档最优 η 表 + 每步字节加权合计 η(vs 推导假设 W8 0.85 / W4 0.75,
  偏离 >10% → 锚修正条款线,判定留控制器)+ 胜出 variant 全记录。
- capture 实测:逐 (fmt, class, M) 胜出配置进 torch.cuda.CUDAGraph 捕获 +
  重放 torch.equal 逐位钉(M6 T5 协议)+ eager launch 逐位对照 +
  set_sync_debug_mode(2) 验 launch 零 host 同步。

输出行前缀:QSPIKE(逐变体)/ QSPIKE_SKIP(编译期资源失败,记账不修)/
QSPIKE_CHECK(kernel 形态数值对拍,spike 级宽松门,非 TOL 标定)/
QSPIKE_BEST / QSPIKE_CAP / QSPIKE_SUMMARY / QSPIKE_ETA / QSPIKE_ANCHOR。
stdout 逐字进 docs;探索级 kernel 的坏数据是微基准的目的,不修到好看。
"""

from __future__ import annotations

import argparse
import datetime
import sys

import torch
import triton
import triton.language as tl

from acorninfer.ops.decode_gemm import (
    _BLOCK_M,
    _REDUCE_BLOCK,
    _ROT_SETS,
    _bench_rot,
    _cfg_shmem_ok,
    _splitk_reduce_kernel,
    _workspace,
)
from acorninfer.quant.quantize import (
    dequant_w4,
    dequant_w8,
    pack_w4,
    quantize_w4,
    quantize_w8,
    unpack_w4,
)

ROOFLINE_GBPS = 1008.0          # 4090 峰值 1.008TB/s(权重字节口径)
ROT_LM = 8                      # lm_head 轮转套数(36 套超显存,见 docstring)
MS = (1, 8)                     # decode 口径两档(BLOCK_M=16 全 pad)
SPLIT_CLASSES = {"kv": (2, 4, 8), "down": (2, 4)}   # N-不足类照 decode_gemm 先例
# spike 级宽松门(正式 TOL Task 1 标定写死)。atol 口径:kernel 侧 dequant 后
# round 到 bf16(每元素 ~2⁻⁹ 相对),fp32 参考不 round → K=4096 点积累计误差
# σ ≈ √K·2⁻⁹ ≈ 0.13(输出尺度 |ref|~64 的 0.2%),max ~4σ ≈ 0.5;atol=1.0 =
# 2× 实测 max(2026-11-01 kv 类首标,见 QSPIKE_CHECK 行);真错(索引/mask)差
# O(10) 起,不可漏网。与 decode_gemm TOL_BF16_GEMM(atol 1e-2,纯归约序)口径
# 不同:那里 kernel/ref 同吃 bf16 w,这里参考是 fp32 dequant,w 舍入是本 kernel
# 数值域的真实组成(Task 1 TOL 对拍锚按此标定)。
SPIKE_TOL = dict(rtol=2e-2, atol=1.0)

CLASSES = {                     # 生产五档(独立模块口径)
    "qo": (4096, 4096),         # q_proj / o_proj
    "kv": (1024, 4096),         # k_proj / v_proj
    "gate_up": (12288, 4096),   # gate_proj / up_proj 各一次
    "down": (4096, 12288),
    "lm_head": (151936, 4096),
}
CALLS_PER_STEP = {"qo": 72, "kv": 72, "gate_up": 72, "down": 36, "lm_head": 1}
ASSUME_ETA = {"w8": 0.85, "w4": 0.75}   # 锚推导假设(权重字节口径)


def qbytes(fmt: str, n: int, k: int) -> int:
    """量化权重字节(含元数据;gb/η 的分子)。"""
    if fmt == "w8":
        return n * k + 2 * n
    return n * k // 2 + 4 * n * (k // 128)


# ---------- 探索级 kernels(spike 内;生产 kernel Task 1 定档) ----------

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
def _w8_kernel(x_ptr, q_ptr, s_ptr, out_ptr, part_ptr,
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


@triton.jit
def _w4_kernel(x_ptr, p_ptr, s_ptr, zp_ptr, out_ptr, part_ptr,
               M, N, K, K_CHUNK,
               SPLIT_K: tl.constexpr, BM: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr,
               UNPACK: tl.constexpr, GMODE: tl.constexpr):
    """C[M,N] = x[M,K] @ dequant_w4(unpack(p), s, zp)[N,K]^T。

    p uint8 [N,K/2] 打包(**低 nibble = 偶数列**,quantize.pack_w4 同约定);
    s/zp fp16 [N, K/128] per-group。
    UNPACK:1 = tl.interleave(p&0xF, p>>4) 织回 [BN,BK] 单 dot(x 连续);
            0 = 位运算移位掩码拆半:偶/奇 nibble 各 [BN,BK/2] dequant,x 偶/奇
                列跨步加载,两次半-K tl.dot,零 shuffle。
    GMODE:  0 = 组对齐(k0 整 128/块内组号标量 → scale 行广播);
            1 = gather 索引 (k0+i)//128 逐元素(允许任意 BLOCK_K)。
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_lo = pid_k * K_CHUNK
    G = K // 128
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    k0 = k_lo
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if UNPACK == 1:
        offs_k = tl.arange(0, BK)
        x_ptrs = x_ptr + offs_m[:, None] * K + k_lo + offs_k[None, :]
        p_ptrs = (p_ptr + offs_n[:, None] * (K // 2) + k_lo // 2
                  + tl.arange(0, BK // 2)[None, :])
        for _ in range(0, K_CHUNK, BK):
            a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            pb = tl.load(p_ptrs)
            q = tl.interleave(pb & 0xF, pb >> 4)        # 偶列 = 低 nibble ✓
            if GMODE == 0:
                s = tl.load(s_ptr + offs_n * G + k0 // 128).to(tl.float32)
                zp = tl.load(zp_ptr + offs_n * G + k0 // 128).to(tl.float32)
                w = (q.to(tl.float32) * s[:, None] + zp[:, None]).to(tl.bfloat16)
            else:
                gidx = (k0 + offs_k) // 128
                s = tl.load(s_ptr + offs_n[:, None] * G + gidx[None, :]).to(tl.float32)
                zp = tl.load(zp_ptr + offs_n[:, None] * G + gidx[None, :]).to(tl.float32)
                w = (q.to(tl.float32) * s + zp).to(tl.bfloat16)
            acc = tl.dot(a, tl.trans(w), acc)
            x_ptrs += BK
            p_ptrs += BK // 2
            k0 += BK
    else:
        offs_kh = tl.arange(0, BK // 2)
        xe_ptrs = x_ptr + offs_m[:, None] * K + k_lo + 2 * offs_kh[None, :]
        xo_ptrs = xe_ptrs + 1
        p_ptrs = (p_ptr + offs_n[:, None] * (K // 2) + k_lo // 2
                  + offs_kh[None, :])
        for _ in range(0, K_CHUNK, BK):
            pb = tl.load(p_ptrs)
            lo = (pb & 0xF).to(tl.float32)
            hi = (pb >> 4).to(tl.float32)
            if GMODE == 0:
                s = tl.load(s_ptr + offs_n * G + k0 // 128).to(tl.float32)
                zp = tl.load(zp_ptr + offs_n * G + k0 // 128).to(tl.float32)
                we = (lo * s[:, None] + zp[:, None]).to(tl.bfloat16)
                wo = (hi * s[:, None] + zp[:, None]).to(tl.bfloat16)
            else:
                ge = (k0 + 2 * offs_kh) // 128
                go = (k0 + 2 * offs_kh + 1) // 128
                se = tl.load(s_ptr + offs_n[:, None] * G + ge[None, :]).to(tl.float32)
                ze = tl.load(zp_ptr + offs_n[:, None] * G + ge[None, :]).to(tl.float32)
                so = tl.load(s_ptr + offs_n[:, None] * G + go[None, :]).to(tl.float32)
                zo = tl.load(zp_ptr + offs_n[:, None] * G + go[None, :]).to(tl.float32)
                we = (lo * se + ze).to(tl.bfloat16)
                wo = (hi * so + zo).to(tl.bfloat16)
            xe = tl.load(xe_ptrs, mask=m_mask[:, None], other=0.0)
            xo = tl.load(xo_ptrs, mask=m_mask[:, None], other=0.0)
            acc = tl.dot(xe, tl.trans(we), acc)
            acc = tl.dot(xo, tl.trans(wo), acc)
            xe_ptrs += BK
            xo_ptrs += BK
            p_ptrs += BK // 2
            k0 += BK
    _quant_store(acc, out_ptr, part_ptr, pid_k, BM, N, SPLIT_K, offs_m, offs_n, m_mask)


# ---------- launch(与 decode_gemm 同款:只查 cfg + launch,零 host 同步) ----------

def launch_w8(x, q, s, out, cfg, N, K):
    bn, bk, warps, stages, sk = cfg
    M = x.shape[0]
    assert N % bn == 0 and K % (sk * bk) == 0, f"w8 形状不整除 N={N} K={K} cfg={cfg}"
    if sk == 1:
        _w8_kernel[(N // bn, 1)](x, q, s, out, out, M, N, K, K,
                                 SPLIT_K=1, BM=_BLOCK_M, BN=bn, BK=bk,
                                 num_warps=warps, num_stages=stages)
    else:
        part = _workspace(N, sk, x.device)      # 常驻缓存,首调分配(捕获前)
        _w8_kernel[(N // bn, sk)](x, q, s, out, part, M, N, K, K // sk,
                                  SPLIT_K=sk, BM=_BLOCK_M, BN=bn, BK=bk,
                                  num_warps=warps, num_stages=stages)
        total = M * N
        _splitk_reduce_kernel[(triton.cdiv(total, _REDUCE_BLOCK),)](
            part, out, total, _BLOCK_M * N, sk, BLOCK=_REDUCE_BLOCK)
    return out


def launch_w4(x, p, s, zp, out, cfg, N, K):
    bn, bk, warps, stages, unpack, gmode, sk = cfg
    M = x.shape[0]
    assert N % bn == 0 and K % (sk * bk) == 0, f"w4 形状不整除 N={N} K={K} cfg={cfg}"
    if unpack == 1:
        if sk == 1:
            _w4_kernel[(N // bn, 1)](x, p, s, zp, out, out, M, N, K, K,
                                     SPLIT_K=1, BM=_BLOCK_M, BN=bn, BK=bk,
                                     UNPACK=1, GMODE=gmode,
                                     num_warps=warps, num_stages=stages)
        else:
            part = _workspace(N, sk, x.device)
            _w4_kernel[(N // bn, sk)](x, p, s, zp, out, part, M, N, K, K // sk,
                                      SPLIT_K=sk, BM=_BLOCK_M, BN=bn, BK=bk,
                                      UNPACK=1, GMODE=gmode,
                                      num_warps=warps, num_stages=stages)
            total = M * N
            _splitk_reduce_kernel[(triton.cdiv(total, _REDUCE_BLOCK),)](
                part, out, total, _BLOCK_M * N, sk, BLOCK=_REDUCE_BLOCK)
        return out
    # UNPACK=0(偶/奇拆半):K_CHUNK 仍整除 BK;g0 附加约束 chunk 整 128(host 端过滤)
    if sk == 1:
        _w4_kernel[(N // bn, 1)](x, p, s, zp, out, out, M, N, K, K,
                                 SPLIT_K=1, BM=_BLOCK_M, BN=bn, BK=bk,
                                 UNPACK=0, GMODE=gmode,
                                 num_warps=warps, num_stages=stages)
    else:
        part = _workspace(N, sk, x.device)
        _w4_kernel[(N // bn, sk)](x, p, s, zp, out, part, M, N, K, K // sk,
                                  SPLIT_K=sk, BM=_BLOCK_M, BN=bn, BK=bk,
                                  UNPACK=0, GMODE=gmode,
                                  num_warps=warps, num_stages=stages)
        total = M * N
        _splitk_reduce_kernel[(triton.cdiv(total, _REDUCE_BLOCK),)](
            part, out, total, _BLOCK_M * N, sk, BLOCK=_REDUCE_BLOCK)
    return out


LAUNCH = {"w8": launch_w8, "w4": launch_w4}


def variant_str(fmt, cfg) -> str:
    if fmt == "w8":
        bn, bk, w, st, sk = cfg
        return f"bn{bn}_bk{bk}_w{w}_s{st}_sk{sk}"
    bn, bk, w, st, u, g, sk = cfg
    return f"bn{bn}_bk{bk}_w{w}_s{st}_u{u}_g{g}_sk{sk}"


# ---------- 网格(shmem 预过滤照 _cfg_shmem_ok;N % BN 整除过滤) ----------

def grid_w8(N):
    out = []
    for bn in (64, 128, 256):
        if N % bn != 0:
            continue
        for bk in (64, 128):
            for w in (4, 8):
                for st in (2, 3, 4):
                    if not _cfg_shmem_ok((bn, bk, w, st, 1)):
                        continue
                    out.append((bn, bk, w, st, 1))
    return out


def grid_w4(N):
    out = []
    for bn in (64, 128, 256):
        if N % bn != 0:
            continue
        for bk in (64, 128):
            for w in (4, 8):
                for st in (2, 3, 4):
                    if not _cfg_shmem_ok((bn, bk, w, st, 1)):
                        continue
                    for u in (0, 1):
                        for g in (0, 1):
                            out.append((bn, bk, w, st, u, g, 1))
    return out


GRID = {"w8": grid_w8, "w4": grid_w4}


# ---------- 数值对拍门(spike 级宽松;正式 TOL Task 1 标定写死) ----------

def check_forms(device):
    """每 kernel 形态一次 kv 小类对拍(fp32 dequant 参考实现为锚)。"""
    n, k = CLASSES["kv"]
    torch.manual_seed(23)
    x = torch.randn(8, k, device=device).to(torch.bfloat16)
    w = torch.randn(n, k, device=device).to(torch.bfloat16)
    q8, s8 = quantize_w8(w.cpu())
    q8, s8 = q8.to(device), s8.to(device)
    ref8 = x.float() @ dequant_w8(q8, s8).T
    q4, s4, zp4 = quantize_w4(w.cpu())
    q4, s4, zp4 = q4.to(device), s4.to(device), zp4.to(device)
    p4 = pack_w4(q4).contiguous()
    ref4 = x.float() @ dequant_w4(q4, s4, zp4).T
    out = torch.empty(8, n, dtype=torch.bfloat16, device=device)
    cases = [("w8", (128, 128, 4, 3, 1), ref8, q8, s8, None),
             ("w8", (128, 64, 4, 3, 4), ref8, q8, s8, None),
             ("w8", (128, 64, 4, 3, 8), ref8, q8, s8, None)]
    for bk in (64, 128):
        for u in (0, 1):
            for g in (0, 1):
                cases.append(("w4", (128, bk, 4, 3, u, g, 1), ref4, p4, s4, zp4))
    cases.append(("w4", (128, 64, 4, 3, 1, 0, 2), ref4, p4, s4, zp4))
    for fmt, cfg, ref, wp, s_, zp_ in cases:
        _call(fmt, LAUNCH[fmt], x, (wp, s_, zp_), out, cfg, n, k)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref, **SPIKE_TOL)
        maxabs = (out.float() - ref).abs().max().item()
        print(f"QSPIKE_CHECK fmt={fmt} form={variant_str(fmt, cfg)} "
              f"ok=True maxabs={maxabs:.4f} tol={SPIKE_TOL}", flush=True)
    del x, w, q8, s8, q4, s4, zp4, p4, ref8, ref4, out
    torch.cuda.empty_cache()


# ---------- capture 实测(M6 T5 协议:捕获 + 重放逐位 + 零 host 同步) ----------

def capture_check(fmt, cls, n, k, m, cfg, w0, s0, zp0, device):
    launch = LAUNCH[fmt]
    x = torch.randn(m, k, device=device).to(torch.bfloat16)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    _call(fmt, launch, x, (w0, s0, zp0), out, cfg, n, k)   # 预编译 + workspace 就位
    torch.cuda.synchronize()
    # launch 路径零 host 同步(set_sync_debug_mode(2) = 出错即抛)
    prev = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode(2)
        _call(fmt, launch, x, (w0, s0, zp0), out, cfg, n, k)
        sync_dbg = "ok"
    except Exception as e:                           # 如实记录,不粉饰
        sync_dbg = f"FAIL({type(e).__name__})"
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    # side-stream warmup ×3 → 捕获(T5 协议;ws[0] 为静态缓冲,类收尾前存活)
    sstr = torch.cuda.Stream()
    sstr.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sstr):
        for _ in range(3):
            _call(fmt, launch, x, (w0, s0, zp0), out, cfg, n, k)
    torch.cuda.current_stream().wait_stream(sstr)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _call(fmt, launch, x, (w0, s0, zp0), out, cfg, n, k)
    g.replay()
    torch.cuda.synchronize()
    r1 = out.clone()
    g.replay()
    torch.cuda.synchronize()
    replay_bitwise = torch.equal(r1, out)
    out_e = torch.empty_like(out)
    _call(fmt, launch, x, (w0, s0, zp0), out_e, cfg, n, k)   # eager 同输入对照
    torch.cuda.synchronize()
    eager_bitwise = torch.equal(r1, out_e)
    note = "ref=skip(lm_head fp32 参考省略,bitwise 钉 + kv CHECK 已覆盖)"
    ref_tag = "skip"
    if cls != "lm_head":                             # fp32 参考锚(spike 宽松)
        ref = _ref_out(fmt, x, w0, s0, zp0)
        torch.testing.assert_close(r1.float(), ref, **SPIKE_TOL)
        maxabs = (r1.float() - ref).abs().max().item()
        ref_tag = f"maxabs={maxabs:.4f}"
        note = f"ref=ok({ref_tag})"
    print(f"QSPIKE_CAP fmt={fmt} class={cls} M={m} variant={variant_str(fmt, cfg)} "
          f"capture=ok replay_bitwise={replay_bitwise} eager_bitwise={eager_bitwise} "
          f"sync_debug={sync_dbg} {note}", flush=True)
    del x, out, out_e, g
    torch.cuda.empty_cache()


def _ref_out(fmt, x, w0, s0, zp0):
    """fp32 参考输出(w0 为量化存储,经 dequant 参考实现还原;W4 先 unpack_w4)。"""
    if fmt == "w8":
        w_f = dequant_w8(w0, s0)
    else:
        w_f = dequant_w4(unpack_w4(w0), s0, zp0)
    return x.float() @ w_f.T


# ---------- 逐类 sweep(两段式:段1 数值网格 → 段2 split 候选) ----------

def build_sets(fmt, n, k, n_sets, device):
    """轮转套:randn 权重过**真实 quantize 函数**(分布逼真:dequant 值 ~N(0,1),
    capture 的 fp32 参考对拍 atol 口径才成立;字节流随机性对带宽无差)。

    显存账:峰值 = 1 套瞬态(bf16 w + fp32 w32 + 量化产物)≈ 4.4GB(lm_head)
    + 已存套;逐套构建即弃 w。
    """
    sets = []
    for _ in range(n_sets):
        w = torch.randn(n, k, device=device).to(torch.bfloat16)
        if fmt == "w8":
            q, s = quantize_w8(w)
            sets.append((q, s))
        else:
            q, s, zp = quantize_w4(w)
            sets.append((pack_w4(q).contiguous(), s, zp))
        del w
    return sets


def sweep_class(fmt, cls, device):
    n, k = CLASSES[cls]
    n_sets = ROT_LM if cls == "lm_head" else _ROT_SETS
    torch.manual_seed(11)
    ws = build_sets(fmt, n, k, n_sets, device)
    best = {}
    for m in MS:
        x = torch.randn(m, k, device=device).to(torch.bfloat16)
        out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
        launch = LAUNCH[fmt]
        results, skipped = [], 0
        for cfg in GRID[fmt](n):
            try:
                us = _bench_rot(
                    lambda t, c=cfg: _call(fmt, launch, x, t, out, c, n, k), ws)
            except Exception as e:
                skipped += 1
                print(f"QSPIKE_SKIP fmt={fmt} variant={variant_str(fmt, cfg)} "
                      f"class={cls} M={m} {type(e).__name__}", flush=True)
                continue
            b = qbytes(fmt, n, k)
            gb = b / (us * 1e-6) / 1e9
            print(f"QSPIKE {fmt} {variant_str(fmt, cfg)} {cls} M={m} "
                  f"us={us:.2f} gb={gb:.1f}", flush=True)
            results.append((us, cfg))
        # 段2:split 候选(N-不足类照 decode_gemm 先例;top-2 × sk)
        if cls in SPLIT_CLASSES:
            results.sort()
            for base in [c for _, c in results[:2]]:
                for sk in SPLIT_CLASSES[cls]:
                    if k % (sk * base[1]) != 0:
                        continue
                    if fmt == "w4" and base[5] == 0 and (k // sk) % 128 != 0:
                        continue   # g0 组对齐:K_CHUNK 须整 128
                    cfg = (*base[:-1], sk) if fmt == "w8" else (*base[:-1], sk)
                    try:
                        us = _bench_rot(lambda t, c=cfg: _call(fmt, launch, x, t, out, c, n, k), ws)
                    except Exception as e:
                        skipped += 1
                        print(f"QSPIKE_SKIP fmt={fmt} variant={variant_str(fmt, cfg)} "
                              f"class={cls} M={m} {type(e).__name__}", flush=True)
                        continue
                    b = qbytes(fmt, n, k)
                    gb = b / (us * 1e-6) / 1e9
                    print(f"QSPIKE {fmt} {variant_str(fmt, cfg)} {cls} M={m} "
                          f"us={us:.2f} gb={gb:.1f}", flush=True)
                    results.append((us, cfg))
        results.sort()
        best_us, best_cfg = results[0]
        gb = qbytes(fmt, n, k) / (best_us * 1e-6) / 1e9
        eta = gb / ROOFLINE_GBPS
        print(f"QSPIKE_BEST fmt={fmt} class={cls} M={m} us={best_us:.2f} "
              f"gb={gb:.1f} eta={eta:.3f} variant={variant_str(fmt, best_cfg)} "
              f"skipped={skipped}", flush=True)
        capture_check(fmt, cls, n, k, m, best_cfg, ws[0][0], ws[0][1],
                      ws[0][2] if fmt == "w4" else None, device)
        best[m] = (best_us, best_cfg)
        del x, out
    del ws
    torch.cuda.empty_cache()
    return best


def _call(fmt, launch, x, t, out, cfg, n, k):
    if fmt == "w8":
        launch(x, t[0], t[1], out, cfg, n, k)
    else:
        launch(x, t[0], t[1], t[2], out, cfg, n, k)


# ---------- 汇总(逐档 η 表 + 每步字节加权 vs 推导假设) ----------

def summarize(fmt, all_best, classes):
    partial = set(classes) != set(CLASSES)
    step_bytes = sum(CALLS_PER_STEP[c] * qbytes(fmt, *CLASSES[c]) for c in classes)
    tag = " partial=classes=" + ",".join(classes) if partial else ""
    for m in MS:
        t_step = sum(CALLS_PER_STEP[c] * all_best[c][m][0] for c in classes)
        for c in classes:
            us, cfg = all_best[c][m]
            b = qbytes(fmt, *CLASSES[c])
            gb = b / (us * 1e-6) / 1e9
            print(f"QSPIKE_SUMMARY fmt={fmt} class={c} M={m} us={us:.2f} "
                  f"gb={gb:.1f} eta={gb / ROOFLINE_GBPS:.3f} "
                  f"variant={variant_str(fmt, cfg)}", flush=True)
        eta_w = step_bytes / (t_step * 1e-6) / 1e9 / ROOFLINE_GBPS
        assume = ASSUME_ETA[fmt]
        dev = (eta_w / assume - 1) * 100
        print(f"QSPIKE_ETA fmt={fmt} M={m}{tag} weighted_eta={eta_w:.3f} "
              f"weighted_gbps={step_bytes / (t_step * 1e-6) / 1e9:.1f} "
              f"assume={assume} dev_pct={dev:+.1f} "
              f"step_bytes_gb={step_bytes / 1e9:.3f} "
              f"step_time_ms={t_step / 1000:.3f} "
              f"(roofline={ROOFLINE_GBPS:.0f}GB/s 权重字节口径)", flush=True)
    devs = []
    for m in MS:
        t_step = sum(CALLS_PER_STEP[c] * all_best[c][m][0] for c in classes)
        eta_w = (step_bytes / (t_step * 1e-6) / 1e9) / ROOFLINE_GBPS
        devs.append((eta_w / ASSUME_ETA[fmt] - 1) * 100)
    print(f"QSPIKE_ANCHOR fmt={fmt}{tag} dev_pct={min(devs):+.1f}..{max(devs):+.1f} "
          f"assume={ASSUME_ETA[fmt]} threshold=±10% "
          f"(>10% → 锚修正条款一次;判定留控制器)", flush=True)


def run_leg(fmt, classes):
    dev = "cuda:0"
    print(f"QSPIKE_HEADER leg={fmt} date={datetime.date.today()} "
          f"torch={torch.__version__} triton={triton.__version__} "
          f"dev={torch.cuda.get_device_name(0)} "
          f"rot_sets={_ROT_SETS}(lm_head={ROT_LM}) block_m={_BLOCK_M} "
          f"ms={MS} roofline_gbps={ROOFLINE_GBPS:.0f} "
          f"classes={','.join(classes)}", flush=True)
    check_forms(dev)
    all_best = {}
    for cls in classes:
        all_best[cls] = sweep_class(fmt, cls, dev)
    summarize(fmt, all_best, classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["w8", "w4"])
    ap.add_argument("--classes", default="kv,qo,gate_up,down,lm_head")
    args = ap.parse_args()
    classes = args.classes.split(",")
    unknown = [c for c in classes if c not in CLASSES]
    if unknown:
        sys.exit(f"未知类:{unknown}(合法:{list(CLASSES)})")
    run_leg(args.leg, classes)


if __name__ == "__main__":
    main()
