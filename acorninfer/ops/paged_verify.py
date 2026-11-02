"""M8 Task 2:分页 verify attention —— paged_decode_attention 的 m 位窗口推广(只读派生)。

源 = ops/paged_attention.py(保护文件,零改动):combine kernel 与数值路径
逐行同源,combine kernel 直接 import 复用;split kernel 新写,唯一本质改动 =
每行读上限的行依赖化。eager-only(v1 spec 模式不走 graph,无 capture 要求)。

语义(vs decode kernel,逐条):
1. q 形状 [B,H,1,D] → [B,m,H,D](m = spec_num + 1,全批统一):展平 query 行
   (b, j),grid 第 0 维 B → B*m;row = b*m + j。
2. 每行可见上限(绝对位置数):end_j = seq_lens[b] − (m − 1 − j) = len + j + 1
   (j 位可见历史 len + 窗口前 j+1 位;窗口内因果)。seq_lens 契约 = 传 len+m
   (含本步全部窗口位)。decode 的 L ≡ verify 的 j = m−1 行退化情形。
3. 页遍历(start/end 页对齐分片、尾块 slot < end−t mask)、k/v 偏移公式
   (((page*P+slot)*KV+kv)*D+d)、online-softmax 两常数更新、空 split 存
   (−inf,0,0)、combine 的 β=exp(m_s−m_new) 合并 —— 全部照原结构逐行派生,
   无数值路径改动;行不变式 L_row = len+j+1 ≥ 1 → split 0 非空 → combine
   首迭代 m_new 有限,与原 kernel 同源(L ≥ 1)。
4. k/v 写入不在本 kernel:调用方 scatter_verify(cache/pager.py,M8 Task 1)
   已把 m 个窗口位写进页,本 kernel 只读页;死槽(接受后 set_lens 回退的
   len+a+1..len+m−1)由下步复写,读侧按 lens 截断天然不可见。

数值策略照原 kernel:q/k/v load 后全程 fp32 计算,仅 combine 末尾 cast 回
输出 dtype;对拍容差按实测标定(tests/test_paged_verify.py TOL_VERIFY,
标定表见 docs/benchmarks.md M8 段 2)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# combine kernel 与 SM 启发式原样复用保护原件(paged_attention.py 零改动)
from acorninfer.ops.paged_attention import _paged_attn_combine_kernel

__all__ = ["paged_verify_attention"]


@triton.jit
def _paged_verify_split_kernel(
    q_ptr,          # [B*M*H, D](行序 (b, j, h))
    k_ptr,          # [NB, P, KV, D]
    v_ptr,
    tables_ptr,     # [B, MP] int32
    lens_ptr,       # [B] int32(= len + m,含窗口全部 m 位)
    acc_ptr,        # [B*M*H, S, D] fp32
    m_ptr, l_ptr,   # [B*M*H, S] fp32
    scale,
    MP,             # tables 行宽(运行期 stride)
    M: tl.constexpr,     # 窗口宽(m = spec_num + 1)
    REP: tl.constexpr,   # H // KV
    KV: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
):
    row = tl.program_id(0)             # 展平 query 行:b*M + j
    hq = tl.program_id(1)              # Q 头;kv = hq // REP
    sp = tl.program_id(2)
    b = row // M
    j = row % M
    kv = hq // REP

    # 行可见上限(与原 kernel 唯一本质差异):decode 的 L 换成
    # L_row = seq_lens[b] − (m−1−j) = len + j + 1(窗口内因果)。
    L = tl.load(lens_ptr + b) - (M - 1 - j)
    # 页对齐分片/尾块 slot mask/页偏移公式:与 _paged_attn_split_kernel 逐行同源。
    npages = (L + P - 1) // P
    pps = (npages + S - 1) // S          # pages per split
    start = sp * pps * P
    end = tl.minimum(start + pps * P, L)

    d = tl.arange(0, D)
    q = tl.load(q_ptr + (row * H + hq) * D + d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    slot = tl.arange(0, P)
    for t in range(start, end, P):
        page = tl.load(tables_ptr + b * MP + t // P)
        kv_mask = slot < (end - t)
        # k_pages[page, slot, kv, :] 的内存偏移(同原 kernel;int32 边界口径同源):
        #   ((((page*P) + slot) * KV) + kv) * D + d
        k_off = (((page * P + slot[:, None]) * KV + kv) * D + d[None, :])
        kt = tl.load(k_ptr + k_off, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * kt, axis=1) * scale       # [P]
        scores = tl.where(kv_mask, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)       # 首块:m_i=-inf → alpha=0;L_row ≥ 1 保证 m_new 有限
        p = tl.exp(scores - m_new)        # -inf → 0
        vt = tl.load(v_ptr + k_off, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * vt, axis=0)
        m_i = m_new

    off = (row * H + hq) * S + sp
    tl.store(acc_ptr + off * D + d, acc)
    tl.store(m_ptr + off, m_i)        # 空分片 → (-inf, 0, 0);combine 侧 beta=exp(-inf)=0 无贡献
    tl.store(l_ptr + off, l_i)


def paged_verify_attention(q, k_pages, v_pages, tables, seq_lens, num_splits: int):
    """verify 步分页窗口因果 attention(M8 Task 2,spec §2.2)。

    q [B, m, H, D](contiguous,已 norm+rope,位置 = len..len+m−1);
    k/v_pages [NB,P,KV,D](只读——写入已由 scatter_verify 完成);
    tables [B,MP] int32(build_tables(lens_override=[len+m]) 的读侧表);
    seq_lens [B] int32 = len + m(含本步全部窗口位);
    num_splits ≥ 1(split 结构照 decode kernel;启发式可复用
    paged_attention.num_splits_for,按 B*m 个 query 行估)。
    返回 [B,m,H,D] 与 q 同 dtype:第 j 位 = 位置 len+j 的窗口因果注意力输出
    (可见集 = 历史 len + 窗口前 j+1 位)。GQA 映射 q 头 h ← kv 头 h // (H//KV),
    与 pager._gather + repeat_interleave 的稠密路径一致。
    dtype:fp32/bf16 已对拍(bf16 TOL_VERIFY 标定),fp16 未单独对拍。"""
    B, M, H, D = q.shape
    NB, P, KV, _ = k_pages.shape
    S = num_splits
    assert M >= 1, "verify window must be non-empty"
    assert D & (D - 1) == 0 and P & (P - 1) == 0, "D and P must be power-of-2"
    assert H % KV == 0, "GQA requires H divisible by KV"
    assert S >= 1
    assert v_pages.shape == k_pages.shape
    assert tables.shape[0] == B and seq_lens.shape[0] == B
    assert k_pages.is_contiguous() and v_pages.is_contiguous()
    assert tables.is_contiguous() and seq_lens.is_contiguous()
    q2 = q.reshape(B * M * H, D).contiguous()
    acc = torch.empty((B * M * H, S, D), dtype=torch.float32, device=q.device)
    m = torch.empty((B * M * H, S), dtype=torch.float32, device=q.device)
    l = torch.empty((B * M * H, S), dtype=torch.float32, device=q.device)
    _paged_verify_split_kernel[(B * M, H, S)](
        q2, k_pages, v_pages, tables, seq_lens, acc, m, l,
        D ** -0.5, tables.shape[1],
        M=M, REP=H // KV, KV=KV, H=H, D=D, P=P, S=S,
    )
    out = torch.empty((B * M * H, D), dtype=q.dtype, device=q.device)
    _paged_attn_combine_kernel[(B * M * H,)](acc, m, l, out, S=S, D=D)
    return out.view(B, M, H, D)
