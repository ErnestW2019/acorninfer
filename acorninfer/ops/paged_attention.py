"""K3: 分页 decode attention —— flash-decoding 两阶段 kernel(torch 参考即单测 oracle)。

Phase 1(split)grid = (B, H, S):按 Q 头铺开,每 program 算一个
(序列, Q 头, KV 分片) 的 online-softmax 部和 (acc, m, l);GQA 下
kv 头 = hq // REP(REP = H // KV)。并行度公式按 Q 头估是 B×H(已含
GQA 因子)。num_splits 启发式固定 SM=128(4090),调参归 Task 8。
Phase 2(combine)grid = (B*H,):以公共 max-m 重标定合并(β=exp(m_s−m_new)),
l 同步累计,末尾 acc/l_j 归一,cast 回 q dtype 单次取整。

输入即 M2 页张量+块表,直读不 gather;q_len≡1(decode 步),无因果掩码。
分片按整页对齐(计划草稿的 token 均分块在非页对齐块首会错位读,缘由见
_paged_attn_split_kernel 内注释)。

数值策略:q/k/v load 后全程 fp32 计算(online softmax 标准两常数更新),
仅 combine 末尾 cast 回输出 dtype —— attention 是 L 长求和,归约序漂移
天然大于逐元素算子,对拍容差按实测标定(见 tests/test_ops_paged_attention.py)。

尾块 mask 只处理"slot ≥ 有效 token 数"一种:块表无空洞、前缀恒稠密
(pager.write 顺序 scatter 保证)。空分片(start ≥ end)存 (m=-inf, l=0,
acc=0),combine 无分支:β=exp(m_s−m_new),空分片 m_s=−inf → β=0,零贡献。
decode 步不变式 L ≥ 1 → split 0 非空 → combine 首迭代 m_new 必有限,全程无 NaN。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SM_COUNT = 128  # RTX 4090;固定启发式,Task 8 压测后调


@triton.jit
def _paged_attn_split_kernel(
    q_ptr,          # [B*H, D]
    k_ptr,          # [NB, P, KV, D]
    v_ptr,
    tables_ptr,     # [B, MP] int32
    lens_ptr,       # [B] int32
    acc_ptr,        # [B*H, S, D] fp32
    m_ptr, l_ptr,   # [B*H, S] fp32
    scale,
    MP,             # tables 行宽(运行期 stride)
    REP: tl.constexpr,   # H // KV
    KV: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
):
    b = tl.program_id(0)
    hq = tl.program_id(1)              # Q 头;kv = hq // REP
    sp = tl.program_id(2)
    kv = hq // REP

    L = tl.load(lens_ptr + b)
    # 页对齐分片:每 split 拿整数页,块首 t 恒满足 t % P == 0 → 块内单个
    # page 标量即可定位,score/尾块 mask(slot < end - t)语义不变。
    #(计划草稿按 token ceil 分块,非页对齐的块首会漏掉 t % P 页内偏移、
    # 从页 0 槽位开始错位读;且测试 CONFIGS "S>页数:部分 split 空" 只在
    # 页对齐分块下成立。空 split:sp ≥ npages → start ≥ end,存 (-inf,0,0)。)
    npages = (L + P - 1) // P
    pps = (npages + S - 1) // S          # pages per split
    start = sp * pps * P
    end = tl.minimum(start + pps * P, L)

    d = tl.arange(0, D)
    q = tl.load(q_ptr + (b * H + hq) * D + d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    slot = tl.arange(0, P)
    for t in range(start, end, P):
        page = tl.load(tables_ptr + b * MP + t // P)
        kv_mask = slot < (end - t)
        # k_pages[page, slot, kv, :] 的内存偏移:
        #   ((((page*P) + slot) * KV) + kv) * D + d
        # int32 边界:page*P*KV*D 在 >~100 万页的池(× P·KV·D)时顶到 int32
        # 上限;M3 范围(max_model_len 512, P=16 → 每序列 ≤32 页)远够,
        # Task 8 若调大池再议 int64。
        k_off = (((page * P + slot[:, None]) * KV + kv) * D + d[None, :])
        kt = tl.load(k_ptr + k_off, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * kt, axis=1) * scale       # [P]
        scores = tl.where(kv_mask, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)       # 首块:m_i=-inf → alpha=0;L≥1 保证 m_new 有限
        p = tl.exp(scores - m_new)        # -inf → 0
        vt = tl.load(v_ptr + k_off, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * vt, axis=0)
        m_i = m_new

    off = (b * H + hq) * S + sp
    tl.store(acc_ptr + off * D + d, acc)
    tl.store(m_ptr + off, m_i)        # 空分片 → (-inf, 0, 0);combine 侧 beta=exp(-inf)=0 无贡献
    tl.store(l_ptr + off, l_i)


@triton.jit
def _paged_attn_combine_kernel(
    acc_ptr, m_ptr, l_ptr, out_ptr,     # acc [B*H,S,D] fp32;out [B*H,D]
    S: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)              # b*H + hq
    d = tl.arange(0, D)
    m_j = float("-inf")
    l_j = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for sp in range(S):                 # S 为 constexpr
        m_s = tl.load(m_ptr + pid * S + sp)
        l_s = tl.load(l_ptr + pid * S + sp)
        m_new = tl.maximum(m_j, m_s)
        alpha = tl.exp(m_j - m_new)     # 首迭代 m_j=-inf → 0;L≥1 ⇒ sp=0 非空,m_new 有限
        beta = tl.exp(m_s - m_new)      # 空分片 m_s=-inf → beta=0
        a = tl.load(acc_ptr + (pid * S + sp) * D + d)
        acc = acc * alpha + a * beta
        l_j = l_j * alpha + l_s * beta
        m_j = m_new
    tl.store(out_ptr + pid * D + d, (acc / l_j).to(out_ptr.dtype.element_ty))


def paged_decode_attention(q, k_pages, v_pages, tables, seq_lens, num_splits: int):
    """q [B,H,D](contiguous,已 norm+rope);k/v_pages [NB,P,KV,D];
    tables [B,MP] int32;seq_lens [B] int32(含本步新 token)。
    返回 [B,H,D] 与 q 同 dtype。GQA 映射 q 头 j ← kv 头 j // (H//KV),
    与 pager._gather + repeat_interleave 的 torch 路径(oracle)一致。
    dtype:fp32/bf16 已对拍,fp16 未单独对拍。"""
    B, H, D = q.shape
    NB, P, KV, _ = k_pages.shape
    S = num_splits
    assert D & (D - 1) == 0 and P & (P - 1) == 0, "D and P must be power-of-2"
    assert H % KV == 0, "GQA requires H divisible by KV"
    assert S >= 1
    assert v_pages.shape == k_pages.shape
    assert tables.shape[0] == B and seq_lens.shape[0] == B
    assert k_pages.is_contiguous() and v_pages.is_contiguous()
    assert tables.is_contiguous() and seq_lens.is_contiguous()
    q2 = q.reshape(B * H, D).contiguous()
    acc = torch.empty((B * H, S, D), dtype=torch.float32, device=q.device)
    m = torch.empty((B * H, S), dtype=torch.float32, device=q.device)
    l = torch.empty((B * H, S), dtype=torch.float32, device=q.device)
    _paged_attn_split_kernel[(B, H, S)](
        q2, k_pages, v_pages, tables, seq_lens, acc, m, l,
        D ** -0.5, tables.shape[1],
        REP=H // KV, KV=KV, H=H, D=D, P=P, S=S,
    )
    out = torch.empty((B * H, D), dtype=q.dtype, device=q.device)
    _paged_attn_combine_kernel[(B * H,)](acc, m, l, out, S=S, D=D)
    return out.view(B, H, D)


def num_splits_for(B: int, num_heads: int, *, sm_count: int = _SM_COUNT,
                   max_splits: int = 8) -> int:
    """CTAs = B*H*S 填满 SM 的固定启发式。"""
    return max(1, min(max_splits, -(-sm_count // (B * num_heads))))


def build_tables(views, page_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """块表 → ([B, max_pages] int32 填充张量, [B] int32 长度含本步新 token)。
    填充值任意:kernel 按 seq_lens 截断循环,永不读填充项。"""
    lens = [int(v.seq_len) for v in views]
    max_pages = max(1, max(-(-L // page_size) for L in lens))
    t = torch.zeros((len(views), max_pages), dtype=torch.int32)
    for i, v in enumerate(views):
        n = -(-lens[i] // page_size)
        t[i, :n] = torch.tensor(v.table[:n], dtype=torch.int32)
    return t.to(device), torch.tensor(lens, dtype=torch.int32, device=device)


def paged_decode_attention_ref(q, k_pages, v_pages, tables, seq_lens):
    """Oracle:逐序列按块表收集(页张量直读,非 view.update 的 gather)→ fp32 softmax 展开。
    GQA 映射与 _repeat_kv 一致:q 头 j ← kv 头 j // (H//KV)。收集表达式与
    pager._gather 逐位同构(token i @ (table[i//P], i%P),已用
    PagedCacheView.write 小例验证)。"""
    B, H, D = q.shape
    NB, P, KV, _ = k_pages.shape
    rep = H // KV
    outs = []
    for b in range(B):
        L = int(seq_lens[b])
        n_pages = -(-L // P)
        idx = tables[b, :n_pages].long()
        kf = k_pages[idx].reshape(-1, KV, D)[:L].permute(1, 0, 2)   # [KV, L, D]
        vf = v_pages[idx].reshape(-1, KV, D)[:L].permute(1, 0, 2)
        kf = kf.repeat_interleave(rep, 0)                            # [H, L, D]
        vf = vf.repeat_interleave(rep, 0)
        s = torch.einsum('hd,hld->hl', q[b].float(), kf.float()) * (D ** -0.5)
        # 计划草稿的 `q[b] @ kf.transpose(1,2)` 是 [H,D]@[H,D,L]:torch 把二维
        # 操作数提升成 (1,H,D) 再广播,q 头轴与 kf 批轴撞成 [H,H,L](p@vf 同理
        # 膨成 [H,H,D])。逐 q 头点积须显式对齐轴,einsum 'hd,hld->hl' 即轴序
        # 推导的直译。
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum('hl,hld->hd', p, vf.float()).to(q.dtype))
    return torch.stack(outs, dim=0)
