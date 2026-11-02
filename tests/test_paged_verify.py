"""M8 Task 2 对拍:verify kernel ↔ 同输入 gather-to-dense 稠密参考(读页张量)。

参考实现 `paged_verify_attention_ref` 在本文件(独立稠密参考:构造思路复用
`paged_decode_attention_ref` —— 按块表 gather → fp32 softmax 展开;逐 (b, j)
行按可见上限 L_j = seq_lens[b] − (m−1−j) 截断)。seq_lens 契约 = len + m
(含本步全部窗口位;len = 该 seq 前写长度)。

TOL 标定(协议:命令/日期写死,实测回填):
- fp32:全局 TOL_F32 紧(索引/mask/边界错必炸);
- bf16:TOL_VERIFY(per-kernel 标定值)。标定:2026-11-02,GPU 1,命令
  `CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest tests/test_paged_verify.py -v -s`:
  B∈{1,4,7}×m∈{2,5}×len∈{1,15,16,17,48} 全 30 组合,27 组合 max|diff|
  = 0.000000(kernel 与参考同 fp32 全程 + 末端单次 cast → 取整后逐位同),
  worst = 0.001953(B=7, m=5, len=15;次高 0.000244,B=7, m=5, len=17)
  = 单 bf16 ulp(@0.5–1.0 值域)。取 atol=4e-3、rtol=4e-3(≈ worst 的 2 ulp
  余量;attention 是 L 长求和,L 增大漂移上升,扩配置时按协议重标定)。
  逐组合 maxabs/maxrel 表见 docs/benchmarks.md M8 段 2。
"""
import pytest
import torch
from triton_utils import TOL_F32

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.ops.paged_verify import paged_verify_attention

# bf16 标定值(2026-11-02 回填;见文件头标定记录)
TOL_VERIFY = dict(rtol=4e-3, atol=4e-3)

# (B, m, len, S):len = 前写历史长度(每序列 hist_b = len + b 变长批,
# kernel 读侧 seq_lens[b] = hist_b + m);S 按页数覆盖 1/空 split/多 split。
_LENS = [1, 15, 16, 17, 48]          # 跨页/页界边界:1 页尾 / 恰页界 / 页界+1 / 3 恰整页
_S_BY_LEN = {1: 4, 15: 2, 16: 1, 17: 4, 48: 8}   # 1: S>页数(空 split);16: S=1;48: S=8>4 页
CONFIGS = [(B, m, L, _S_BY_LEN[L]) for B in (1, 4, 7) for m in (2, 5) for L in _LENS]


def paged_verify_attention_ref(q, k_pages, v_pages, tables, seq_lens):
    """稠密 oracle:逐序列按块表收集 → 逐 query 行 (b, j) 按可见上限截断 softmax。
    GQA 映射与 _repeat_kv 一致:q 头 j ← kv 头 j // (H//KV);收集表达式与
    paged_decode_attention_ref 逐位同构(token i @ (table[i//P], i%P)),
    唯一差异 = 行内截断 L_j = seq_lens[b] − (m−1−j)。"""
    B, M, H, D = q.shape
    NB, P, KV, _ = k_pages.shape
    rep = H // KV
    outs = []
    for b in range(B):
        Lfull = int(seq_lens[b])
        n_pages = -(-Lfull // P)
        idx = tables[b, :n_pages].long()
        kf = k_pages[idx].reshape(-1, KV, D)[:Lfull].permute(1, 0, 2)   # [KV, Lfull, D]
        vf = v_pages[idx].reshape(-1, KV, D)[:Lfull].permute(1, 0, 2)
        kf = kf.repeat_interleave(rep, 0)                               # [H, Lfull, D]
        vf = vf.repeat_interleave(rep, 0)
        for j in range(M):
            Lj = Lfull - (M - 1 - j)          # 行可见上限 = hist + j + 1
            s = torch.einsum('hd,hld->hl', q[b, j].float(), kf[:, :Lj, :].float()) * (D ** -0.5)
            p = torch.softmax(s, dim=-1)
            outs.append(torch.einsum('hl,hld->hd', p, vf[:, :Lj, :].float()).to(q.dtype))
    return torch.stack(outs, dim=0).view(B, M, H, D)


def _gen(dtype):
    g = torch.Generator().manual_seed(8)
    H, KV, D, P = 4, 2, 16, 16
    for B, M, L, S in CONFIGS:
        lens_list = [L + b + M for b in range(B)]     # hist_b = L + b(变长批)
        max_pages = max(-(-lf // P) for lf in lens_list)
        nb = max_pages + 2
        q = torch.randn(B, M, H, D, generator=g).to("cuda").to(dtype)
        k_pages = torch.randn(nb, P, KV, D, generator=g).to("cuda").to(dtype)
        v_pages = torch.randn(nb, P, KV, D, generator=g).to("cuda").to(dtype)
        # 块表:序列内页 id 互异(真实语义);填充列指向任意页(kernel 按 lens 截断
        # 循环,永不读填充项)。
        tables = torch.empty(B, max_pages, dtype=torch.int32)
        for i, lf in enumerate(lens_list):
            n = -(-lf // P)
            tables[i, :n] = torch.randperm(nb, generator=g)[:n]
            if n < max_pages:
                tables[i, n:] = torch.randint(0, nb, (max_pages - n,), generator=g)
        tables = tables.to("cuda")
        lens_t = torch.tensor(lens_list, device="cuda", dtype=torch.int32)
        yield q, k_pages, v_pages, tables, lens_t, S


@pytest.mark.gpu
def test_verify_matches_dense_reference():
    """B∈{1,4,7} × m∈{2,5} × len∈{1,15,16,17,48} 全矩阵(30 组合,变长批):
    fp32 紧容差验索引/行界/mask;bf16 按标定容差验数值路径;打印逐组合标定表。"""
    for q, kp, vp, tb, lens, S in _gen(torch.float32):
        ref = paged_verify_attention_ref(q, kp, vp, tb, lens)
        out = paged_verify_attention(q, kp, vp, tb, lens, S)
        torch.testing.assert_close(out, ref, **TOL_F32)

    worst_abs = 0.0
    print(f"\n{'B':>2} {'m':>2} {'len':>4} {'S':>2} | {'maxabs':>10} {'maxrel':>10}")
    for (B, M, L, S), (q, kp, vp, tb, lens, _) in zip(CONFIGS, _gen(torch.bfloat16)):
        ref = paged_verify_attention_ref(q, kp, vp, tb, lens)
        out = paged_verify_attention(q, kp, vp, tb, lens, S)
        diff = (out.float() - ref.float()).abs()
        maxabs = diff.max().item()
        maxrel = (maxabs / ref.float().abs().max().clamp_min(1e-9)).item()
        worst_abs = max(worst_abs, maxabs)
        print(f"{B:>2} {M:>2} {L:>4} {S:>2} | {maxabs:>10.6f} {maxrel:>10.2e}")
        torch.testing.assert_close(out, ref, **TOL_VERIFY)
    print(f"[paged_verify bf16 worst max|diff| = {worst_abs:.6f}]")  # 标定输出,回填 TOL


@pytest.mark.gpu
def test_window_causal_exact():
    """窗口因果精确钉:k ≡ 0 → 可见分数恒 0 → softmax 严格均匀 → 每行输出 =
    可见集 v 的精确均值(与 q 无关,独立于 gather 式参考)。三重 tripwire:
    ①窗口后位(hist+j+1..hist+m−1 的 v ×100):行 j 泄漏任一后位 → 均值量级爆炸;
    ②lens 外槽位(≥ hist_b+m 的页内槽 = NaN;表填充项指向全 NaN 陷阱页):
      任何越 lens 读 → NaN 传播;
    ③逐行 = 精确均匀均值(fp32 紧容差)——可见集组成错(多读/少读一位)即炸。"""
    g = torch.Generator().manual_seed(11)
    B, M, H, KV, D, P = 3, 5, 4, 2, 16, 16
    hist0 = 8
    nb = B + 1                          # 每 seq 独占 1 页(≤15 token);末页 = NaN 陷阱页
    k_pages = torch.zeros(nb, P, KV, D)
    v_pages = torch.randn(nb, P, KV, D, generator=g)
    k_pages[nb - 1].fill_(float("nan"))
    v_pages[nb - 1].fill_(float("nan"))

    hist = [hist0 + b for b in range(B)]           # 8, 9, 10(变长批)
    tables = torch.full((B, 2), nb - 1, dtype=torch.int32)  # 填充列 → 陷阱页
    dense_v = []                        # 每序列稠密 v:[KV, hist_b + M, D](构造值,非 gather)
    for b in range(B):
        tables[b, 0] = b
        hb, Lb = hist[b], hist[b] + M
        dv = torch.randn(KV, Lb, D, generator=g)
        for t in range(hb + 1, hb + M):            # 窗口位 1..m−1(位置 hb+1..hb+M−1):
            dv[:, t, :] = dv[:, t, :] * 100.0      # 对行 j < t−hb 是"窗口后位"毒值
        # lens 外槽位(t ≥ Lb)置 NaN:任何越 lens 读 → NaN
        k_pages[b] = 0.0                           # 全部槽位 k=0(可见分数恒 0)
        v_pages[b] = 0.0
        v_pages[b, :Lb] = dv.permute(1, 0, 2)      # 页内 token-major:[Lb, KV, D]
        v_pages[b, Lb:] = float("nan")             # 页尾越 lens 槽 = NaN
        k_pages[b, Lb:] = float("nan")
        dense_v.append(dv)

    k_pages, v_pages, tables = k_pages.to("cuda"), v_pages.to("cuda"), tables.to("cuda")
    lens = torch.tensor([hb + M for hb in hist], device="cuda", dtype=torch.int32)
    q = torch.randn(B, M, H, D, generator=g).to("cuda")    # 任意 q:均匀性与 q 无关

    out = paged_verify_attention(q, k_pages, v_pages, tables, lens, 4)

    assert torch.isfinite(out).all(), "越 lens 读到 NaN 槽位/陷阱页(窗口后位或历史外位泄漏)"
    for b in range(B):
        hb = hist[b]
        for j in range(M):
            Lj = hb + j + 1                               # 行可见集 = [0, hb+j]
            for h in range(H):
                expect = dense_v[b][h // (H // KV), :Lj, :].mean(dim=0).to(out.device)
                torch.testing.assert_close(
                    out[b, j, h].float(), expect,
                    rtol=1e-4, atol=1e-2,
                    msg=f"row (b={b}, j={j}, h={h}):可见集均匀均值不匹配(窗口因果/组成错误)",
                )


@pytest.mark.gpu
def test_rollback_then_reverify_numerics():
    """死槽复用数值闭环(真实 pager 路径):verify 写 m=5 → set_lens 回 a=2 →
    下步 ensure_n(5) 复用死槽(p0 页尾 2 槽 → p1 槽 0,1)+ 同页续接 → 复写 →
    verify 读侧结果 ≡ 从未写过死槽的孪生 cache(页值 + 输出逐位 torch.equal)
    ≡ 独立稠密参考(fp32 紧容差)。死槽复用数值闭环:任何未复写残留/越 lens
    读都会打破逐位相等。"""
    torch.manual_seed(3)
    KV, D, P, H, m = 2, 16, 16, 4, 5
    base, a = 13, 2                     # p0 页中段起,跨页界(13..15 ∈ p0,16,17 ∈ p1)
    device = "cuda"

    def make():
        cache = PagedKVCache(1, KV, D, P, BlockAllocator(8), device, torch.float32)
        return cache, PagedCacheView(cache)

    pre, vpre = torch.randn(1, KV, base, D).to(device), torch.randn(1, KV, base, D).to(device)
    k1, v1 = torch.randn(1, KV, m, D).to(device), torch.randn(1, KV, m, D).to(device)
    k2, v2 = torch.randn(1, KV, m, D).to(device), torch.randn(1, KV, m, D).to(device)

    # ---- 路径 A:写 m=5 → set_lens 回 a=2 → 复写(引擎死槽复用流)----
    cacheA, viewA = make()
    viewA.write(0, pre, vpre)
    t1 = viewA.ensure_append_capacity_n(0, m)
    pages1 = torch.tensor([[p for p, _ in t1]], dtype=torch.long, device=device)
    slots1 = torch.tensor([[s for _, s in t1]], dtype=torch.long, device=device)
    cacheA.scatter_verify([viewA], 0, k1, v1, pages1, slots1)
    assert viewA.seq_len == base + m == 18 and len(viewA.table) == 2

    viewA.set_lens(0, base + a + 1)               # = 16;死槽 = 位置 16,17(p1 槽 0,1)
    t2 = viewA.ensure_append_capacity_n(0, m)
    assert t2[:2] == t1[3:5]                      # 死槽复用(页表项已在,零分配)
    assert t2[2:] == [(viewA.table[1], s) for s in (2, 3, 4)]  # 同页续接
    pages2 = torch.tensor([[p for p, _ in t2]], dtype=torch.long, device=device)
    slots2 = torch.tensor([[s for _, s in t2]], dtype=torch.long, device=device)
    cacheA.scatter_verify([viewA], 0, k2, v2, pages2, slots2)
    assert viewA.seq_len == base + a + 1 + m == 21

    # ---- 路径 B:从未写过死槽(等价终态:pre ∪ k1 幸存前缀 ∪ k2 直接写)----
    cacheB, viewB = make()
    viewB.write(0, pre, vpre)
    viewB.write(0, k1[:, :, :a + 1], v1[:, :, :a + 1])
    viewB.write(0, k2, v2)
    assert viewB.seq_len == 21 and viewB.table == viewA.table
    for pg in range(len(viewA.table)):
        assert torch.equal(cacheA.k_pages[0][pg], cacheB.k_pages[0][pg])
        assert torch.equal(cacheA.v_pages[0][pg], cacheB.v_pages[0][pg])

    # ---- verify 读侧:同输入 → 输出逐位一致;且 = 独立稠密参考 ----
    g = torch.Generator().manual_seed(4)
    q = torch.randn(1, m, H, D, generator=g).to(device)
    tables = torch.tensor([viewA.table], dtype=torch.int32, device=device)
    lens = torch.tensor([21], dtype=torch.int32, device=device)
    outA = paged_verify_attention(q, cacheA.k_pages[0], cacheA.v_pages[0], tables, lens, 4)
    outB = paged_verify_attention(q, cacheB.k_pages[0], cacheB.v_pages[0], tables, lens, 4)
    assert torch.equal(outA, outB), "复写后读侧 ≠ 从未写过死槽(死槽残留/越 lens 读)"

    kd = torch.cat([pre[0], k1[0][:, :a + 1], k2[0]], dim=1)      # [KV, 21, D]
    vd = torch.cat([vpre[0], v1[0][:, :a + 1], v2[0]], dim=1)
    kf, vf = kd.repeat_interleave(H // KV, 0), vd.repeat_interleave(H // KV, 0)
    for j in range(m):
        Lj = 21 - (m - 1 - j)
        s = torch.einsum('hd,hld->hl', q[0, j].float(), kf[:, :Lj, :].float()) * (D ** -0.5)
        p = torch.softmax(s, dim=-1)
        expect = torch.einsum('hl,hld->hd', p, vf[:, :Lj, :].float()).to(q.dtype)
        torch.testing.assert_close(outA[0, j], expect, **TOL_F32)
