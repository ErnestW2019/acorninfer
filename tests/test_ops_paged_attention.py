"""K3 对拍:kernel ↔ 逐序列 torch 展开 oracle(读页张量)。

CONFIGS 覆盖:L=1 / 恰整页 / 跨页尾残 / 变长批 / S>页数(空 split)/ P=4。
fp32 紧容差验索引与 mask;bf16 标定容差验数值策略(标定输出回填 TOL_BF16)。

语义契约(pager._gather 同构,已用 PagedCacheView.write 小例逐位验证):
token i 位于 (table[i//P], i%P),即 k_pages[table[t//P]][t%P] 为第 t 个 token;
GQA 映射 q 头 j ← kv 头 j // (H//KV),与 kernel 的 kv = hq // REP 互为镜像。
空 split 路径(start ≥ end)两条触发线:config (3,...,[33,1,48],4) 的 L=1
序列(L=1, S=4 → pps=1,split 1..3 全空)与 config [40],8(3 页 8 splits,
pps=1 → split 3..7 全空)。
"""
import pytest
import torch
from triton_utils import TOL_BF16, TOL_F32, max_abs_diff

from acorninfer.ops.paged_attention import (
    build_tables,
    num_splits_for,
    paged_decode_attention,
    paged_decode_attention_ref,
)

# bf16 标定:2026-10-30,命令 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest
# tests/test_ops_paged_attention.py -v -s`,实测 max|diff| = 0.000000(≤ 全局
# TOL_BF16 1e-2 → 按协议直接用全局值,不设 per-kernel 常量)。kernel 与 oracle
# 全程 fp32、仅末端一次 cast,fp32 归约序漂移 ~1e-7 远低于 bf16 格点,取整后
# 逐位相同。注意:attention 是 L 长求和,L/D 增大后漂移上升,Task 8 扩配置时
# 按协议重标定。

# (B, H, KV, D, P, lens, S)
CONFIGS = [
    (1, 4, 2, 16, 16, [1], 1),
    (1, 4, 2, 16, 16, [16], 1),      # 恰一整页
    (1, 4, 2, 16, 16, [17], 2),      # 跨 2 页,尾页 1 token
    (2, 4, 2, 16, 4, [7, 20], 2),    # P=4 + 变长
    (3, 8, 2, 16, 16, [33, 1, 48], 4),
    (1, 4, 2, 16, 16, [40], 8),      # S > 页数:部分 split 空
]


def _gen(dtype):
    g = torch.Generator().manual_seed(23)
    for B, H, KV, D, P, lens, S in CONFIGS:
        max_pages = max(-(-L // P) for L in lens)
        nb = max_pages + 2
        q = torch.randn(B, H, D, generator=g).to("cuda").to(dtype)
        k_pages = torch.randn(nb, P, KV, D, generator=g).to("cuda").to(dtype)
        v_pages = torch.randn(nb, P, KV, D, generator=g).to("cuda").to(dtype)
        # 计划陷阱修正:去重按序列内进行(同一序列的块表内页 id 互异 —— 真实
        # 系统语义;计划草稿的跨序列共享 seen 集合会在 config 5 上死循环:
        # 需 3+1+3=7 个互异页而池子只有 nb=5)。跨序列共享页对数值对拍无害
        #(oracle 按块表逐序列读,读到同样的页内容是良定义的)。
        # 填充列任意值:kernel 按 seq_lens 截断循环,永不读填充项。
        tables = torch.empty(B, max_pages, dtype=torch.int32)
        for i, L in enumerate(lens):
            n = -(-L // P)
            tables[i, :n] = torch.randperm(nb, generator=g)[:n]
            if n < max_pages:
                tables[i, n:] = torch.randint(0, nb, (max_pages - n,), generator=g)
        tables = tables.to("cuda")
        lens_t = torch.tensor(lens, device="cuda", dtype=torch.int32)
        yield q, k_pages, v_pages, tables, lens_t, S


@pytest.mark.gpu
def test_matches_ref_fp32_tight():
    for q, kp, vp, tb, lens, S in _gen(torch.float32):
        ref = paged_decode_attention_ref(q, kp, vp, tb, lens)
        out = paged_decode_attention(q, kp, vp, tb, lens, S)
        torch.testing.assert_close(out, ref, **TOL_F32)


@pytest.mark.gpu
def test_matches_ref_bf16_calibrated():
    worst = 0.0
    for q, kp, vp, tb, lens, S in _gen(torch.bfloat16):
        ref = paged_decode_attention_ref(q, kp, vp, tb, lens)
        out = paged_decode_attention(q, kp, vp, tb, lens, S)
        worst = max(worst, max_abs_diff(out, ref))
        torch.testing.assert_close(out, ref, **TOL_BF16)
    print(f"\n[paged_attn bf16 calibrated max|diff| = {worst:.6f}]")  # 标定输出,回填 TOL


@pytest.mark.gpu
def test_deterministic_double_run():
    for q, kp, vp, tb, lens, S in _gen(torch.float32):
        a = paged_decode_attention(q, kp, vp, tb, lens, S)
        b = paged_decode_attention(q, kp, vp, tb, lens, S)
        assert torch.equal(a, b)


@pytest.mark.gpu
def test_num_splits_heuristic():
    assert num_splits_for(1, 8) == 8      # ceil(128/8)=16 → clamp 8
    assert num_splits_for(8, 8) == 2      # ceil(128/64)
    assert num_splits_for(1, 32) == 4     # B*H=32 → ceil(128/32)
    assert num_splits_for(4, 32) == 1


@pytest.mark.gpu
def test_build_tables_shapes():
    class FakeView:  # 只需 seq_len 属性 + table 列表
        def __init__(self, table, L):
            self.table, self._L = table, L
        @property
        def seq_len(self):
            return self._L

    views = [FakeView([3, 4, 5], 40), FakeView([7], 1)]
    t, lens = build_tables(views, 16, "cuda")
    assert t.shape == (2, 3) and t.dtype == torch.int32
    assert t[0].tolist() == [3, 4, 5] and lens.tolist() == [40, 1]
