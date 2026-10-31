"""TP=2 切分原语结构单测(M5 Task 3,单卡即可跑,不起进程组)。

结构钉子:shard_qkv / shard_gate_up 输出形状 + 行内容 == 手工切片对照;
Column/Row 分片前向数值 == 全宽 GEMM 的对应切片重构(TOL);词并行嵌入
越界行置零、两分片求和 == 全宽查表(逐位)。
"""

import pytest
import torch
import torch.nn.functional as F

from triton_utils import TOL_BF16, max_abs_diff

from acorninfer.models.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    shard_cols,
    shard_gate_up,
    shard_qkv,
    shard_rows,
)


def test_shard_qkv_structure():
    """融合 qkv [(n_q+2n_kv)·hd, hidden] 按头列切:q 取 rank 段,k/v 同,
    concat 保持 q/k/v 序(spec §4.1);手工切片逐位对照。"""
    n_q, n_kv, hd, hidden = 4, 2, 2, 3
    w = torch.arange((n_q + 2 * n_kv) * hd * hidden, dtype=torch.float32)
    w = w.reshape(n_q * hd + 2 * n_kv * hd, hidden)
    for rank in range(2):
        got = shard_qkv(w, n_q, n_kv, hd, rank, 2)
        q_rows, kv_rows = (n_q // 2) * hd, (n_kv // 2) * hd
        q0 = rank * q_rows
        k0 = n_q * hd + rank * kv_rows
        v0 = (n_q + n_kv) * hd + rank * kv_rows
        want = torch.cat([
            w[q0:q0 + q_rows],
            w[k0:k0 + kv_rows],
            w[v0:v0 + kv_rows],
        ], dim=0)
        assert got.shape == ((n_q // 2 + 2 * (n_kv // 2)) * hd, hidden)
        assert torch.equal(got, want)


def test_shard_gate_up_structure():
    """融合 gate_up [2·inter, hidden] 列切:gate 取半 + up 取半,保持序。"""
    inter, hidden = 6, 3
    w = torch.arange(2 * inter * hidden, dtype=torch.float32).reshape(2 * inter, hidden)
    for rank in range(2):
        got = shard_gate_up(w, rank, 2)
        h = inter // 2
        want = torch.cat([w[rank * h:(rank + 1) * h],
                          w[inter + rank * h:inter + (rank + 1) * h]], dim=0)
        assert got.shape == (inter, hidden)
        assert torch.equal(got, want)


def test_shard_rows_cols():
    """V 维行切(embed/lm_head)与 K 维列切(o/down)的连续等分。"""
    w = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    assert torch.equal(shard_rows(w, 0, 2), w[:3])
    assert torch.equal(shard_rows(w, 1, 2), w[3:])
    assert torch.equal(shard_cols(w, 0, 2), w[:, :2])
    assert torch.equal(shard_cols(w, 1, 2), w[:, 2:])
    with pytest.raises(AssertionError):
        shard_rows(torch.zeros(5, 2), 0, 2)  # 不可整除 → 构造期拒绝


@pytest.mark.gpu
def test_column_parallel_matches_full_slice():
    """列并行分片前向 == 全宽 GEMM 对应输出列切片(TOL;bf16 合批 GEMM
    形状差与 M4.1 对拍同源)。"""
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(5, 8, device=device, dtype=torch.bfloat16)
    w_full = torch.randn(12, 8, device=device, dtype=torch.bfloat16)
    full = F.linear(x, w_full)
    for rank in range(2):
        lin = ColumnParallelLinear(8, 12, rank, 2).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(shard_rows(w_full, rank, 2))
        part = lin(x)
        want = full[:, rank * 6:(rank + 1) * 6]
        assert part.shape == (5, 6)
        assert max_abs_diff(part, want) <= TOL_BF16["rtol"] * 0 + TOL_BF16["atol"]


@pytest.mark.gpu
def test_row_parallel_partials_sum_to_full():
    """行并行 partial(skip_reduce=True)求和 == 全宽 GEMM(标定容差);
    单测不起进程组,allreduce 路径由 TP 双卡对拍端到端覆盖。

    容差标定(TOL 协议):两个 K/2 分片各自 fp32 累加后 bf16 舍入、再求和,
    与单次 K 全长 GEMM 的归约序差为固有舍入。标定:2026-10-31,命令
    `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest
    tests/test_parallel.py -k row_parallel -v`,实测 max|diff| = 0.03125
    (|ref|≈4–6 处恰 2 个 bf16 ulp)。写死 rtol=1.6e-2(≈2 bf16 ulp 相对界)
    + atol=1e-2(覆盖小 |ref| 元素);再触发即超标。"""
    device = "cuda"
    torch.manual_seed(1)
    x = torch.randn(5, 8, device=device, dtype=torch.bfloat16)
    w_full = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    full = F.linear(x, w_full)
    total = torch.zeros(5, 6, device=device, dtype=torch.float32)
    for rank in range(2):
        lin = RowParallelLinear(8, 6, rank, 2).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(shard_cols(w_full, rank, 2))
        part = lin(x[:, rank * 4:(rank + 1) * 4], skip_reduce=True)  # 行并行输入 = 本 rank K 切片
        total += part.float()
    torch.testing.assert_close(total.to(torch.bfloat16), full,
                               rtol=1.6e-2, atol=1e-2)


@pytest.mark.gpu
def test_vocab_parallel_embedding_mask_and_sum():
    """词并行嵌入:两分片 partial(skip_reduce=True)求和 == 全宽查表
    逐位一致;各自 rank 段外的 id 贡献必须为 0(越界置零)。"""
    device = "cuda"
    torch.manual_seed(2)
    table = torch.randn(8, 4, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([[0], [3], [7], [5]], device=device)
    full = F.embedding(ids, table)  # [4, 1, 4]
    total = torch.zeros(4, 1, 4, device=device, dtype=torch.float32)
    for rank in range(2):
        emb = VocabParallelEmbedding(8, 4, rank, 2).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            emb.weight.copy_(shard_rows(table, rank, 2))
        part = emb(ids, skip_reduce=True)
        # rank0 只对 id<4 非零,rank1 只对 id>=4 非零
        lo, hi = rank * 4, (rank + 1) * 4
        outside = (ids < lo) | (ids >= hi)
        assert torch.equal(part.float()[outside.ravel()], torch.zeros_like(part.float()[outside.ravel()]))
        total += part.float()
    assert torch.equal(total.to(torch.bfloat16), full)
