"""TP=2 张量并行原语(M5 Task 3):结构化切分纯函数 + 分片线性层。

切分纪律(spec §4.1):结构化切分,禁止裸 dim 对半 ——
- 融合 qkv 按头切(q 取 rank 段 16 头、k/v 各取 4 kv 头,concat 保持 q/k/v 序);
- 融合 gate_up 按列切(gate 半 + up 半,concat 保持 gate/up 序);
- o / down 行并行(K 维整块列切,partial → allreduce);
- embed / lm_head 词并行行切(V 维整块)。
norms 复制不切。本模块不持有进程组:allreduce/all-gather 由调用方
(acorninfer.distributed 或 RowParallel/VocabParallel 的默认 forward)执行。
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def shard_rows(w: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """dim0 连续等分(V 维行切:embed/lm_head 词并行)。"""
    n = w.shape[0]
    assert n % world == 0, f"dim0 {n} 不可被 world={world} 整除"
    c = n // world
    return w[rank * c:(rank + 1) * c].contiguous()


def shard_cols(w: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """dim1 连续等分(K 维列切:o / down 的行并行输入切)。"""
    n = w.shape[1]
    assert n % world == 0, f"dim1 {n} 不可被 world={world} 整除"
    c = n // world
    return w[:, rank * c:(rank + 1) * c].contiguous()


def shard_qkv(
    w: torch.Tensor, n_q: int, n_kv: int, head_dim: int, rank: int, world: int
) -> torch.Tensor:
    """融合 qkv [(n_q+2·n_kv)·hd, hidden] 的按头列切(spec §4.1)。

    q 行段 [0, n_q·hd) 取 rank 段 (n_q/world) 头;k 行段 [n_q·hd, (n_q+n_kv)·hd)
    与 v 行段各取 rank 段 (n_kv/world) 头;concat 保持 q/k/v 序。
    注:spec §4.1 融合布局的参考实现;生产路径走分立投影 + shard_rows/cols,
    本函数由单测守门。
    """
    cq = (n_q // world) * head_dim
    ck = (n_kv // world) * head_dim
    off_k = n_q * head_dim
    off_v = off_k + n_kv * head_dim
    return torch.cat([
        w[rank * cq:(rank + 1) * cq],
        w[off_k + rank * ck:off_k + (rank + 1) * ck],
        w[off_v + rank * ck:off_v + (rank + 1) * ck],
    ], dim=0).contiguous()


def shard_gate_up(w: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """融合 gate_up [2·inter, hidden] 列切:gate 行段取半 + up 行段取半,
    concat 保持 gate/up 序(spec §4.1)。
    注:spec §4.1 融合布局的参考实现;生产路径走分立投影 + shard_rows/cols,
    本函数由单测守门。
    """
    inter = w.shape[0] // 2
    h = inter // world
    return torch.cat([
        w[rank * h:(rank + 1) * h],
        w[inter + rank * h:inter + (rank + 1) * h],
    ], dim=0).contiguous()


class ColumnParallelLinear(nn.Module):
    """列并行(V 维切):weight [out/world, in];forward 输出 partial,不做
    gather_output —— allreduce(MLP/o 后)或 all-gather(lm_head 后)由调用方
    接线。"""

    def __init__(self, in_features: int, out_features: int, rank: int, world: int):
        super().__init__()
        assert out_features % world == 0, f"out {out_features} 不可被 world={world} 整除"
        self.rank = rank
        self.world = world
        self.weight = nn.Parameter(torch.zeros(out_features // world, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """行并行(K 维切):weight [out, in/world];forward 输出 partial,
    skip_reduce=False(默认)时就地 allreduce(各 rank 和 = 全宽输出);
    skip_reduce=True 交调用方决定归并时机。"""

    def __init__(self, in_features: int, out_features: int, rank: int, world: int):
        super().__init__()
        assert in_features % world == 0, f"in {in_features} 不可被 world={world} 整除"
        self.rank = rank
        self.world = world
        self.weight = nn.Parameter(torch.zeros(out_features, in_features // world))

    def forward(self, x: torch.Tensor, skip_reduce: bool = False) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if not skip_reduce:
            dist.all_reduce(out)
        return out


class VocabParallelEmbedding(nn.Module):
    """词并行嵌入:weight [vocab/world, in];本 rank 词段外的 id 查表行置 0,
    结果 allreduce —— x + 0 = x,与全宽查表逐位等价。"""

    def __init__(self, vocab_size: int, embed_dim: int, rank: int, world: int):
        super().__init__()
        assert vocab_size % world == 0, f"vocab {vocab_size} 不可被 world={world} 整除"
        self.rank = rank
        self.world = world
        self.vocab_local = vocab_size // world
        self.vocab_start = rank * self.vocab_local
        self.weight = nn.Parameter(torch.zeros(self.vocab_local, embed_dim))

    def forward(self, ids: torch.Tensor, skip_reduce: bool = False) -> torch.Tensor:
        mask = (ids >= self.vocab_start) & (ids < self.vocab_start + self.vocab_local)
        local = (ids - self.vocab_start).clamp(0, self.vocab_local - 1)
        out = F.embedding(local, self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        if not skip_reduce:
            dist.all_reduce(out)
        return out
