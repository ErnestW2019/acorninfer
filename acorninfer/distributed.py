"""TP=2 进程组原语(M5 Task 3):nccl setup / broadcast / all-gather / 收尾。

简洁为主,不为未来留抽象:world 由 init_process_group 固定;张量封装只做
当前 TP 模型用到的两件(broadcast 入步 token、all-gather partial logits)。
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def setup(rank: int, world_size: int, init_method: str) -> None:
    """nccl 初始化 + 当前进程绑定 rank 号卡。"""
    dist.init_process_group(
        "nccl", init_method=init_method, rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)


def destroy() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_object(obj, src: int = 0):
    """任意可 pickle 对象广播(src → 全员;Engine 下发元数据用)。"""
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def broadcast_tokens(tokens: torch.Tensor, src: int = 0) -> torch.Tensor:
    """步 token 张量广播(src → 全员,SPMD 各 rank 喂同一输入)。"""
    dist.broadcast(tokens, src=src)
    return tokens


def all_gather_logits(logits: torch.Tensor) -> torch.Tensor:
    """partial logits [B, V/world] 沿词表维 all-gather concat → [B, V]。"""
    world = dist.get_world_size()
    packed = logits.contiguous()
    parts = [torch.empty_like(packed) for _ in range(world)]
    dist.all_gather(parts, packed)
    return torch.cat(parts, dim=-1)
