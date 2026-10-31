"""Loading model weights from a safetensors sharded checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_safetensors(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load all shards named by model.safetensors.index.json into one dict."""
    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))

    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_names:
        loaded = load_file(model_dir / shard, device=str(device))
        tensors.update(loaded)
    return {name: t.to(dtype=dtype) for name, t in tensors.items()}


def load_safetensors_sharded(
    model_dir: str | Path,
    shard_fn,
    rank: int,
    world_size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """TP 分片加载(M5 Task 3):全量载 CPU → shard_fn(name, t, rank, world)
    逐名切(返回 None = 该 rank 丢弃此名)→ 搬 device/dtype → 释放全量。

    兼容 index 版(model.safetensors.index.json)与单文件版
    (model.safetensors)checkpoint;shard_fn 由模型侧提供(见
    qwen3_tp.build_shard_fn)。load_safetensors 强依赖 index,本函数是
    其分片推广,原函数一行不动。
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        files = [model_dir / s for s in shard_names]
    elif (model_dir / "model.safetensors").exists():
        files = [model_dir / "model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors checkpoint under {model_dir}")

    tensors: dict[str, torch.Tensor] = {}
    for f in files:  # 全量驻留 CPU(与 from_pretrained 的单副本装载口径一致)
        tensors.update(load_file(str(f), device="cpu"))
    sharded: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        piece = shard_fn(name, t, rank, world_size)
        if piece is not None:
            sharded[name] = piece.to(device=device, dtype=dtype)
    del tensors  # 显式释放全量(峰值 = 全量 + 分片驻留)
    return sharded
