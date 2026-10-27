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
