"""Model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen3Config:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: list[int]

    @classmethod
    def from_json(cls, path: Path | str) -> Qwen3Config:
        raw = json.loads(Path(path).read_text())
        eos = raw["eos_token_id"]
        return cls(
            num_hidden_layers=raw["num_hidden_layers"],
            hidden_size=raw["hidden_size"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            intermediate_size=raw["intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            vocab_size=raw["vocab_size"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            bos_token_id=raw["bos_token_id"],
            eos_token_id=eos if isinstance(eos, list) else [eos],
        )

    @classmethod
    def tiny(cls) -> Qwen3Config:
        """Small config for cheap unit tests."""
        return cls(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            vocab_size=256,
            tie_word_embeddings=False,
            bos_token_id=0,
            eos_token_id=[1],
        )
