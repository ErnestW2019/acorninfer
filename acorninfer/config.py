"""Model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

# Qwen3 生成终止符:<|endoftext|>(基座)、<|im_end|>(chat)。
# 引擎默认值;请求级 eos 覆盖在 Engine.submit 传入。
QWEN3_EOS_IDS: frozenset[int] = frozenset({151643, 151645})


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


@dataclass(frozen=True)
class EngineConfig:
    """Engine-level configuration (M2)."""

    model_dir: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    num_blocks: int = 1024      # 物理块数;Qwen3-8B 每块 16 token ≈ 2.3MB(全 36 层)
    page_size: int = 16
    max_running: int = 8        # decode batch 上限
    watermark_blocks: int = 8   # 水位:free 低于此值不再准入,降低抢占概率
    attention_backend: str = "sdpa"  # "sdpa" | "triton"(M3 paged attention)
    fused_ops: bool = False          # True → decode 走 K1/K2 融合算子(M3)

    def __post_init__(self) -> None:
        # CLI(Task 10 bench/server)直接透传参数,坏值在构造期爆掉而非运行期排队卡死
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        if self.max_running < 1:
            raise ValueError(f"max_running must be >= 1, got {self.max_running}")
        if self.watermark_blocks < 0:
            raise ValueError(f"watermark_blocks must be >= 0, got {self.watermark_blocks}")
        if self.attention_backend not in ("sdpa", "triton"):
            raise ValueError(
                f"attention_backend must be 'sdpa'|'triton', got {self.attention_backend!r}"
            )
