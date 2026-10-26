"""Greedy generation without a KV cache (M0).

Each step re-runs the full sequence: O(n^2) total, correct and simple.
The KV-cache path replaces this in M1; this loop is the golden oracle.
"""

from __future__ import annotations

import torch

from suiren.models.qwen3 import Qwen3ForCausalLM
from suiren.tokenizer import Tokenizer

EOS_IDS = frozenset({151643, 151645})  # <|endoftext|>, <|im_end|>


@torch.inference_mode()
def generate_token_ids(
    model: Qwen3ForCausalLM,
    prompt_ids: list[int] | None = None,
    *,
    tokenizer: Tokenizer | None = None,
    prompt: str | None = None,
    max_new_tokens: int = 64,
) -> list[int]:
    """Generate greedily; returns prompt ids + generated ids (prompt included)."""
    if prompt_ids is None:
        assert tokenizer is not None and prompt is not None
        prompt_ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device

    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([out], device=device, dtype=torch.long)
        logits = model(input_ids)
        next_id = int(logits[0, -1].argmax())
        if next_id in EOS_IDS:
            break
        out.append(next_id)
    return out
