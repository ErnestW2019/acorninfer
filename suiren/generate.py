"""Greedy generation without a KV cache (M0).

Each step re-runs the full sequence: O(n^2) total, correct and simple.
The KV-cache path replaces this in M1; this loop is the golden oracle.
"""

from __future__ import annotations

import torch

from suiren.cache import KVCache
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
    eos_ids: frozenset[int] = EOS_IDS,
) -> list[int]:
    """Generate greedily; returns prompt ids + generated ids (prompt included).

    Mirrors HF semantics: a generated EOS id is APPENDED to the output before
    stopping, so the result may end with a trailing EOS id.
    """
    if prompt_ids is None:
        if tokenizer is None or prompt is None:
            raise ValueError("generate_token_ids needs prompt_ids, or both tokenizer and prompt")
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt encoded to zero tokens; pass a non-empty prompt")
    device = next(model.parameters()).device

    out = list(prompt_ids)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([out], device=device, dtype=torch.long)
        logits = model(input_ids)
        next_id = int(logits[0, -1].argmax())
        out.append(next_id)
        if next_id in eos_ids:
            break
    return out


@torch.inference_mode()
def generate_cached(
    model: Qwen3ForCausalLM,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 64,
    eos_ids: frozenset[int] = EOS_IDS,
    sampler=None,
    cache: KVCache | None = None,
) -> tuple[list[int], list[float]]:
    """Prefill/decode generation with a KV cache: O(n) total instead of O(n^2).

    prefill  = one forward over the whole prompt (cache empty -> causal);
    decode   = one forward per new token (q_len == 1, attends to all cache).

    Returns (ids, step_logprobs): prompt included; step_logprobs holds the
    sampler's logprob for each chosen token (empty list when greedy).
    EOS semantics mirror HF: a generated EOS id is appended, then we stop.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    device = next(model.parameters()).device
    if cache is None:
        cache = KVCache(num_layers=len(model.layers))

    out = list(prompt_ids)
    logprobs: list[float] = []

    # prefill:整段 prompt 一次前向,取最后位置的分布
    last = model(torch.tensor([out], device=device, dtype=torch.long), cache)[0, -1]

    for _ in range(max_new_tokens):
        if sampler is None:
            next_id, lp = int(last.argmax()), None
        else:
            next_id, lp = sampler.sample(last)
        out.append(next_id)
        if lp is not None:
            logprobs.append(lp)
        if next_id in eos_ids:
            break
        # decode:单 token 增量前向
        last = model(torch.tensor([[next_id]], device=device, dtype=torch.long), cache)[
            0, -1
        ]
    return out, logprobs
