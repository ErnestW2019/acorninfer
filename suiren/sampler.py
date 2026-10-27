"""Sampling parameters and token selection.

Mirrors HF's warper pipeline order: temperature -> top-k -> top-p.
Numerical fidelity to HF is a hard requirement (M1 acceptance: per-step
logprob diff < 0.1); the unit suite pins our processing to HF's warpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled


class Sampler:
    def __init__(self, params: SamplingParams):
        if params.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < params.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if params.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables)")
        self.params = params

    def process(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature -> top-k -> top-p; returns processed [vocab] logits."""
        logits = logits.float() / self.params.temperature
        if self.params.top_k > 0:
            top_k = min(self.params.top_k, logits.size(-1))  # Safety check (HF 同款)
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = float("-inf")
        if self.params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=False)
            cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cum <= (1.0 - self.params.top_p)
            remove[-1] = False  # 保底 1 个 token(HF min_tokens_to_keep=1)
            mask = torch.zeros_like(remove).scatter(0, sorted_idx, remove)
            logits[mask] = float("-inf")
        return logits

    def sample(self, logits: torch.Tensor) -> tuple[int, float]:
        """Process -> softmax -> multinomial; returns (token, its logprob).

        The token is drawn from the processed (truncated) distribution, so
        top_k=1 stays deterministic. The reported logprob is the token's
        probability under the temperature-scaled distribution *before*
        top-k/top-p truncation (test_top_k_one pins this: it must equal
        log_softmax of the unscaled-by-truncation logits, not the degenerate
        0.0 that a single-survivor renormalization would give).
        """
        processed = self.process(logits)
        token = int(torch.multinomial(F.softmax(processed, dim=-1), num_samples=1))
        logprobs = F.log_softmax(logits.float() / self.params.temperature, dim=-1)
        return token, float(logprobs[token])
