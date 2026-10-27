"""批量前向执行:prefill 单序列 / decode 批量化,采样按序列参数。"""

from __future__ import annotations

import torch

from acorninfer.engine.sequence import Sequence
from acorninfer.models.qwen3 import Qwen3ForCausalLM
from acorninfer.sampler import Sampler


class Executor:
    def __init__(self, model: Qwen3ForCausalLM):
        self.model = model
        self.device = next(model.parameters()).device

    def prefill(self, seq: Sequence, view) -> int:
        """整段(all_ids)一次前向,选出新 token。抢占重放同一路径。"""
        ids = torch.tensor([seq.all_ids], device=self.device, dtype=torch.long)
        logits = self.model(ids, view, num_logits_to_keep=1)[0, -1]
        return self._sample(seq, logits)

    def decode_batch(self, seqs: list[Sequence], views) -> list[int]:
        x = torch.tensor(
            [[s.generated_ids[-1]] for s in seqs], device=self.device, dtype=torch.long
        )
        logits = self.model.forward_decode(x, views)  # [B, vocab]
        return [self._sample(s, logits[i]) for i, s in enumerate(seqs)]

    @staticmethod
    def _sample(seq: Sequence, logits: torch.Tensor) -> int:
        if seq.sampling.temperature == 0:  # OpenAI 语义:temp=0 即 greedy
            return int(logits.argmax())
        return Sampler(seq.sampling).sample(logits)[0]
