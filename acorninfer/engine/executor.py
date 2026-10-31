"""批量前向执行:prefill 单序列/同批多序列 / decode 批量化,采样按序列参数。"""

from __future__ import annotations

import torch

from acorninfer.engine.graph_runner import BUCKETS, DecodeGraphRunner
from acorninfer.engine.sequence import Sequence
from acorninfer.models.qwen3 import Qwen3ForCausalLM
from acorninfer.sampler import Sampler


class Executor:
    def __init__(self, model: Qwen3ForCausalLM, graph_runner: DecodeGraphRunner | None = None):
        self.model = model
        self.graph_runner = graph_runner
        self.device = next(model.parameters()).device

    def prefill(self, seq: Sequence, view) -> int:
        """整段(all_ids)一次前向,选出新 token。抢占重放同一路径。"""
        ids = torch.tensor([seq.all_ids], device=self.device, dtype=torch.long)
        logits = self.model(ids, view, num_logits_to_keep=1)[0, -1]
        return self._sample(seq, logits)

    def prefill_batch(self, seqs: list[Sequence], views) -> list[int]:
        """多序列同批 prefill(M4.1):右 padding 合批前向,逐行采样
        (_sample 语义不变:temp==0 argmax;temp>0 逐行 Sampler)。"""
        lens = [len(s.all_ids) for s in seqs]
        Lmax = max(lens)
        ids = torch.zeros(len(seqs), Lmax, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids[i, : lens[i]] = torch.tensor(s.all_ids, dtype=torch.long,
                                             device=self.device)
        logits = self.model.forward_prefill_batch(ids, views, real_lens=lens)
        return [self._sample(s, logits[i]) for i, s in enumerate(seqs)]

    def extend(self, seqs: list[Sequence], views,
               chunks: list[tuple[Sequence, int, int]]) -> list[int | None]:
        """chunk 续读(M5):chunks = [(seq, start, length)] 与 seqs/views 同序
        对齐。ids [N, Lc_max] 右 padding(real_lens=chunk_lens)→
        forward_extend;仅 prompt 完成(start+length == len(all_ids))的行
        _sample(与 prefill_batch 同语义);未完成行返回 None —— 中间块不发
        token,部分 prompt 条件下的采样是垃圾,绝不外泄。"""
        starts = [start for _, start, _ in chunks]
        lens = [length for _, _, length in chunks]
        Lc = max(lens)
        ids = torch.zeros(len(seqs), Lc, dtype=torch.long, device=self.device)
        for i, (s, start, length) in enumerate(chunks):
            ids[i, :length] = torch.tensor(
                s.all_ids[start: start + length], dtype=torch.long,
                device=self.device,
            )
        logits = self.model.forward_extend(ids, views, starts=starts,
                                           chunk_lens=lens)
        return [
            self._sample(s, logits[i])
            if start + length == len(s.all_ids) else None
            for i, (s, start, length) in enumerate(chunks)
        ]

    def decode_batch(self, seqs: list[Sequence], views) -> list[int]:
        # 全批 greedy 谓词只求值一次:runner 派发与 eager 合批共用同一局部值
        greedy = all(s.sampling.temperature == 0 for s in seqs)
        # runner 契约 = triton(构造期钉死)+ greedy(与 _sample 的 argmax 分支同义)+ B ≤ 8
        if (self.graph_runner is not None
                and len(views) <= BUCKETS[-1]
                and greedy):
            return self.graph_runner.decode_batch(seqs, views)
        x = torch.tensor(
            [[s.generated_ids[-1]] for s in seqs], device=self.device, dtype=torch.long
        )
        logits = self.model.forward_decode(x, views)  # [B, vocab]
        if greedy:
            # 全 greedy:与 runner 同款合批 argmax(单次 D2H)
            return [int(t) for t in logits.argmax(dim=1).tolist()]
        return [self._sample(s, logits[i]) for i, s in enumerate(seqs)]

    @staticmethod
    def _sample(seq: Sequence, logits: torch.Tensor) -> int:
        """temp=0 → greedy(OpenAI 语义,Sampler 拒绝 temp=0,须先分流)。

        temp>0 走全局 RNG(generator=None):同进程多种子序列交织后,
        单请求不可复现 —— 契约与 Sampler(generator) 的按请求可复现性不同,
        bench 的确定性测试一律用 temp=0。
        """
        if seq.sampling.temperature == 0:
            return int(logits.argmax())
        return Sampler(seq.sampling).sample(logits)[0]
