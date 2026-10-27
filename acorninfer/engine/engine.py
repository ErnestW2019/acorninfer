"""Engine:请求生命周期 + step 循环。单线程契约(同一时刻一个 step 调用方)。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import QWEN3_EOS_IDS, EngineConfig
from acorninfer.engine.executor import Executor
from acorninfer.engine.scheduler import Scheduler
from acorninfer.engine.sequence import RequestStatus, Sequence
from acorninfer.models.qwen3 import Qwen3ForCausalLM, from_pretrained
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer


@dataclass
class TokenOutput:
    seq_id: int
    token_id: int
    finished: bool
    stop_reason: str | None = None


class Engine:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        tokenizer: Tokenizer | None,
        *,
        num_blocks: int = 1024,
        page_size: int = 16,
        max_running: int = 8,
        watermark_blocks: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        cfg = model.cfg
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.allocator = BlockAllocator(num_blocks)
        self.cache = PagedKVCache(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            page_size,
            self.allocator,
            device,
            dtype,
        )
        self.scheduler = Scheduler(self.allocator, page_size, max_running, watermark_blocks)
        self.executor = Executor(model)
        self.page_size = page_size
        self.seqs: dict[int, Sequence] = {}
        self._views: dict[int, PagedCacheView] = {}
        self._next_id = 0

    @classmethod
    def from_config(cls, cfg: EngineConfig, tokenizer: Tokenizer) -> "Engine":
        model = from_pretrained(cfg.model_dir, cfg.device, cfg.dtype)
        return cls(
            model,
            tokenizer,
            num_blocks=cfg.num_blocks,
            page_size=cfg.page_size,
            max_running=cfg.max_running,
            watermark_blocks=cfg.watermark_blocks,
        )

    def submit(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        *,
        max_tokens: int,
        eos_ids: frozenset[int] | None = None,
    ) -> int:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if -(-len(prompt_ids) // self.page_size) > self.allocator.total:
            raise ValueError("prompt longer than total KV capacity; reject early")
        seq = Sequence(
            seq_id=self._next_id,
            prompt_ids=list(prompt_ids),
            sampling=sampling,
            max_tokens=max_tokens,
            eos_ids=frozenset(eos_ids) if eos_ids is not None else QWEN3_EOS_IDS,
        )
        self._next_id += 1
        self.seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    @torch.inference_mode()
    def step(self) -> list[TokenOutput]:
        prefills, decodes = self.scheduler.schedule()
        outputs: list[TokenOutput] = []
        for seq in prefills:
            # 块表共享同一 list 对象:调度器 _release 清空它即同时清空视图
            view = PagedCacheView(self.cache, seq.block_table)
            self._views[seq.seq_id] = view
            token = self.executor.prefill(seq, view)
            seq.num_pages = len(view.table)
            self._record(seq, token, outputs)
        if decodes:
            views = [self._views[s.seq_id] for s in decodes]
            tokens = self.executor.decode_batch(decodes, views)
            for seq, token in zip(decodes, tokens):
                seq.num_pages = len(self._views[seq.seq_id].table)
                self._record(seq, token, outputs)
        # 退休:释放已完成序列的块(调度器不退休,见 Scheduler docstring)
        for seq in [s for s in self.scheduler.running if s.is_finished]:
            self.scheduler.running.remove(seq)
            seq.status = RequestStatus.FINISHED
            self._views.pop(seq.seq_id)  # 视图丢弃;块经共享块表释放
            for b in seq.block_table:
                self.allocator.free(b)
            seq.block_table.clear()
            seq.num_pages = 0
        return outputs

    def _record(self, seq: Sequence, token_id: int, outputs: list[TokenOutput]) -> None:
        seq.generated_ids.append(token_id)
        now = time.perf_counter()
        if seq.first_token_at is None:
            seq.first_token_at = now
        seq.last_token_at = now
        finished, reason = False, None
        if token_id in seq.eos_ids:
            finished, reason = True, "stop"
        elif len(seq.generated_ids) >= seq.max_tokens:
            finished, reason = True, "length"
        if finished:
            seq.stop_reason = reason
            seq.status = RequestStatus.FINISHED
        outputs.append(TokenOutput(seq.seq_id, token_id, finished, seq.stop_reason))

    def has_pending(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def assert_balanced(self) -> None:
        self.allocator.assert_balanced()
