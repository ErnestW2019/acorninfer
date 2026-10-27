"""Sequence: 引擎内的请求状态机。"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from acorninfer.sampler import SamplingParams


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt_ids: list[int]
    sampling: SamplingParams
    max_tokens: int
    eos_ids: frozenset[int]
    generated_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)  # 与 PagedCacheView 共享
    num_pages: int = 0                                    # engine 每次前向后同步
    status: RequestStatus = RequestStatus.WAITING
    stop_reason: str | None = None
    created_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    last_token_at: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.generated_ids)

    @property
    def all_ids(self) -> list[int]:
        """prompt + generated;抢占 recompute 的 prefill 输入。"""
        return self.prompt_ids + self.generated_ids

    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED
