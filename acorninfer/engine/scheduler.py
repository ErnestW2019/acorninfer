"""FCFS 调度器:准入 + 水位 + 抢占最老(recompute)。spec §5。

策略:KV 不足先排队;running 增长可能耗尽块时,抢占最老请求
(释放其全部块,回 waiting 队首,重被准入时对 all_ids 重新 prefill)。
"""

from __future__ import annotations

from collections import deque

from acorninfer.cache.pager import BlockAllocator
from acorninfer.engine.sequence import RequestStatus, Sequence


class Scheduler:
    def __init__(
        self,
        allocator: BlockAllocator,
        page_size: int,
        max_running: int = 8,
        watermark_blocks: int = 0,
    ):
        self.allocator = allocator
        self.page_size = page_size
        self.max_running = max_running
        self.watermark = watermark_blocks
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def _release(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.allocator.free(b)
        seq.block_table.clear()
        seq.num_pages = 0

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """返回 (本轮 prefill, 本轮 decode);副作用:抢占与准入。"""
        # 抢占:保证每个 running 序列本轮 decode 至多再要 1 块
        preempted_ids: set[int] = set()
        while True:
            deficit = sum(
                1 for s in self.running if s.num_tokens == s.num_pages * self.page_size
            )
            if self.allocator.num_free >= deficit:
                break
            victim = self.running.pop(0)  # 最老
            victim.status = RequestStatus.WAITING
            self._release(victim)
            self.waiting.appendleft(victim)
            preempted_ids.add(victim.seq_id)

        # 准入:FCFS,受水位与 max_running 约束;刚被抢占者本轮不准入(防抖:
        # 否则同一步"抢占→重准入"空转,下轮重试)
        prefills: list[Sequence] = []
        pending_pages = 0  # 本轮已准入序列将消耗的块数:Engine 尚未真实分配,先记账
        while self.waiting and len(self.running) < self.max_running:
            cand = self.waiting[0]
            if cand.seq_id in preempted_ids:
                break
            need = -(-cand.num_tokens // self.page_size) - cand.num_pages
            if self.allocator.num_free - pending_pages - self.watermark < need:
                break
            self.waiting.popleft()
            pending_pages += need
            cand.status = RequestStatus.RUNNING
            self.running.append(cand)
            prefills.append(cand)

        prefill_ids = {s.seq_id for s in prefills}
        decodes = [s for s in self.running if s.seq_id not in prefill_ids]
        return prefills, decodes
