"""FCFS 调度器:准入 + 水位 + 抢占最老(recompute)。spec §5。

策略:KV 不足先排队;running 增长可能耗尽块时,抢占最老请求
(释放其全部块,回 waiting 队首,重被准入时对 all_ids 重新 prefill)。
"""

from __future__ import annotations

from collections import deque

from acorninfer.cache.pager import BlockAllocator
from acorninfer.engine.sequence import RequestStatus, Sequence


class Scheduler:
    """FCFS 调度器。

    退休契约(Task 8 Engine 负责):序列结束后由引擎从 running 移除、
    状态置 FINISHED、经 view 的共享块表释放块 —— 调度器自身不退休序列,
    忘记退休的序列会一直占 deficit 与 max_running 名额。
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        page_size: int,
        max_running: int = 8,
        watermark_blocks: int = 0,
        chunked_budget: int | None = None,
    ):
        self.allocator = allocator
        self.page_size = page_size
        self.max_running = max_running
        self.watermark = watermark_blocks
        self.chunked_budget = chunked_budget
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def _release(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.allocator.free(b)
        seq.block_table.clear()
        seq.num_pages = 0
        seq.prefill_cursor = 0  # 被抢占者重算:KV 已清空,chunk 化与 decode 级受害者同归零

    def _is_mid_prefill(self, seq: Sequence) -> bool:
        """prompt prefill 未完成:尚无生成 token 且 cursor < prompt 长度。"""
        return not seq.generated_ids and seq.prefill_cursor < len(seq.prompt_ids)

    def _plan_running_chunks(
        self,
    ) -> tuple[list[tuple[Sequence, int, int]], int, int | None]:
        """running 中 mid-prefill 序列的本步续读 chunk(FCFS 按 running 序)。

        返回 (chunks, 本步追加页数, 剩余预算)。追加页 = ceil((cursor+len)/page)
        - num_pages,属"步内即时占块"(与页满 decode 同理,准入不得动用)。
        预算耗尽时该序列本步停滞(不 chunk 也不 decode),下一轮仍 FCFS 优先。
        """
        chunks: list[tuple[Sequence, int, int]] = []
        append_pages = 0
        budget_left = self.chunked_budget
        for seq in self.running:
            if not self._is_mid_prefill(seq):
                continue
            remaining = len(seq.prompt_ids) - seq.prefill_cursor
            length = remaining if budget_left is None else min(remaining, budget_left)
            if length <= 0:
                continue  # 预算耗尽:本步停滞
            if budget_left is not None:
                budget_left -= length
            append = -(-(seq.prefill_cursor + length) // self.page_size) - seq.num_pages
            append_pages += max(append, 0)
            chunks.append((seq, seq.prefill_cursor, length))
        return chunks, append_pages, budget_left

    def schedule(
        self,
    ) -> tuple[list[tuple[Sequence, int, int]], list[Sequence]]:
        """返回 (本轮 prefill chunks, 本轮 decode);副作用:抢占与准入。

        chunks = list[(Sequence, start, length)]:start = seq.prefill_cursor,
        每序列每步至多一个 chunk。chunked_budget=None(默认)时每序列一条全量
        (start=0,length=len(all_ids)),且 running 中不存在 mid-prefill 序列
        (准入步即写完全量),故调度决策与本函数 chunk 化之前逐字等价。

        正确性不变式:本轮页满 decode 的占块发生在步内(下次 schedule() 的
        抢占循环来不及反应),因此准入预算必须预留这部分(reserve);
        watermark 只是额外策略余量,不再承担正确性。chunk 化后 running 中
        mid-prefill 序列本步 chunk 的追加页同属"步内即时占块",一并入 reserve
        (chunked_budget=None 时该项恒为 0)——否则会出现 chunking 之前
        不存在的步内 BlockExhausted。

        预算分配:FCFS —— 先 running 中 mid-prefill,后 waiting 新准入
        (保持准入顺序);waiting 可被部分准入(首块截断)。抢占的重算候选
        (generated 非空)豁免预算:重算走既有全量 prefill 路径,是正确性
        路径而非吞吐路径。
        """
        # 抢占:保证 running 中 decode 级序列本轮 decode 至多再要 1 块,且
        # mid-prefill 序列本步 chunk 的追加页有着落。
        # 精确判据是 >:调度时 KV 已写 num_tokens-1 个 token,本轮 decode 写第
        # num_tokens 个 —— T==S·page 时恰好落进最后一格(不分配),溢出一格
        # (T>S·page,如刚 prefill 完 T=S+1)才真要块;漏记必崩。
        # mid-prefill 序列不计入 deficit(本步不 decode),其需求由 chunk
        # 追加页单独计 —— None 预算下不存在 mid-prefill,公式退化为原样。
        preempted_ids: set[int] = set()
        while True:
            chunks, append_pages, budget_left = self._plan_running_chunks()
            deficit = sum(
                1
                for s in self.running
                if not self._is_mid_prefill(s)
                and s.num_tokens > s.num_pages * self.page_size
            )
            if self.allocator.num_free >= deficit + append_pages:
                break
            victim = self.running.pop(0)  # 最老
            victim.status = RequestStatus.WAITING
            self._release(victim)
            self.waiting.appendleft(victim)
            preempted_ids.add(victim.seq_id)
        reserve = deficit + append_pages  # 本轮步内即时占块;准入不得动用

        # 准入:FCFS,受水位与 max_running 约束;刚被抢占者本轮不准入(防抖:
        # 否则同一步"抢占→重准入"空转,下轮重试)。准入仍按全 prompt/all_ids
        # 预留页(need 覆盖最终全部页),chunk 只是本步落地的前缀。
        pending_pages = 0  # 本轮已准入序列将消耗的块数:Engine 尚未真实分配,先记账
        while self.waiting and len(self.running) < self.max_running:
            cand = self.waiting[0]
            if cand.seq_id in preempted_ids:
                break
            need = -(-cand.num_tokens // self.page_size) - cand.num_pages
            if self.allocator.num_free - pending_pages - reserve - self.watermark < need:
                break
            if cand.generated_ids:
                # 重算候选(抢占的 decode 级序列):全量 all_ids,豁免预算,
                # 引擎按 start==0 且 length==len(all_ids) 路由到既有 prefill 路径
                start, length = 0, len(cand.all_ids)
            else:
                start = cand.prefill_cursor  # waiting 序列 cursor 恒为 0(新提交或被抢占者已归零)
                remaining = len(cand.prompt_ids) - cand.prefill_cursor
                length = remaining if budget_left is None else min(remaining, budget_left)
                if length <= 0:
                    break  # 本步 prefill 预算耗尽:FCFS 队头阻塞,后续一并不准入
                if budget_left is not None:
                    budget_left -= length
            self.waiting.popleft()
            pending_pages += need
            cand.status = RequestStatus.RUNNING
            self.running.append(cand)
            chunks.append((cand, start, length))

        chunked_ids = {s.seq_id for s, _, _ in chunks}
        decodes = [s for s in self.running if s.seq_id not in chunked_ids]
        return chunks, decodes
