"""Paged KV cache: block allocator (this file grows PagedKVCache in Task 4).

Spec §3.5: 块表 + page size 16;分配器先做线性空闲表,M4 加哈希前缀匹配。
Spec §5:   KV 泄漏 debug 守卫 —— assert_balanced 在引擎排空后必须通过。
"""

from __future__ import annotations


class BlockExhausted(Exception):
    """物理块耗尽;调度器以此触发排队/抢占。"""


class BlockAllocator:
    """Linear free-list allocator for physical KV blocks (LIFO reuse)."""

    def __init__(self, num_blocks: int):
        self._free: list[int] = list(range(num_blocks))[::-1]  # LIFO
        self._live: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_live(self) -> int:
        return len(self._live)

    @property
    def total(self) -> int:
        return len(self._free) + len(self._live)

    def allocate(self) -> int:
        if not self._free:
            raise BlockExhausted("no free KV blocks")
        b = self._free.pop()
        self._live.add(b)
        return b

    def free(self, block: int) -> None:
        if block not in self._live:
            raise ValueError(f"double free: block {block} is not live")
        self._live.discard(block)
        self._free.append(block)

    def assert_balanced(self) -> None:
        """Debug 守卫:引擎排空后必须零 live 块,否则 = KV 泄漏。"""
        assert not self._live, f"KV leak: blocks {sorted(self._live)} never freed"
