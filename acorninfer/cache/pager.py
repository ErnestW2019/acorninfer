"""Paged KV cache: block allocator (this file grows PagedKVCache in Task 4).

Spec §3.5: 块表 + page size 16;分配器先做线性空闲表,M4 加哈希前缀匹配。
Spec §5:   KV 泄漏 debug 守卫 —— assert_balanced 在引擎排空后必须通过。
"""

from __future__ import annotations

import torch


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
            raise ValueError(f"block {block} is not live (double free or invalid)")
        self._live.remove(block)
        self._free.append(block)

    def assert_balanced(self) -> None:
        """Debug 守卫:引擎排空后必须零 live 块,否则 = KV 泄漏。

        显式 raise 而非 assert 语句:本方法的存在意义就是这个检查,
        不能在 `python -O` 下静默失效(spec §5 验收依赖它)。
        """
        if self._live:
            raise AssertionError(f"KV leak: blocks {sorted(self._live)} never freed")


class PagedKVCache:
    """物理页存储:每层 k_pages/v_pages [num_blocks, page_size, kv_heads, head_dim]。

    序列通过块表(list[block_id])引用自己的页;token 顺序 = 块表序 × 页内偏移。
    M3 的 Triton paged attention 将直接按页读取;本实现的 gather-to-contiguous
    就是它的数值 reference。
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        allocator: BlockAllocator,
        device,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.allocator = allocator
        self.device = device
        self.dtype = dtype
        shape = (allocator.total, page_size, num_kv_heads, head_dim)
        self.k_pages = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v_pages = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]

    def scatter_decode(
        self,
        views: list[PagedCacheView],
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        pages: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """decode 步合批写入(M3 Task 9):单次高级索引 scatter + 逐 view 提交 _lens。

        k/v: [B, kv_heads, 1, head_dim];pages/slots: [B] long(各序列追加位,
        由 PagedCacheView.ensure_append_capacity 在步首算好/分配好)。
        位值语义 ≡ 逐 view.write(layer_idx, k[b:b+1], v[b:b+1])(单测钉住):
        无算术、无 cast 舍入差异,同一 (page,slot) 写入同值。
        容量分配不在此发生(步首 ensure 已完成),故本方法不可能抛 BlockExhausted;
        _lens 提交 = 各 view 现长 +1(decode 每层恰追加 1 token)。
        层间锁步(各层等长,引擎路径恒成立)由调用方保证。
        """
        B = len(views)
        assert k.shape[2] == 1 and v.shape[2] == 1, "scatter_decode is decode-only (s=1)"
        assert B == k.shape[0] == v.shape[0] == pages.shape[0] == slots.shape[0]
        # 与 write 的逐 token 行对齐同构:token 轴转前 (permute(1,0,2) 的 s=1 形态)
        self.k_pages[layer_idx][pages, slots] = k[:, :, 0, :].to(self.dtype)
        self.v_pages[layer_idx][pages, slots] = v[:, :, 0, :].to(self.dtype)
        for view in views:
            view._lens[layer_idx] = view._lens.get(layer_idx, 0) + 1


class PagedCacheView:
    """每序列视图:块表 + update()/seq_len,与 M1 KVCache 接口镜像。

    长度按层独立记录(forward 每层各调一次 update);seq_len 读 layer 0,
    与 M1 KVCache"层间锁步、layer 0 为准"的约定一致。
    table 与 Sequence.block_table 共享同一 list 对象(engine 负责传递;
    传入副本会在引擎释放/复用块时造成静默的跨序列 KV 污染)。
    """

    def __init__(self, cache: PagedKVCache, table: list[int] | None = None):
        self.cache = cache
        self.table: list[int] = table if table is not None else []
        self._lens: dict[int, int] = {}  # per-layer written length

    @property
    def seq_len(self) -> int:
        return self._lens.get(0, 0)

    def write(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """纯 scatter:写入新 token 的 k/v;不读不 gather(读侧由调用方定:
        update() = write + gather;M3 Triton 后端只 write,attention 直接读页)。
        分配在 scatter 之前全部完成:BlockExhausted 中断零部分写入、_lens 未提交。

        k_new/v_new: [1, kv_heads, s, head_dim];块不足时按需 allocate。
        """
        c = self.cache
        s = k_new.shape[2]
        page = c.page_size
        base = self._lens.get(layer_idx, 0)
        needed = (base + s + page - 1) // page
        while len(self.table) < needed:
            self.table.append(c.allocator.allocate())
        pos = torch.arange(base, base + s)
        # dtype=torch.long 必须显式:s=0 时空 list 会默认造出 float 张量,当索引会炸
        rows = torch.tensor(
            [self.table[p // page] for p in pos.tolist()], device=c.device, dtype=torch.long
        )
        slots = (pos % page).to(c.device)
        # scatter 目标 [rows,slots] 是 [s, kv, D](token 轴在前);k_new[0] 是
        # [kv, s, D] → permute(1,0,2) 后逐 token 对齐(rows[i],slots[i]) <- token i
        c.k_pages[layer_idx][rows, slots] = k_new[0].permute(1, 0, 2).to(c.dtype)
        c.v_pages[layer_idx][rows, slots] = v_new[0].permute(1, 0, 2).to(c.dtype)
        self._lens[layer_idx] = base + s

    def ensure_append_capacity(self, layer_idx: int) -> tuple[int, int]:
        """decode 合批路径的步首容量确保(M3 Task 9):若在 _lens[layer_idx] 处
        追加 1 token 会跨页(_lens[layer_idx] % page_size == 0),先分配新页号
        追加进 table,否则 no-op(页内余量足够)。

        返回追加位 (page_id, slot_id) —— scatter_decode 的写入目标。
        BlockExhausted 语义与 write 一致:分配循环中途抛出时,已拿到的页
        留在 table、_lens 未提交(释放后重试连贯,同 update 的钉死语义)。
        """
        base = self._lens.get(layer_idx, 0)
        page = self.cache.page_size
        needed = base // page + 1  # 追加 1 token 落在第 base//page 页
        while len(self.table) < needed:
            self.table.append(self.cache.allocator.allocate())
        return self.table[base // page], base % page

    def update(
        self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写入并返回全量 [1, kv_heads, seq_len, head_dim](write + gather)。"""
        self.write(layer_idx, k_new, v_new)
        return (self._gather(layer_idx, self.cache.k_pages),
                self._gather(layer_idx, self.cache.v_pages))

    def _gather(self, layer_idx: int, pages: list[torch.Tensor]) -> torch.Tensor:
        c = self.cache
        # dtype=torch.long 与 update 的 rows 对称:空表时 torch.tensor([]) 会默认 float
        t = pages[layer_idx][torch.tensor(self.table, device=c.device, dtype=torch.long)]
        t = t.reshape(-1, c.num_kv_heads, c.head_dim)[: self._lens.get(layer_idx, 0)]
        return t.permute(1, 0, 2)[None]                                  # [1, kv, len, D]
