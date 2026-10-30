import pytest
import torch

from acorninfer.cache.pager import BlockAllocator, BlockExhausted, PagedCacheView, PagedKVCache

PAGE = 4  # 小页方便测试跨页


def make_cache(num_blocks=4, kv_heads=2, head_dim=8, layers=2):
    alloc = BlockAllocator(num_blocks)
    return PagedKVCache(layers, kv_heads, head_dim, PAGE, alloc, "cpu", torch.float32)


def test_allocate_returns_unique_blocks_then_exhausts():
    a = BlockAllocator(2)
    b1, b2 = a.allocate(), a.allocate()
    assert {b1, b2} == {0, 1}
    assert a.num_free == 0 and a.num_live == 2 and a.total == 2
    with pytest.raises(BlockExhausted):
        a.allocate()


def test_free_returns_block_to_pool_lifo():
    a = BlockAllocator(2)
    x, y = a.allocate(), a.allocate()
    a.free(x)
    a.free(y)
    assert a.allocate() == y  # LIFO:后释放先复用
    assert a.allocate() == x


def test_double_free_raises():
    a = BlockAllocator(1)
    b = a.allocate()
    a.free(b)
    with pytest.raises(ValueError, match="double free"):
        a.free(b)


def test_assert_balanced_detects_leak():
    a = BlockAllocator(1)
    b = a.allocate()
    with pytest.raises(AssertionError, match="leak"):
        a.assert_balanced()
    a.free(b)
    a.assert_balanced()  # 不抛


def test_view_update_grows_and_gathers_like_concat():
    cache = make_cache()
    view = PagedCacheView(cache)
    k1 = torch.randn(1, 2, 3, 8)
    v1 = torch.randn(1, 2, 3, 8)
    k_full, v_full = view.update(0, k1, v1)
    assert view.seq_len == 3 and len(view.table) == 1
    torch.testing.assert_close(k_full, k1)
    # 跨页:3 + 3 = 6 > PAGE=4,需要第 2 块
    k2 = torch.randn(1, 2, 3, 8)
    v2 = torch.randn(1, 2, 3, 8)
    k_full, v_full = view.update(0, k2, v2)
    assert view.seq_len == 6 and len(view.table) == 2
    assert torch.equal(k_full[:, :, :3], k1)
    assert torch.equal(k_full[:, :, 3:], k2)
    assert torch.equal(v_full[:, :, :3], v1)
    assert torch.equal(v_full[:, :, 3:], v2)


def test_pages_independent_across_layers_and_views():
    """层间与视图间互不串扰;seq_len 读 layer 0(M1 KVCache 锁步约定)。"""
    cache = make_cache()
    va, vb = PagedCacheView(cache), PagedCacheView(cache)
    ka = torch.randn(1, 2, 2, 8)
    va.update(0, ka, torch.randn(1, 2, 2, 8))
    kb = torch.randn(1, 2, 5, 8)
    vb.update(1, kb, torch.randn(1, 2, 5, 8))  # vb 只写了 layer 1
    assert va.seq_len == 2 and vb.seq_len == 0  # seq_len 以 layer 0 为准
    k1, _ = vb.update(1, torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))
    assert torch.equal(k1[:, :, :5], kb)  # vb 的 layer-1 数据完好,新 token 接在其后
    k0, _ = va.update(0, torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))
    assert torch.equal(k0[:, :, :2], ka)  # 另一 view 的写入不影响本 view


def test_gather_matches_naive_concat_exactly():
    """分页存储 + 块表 gather 必须与朴素拼接逐位一致(M2 等价性的根)。"""
    cache = make_cache()
    view = PagedCacheView(cache)
    naive_k = []
    for s in (5, 1, 4):  # 故意跨 3 块:5+1+4=10 > 2 页
        k = torch.randn(1, 2, s, 8)
        naive_k.append(k[0])
        view.update(0, k, torch.randn(1, 2, s, 8))
    k_full, _ = view.update(0, torch.randn(1, 2, 0, 8), torch.randn(1, 2, 0, 8))
    # k[0] 是 [kv, s, D]:token 轴在 dim 1,朴素拼接按 token 轴
    assert torch.equal(k_full[0], torch.cat(naive_k, dim=1))


def test_view_allocation_exhausts_pool():
    alloc = BlockAllocator(2)
    cache = PagedKVCache(2, 2, 8, PAGE, alloc, "cpu", torch.float32)
    view = PagedCacheView(cache)
    with pytest.raises(BlockExhausted):
        view.update(0, torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8))
    # 中断的 update 已把拿到的块记进 table;释放后守卫必须通过
    for b in view.table:
        alloc.free(b)
    alloc.assert_balanced()


def test_update_retry_after_exhaustion_is_coherent():
    """耗尽 → 释放 → 重试:零部分写入、无重复写(Task 8 抢占-重准入路径依赖)。"""
    alloc = BlockAllocator(2)
    cache = PagedKVCache(2, 2, 8, PAGE, alloc, "cpu", torch.float32)
    view = PagedCacheView(cache)
    with pytest.raises(BlockExhausted):
        view.update(0, torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8))
    assert view.seq_len == 0  # 中断的 update 未提交任何长度
    for b in view.table:
        alloc.free(b)
    view.table.clear()
    k = torch.randn(1, 2, 6, 8)
    k_full, _ = view.update(0, k, torch.randn(1, 2, 6, 8))
    assert view.seq_len == 6
    assert torch.equal(k_full, k)  # 重试写入正确,无残留脏数据


def test_write_then_gather_equals_update():
    """write + gather ≡ update(逐位);块表分配行为一致。"""
    cache = make_cache(num_blocks=8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    t1: list[int] = []
    view_a = PagedCacheView(cache, t1)
    k_up, v_up = view_a.update(0, k, v)

    t2: list[int] = []
    view_b = PagedCacheView(cache, t2)
    view_b.write(0, k, v)
    k_g = view_b._gather(0, cache.k_pages)
    v_g = view_b._gather(0, cache.v_pages)

    assert torch.equal(k_up, k_g) and torch.equal(v_up, v_g)
    # 同一共享分配器(LIFO)给两个视图的物理块号必然不同;可比的是分配行为:
    # 相同写入形状 → 相同的块表长度。
    assert len(t1) == len(t2) == 1 and view_a.seq_len == view_b.seq_len == 3


# ---- M3 Task 9:decode 发射合批的新 API(ensure_append_capacity + scatter_decode)----


def test_ensure_append_capacity_allocates_only_on_page_boundary():
    """跨页序列(seq_len % page == 0)追加需新页;否则 no-op。

    返回追加位 (page_id, slot_id);容量确保绝不触碰 _lens(提交是 scatter 的职责)。
    """
    cache = make_cache(num_blocks=8)
    view = PagedCacheView(cache)
    view.write(0, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))  # 3 < PAGE:页内有余量
    live_before, free_before = cache.allocator.num_live, cache.allocator.num_free
    assert view.ensure_append_capacity(0) == (view.table[0], 3)
    assert len(view.table) == 1  # 未跨页:零分配
    assert (cache.allocator.num_live, cache.allocator.num_free) == (live_before, free_before)
    view.write(0, torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))  # → 4:页满
    old_table = list(view.table)
    assert view.ensure_append_capacity(0) == (view.table[1], 0)  # 跨页:新页 + slot 0
    assert len(view.table) == 2 and view.table[:1] == old_table
    assert view.seq_len == 4  # 容量确保不提交长度


def test_ensure_capacity_noop_when_all_sequences_have_page_room():
    """全部序列不跨页:对每个 view 均 no-op,分配器零扰动(步首热路径的常态)。"""
    cache = make_cache(num_blocks=8)
    views = []
    for n in (1, 2, 3):  # 全部 < PAGE
        v = PagedCacheView(cache)
        v.write(0, torch.randn(1, 2, n, 8), torch.randn(1, 2, n, 8))
        views.append(v)
    live, free = cache.allocator.num_live, cache.allocator.num_free
    targets = [v.ensure_append_capacity(0) for v in views]
    assert (cache.allocator.num_live, cache.allocator.num_free) == (live, free)
    assert targets == [(views[i].table[0], i + 1) for i in range(3)]
    assert all(len(v.table) == 1 for v in views)


def test_batched_ensure_exhaustion_keeps_allocated_pages_lens_uncommitted():
    """步首批量容量确保中途 BlockExhausted:已分配页留在各自 table
    (与 write/update 的重试语义同:test_update_retry_after_exhaustion_is_coherent),
    所有 _lens 未提交 —— 释放后可重试。"""
    alloc = BlockAllocator(4)
    cache = PagedKVCache(1, 2, 8, PAGE, alloc, "cpu", torch.float32)
    views = [PagedCacheView(cache) for _ in range(3)]
    for v in views:  # 各恰占满 1 页:后续追加全部跨页
        v.write(0, torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    with pytest.raises(BlockExhausted):
        [v.ensure_append_capacity(0) for v in views]
    assert len(views[0].table) == 2  # 已成功跨页拿到的页留在表内
    assert all(len(v.table) == 1 for v in views[1:])
    assert all(v.seq_len == 4 for v in views)  # ensure 不提交长度
    for v in views:
        for b in v.table:
            alloc.free(b)
    alloc.assert_balanced()


def test_scatter_decode_equals_per_view_write_bitwise():
    """合批 scatter ≡ 逐序列 write(逐位):k/v 页内容、块表、_lens 全同。

    覆盖三种槽位形态:页内尾槽、恰跨页(slot 0)、页中槽;
    scatter 只提交被写层的 _lens,其余层不受扰(层间锁步由调用方保证)。
    """
    torch.manual_seed(0)
    B, layers = 3, 2
    lens = (3, 4, 5)  # view1 恰在页边界(PAGE=4):追加必跨页
    pre_k = [torch.randn(1, 2, n, 8) for n in lens]
    pre_v = [torch.randn(1, 2, n, 8) for n in lens]
    k_new = torch.randn(B, 2, 1, 8)
    v_new = torch.randn(B, 2, 1, 8)

    def run(mode):
        cache = PagedKVCache(layers, 2, 8, PAGE, BlockAllocator(8), "cpu", torch.float32)
        views = [PagedCacheView(cache) for _ in range(B)]
        for i in range(B):  # 预置等长等内容的页数据:两 run 分配序逐块一致
            for layer in range(layers):
                views[i].write(layer, pre_k[i], pre_v[i])
        if mode == "loop":
            for b in range(B):
                views[b].write(0, k_new[b : b + 1], v_new[b : b + 1])
            targets = None
        else:
            targets = [views[b].ensure_append_capacity(0) for b in range(B)]
            pages = torch.tensor([p for p, _ in targets], dtype=torch.long)
            slots = torch.tensor([s for _, s in targets], dtype=torch.long)
            cache.scatter_decode(views, 0, k_new, v_new, pages, slots)
        return cache, views, targets

    cache_a, views_a, _ = run("loop")
    cache_b, views_b, targets_b = run("batch")
    # 槽位形态自查:3→页内尾槽、4→跨页新页 slot 0、5→次页 slot 1
    assert targets_b == [
        (views_b[0].table[0], 3), (views_b[1].table[1], 0), (views_b[2].table[1], 1),
    ]
    for i in range(B):
        assert views_a[i].table == views_b[i].table
        assert views_a[i]._lens == views_b[i]._lens
    assert views_b[1]._lens == {0: 5, 1: 4}  # 只提交被写层(layer 0 = pre 4 + 1)
    for layer in range(layers):
        assert torch.equal(cache_a.k_pages[layer], cache_b.k_pages[layer])
        assert torch.equal(cache_a.v_pages[layer], cache_b.v_pages[layer])


def test_write_exhaustion_leaves_no_partial_state():
    """块不足时 write 原样抛 BlockExhausted、_lens 未提交(与 update 同语义)。"""
    cache = make_cache(num_blocks=1, layers=1)
    t: list[int] = []
    view = PagedCacheView(cache, t)
    k1 = torch.randn(1, 2, 3, 8)
    view.write(0, k1, torch.randn(1, 2, 3, 8))  # 占 1 块
    with pytest.raises(BlockExhausted):
        view.write(0, torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8))
    assert view.seq_len == 3  # 失败写不提交
    # 页面内容也不许有部分写入:失败写零 scatter,已提交数据在页中原样
    assert torch.equal(view._gather(0, cache.k_pages), k1)
