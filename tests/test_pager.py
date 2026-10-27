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
