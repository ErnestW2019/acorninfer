import pytest

from acorninfer.cache.pager import BlockAllocator, BlockExhausted


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
