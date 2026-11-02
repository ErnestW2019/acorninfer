"""M8 Task 1:pager 多槽游标(唯一保护解禁;spec §2.3)。

三个增量方法的等价钉:ensure_append_capacity_n(n=1 ≡ 旧 ensure 逐字节)、
scatter_verify(m=1 ≡ scatter_decode 逐位;m>1 多槽 + _lens 提交 +m)、
set_lens(只回退不前进)。纯 CPU;既有方法零触碰,fast 全套件零换金。
"""

import pytest
import torch

from acorninfer.cache.pager import BlockAllocator, BlockExhausted, PagedCacheView, PagedKVCache

PAGE = 4  # 小页方便测试跨页;page_size=16 的口径在 parametrize 里覆盖


def make_cache(num_blocks=8, kv_heads=2, head_dim=8, layers=2, page_size=PAGE):
    alloc = BlockAllocator(num_blocks)
    return PagedKVCache(layers, kv_heads, head_dim, page_size, alloc, "cpu", torch.float32)


# ---- ensure_append_capacity_n:n=1 等价钉 + 跨页分配 + BlockExhausted 语义 ----


@pytest.mark.parametrize("page_size", [4, 16])
def test_ensure_n_equals_old_for_n1(page_size):
    """ensure_append_capacity_n(layer,1) 与旧 ensure_append_capacity 逐字节同 (page,slot):
    覆盖空表/页内中段/页尾/恰跨页各形态;ensure 家族均不提交 _lens。"""
    kv, dim = 2, 8
    for base in (0, 3, page_size - 1, page_size, page_size + 2):
        k = torch.randn(1, kv, base, dim)
        v = torch.randn(1, kv, base, dim)
        view_old = PagedCacheView(make_cache(page_size=page_size))
        view_new = PagedCacheView(make_cache(page_size=page_size))
        if base:
            view_old.write(0, k, v)
            view_new.write(0, k, v)
        old = view_old.ensure_append_capacity(0)
        new = view_new.ensure_append_capacity_n(0, 1)
        assert new == [old]
        assert len(view_old.table) == len(view_new.table)
        assert view_old.seq_len == view_new.seq_len == base  # ensure 家族不提交长度


def test_ensure_n_allocates_across_pages():
    """n=5、page=4 从 base=3 起跨 2 页:分配数 = write() 的 needed 公式
    (base+n+page-1)//page;返回游标 = 逐位置 (table[p//page], p%page);
    BlockExhausted 语义同 write:中途抛时已拿页留 table、_lens 未提交。"""
    cache = make_cache(layers=1)
    view = PagedCacheView(cache)
    view.write(0, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))  # base=3,占 1 页
    live_before = cache.allocator.num_live
    targets = view.ensure_append_capacity_n(0, 5)
    # needed 公式与 write 逐字同源:5 个新位从槽 3 起 → 恰补 1 页(共 2 页)
    assert len(view.table) == (3 + 5 + PAGE - 1) // PAGE == 2
    assert cache.allocator.num_live - live_before == 1
    assert targets == [
        (view.table[0], 3),
        (view.table[1], 0), (view.table[1], 1), (view.table[1], 2), (view.table[1], 3),
    ]
    assert view.seq_len == 3  # ensure 不提交长度

    # 分配行为 ≡ write:同 base 同 s,write 5 个 token 后表长同公式
    view_w = PagedCacheView(make_cache(layers=1))
    view_w.write(0, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    view_w.write(0, torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))
    assert len(view_w.table) == 2 and view_w.seq_len == 8

    # BlockExhausted 同 write:needed=2 只供 1 块 → 首次分配成功(留 table),次轮抛
    alloc = BlockAllocator(1)
    cache_x = PagedKVCache(1, 2, 8, PAGE, alloc, "cpu", torch.float32)
    view_x = PagedCacheView(cache_x)
    with pytest.raises(BlockExhausted):
        view_x.ensure_append_capacity_n(0, 5)
    assert len(view_x.table) == 1  # 已拿到的页留在 table(释放后重试连贯,同 update 钉死语义)
    assert view_x.seq_len == 0     # _lens 未提交
    for b in view_x.table:
        alloc.free(b)
    alloc.assert_balanced()


# ---- scatter_verify:m=1 等价钉 + 多槽写入/_lens 提交 ----


def test_scatter_verify_m1_equals_scatter_decode():
    """m=1:两方法对同一输入写页逐位一致 + _lens 提交一致。

    覆盖三种槽位形态:页内尾槽、恰跨页(slot 0)、页中槽(与 scatter_decode
    的既有等价钉同构);层间锁步:只提交被写层。
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
        if mode == "decode":
            targets = [views[b].ensure_append_capacity(0) for b in range(B)]
            pages = torch.tensor([p for p, _ in targets], dtype=torch.long)
            slots = torch.tensor([s for _, s in targets], dtype=torch.long)
            cache.scatter_decode(views, 0, k_new, v_new, pages, slots)
        else:
            targets = [views[b].ensure_append_capacity_n(0, 1) for b in range(B)]
            pages = torch.tensor([[p for p, _ in t] for t in targets], dtype=torch.long)
            slots = torch.tensor([[s for _, s in t] for t in targets], dtype=torch.long)
            cache.scatter_verify(views, 0, k_new, v_new, pages, slots)
        return cache, views

    cache_a, views_a = run("decode")
    cache_b, views_b = run("verify")
    for b in range(B):
        assert views_a[b].table == views_b[b].table
        assert views_a[b]._lens == views_b[b]._lens
    for layer in range(layers):
        assert torch.equal(cache_a.k_pages[layer], cache_b.k_pages[layer])
        assert torch.equal(cache_a.v_pages[layer], cache_b.v_pages[layer])


def test_scatter_verify_multislot_and_lens():
    """m=5、B=3:写 [B,kv,5,D] 于 [B,5] 目标(ensure_n 步首算好);lens +5;
    页值逐位核对 (pages[b,j],slots[b,j]) <- k[b,:,j,:];只提交被写层(层间锁步)。"""
    torch.manual_seed(1)
    B, m, layers = 3, 5, 2
    lens = (2, 4, 6)  # 槽位形态各异:页中段起(跨页)/恰边界起/页内跨页
    cache = PagedKVCache(layers, 2, 8, PAGE, BlockAllocator(8), "cpu", torch.float32)
    views = [PagedCacheView(cache) for _ in range(B)]
    pre_k = [torch.randn(1, 2, n, 8) for n in lens]
    pre_v = [torch.randn(1, 2, n, 8) for n in lens]
    for i in range(B):
        for layer in range(layers):
            views[i].write(layer, pre_k[i], pre_v[i])

    k = torch.randn(B, 2, m, 8)
    v = torch.randn(B, 2, m, 8)
    targets = [views[b].ensure_append_capacity_n(0, m) for b in range(B)]
    pages = torch.tensor([[p for p, _ in t] for t in targets], dtype=torch.long)   # [B, m]
    slots = torch.tensor([[s for _, s in t] for t in targets], dtype=torch.long)  # [B, m]
    cache.scatter_verify(views, 0, k, v, pages, slots)

    assert targets[0][0] == (views[0].table[0], 2)  # 页中段起
    assert targets[0][2] == (views[0].table[1], 0)  # 跨页续接(slot 0)
    assert targets[1][0] == (views[1].table[1], 0)  # 恰边界起
    for b in range(B):
        assert views[b]._lens == {0: lens[b] + m, 1: lens[b]}  # 只提交被写层,+m
        for j in range(m):
            p, s = targets[b][j]
            assert torch.equal(cache.k_pages[0][p, s], k[b, :, j, :])
            assert torch.equal(cache.v_pages[0][p, s], v[b, :, j, :])


# ---- set_lens:只回退 + 死槽复用闭环 ----


def test_set_lens_rollback_guarded():
    """set_lens 只回退不前进:new_len ≤ 已写 len 合法(含相等);越界 ValueError;
    回退后 seq_len 即新值,gather 读侧截到新长(死槽 [new_len, 旧 len) 不可见)。
    engine 侧 36 层循环由锁步调用方保证(注释口径同 scatter_decode)。"""
    cache = make_cache(layers=1)
    view = PagedCacheView(cache)
    with pytest.raises(ValueError):
        view.set_lens(0, 1)  # 空层(现长 0)前进越界
    view.set_lens(0, 0)      # 相等:合法

    k = torch.randn(1, 2, 6, 8)
    view.write(0, k, torch.randn(1, 2, 6, 8))
    with pytest.raises(ValueError):
        view.set_lens(0, 7)  # 前进越界(前进只属 write/scatter 家族)
    assert view.seq_len == 6
    view.set_lens(0, 6)      # 相等 no-op:合法
    view.set_lens(0, 4)      # 回退(engine 场景:接受 a=2 → len+a+1)
    assert view.seq_len == 4
    k_g = view._gather(0, cache.k_pages)
    assert k_g.shape == (1, 2, 4, 8)
    assert torch.equal(k_g, k[:, :, :4])  # 读侧只到 lens:死槽位不出现


def test_dead_slot_reuse_roundtrip():
    """写 m=5 → set_lens 回 a=2 → 下步 ensure_n(5) 从 len'=base+3 起复用槽位
    (已分配的页内槽直接复用、跨页续接分配)→ 复写 → gather 读侧只到 lens:
    死槽永不可见(spec §2.3:零泄漏、零分配器回滚、free-list 语义不变)。"""
    torch.manual_seed(2)
    kv, dim, m = 2, 8, 5
    cache = make_cache(num_blocks=8, layers=1)
    view = PagedCacheView(cache)
    base = 2
    pre = torch.randn(1, kv, base, dim)
    view.write(0, pre, torch.randn(1, kv, base, dim))

    # 步 1:verify 写 m=5 于游标 [base, base+m):lens=7、占 2 页
    k1, v1 = torch.randn(1, kv, m, dim), torch.randn(1, kv, m, dim)
    t1 = view.ensure_append_capacity_n(0, m)
    pages1 = torch.tensor([[p for p, _ in t1]], dtype=torch.long)
    slots1 = torch.tensor([[s for _, s in t1]], dtype=torch.long)
    cache.scatter_verify([view], 0, k1, v1, pages1, slots1)
    assert view.seq_len == base + m == 7 and len(view.table) == 2

    # 步 2:接受 a=2 → lens = base+a+1 = 5;死槽 = 位置 5,6(t1[3], t1[4])
    a = 2
    view.set_lens(0, base + a + 1)
    assert view.seq_len == 5

    # 步 3:下步 verify(m=5)从 len' 起要 5 个游标位:位置 5,6 是死槽
    # (页表项已在 → 直接复用,零分配);位置 7,8,9 跨页续接(恰新拿 1 页)
    live, free = cache.allocator.num_live, cache.allocator.num_free
    t2 = view.ensure_append_capacity_n(0, m)
    assert t2[:2] == t1[3:5] == [(view.table[1], 1), (view.table[1], 2)]  # 死槽复用
    assert t2[2] == (view.table[1], 3)  # 已分配页的未用槽:页表项已在,零分配
    assert t2[3:] == [(view.table[2], 0), (view.table[2], 1)]  # 跨页续接分配
    assert len(view.table) == 3
    assert (cache.allocator.num_live, cache.allocator.num_free) == (live + 1, free - 1)

    # 步 4:复写死槽 + 续接位 → lens 前进到 10
    k2, v2 = torch.randn(1, kv, m, dim), torch.randn(1, kv, m, dim)
    pages2 = torch.tensor([[p for p, _ in t2]], dtype=torch.long)
    slots2 = torch.tensor([[s for _, s in t2]], dtype=torch.long)
    cache.scatter_verify([view], 0, k2, v2, pages2, slots2)
    assert view.seq_len == 10

    # 步 5:读侧(gather)只到 lens:位置 <5 = 已提交前缀(pre + k1 存活前缀),
    # 位置 5..9 = 复写后的新值(死槽旧值被覆盖,无残留、无越 lens 暴露)
    k_g = view._gather(0, cache.k_pages)
    v_g = view._gather(0, cache.v_pages)
    assert k_g.shape == (1, kv, 10, dim)
    assert torch.equal(k_g[:, :, :base], pre[0][None])                        # [kv, len, D] 布局
    assert torch.equal(k_g[:, :, base:base + a + 1], k1[0][:, : a + 1][None])
    assert torch.equal(k_g[:, :, base + a + 1:], k2[0][None])
    assert torch.equal(v_g[:, :, base + a + 1:], v2[0][None])
