"""单序列分页路径 == 朴素 cache 路径 == 无 cache 路径(逐位)。

gather 只做拷贝不改数值,SDPA 输入逐位相同 → 输出必须 torch.equal。
任何一处 fail 都意味着分页语义(块表序、页内偏移、部分页截断)有 bug。
"""

import pytest
import torch

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache


@pytest.fixture
def env(tiny_model):
    alloc = BlockAllocator(16)
    cache = PagedKVCache(
        tiny_model.cfg.num_hidden_layers,
        tiny_model.cfg.num_key_value_heads,
        tiny_model.cfg.head_dim,
        4,  # 小页,强制频繁跨页
        alloc,
        "cpu",
        torch.bfloat16,
    )
    return tiny_model, cache, alloc


def _fresh_view(cache):
    return PagedCacheView(cache)


def _free_view(view, alloc):
    for b in view.table:
        alloc.free(b)


def test_paged_prefill_equals_no_cache(env):
    model, cache, alloc = env
    ids = torch.randint(0, 256, (1, 9))
    with torch.inference_mode():
        plain = model(ids)
        view = _fresh_view(cache)
        paged = model(ids, view)
    assert torch.equal(paged, plain)
    _free_view(view, alloc)


def test_paged_decode_step_equals_full_forward(env):
    model, cache, alloc = env
    ids = torch.randint(0, 256, (1, 6))
    with torch.inference_mode():
        full = model(ids)
        view = _fresh_view(cache)
        _ = model(ids[:, :5], view)
        stepped = model(ids[:, 5:], view)
    assert torch.equal(stepped[:, -1], full[:, -1])
    _free_view(view, alloc)


def test_paged_multi_step_equals_full_forward(env):
    model, cache, alloc = env
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 6))
    with torch.inference_mode():
        full = model(ids)
        view = _fresh_view(cache)
        last = model(ids[:, :3], view)
        for t in range(3, 6):
            last = model(ids[:, t : t + 1], view)
    assert torch.equal(last[:, -1], full[:, -1])
    _free_view(view, alloc)


def test_paged_chunked_prefill_equals_full_forward(env):
    """分支 3(非空 cache + 多 token)走分页 gather。"""
    model, cache, alloc = env
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (1, 11))  # 11 token,页=4 → 至少跨 2 次页边界
    with torch.inference_mode():
        full = model(ids)
        view = _fresh_view(cache)
        _ = model(ids[:, :3], view)
        mid = model(ids[:, 3:7], view)   # 4-token 块,显式掩码分支
        end = model(ids[:, 7:], view)
    assert torch.equal(mid, full[:, 3:7])
    assert torch.equal(end, full[:, 7:])
    _free_view(view, alloc)
