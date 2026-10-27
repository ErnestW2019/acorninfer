import torch

from suiren.cache import KVCache


def shape(s):
    return (1, 2, s, 4)  # [batch, kv_heads, seq, head_dim] 的玩具形状


def test_empty_cache_seq_len_zero():
    cache = KVCache(num_layers=2)
    assert cache.seq_len == 0


def test_update_concatenates_per_layer():
    cache = KVCache(num_layers=2)
    k1 = torch.randn(*shape(3))
    v1 = torch.randn(*shape(3))
    k_full, v_full = cache.update(0, k1, v1)
    assert k_full.shape == shape(3)
    assert cache.seq_len == 3

    k2 = torch.randn(*shape(1))
    v2 = torch.randn(*shape(1))
    k_full, v_full = cache.update(0, k2, v2)
    assert k_full.shape == shape(4)
    assert torch.equal(k_full[:, :, :3], k1)
    assert torch.equal(k_full[:, :, 3:], k2)
    assert torch.equal(v_full[:, :, :3], v1)
    assert torch.equal(v_full[:, :, 3:], v2)


def test_chunked_prefill_multi_token_append():
    cache = KVCache(num_layers=1)
    cache.update(0, torch.randn(*shape(3)), torch.randn(*shape(3)))
    k_full, v_full = cache.update(0, torch.randn(*shape(2)), torch.randn(*shape(2)))
    assert k_full.shape == shape(5)
    assert cache.seq_len == 5


def test_layers_are_independent():
    # 两层各自 update、互不干扰;seq_len 以 layer 0 为准,
    # 各层长度一致是调用方(模型)的职责,cache 不做断言。
    cache = KVCache(num_layers=2)
    cache.update(0, torch.zeros(*shape(2)), torch.zeros(*shape(2)))
    k1, _ = cache.update(1, torch.ones(*shape(2)), torch.ones(*shape(2)))
    assert k1.shape == shape(2)
    assert cache.seq_len == 2
