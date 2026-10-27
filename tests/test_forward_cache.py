import torch

from suiren.cache import KVCache
from suiren.config import Qwen3Config
from suiren.models.qwen3 import Qwen3ForCausalLM


def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config.tiny()).to(dtype=torch.bfloat16)


def test_prefill_with_cache_equals_no_cache():
    model = tiny_model()
    ids = torch.randint(0, 256, (1, 8))
    with torch.inference_mode():
        plain = model(ids)
        cached = model(ids, KVCache(num_layers=len(model.layers)))
    torch.testing.assert_close(cached, plain)


def test_decode_step_matches_full_forward():
    model = tiny_model()
    ids = torch.randint(0, 256, (1, 6))
    with torch.inference_mode():
        full = model(ids)                     # 一次性 6 token
        cache = KVCache(num_layers=len(model.layers))
        _ = model(ids[:, :5], cache)          # prefill 前 5
        stepped = model(ids[:, 5:], cache)    # decode 第 6 个
    torch.testing.assert_close(stepped[:, -1], full[:, -1])


def test_multi_step_decode_matches_full_forward():
    model = tiny_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 6))
    with torch.inference_mode():
        full = model(ids)
        cache = KVCache(num_layers=len(model.layers))
        last = model(ids[:, :3], cache)
        for t in range(3, 6):
            last = model(ids[:, t : t + 1], cache)
    torch.testing.assert_close(last[:, -1], full[:, -1])


def test_chunked_prefill_matches_full_forward():
    # 分支 3:非空 cache + 多 token(分块 prefill),显式右下对齐掩码路径
    model = tiny_model()
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (1, 5))
    with torch.inference_mode():
        full = model(ids)
        cache = KVCache(num_layers=len(model.layers))
        _ = model(ids[:, :3], cache)          # prefill 3
        chunked = model(ids[:, 3:], cache)    # 2-token chunk,走显式掩码分支
    torch.testing.assert_close(chunked, full[:, 3:])
