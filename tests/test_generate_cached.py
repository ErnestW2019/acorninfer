import pytest
import torch

from acorninfer.cache import KVCache
from acorninfer.generate import generate_cached, generate_token_ids
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


def test_cached_greedy_matches_oracle_tiny(tiny_model):
    model = tiny_model
    with torch.inference_mode():
        oracle = generate_token_ids(model, prompt_ids=[5, 6, 7], max_new_tokens=24)
        cached, _ = generate_cached(model, prompt_ids=[5, 6, 7], max_new_tokens=24)
    assert cached == oracle


def test_cached_returns_prompt_and_logprobs_empty_when_greedy(tiny_model):
    model = tiny_model
    ids, lps = generate_cached(model, prompt_ids=[5, 6, 7], max_new_tokens=4)
    assert ids[:3] == [5, 6, 7]
    assert lps == []


def test_sampler_path_call_shape_and_alignment(tiny_model):
    """Task 7 前 pin 住 sampler 的调用形状与 step_logprobs 对齐契约。"""

    class StubSampler:
        def sample(self, logits):
            assert logits.shape == (256,) and logits.dtype == torch.float32
            return int(logits.argmax()), -0.5

    model = tiny_model
    ids, lps = generate_cached(model, prompt_ids=[5, 6, 7], max_new_tokens=4, sampler=StubSampler())
    assert len(lps) == len(ids) - 3  # 1:1 对齐:每个生成 token 一条 logprob
    assert lps == [-0.5] * len(lps)


@pytest.mark.slow
def test_cached_greedy_matches_oracle_real_model():
    """真实权重上 cached == oracle:cache 正确性的最强证明。"""
    model = from_pretrained(QWEN3_8B)
    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    prompt_ids = tok.encode("The first three prime numbers are")
    with torch.inference_mode():
        oracle = generate_token_ids(model, prompt_ids=prompt_ids, max_new_tokens=40)
        cache = KVCache(num_layers=len(model.layers))  # 显式传 cache 复用同模型
        cached, _ = generate_cached(
            model, prompt_ids=prompt_ids, max_new_tokens=40, cache=cache
        )
    assert cached == oracle

    model.to("cpu")
    torch.cuda.empty_cache()
