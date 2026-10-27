import pytest
import torch

from suiren.cache import KVCache
from suiren.config import Qwen3Config
from suiren.generate import generate_cached, generate_token_ids
from suiren.models.qwen3 import Qwen3ForCausalLM, from_pretrained
from suiren.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config.tiny()).to(dtype=torch.bfloat16)


def test_cached_greedy_matches_oracle_tiny():
    model = tiny_model()
    with torch.inference_mode():
        oracle = generate_token_ids(model, prompt_ids=[5, 6, 7], max_new_tokens=24)
        cached, _ = generate_cached(model, prompt_ids=[5, 6, 7], max_new_tokens=24)
    assert cached == oracle


def test_cached_returns_prompt_and_logprobs_empty_when_greedy():
    model = tiny_model()
    ids, lps = generate_cached(model, prompt_ids=[5, 6, 7], max_new_tokens=4)
    assert ids[:3] == [5, 6, 7]
    assert lps == []


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
