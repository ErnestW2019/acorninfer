import pytest
import torch

from suiren.config import Qwen3Config
from suiren.generate import generate_token_ids
from suiren.models.qwen3 import Qwen3ForCausalLM, map_hf_state_dict
from suiren.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


def test_generate_tiny_model_runs():
    torch.manual_seed(0)
    cfg = Qwen3Config.tiny()
    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    ids = generate_token_ids(model, prompt_ids=[5, 6, 7], max_new_tokens=8)
    assert len(ids) > 3  # prompt 保留了
    assert all(0 <= i < 256 for i in ids)


@pytest.mark.slow
def test_generate_real_model_factual():
    from suiren.weights import load_safetensors

    cfg = Qwen3Config.from_json(f"{QWEN3_8B}/config.json")
    # CPU 上构建 → assign 加载 CPU 权重 → 再整体上卡,
    # 峰值显存只有一份权重(16GB),与 Task 5 的安全编排一致。
    model = Qwen3ForCausalLM(cfg)
    mapped = map_hf_state_dict(load_safetensors(QWEN3_8B))
    model.load_state_dict(mapped, strict=True, assign=True)
    model.to(device="cuda:0", dtype=torch.bfloat16)

    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    ids = generate_token_ids(
        model, tokenizer=tok, prompt="The capital of France is", max_new_tokens=12
    )
    text = tok.decode(ids)
    assert "Paris" in text or "paris" in text

    del model  # 释放 16GB,避免套件内后续用例 OOM
    torch.cuda.empty_cache()
