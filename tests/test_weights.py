import pytest
import torch

from suiren.weights import load_safetensors

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


@pytest.mark.slow
def test_load_real_weights_names_and_shapes():
    tensors = load_safetensors(QWEN3_8B, device="cpu")
    assert tensors["lm_head.weight"].shape == (151936, 4096)
    assert tensors["model.embed_tokens.weight"].shape == (151936, 4096)
    assert tensors["model.layers.0.self_attn.q_proj.weight"].shape == (4096, 4096)
    assert tensors["model.layers.0.self_attn.k_proj.weight"].shape == (1024, 4096)
    assert tensors["model.layers.0.self_attn.v_proj.weight"].shape == (1024, 4096)
    assert tensors["model.layers.0.self_attn.o_proj.weight"].shape == (4096, 4096)
    assert tensors["model.layers.0.self_attn.q_norm.weight"].shape == (128,)
    assert tensors["model.layers.0.mlp.gate_proj.weight"].shape == (12288, 4096)
    assert tensors["model.layers.0.mlp.down_proj.weight"].shape == (4096, 12288)
    assert tensors["model.norm.weight"].shape == (4096,)
    assert all(t.dtype == torch.bfloat16 for t in tensors.values())
    assert len(tensors) == 399
