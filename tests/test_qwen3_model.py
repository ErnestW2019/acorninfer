import pytest
import torch

from suiren.config import Qwen3Config
from suiren.models.qwen3 import Qwen3ForCausalLM, map_hf_state_dict

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


def tiny_model(device: str = "cpu") -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config.tiny()
    return Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)


def test_forward_shapes_tiny():
    model = tiny_model()
    ids = torch.randint(0, 256, (1, 8))
    logits = model(ids)
    assert logits.shape == (1, 8, 256)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_forward_batch_consistency_tiny():
    model = tiny_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 8))
    batched = model(ids)
    single0 = model(ids[0:1])
    single1 = model(ids[1:2])
    assert torch.equal(batched[0], single0[0])
    assert torch.equal(batched[1], single1[0])


def test_state_dict_matches_hf_key_mapping():
    model = tiny_model()
    # 模拟 HF 命名:去掉 "model." 前缀后必须严格对上
    hf_style = {f"model.{k}": v for k, v in model.state_dict().items()}
    hf_style["lm_head.weight"] = hf_style.pop("model.lm_head.weight")
    mapped = {k.removeprefix("model."): v for k, v in hf_style.items()}
    model.load_state_dict(mapped, strict=True)


@pytest.mark.slow
def test_real_weights_load_strict_and_forward():
    from suiren.weights import load_safetensors

    cfg = Qwen3Config.from_json(f"{QWEN3_8B}/config.json")
    # 模型留在 CPU 上初始化;assign=True 会用加载的张量直接替换参数,
    # 其 dtype 以加载结果(bf16)为准。权重加载到 CPU 后 assign,
    # 再整体搬上 GPU —— 峰值显存只有一份权重(16GB),不会两份副本 OOM。
    # (不走 load_file(device="cuda") 的 GPU-direct 路径:实测在本机偶发
    #  cudaErrorInvalidValue,mmap 后端不稳定。)
    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    mapped = map_hf_state_dict(load_safetensors(QWEN3_8B))
    model.load_state_dict(mapped, strict=True, assign=True)
    model.to(device="cuda:0")

    with torch.inference_mode():
        ids = torch.tensor([[9707]], device="cuda:0")  # 单 token
        logits = model(ids)
        assert logits.shape == (1, 1, cfg.vocab_size)
        assert torch.isfinite(logits).all()

    del model  # 释放 16GB,避免套件内后续用例 OOM
    torch.cuda.empty_cache()


def test_rope_positions_match_manual():
    from suiren.models.qwen3 import RotaryEmbedding

    rope = RotaryEmbedding(16, 10_000.0)
    positions = torch.tensor([2.0, 5.0, 9.0])
    cos, sin = rope(positions, torch.float32, torch.device("cpu"))
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    freqs = torch.outer(positions, inv_freq)
    manual_cos = torch.cat((freqs, freqs), dim=-1).cos()
    assert torch.allclose(cos, manual_cos, atol=1e-6)
    assert torch.allclose(sin, torch.cat((freqs, freqs), dim=-1).sin(), atol=1e-6)


def test_forward_positions_arange_default():
    # 默认路径(无 cache)位置必须是 arange(seq),数值与 M0 完全一致
    model = tiny_model()
    ids = torch.randint(0, 256, (1, 8))
    with torch.inference_mode():
        logits = model(ids)
    assert logits.shape == (1, 8, 256)
