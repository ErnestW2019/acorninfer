import pytest
import torch

from acorninfer.config import Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM, map_hf_state_dict

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
    from acorninfer.weights import load_safetensors

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
    from acorninfer.models.qwen3 import RotaryEmbedding

    rope = RotaryEmbedding(16, 10_000.0)
    # 独立锚点:inv_freq[0] == 1.0(θ^0),所以 dim0 的 cos/sin 就是 cos(pos)/sin(pos) 本身
    positions = torch.tensor([2.0, 3.0, 9.0])
    cos, sin = rope(positions, torch.float32, torch.device("cpu"))
    assert torch.allclose(cos[:, 0], torch.cos(positions), atol=1e-6)
    assert torch.allclose(sin[:, 0], torch.sin(positions), atol=1e-6)
    # 高维分量与手工公式对照(频率递减θ^{-2i/d})
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    freqs = torch.outer(positions, inv_freq)
    assert torch.allclose(cos, torch.cat((freqs, freqs), dim=-1).cos(), atol=1e-6)
    assert torch.allclose(sin, torch.cat((freqs, freqs), dim=-1).sin(), atol=1e-6)
    # bf16 cast 路径:生产总是跑 cast,必须钉住输出 dtype
    cos_b, sin_b = rope(torch.tensor([3]), torch.bfloat16, torch.device("cpu"))
    assert cos_b.dtype == torch.bfloat16 and sin_b.dtype == torch.bfloat16
    assert torch.allclose(cos_b.float(), cos[1:2], atol=2e-2)  # pos=3 行,bf16 容差


def test_forward_position_dependence():
    # 因果 + 位置是逐调用 arange:同一前缀在两种上下文长度下,pos=3 的输出必须逐位一致。
    # 这才是"位置正确"的黑盒钉子(绝对位置经相对旋转不可观测,不同输入的对比是空洞断言)。
    model = tiny_model()
    torch.manual_seed(1)
    ids_b = torch.randint(0, 256, (1, 8))
    ids_a = ids_b[:, :4].clone()
    with torch.inference_mode():
        out_a, out_b = model(ids_a), model(ids_b)
    assert out_a.shape == (1, 4, 256)
    assert out_b.shape == (1, 8, 256)
    assert torch.equal(out_a[:, -1], out_b[:, 3])
