import pytest
import torch

from suiren.config import Qwen3Config
from suiren.models.qwen3 import Qwen3ForCausalLM

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
    mapped = {k.removeprefix("model."): v for k, v in load_safetensors(QWEN3_8B).items()}
    model.load_state_dict(mapped, strict=True, assign=True)
    model.to(device="cuda:0")

    ids = torch.tensor([[9707]], device="cuda:0")  # 单 token
    logits = model(ids)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    del model  # 释放 16GB,避免套件内后续用例 OOM
    torch.cuda.empty_cache()
