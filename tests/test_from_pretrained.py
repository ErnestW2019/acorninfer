import pytest
import torch

from acorninfer.models.qwen3 import from_pretrained

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


@pytest.mark.slow
def test_from_pretrained_loads_real_model():
    model = from_pretrained(QWEN3_8B)
    assert model.cfg.vocab_size == 151936
    assert next(model.parameters()).dtype == torch.bfloat16
    assert not model.training
    ids = torch.tensor([[9707]], device="cuda:0")
    with torch.inference_mode():
        logits = model(ids)
    assert logits.shape == (1, 1, 151936)
    assert torch.isfinite(logits).all()

    model.to("cpu")
    torch.cuda.empty_cache()
