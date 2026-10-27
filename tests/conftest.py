"""Shared test helpers."""

import pytest
import torch

from acorninfer.config import Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


@pytest.fixture
def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config.tiny()).to(dtype=torch.bfloat16)
