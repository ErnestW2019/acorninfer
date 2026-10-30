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


def pytest_collection_modifyitems(config, items):
    """@pytest.mark.gpu 的用例在无 CUDA 设备时自动 skip;slow 的去留仍由 addopts 决定。"""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
