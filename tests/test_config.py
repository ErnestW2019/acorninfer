from pathlib import Path

import pytest

from acorninfer.config import EngineConfig, Qwen3Config

QWEN3_8B = Path("/home/ernest/model-weight/Qwen3-8B")


def test_from_json_reads_real_config():
    cfg = Qwen3Config.from_json(QWEN3_8B / "config.json")
    assert cfg.num_hidden_layers == 36
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 12288
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.rope_theta == 1_000_000
    assert cfg.vocab_size == 151936
    assert cfg.tie_word_embeddings is False
    assert cfg.eos_token_id == [151645]


def test_tiny_config_for_unit_tests():
    cfg = Qwen3Config.tiny()
    assert cfg.num_hidden_layers == 2
    assert cfg.vocab_size == 256
    assert cfg.hidden_size == 64


def test_engine_config_defaults_and_backend_validation():
    cfg = EngineConfig(model_dir="x")
    assert cfg.attention_backend == "sdpa"
    assert cfg.fused_ops is False
    with pytest.raises(ValueError, match="attention_backend"):
        EngineConfig(model_dir="x", attention_backend="sdpaX")
