import pytest
import torch
import torch.nn.functional as F

from suiren.sampler import Sampler, SamplingParams

LOGITS = torch.tensor([2.0, 1.0, 0.5, -1.0])


def test_top_k_one_is_deterministic_argmax():
    s = Sampler(SamplingParams(temperature=1.0, top_k=1))
    token, lp = s.sample(LOGITS.clone())
    assert token == 0
    assert pytest.approx(lp, abs=1e-5) == float(F.log_softmax(LOGITS, -1)[0])


def test_temperature_scales_logits():
    s = Sampler(SamplingParams(temperature=2.0))
    processed = s.process(LOGITS.clone())
    assert torch.allclose(processed, LOGITS / 2.0)


def test_top_k_filters_all_but_k():
    s = Sampler(SamplingParams(temperature=1.0, top_k=2))
    processed = s.process(LOGITS.clone())
    assert processed[0].isfinite() and processed[1].isfinite()
    assert processed[2].isinf() and processed[3].isinf()


def test_top_p_keeps_first_threshold_crossing():
    # softmax probs ≈ [0.609, 0.224, 0.136, 0.030];升序侧累计质量 0.030+0.136+0.224=0.391,
    # top_p=0.6 时 (1-top_p)=0.4 ≥ 0.391 → 低位的三个全被截掉,只剩 token0
    s = Sampler(SamplingParams(temperature=1.0, top_p=0.6))
    processed = s.process(LOGITS.clone())
    assert processed[0].isfinite()
    assert processed[1:].isinf().all()


def test_matches_hf_warpers_on_random_logits():
    """与安装的 transformers warper 逐位对齐(oracle 只进测试)。

    以安装源码的构造签名为准,按 HF generate 的固定顺序 temperature → top-k → top-p 应用。
    """
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    torch.manual_seed(3)
    g = torch.Generator().manual_seed(7)
    warpers = [
        TemperatureLogitsWarper(temperature=0.7),
        TopKLogitsWarper(top_k=20, min_tokens_to_keep=1),
        TopPLogitsWarper(top_p=0.95, min_tokens_to_keep=1),
    ]
    for _ in range(20):
        logits = torch.randn(151936, generator=g)
        ours = Sampler(SamplingParams(temperature=0.7, top_k=20, top_p=0.95)).process(
            logits.clone()
        )
        hf = logits[None]
        for w in warpers:
            hf = w(torch.zeros(1, 1, dtype=torch.long), hf)
        assert torch.allclose(ours, hf[0]), f"max diff {(ours - hf[0]).abs().max()}"
