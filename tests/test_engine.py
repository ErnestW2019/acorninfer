"""Engine 端到端(tiny 模型,CPU):并发等价、连续提交、抢占回收、泄漏守卫。"""

import pytest
import torch

from acorninfer.engine.engine import Engine
from acorninfer.generate import generate_cached
from acorninfer.sampler import SamplingParams


@pytest.fixture
def engine(tiny_model):
    eng = Engine(
        tiny_model,
        tokenizer=None,
        num_blocks=64,
        page_size=4,
        max_running=4,
        watermark_blocks=0,
    )
    yield eng
    # 未完成的序列(如有)排空后再验平衡,避免中间态泄漏误报
    while eng.has_pending():
        eng.step()
    eng.assert_balanced()


def _ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)[0].tolist()


def test_two_concurrent_greedy_match_individual(engine):
    p1, p2 = _ids(5, 0), _ids(7, 1)
    with torch.inference_mode():
        ref1, _ = generate_cached(engine.model, p1, max_new_tokens=8)
        ref2, _ = generate_cached(engine.model, p2, max_new_tokens=8)
    s1 = engine.submit(p1, SamplingParams(temperature=0.0), max_tokens=8)
    s2 = engine.submit(p2, SamplingParams(temperature=0.0), max_tokens=8)
    while engine.has_pending():
        engine.step()
    assert engine.seqs[s1].all_ids == ref1
    assert engine.seqs[s2].all_ids == ref2
    assert {engine.seqs[s].stop_reason for s in (s1, s2)} == {"length"}


def test_midflight_submission_continuous_batching(engine):
    p1, p2 = _ids(5, 0), _ids(7, 1)
    with torch.inference_mode():
        ref1, _ = generate_cached(engine.model, p1, max_new_tokens=6)
        ref2, _ = generate_cached(engine.model, p2, max_new_tokens=6)
    s1 = engine.submit(p1, SamplingParams(temperature=0.0), max_tokens=6)
    engine.step()  # s1 prefill
    s2 = engine.submit(p2, SamplingParams(temperature=0.0), max_tokens=6)  # 中途插入
    while engine.has_pending():
        engine.step()
    assert engine.seqs[s1].all_ids == ref1
    assert engine.seqs[s2].all_ids == ref2


def test_metrics_timestamps_and_first_token(engine):
    s = engine.submit(_ids(5, 0), SamplingParams(temperature=0.0), max_tokens=3)
    outs = engine.step()
    assert len(outs) == 1 and outs[0].seq_id == s
    seq = engine.seqs[s]
    assert seq.first_token_at is not None and seq.last_token_at is not None
    assert seq.first_token_at >= seq.created_at


def test_stop_reason_eos(engine):
    # 让 eos 等于该序列第一步必选的 greedy token → 一步即停,reason=stop
    with torch.inference_mode():
        probe = engine.submit(_ids(5, 0), SamplingParams(temperature=0.0), max_tokens=1)
        engine.step()
        t0 = engine.seqs[probe].generated_ids[0]
    s = engine.submit(
        _ids(5, 0), SamplingParams(temperature=0.0), max_tokens=8, eos_ids=frozenset({t0})
    )
    outs = engine.step()
    assert outs[-1].finished and engine.seqs[s].stop_reason == "stop"


def test_preemption_under_block_pressure(tiny_model):
    """3 块小池(page=4 → 单序列上限 12 token):两条互相挤兑 → 必然抢占;
    输出仍与无压力运行逐位一致(recompute 的 greedy 决定论)。
    参数推算:prompt 3 + max_tokens 8 = 11 token ≤ 12,必须可完成,防死循环。"""
    eng = Engine(tiny_model, tokenizer=None, num_blocks=3, page_size=4,
                 max_running=4, watermark_blocks=0)
    p1, p2 = _ids(3, 0), _ids(3, 1)
    with torch.inference_mode():
        ref1, _ = generate_cached(tiny_model, p1, max_new_tokens=8)
        ref2, _ = generate_cached(tiny_model, p2, max_new_tokens=8)
    eng.submit(p1, SamplingParams(temperature=0.0), max_tokens=8)
    eng.submit(p2, SamplingParams(temperature=0.0), max_tokens=8)
    preempted = False
    while eng.has_pending():
        before = {s.seq_id: s.num_pages for s in eng.scheduler.running}
        eng.step()
        for s in eng.scheduler.waiting:
            if s.num_pages == 0 and s.seq_id in before and before[s.seq_id] > 0:
                preempted = True
    assert preempted, "场景未触发抢占,压力设置失效"
    assert all(s.is_finished for s in eng.seqs.values())
    assert [s.all_ids for s in eng.seqs.values()] == [ref1, ref2] or \
           [s.all_ids for s in eng.seqs.values()] == [ref2, ref1]
    eng.assert_balanced()
