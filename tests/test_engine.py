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


def test_submit_rejects_prompt_over_schedulable_capacity(tiny_model):
    """拒绝阈值是可调度容量(total - watermark),不是 total。

    回归:落在 (total-watermark, total] 缝隙的 prompt 被接受却永远无法准入
    (准入预算按 free - watermark 结算)→ 队首饿死,步循环卡死。
    """
    eng = Engine(tiny_model, tokenizer=None, num_blocks=8, page_size=4,
                 max_running=4, watermark_blocks=2)
    # 8 块 - 水位 2 = 6 块可调度:≤24 token 可入,25+ 必拒绝
    s = eng.submit(_ids(24, 0), SamplingParams(temperature=0.0), max_tokens=1)
    eng.step()  # max_tokens=1:prefill 即完成并退休
    eng.drop(s)  # drop 后 seqs 不留痕(server 的消费后释放契约)
    assert s not in eng.seqs
    with pytest.raises(ValueError, match="capacity"):
        eng.submit(_ids(25, 1), SamplingParams(temperature=0.0), max_tokens=1)
    # 水位 ≥ 块数的配置直接在构造期拒绝(否则任何 prompt 都无法准入)
    with pytest.raises(ValueError, match="watermark"):
        Engine(tiny_model, tokenizer=None, num_blocks=8, page_size=4,
               max_running=4, watermark_blocks=8)


def test_temperature_sampling_end_to_end(engine):
    """temp>0 走 Sampler 路径:只做结构性断言(全局 RNG,值不可复现)。"""
    torch.manual_seed(3)
    s = engine.submit(
        _ids(5, 2), SamplingParams(temperature=0.8, top_p=0.95), max_tokens=3
    )
    for _ in range(4):
        if not engine.has_pending():
            break
        engine.step()
    seq = engine.seqs[s]
    assert seq.is_finished and seq.stop_reason == "length"
    assert len(seq.generated_ids) == 3
    assert all(0 <= t < 256 for t in seq.generated_ids)
    engine.assert_balanced()
