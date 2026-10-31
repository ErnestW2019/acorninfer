"""Engine 端到端(tiny 模型,CPU;decode_graph 等价两例走 GPU):
并发等价、连续提交、抢占回收、泄漏守卫。
"""

import pytest
import torch

from acorninfer.config import Qwen3Config
from acorninfer.engine.engine import Engine
from acorninfer.generate import generate_cached
from acorninfer.models.qwen3 import Qwen3ForCausalLM
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


@pytest.mark.gpu
def test_engine_decode_graph_tokens_bitwise():
    """graph 引擎 vs eager 引擎:同 submits 同步序,生成 token 逐位一致;
    runner 启用钉死(构造期捕获,属性翻转不重建 runner 则必然不一致);
    排空后 KV 平衡(graph 常驻 scratch 例外,见 Engine.assert_balanced)。"""
    outs = []
    for graph in (False, True):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
        model.attention_backend = "triton"
        model.fused_ops = True
        model.decode_graph = graph          # 必须在 Engine 构造前置位
        engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
        assert (engine.graph_runner is not None) == graph
        gen = torch.Generator(device="cuda").manual_seed(7)
        sids = []
        for L in [6, 10, 14, 18]:
            ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
            sids.append(engine.submit(ids[0].tolist(),
                                      SamplingParams(temperature=0.0), max_tokens=6))
        toks = {sid: [] for sid in sids}
        while engine.has_pending():
            for o in engine.step():
                toks[o.seq_id].append(o.token_id)
        for sid in sids:
            engine.drop(sid)
        engine.assert_balanced()
        outs.append([toks[sid] for sid in sids])
        if graph:
            assert engine.graph_runner.last_logits is not None  # dispatch 确曾走过 runner(防假绿)
        model.to("cpu")
    assert outs[0] == outs[1]


@pytest.mark.gpu
def test_engine_decode_graph_eos_shrinks_batch():
    """decode 中途 EOS 退休使批缩小、桶切换(4→2)。

    tiny 词表(256)吐不出 QWEN3_EOS_IDS,max_tokens=1 又在 prefill 就终
    (缩不了 decode 批)—— 用两阶段陷阱:先跑无陷阱 eager 拿全量 token 流,
    取各陷阱序列的第 2 个生成 token(= 首个 decode 产出)作 eos_ids 覆写,
    两引擎同构重跑;批在第 1 个 decode 步后 4→2,余者仍逐位、跑满 max_tokens。
    """

    def run(graph, traps):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
        model.attention_backend = "triton"
        model.fused_ops = True
        model.decode_graph = graph
        engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
        gen = torch.Generator(device="cuda").manual_seed(7)
        prompts = []
        for L in [6, 10, 14, 18]:
            ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
            prompts.append(ids[0].tolist())
        sids = []
        for i, p in enumerate(prompts):
            kw = {"eos_ids": frozenset({traps[i]})} if i in traps else {}
            sids.append(engine.submit(p, SamplingParams(temperature=0.0),
                                      max_tokens=8, **kw))
        toks = {sid: [] for sid in sids}
        while engine.has_pending():
            for o in engine.step():
                toks[o.seq_id].append(o.token_id)
        for sid in sids:
            engine.drop(sid)
        engine.assert_balanced()
        model.to("cpu")
        return [toks[sid] for sid in sids]

    full = run(False, {})                              # 阶段 1:无陷阱全量流
    traps = {0: full[0][1], 1: full[1][1]}             # 生成流第 2 个 token = 首个 decode 产出
    eager = run(False, traps)
    graph = run(True, traps)
    assert eager == graph                              # 缩批后 token 仍逐位
    assert len(eager[0]) == 2 and len(eager[1]) == 2   # 陷阱序列恰在第 1 个 decode 步终止
    assert all(len(s) == 8 for s in eager[2:])         # 余者跑满 → 桶 4→2 切换确有发生


@pytest.mark.gpu
def test_executor_decode_batch_greedy_matches_per_row_argmax():
    """eager decode(graph off)全 greedy:合批 argmax(dim=1) 与逐行 argmax
    token 逐位一致(防实现滑步);这正是 bench eager 锚点走的路径。

    两腿同 seed 独立重建引擎(同实现双跑对拍锚):合批改造前后都必须绿 ——
    改造只许把 B 次 D2H 同步并为 1 次,不许改变任何 token。
    """

    def run_leg():
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
        model.attention_backend = "triton"
        model.fused_ops = True
        model.decode_graph = False                 # Executor 无 runner → eager 路径
        engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
        gen = torch.Generator(device="cuda").manual_seed(3)
        sids = []
        for L in [6, 10]:
            ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
            sids.append(engine.submit(ids[0].tolist(),
                                      SamplingParams(temperature=0.0), max_tokens=4))
        toks = {sid: [] for sid in sids}
        while engine.has_pending():
            for o in engine.step():
                toks[o.seq_id].append(o.token_id)
        for sid in sids:
            engine.drop(sid)
        engine.assert_balanced()
        model.to("cpu")
        return [toks[s] for s in sids]

    assert run_leg() == run_leg()


@pytest.mark.gpu
def test_executor_decode_batch_mixed_temperature_falls_back_per_row():
    """混合温度批(graph off):temp>0 行必须走逐行 _sample/Sampler(RNG 交织
    契约),全 greedy 才允许合批 argmax。钉住分流条件。

    同一 view 连跑两次会推进 KV _lens(前向只增不减),无碍断言:两问的
    x/views 同构,greedy 行 logits 逐位相同 → argmax 与批构成无关。
    """
    from types import SimpleNamespace

    from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
    from acorninfer.engine.executor import Executor

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    ex = Executor(model)   # graph_runner=None → eager
    greedy = SimpleNamespace(sampling=SamplingParams(temperature=0.0),
                             generated_ids=[7])
    sampled = SimpleNamespace(sampling=SamplingParams(temperature=0.5, top_p=0.9),
                              generated_ids=[11])
    alloc = BlockAllocator(32)
    cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    view = PagedCacheView(cache, [])
    model(torch.full((1, 4), 5, device="cuda", dtype=torch.long), view)  # 预热 KV
    torch.manual_seed(123)
    mixed = ex.decode_batch([greedy, sampled], [view, view])
    torch.manual_seed(123)
    all_greedy = ex.decode_batch(
        [greedy, SimpleNamespace(sampling=SamplingParams(temperature=0.0),
                                 generated_ids=[11])], [view, view])
    assert mixed[0] == all_greedy[0]   # greedy 行与批构成无关(确定性)
    assert isinstance(mixed[1], int)
