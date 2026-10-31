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
    """混合温度批(graph off)的分流钉子,三重:
    1. 独立 KV 对拍:两腿各自 allocator/cache/view(同模型同预热输入 → row-0
       上下文逐位同构),greedy 行 token 与批构成无关 —— 结构性断言,非巧合;
    2. mixed 批(temp=0 + temp=0.5)→ 整批恰走逐行 _sample(2 次;逐行 fallback
       对全批循环,temp==0 行在 _sample 内部 argmax 分支解析),其中恰 1 行
       到达 Sampler(temp>0)—— 合批 gate 若被删、argmax 吞掉 temp>0 行 → 0 → 红;
    3. all-greedy 批 0 次 _sample:真走合批 argmax;实现退化成逐行 → 2 → 红。

    spy = 实例属性 shadow Executor._sample,转发真函数,不改返回语义与
    RNG 消耗;计数每次 decode_batch 前清零,用完 del 卸 shadow。
    """
    from types import SimpleNamespace

    from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
    from acorninfer.engine.executor import Executor

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()

    def make_view():
        # 独立 KV:同模型同预热输入 → 各腿起始 KV 逐位同构,互不污染
        alloc = BlockAllocator(32)
        cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                             device="cuda", dtype=torch.bfloat16)
        view = PagedCacheView(cache, [])
        model(torch.full((1, 4), 5, device="cuda", dtype=torch.long), view)  # 预热 KV
        return view

    with torch.inference_mode():
        view_a = make_view()   # 腿 A:mixed 批
        view_b = make_view()   # 腿 B:all-greedy 批
        ex = Executor(model)   # graph_runner=None → eager
        real = ex._sample
        calls = {"total": 0, "sampled": 0}   # sampled = temp>0 行(到 Sampler 分支)

        def spy(s, logits):
            calls["total"] += 1
            if s.sampling.temperature > 0:
                calls["sampled"] += 1
            return real(s, logits)

        ex._sample = spy
        try:
            greedy = SimpleNamespace(sampling=SamplingParams(temperature=0.0),
                                     generated_ids=[7])
            torch.manual_seed(123)
            calls["total"] = calls["sampled"] = 0
            mixed = ex.decode_batch(
                [greedy,
                 SimpleNamespace(sampling=SamplingParams(temperature=0.5, top_p=0.9),
                                 generated_ids=[11])],
                [view_a, view_a])
            assert calls["total"] == 2     # 逐行 fallback 循环全批,两行都过 _sample
            assert calls["sampled"] == 1   # 仅 temp>0 行到 Sampler;gate 删/吞行 → 0 → 红
            assert isinstance(mixed[1], int)
            torch.manual_seed(123)
            calls["total"] = calls["sampled"] = 0
            all_greedy = ex.decode_batch(
                [greedy,
                 SimpleNamespace(sampling=SamplingParams(temperature=0.0),
                                 generated_ids=[11])],
                [view_b, view_b])
            assert calls["total"] == 0     # 全 greedy 真走合批 argmax;退化逐行 → 2 → 红
            assert calls["sampled"] == 0
            assert mixed[0] == all_greedy[0]   # greedy 行与批构成无关(结构性)
        finally:
            del ex._sample


@pytest.mark.gpu
def test_engine_prefill_batch_matches_staggered_bitwise():
    """一次性提交 4 序列(一步批量 prefill)vs 逐条提交逐条排空(N=1 原路径):
    生成 token 逐位一致(经验锚:批量 GEMM 与逐条 GEMM 有 ~1 bf16 ulp 差异,
    greedy argmax 翻转概率极低但非零 —— 本测试钉的是本机/本构建的回归不变量);
    钉死批量腿确曾走 prefill_batch(spy),逐条腿零批量调用。"""

    def run(stagger, pin_calls):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
        model.attention_backend = "sdpa"
        model.fused_ops = False
        model.decode_graph = False
        engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
        gen = torch.Generator(device="cuda").manual_seed(13)
        prompts = []
        for L in [6, 10, 14, 5]:
            ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
            prompts.append(ids[0].tolist())
        calls = {"n": 0}
        if pin_calls:
            real = engine.executor.prefill_batch   # Executor 无此方法 → 红

            def spy(seqs, views):
                calls["n"] += 1
                return real(seqs, views)

            engine.executor.prefill_batch = spy   # 实例 shadow,照 Task 1 的 _sample spy 模式
        try:
            sids, toks = [], {}
            for p in prompts:
                sid = engine.submit(p, SamplingParams(temperature=0.0), max_tokens=6)
                sids.append(sid)
                toks[sid] = []
                if stagger:   # 逐条腿:提交一条立刻排空(每步 prefills 恰 1 → N=1 原路径)
                    while engine.has_pending():
                        for o in engine.step():
                            toks[o.seq_id].append(o.token_id)
            if not stagger:   # 批量腿:全部提交后一次排空(首步 4 序列同批 prefill)
                while engine.has_pending():
                    for o in engine.step():
                        toks[o.seq_id].append(o.token_id)
            for sid in sids:
                engine.drop(sid)
            engine.assert_balanced()
            model.to("cpu")
            return [toks[s] for s in sids], calls["n"]
        finally:
            if pin_calls:
                del engine.executor.prefill_batch

    batched, calls_b = run(stagger=False, pin_calls=True)
    staggered, calls_s = run(stagger=True, pin_calls=True)
    assert calls_b >= 1       # 批量腿确曾走 prefill_batch(防假绿)
    assert calls_s == 0       # 逐条腿零批量调用(N=1 原路径未动)
    assert batched == staggered   # 生成 token 逐位一致(经验锚,见 docstring)


@pytest.mark.gpu
def test_executor_prefill_batch_mixed_temperature_per_row_sampling():
    """批量 prefill 采样结构:一次 forward_prefill_batch + 每行恰一次 _sample
    (temp>0 行走 Sampler,greedy 行 argmax);greedy 行 token == 同一 logits
    对象的 argmax(同 logits 下精确,非跨路径巧合)。

    不断言跨路径 token 相等:批量与逐条 forward 有 ~1 bf16 ulp 差异,
    temp>0 采样对 1-ulp 概率扰动敏感 —— 故用结构性钉子(spy 计数 +
    同 logits argmax 对照 + 各行 KV 只写到各自 real_len),不锁采样值。
    """
    from types import SimpleNamespace

    from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
    from acorninfer.engine.executor import Executor

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()

    def make_view():
        # 独立 KV:各行 cache 状态互不污染(页表/Lens 各自持有)
        alloc = BlockAllocator(32)
        cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                             device="cuda", dtype=torch.bfloat16)
        return PagedCacheView(cache)

    with torch.inference_mode():
        views = [make_view() for _ in range(2)]
        ex = Executor(model)   # graph_runner=None;prefill_batch 不经 runner
        seqs = [
            SimpleNamespace(all_ids=_ids(7, 0),
                            sampling=SamplingParams(temperature=0.0)),
            SimpleNamespace(all_ids=_ids(4, 1),
                            sampling=SamplingParams(temperature=0.5, top_p=0.9)),
        ]
        captured = {}
        real_fpfb = ex.model.forward_prefill_batch   # 捕获批量 logits,转发真函数

        def spy_fpfb(ids, vs, *, real_lens):
            logits = real_fpfb(ids, vs, real_lens=real_lens)
            captured["logits"] = logits
            return logits

        ex.model.forward_prefill_batch = spy_fpfb
        real_sample = ex._sample
        sample_calls = {"n": 0}

        def spy_sample(s, logits):
            sample_calls["n"] += 1
            return real_sample(s, logits)

        ex._sample = spy_sample
        try:
            returned = ex.prefill_batch(seqs, views)
        finally:
            del ex.model.forward_prefill_batch
            del ex._sample
        logits = captured["logits"]
        assert logits.shape == (2, 256)
        assert sample_calls["n"] == 2   # 每行恰一次 _sample(吞掉逐行采样 → <2 → 红)
        assert returned[0] == int(logits[0].argmax())   # greedy 行 == 同 logits argmax
        assert isinstance(returned[1], int)   # temp>0 行走 Sampler,返回 int
        # 变长右 padding:各行 KV 只写到各自 real_len(pad 永不入 KV;_lens 为
        # per-layer dict,对照法同 Task 4 的 prefill_batch 测试)
        assert views[0]._lens == {0: 7, 1: 7}
        assert views[1]._lens == {0: 4, 1: 4}


# ---- M5 Task 1:chunked prefill 端到端 ----


def _mk_engine(tiny_model, budget):
    return Engine(
        tiny_model,
        tokenizer=None,
        num_blocks=64,
        page_size=4,
        max_running=4,
        watermark_blocks=0,
        chunked_budget=budget,
    )


def test_engine_chunked_matches_unchunked(tiny_model):
    """budget 强制切块 vs None:同 seed 同 prompts,token 流逐位一致(贪心
    确定性;数值同源 = prefill_batch + forward_extend 对拍,TOL 门在
    test_forward_extend_chunked_matches_full_prefill)。"""
    prompts = [_ids(9, 0), _ids(5, 1), _ids(13, 2)]
    streams = {}
    for budget in (None, 2):
        eng = _mk_engine(tiny_model, budget)
        with torch.inference_mode():
            refs = [generate_cached(eng.model, p, max_new_tokens=6)[0]
                    for p in prompts]
        sids = [eng.submit(p, SamplingParams(temperature=0.0), max_tokens=6)
                for p in prompts]
        while eng.has_pending():
            eng.step()
        streams[budget] = {sid: eng.seqs[sid].all_ids for sid in sids}
        for sid, ref in zip(sids, refs):  # 两腿各自都对 generate_cached 参照
            assert eng.seqs[sid].all_ids == ref
        eng.assert_balanced()
    assert streams[None] == streams[2]


def test_engine_mixed_step_chunk_and_decode(tiny_model):
    """混合步:2 条 decode 中 + 1 条长 prompt chunking,两类同一步推进
    (与 prefill+decode 混合步同构:chunk 前向 + decode 同步内并存)。"""
    eng = _mk_engine(tiny_model, 6)
    p1, p2, p3 = _ids(3, 3), _ids(3, 4), _ids(8, 5)
    with torch.inference_mode():
        ref1, _ = generate_cached(eng.model, p1, max_new_tokens=4)
        ref2, _ = generate_cached(eng.model, p2, max_new_tokens=4)
        ref3, _ = generate_cached(eng.model, p3, max_new_tokens=3)
    s1 = eng.submit(p1, SamplingParams(temperature=0.0), max_tokens=4)
    s2 = eng.submit(p2, SamplingParams(temperature=0.0), max_tokens=4)
    eng.step()  # 两条短 prompt 全量 prefill(budget 6 = 3+3 恰好)
    s3 = eng.submit(p3, SamplingParams(temperature=0.0), max_tokens=3)
    outs = eng.step()  # 混合步:s3 首块 (0,6) 截断 + s1/s2 decode
    assert [o.seq_id for o in outs] == [s1, s2]  # s3 中间块不发 token
    assert eng.seqs[s3].prefill_cursor == 6
    assert eng.seqs[s3].first_token_at is None
    outs2 = eng.step()  # 混合步:s3 尾块 (6,2) 完成 + s1/s2 decode
    got = {o.seq_id for o in outs2}
    assert got == {s1, s2, s3}  # 两类同步推进
    while eng.has_pending():
        eng.step()
    assert eng.seqs[s1].all_ids == ref1
    assert eng.seqs[s2].all_ids == ref2
    assert eng.seqs[s3].all_ids == ref3
    eng.assert_balanced()


def test_engine_midchunk_emits_nothing(tiny_model):
    """中间块不发 token:输出与 TTFT 时间戳都只在完成块出现。"""
    eng = _mk_engine(tiny_model, 2)
    p = _ids(6, 6)
    with torch.inference_mode():
        ref, _ = generate_cached(eng.model, p, max_new_tokens=3)
    s = eng.submit(p, SamplingParams(temperature=0.0), max_tokens=3)
    o1 = eng.step()  # (0,2) 中间块
    assert o1 == []
    assert eng.seqs[s].prefill_cursor == 2
    assert eng.seqs[s].first_token_at is None
    o2 = eng.step()  # (2,2) 中间块
    assert o2 == []
    assert eng.seqs[s].first_token_at is None
    o3 = eng.step()  # (4,2) 完成块 → 恰 1 token,TTFT 在此
    assert [o.seq_id for o in o3] == [s]
    assert eng.seqs[s].first_token_at is not None
    while eng.has_pending():
        eng.step()
    assert eng.seqs[s].all_ids == ref
    eng.assert_balanced()
