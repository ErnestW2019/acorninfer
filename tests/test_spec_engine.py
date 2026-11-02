"""M8 Task 3:forward_verify + 引擎接线 + 无损性硬门(spec §2.2/§2.4)。

契约(plan Task 3 Step 1 五测试 + 任务文本边界钉):
- EngineConfig spec 字段校验(spec_ngram ∈ 0..8;spec_num ∈ [1,8];spec>0 须
  triton 后端;spec × decode_graph 构造期 ValueError;spec × tp 互斥在
  server._build_engine 照 quant 先例);
- A1 无损性(硬门):tiny 模型同 prompts spec on/off 全量生成逐位一致
  —— 随机模型(零接受为主)+ 复写器 rig(真实接受/bonus)双场景;
- 多 TokenOutput 顺序 = 位置序 + eos/max_tokens 截断语义;
- 连续 verify 步:KV 对拍(lens/gather/双窗口宽度自洽)+ 页表复用断言
  (allocate 次数钉死复用)+ assert_balanced;
- server --spec-ngram/--spec-num 透传 + interdicts;
- 边界:剩余生成预算 < m 的 seq 该步降级 m=1 纯 decode(引擎级,正确性优先)。

复写器 rig(copier):全零层权重 + lm_head.weight = embed_tokens.weight →
残差流 ≈ embed(末位) → 任意位置 argmax = 该位输入 token(自相关占优)。
尾部重复 prompt → n-gram draft 全中 → 每步真实接受 K + bonus(多 token
发射可确定性构造,随机模型办不到)。
"""

import types

import pytest
import torch

from acorninfer.config import EngineConfig, Qwen3Config
from acorninfer.engine.engine import Engine
from acorninfer.models.qwen3 import Qwen3ForCausalLM
from acorninfer.sampler import SamplingParams


# ---------- helpers ----------


def _ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)[0].tolist()


def _make_model(device="cuda"):
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config.tiny()).to(device, torch.bfloat16).eval()


def _make_engine(model, *, spec=False, num_blocks=64, page_size=16,
                 max_running=4):
    kw = dict(spec_ngram=2, spec_num=4) if spec else {}
    model.attention_backend = "triton"
    model.fused_ops = False
    model.decode_graph = False
    eng = Engine(model, None, num_blocks=num_blocks, page_size=page_size,
                 max_running=max_running, watermark_blocks=0, **kw)
    return eng


def _run_leg(prompts, *, spec, max_tokens=12):
    """建引擎 → 全部提交 → 排空;返回 (每 seq 的 all_ids, 每步 per-seq 发射数)。"""
    model = _make_model()
    eng = _make_engine(model, spec=spec)
    sids = [eng.submit(p, SamplingParams(temperature=0.0), max_tokens=max_tokens)
            for p in prompts]
    streams = {s: [] for s in sids}
    per_step = []  # [(步号, {seq_id: 本步发射 token 数})]
    step_no = 0
    while eng.has_pending():
        step_no += 1
        counts: dict[int, int] = {}
        for o in eng.step():
            streams[o.seq_id].append(o.token_id)
            counts[o.seq_id] = counts.get(o.seq_id, 0) + 1
        per_step.append((step_no, counts))
    full = [eng.seqs[s].all_ids for s in sids]
    for s in sids:
        eng.drop(s)
    eng.assert_balanced()
    model.to("cpu")
    torch.cuda.empty_cache()
    return full, per_step, eng


def _count_spy(eng, name):
    """wrap-and-forward spy(照 test_engine 先例):计数 + 转发真方法。"""
    real = getattr(eng.executor, name)
    calls = {"n": 0}

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    setattr(eng.executor, name, spy)
    return calls


# ---------- 1. config 校验 ----------


def test_config_spec_validation():
    """spec_ngram ∈ 0..8;spec_num ∈ [1,8];spec>0 须 triton;spec ×
    decode_graph 构造期 ValueError;spec × tp 互斥在 _build_engine(quant 先例)。"""
    cfg = EngineConfig(model_dir="/m")
    assert cfg.spec_ngram == 0 and cfg.spec_num == 4  # 缺省关(逐字节现行为)
    for n in (1, 2, 8):
        assert EngineConfig(model_dir="/m", attention_backend="triton",
                            spec_ngram=n).spec_ngram == n
    # spec_num 边界(1 与 8 合法;0 与 9 非法,且与 spec 开关无关)
    EngineConfig(model_dir="/m", spec_num=1)
    with pytest.raises(ValueError, match="spec_num"):
        EngineConfig(model_dir="/m", spec_num=0)
    with pytest.raises(ValueError, match="spec_num"):
        EngineConfig(model_dir="/m", attention_backend="triton",
                     spec_ngram=2, spec_num=9)
    with pytest.raises(ValueError, match="spec_ngram"):
        EngineConfig(model_dir="/m", attention_backend="triton", spec_ngram=9)
    with pytest.raises(ValueError, match="spec_ngram"):
        EngineConfig(model_dir="/m", attention_backend="triton", spec_ngram=-1)
    # spec > 0 须 triton 后端(verify kernel 只有 triton 路径)
    with pytest.raises(ValueError, match="triton"):
        EngineConfig(model_dir="/m", spec_ngram=2)
    # spec × decode_graph 构造期互斥(v1 eager-only,spec §2.4 D1)
    with pytest.raises(ValueError, match="decode_graph"):
        EngineConfig(model_dir="/m", attention_backend="triton",
                     spec_ngram=2, decode_graph=True)
    # spec × tp 互斥在 _build_engine(EngineConfig 无 tp 字段,quant 先例)
    import acorninfer.server as server_mod
    with pytest.raises(ValueError, match="tp"):
        server_mod._build_engine(types.SimpleNamespace(
            tp=2, quant="none", spec_ngram=2))


# ---------- 2. A1 无损性(tiny 全量,硬门) ----------


@pytest.mark.gpu
def test_spec_greedy_bitwise_tiny():
    """A1 核心:tiny 同 prompts,spec on/off 全量生成逐位一致。

    双场景:①随机模型(重复/随机/混合三类 prompt;n-gram 命中率低 → 零接受
    + bonus 为主);②复写器 rig(尾部重复 prompt → draft 全中 → 真实接受
    K + bonus 多 token 发射)。spec 腿 spy 钉死 verify_batch 确曾调用(防
    全降级假绿)。"""
    prompts = [_ids(8, 0) + _ids(8, 0),   # 重复段:索引有真实后继(接受候选)
               _ids(9, 3),                # 随机:后继基本未命中 → FILLER → 零接受
               _ids(6, 7) + _ids(3, 9),   # 混合形态
               ]

    # 场景 ①:随机模型
    off1, _, _ = _run_leg(prompts, spec=False)
    on1, _, _ = _run_leg(prompts, spec=True)
    assert off1 == on1                    # A1:逐位(含 prompt 前缀逐位同)

    # 场景 ②:复写器 rig(真实接受 + bonus;多 token 发射)
    def copier():
        m = _make_model()
        for layer in m.layers:  # 全零层:残差流 = embed,attention/mlp 输出恒 0
            for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.o_proj,
                        layer.mlp.gate_proj, layer.mlp.up_proj,
                        layer.mlp.down_proj):
                mod.weight.data.zero_()
        m.lm_head.weight.data = m.embed_tokens.weight.data.clone()
        return m

    def run_copier(spec):
        model = copier()
        eng = _make_engine(model, spec=spec)
        # 尾部 6 个重复 token:后继列表 5 条 → drafts = [77]×4 全真(无 FILLER);
        # copier argmax 恒 = 末位 → 全中 K + bonus,单步 5 token 发射
        p = [11, 22, 33, 44, 55, 66, 77, 77, 77, 77, 77, 77]
        sid = eng.submit(p, SamplingParams(temperature=0.0), max_tokens=12)
        spy = _count_spy(eng, "verify_batch")
        streams, per_step = [], []
        while eng.has_pending():
            outs = eng.step()
            emitted = {}
            for o in outs:
                streams.append(o.token_id)
                emitted[o.seq_id] = emitted.get(o.seq_id, 0) + 1
            per_step.append(emitted)
        full = eng.seqs[sid].all_ids
        n_verify = spy["n"]
        eng.drop(sid)
        eng.assert_balanced()
        model.to("cpu")
        torch.cuda.empty_cache()
        return full, streams, per_step, n_verify

    full_off, _, _, _ = run_copier(spec=False)
    full_on, streams_on, per_step_on, n_verify = run_copier(spec=True)
    assert full_off == full_on                        # A1:真实接受路径逐位
    assert n_verify >= 1                              # spec 腿确曾 verify(防全降级假绿)
    assert any(v >= 2 for step in per_step_on for v in step.values())


# ---------- 3. 多 TokenOutput 顺序 + eos/max_tokens 截断 ----------


@pytest.mark.gpu
def test_spec_multitoken_outputs_order():
    """接受多 token 时 TokenOutput 顺序 = 位置序(单步 5 发射:copier 全中
    K=4 + bonus);max_tokens 截断恰在剩余预算处(length);eos 命中即截断
    (stop,其后窗口位不发);发射序内 finished 只在末 token。"""
    # --- copier rig:确定性全中,单步 5 token ---
    model = _make_model()
    for layer in model.layers:
        for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                    layer.self_attn.v_proj, layer.self_attn.o_proj,
                    layer.mlp.gate_proj, layer.mlp.up_proj,
                    layer.mlp.down_proj):
            mod.weight.data.zero_()
    model.lm_head.weight.data = model.embed_tokens.weight.data.clone()
    eng = _make_engine(model, spec=True)
    e = 77
    # 尾部 6 个重复:后继列表 5 条 → drafts [e]×4 全真无 FILLER → 全中 K+bonus
    sid = eng.submit([11, 22, 33, 44, 55, 66, e, e, e, e, e, e],
                     SamplingParams(temperature=0.0), max_tokens=8)
    outs = eng.step()          # prefill:argmax = e → 1 token
    assert [(o.token_id, o.finished) for o in outs] == [(e, False)]
    outs2 = eng.step()         # verify 全中:5 token,e 达 6 < 8
    assert [o.seq_id for o in outs2] == [sid] * 5              # 位置序,同 seq 连续
    assert [o.token_id for o in outs2] == [e] * 5
    assert [o.finished for o in outs2] == [False] * 4 + [False]
    assert len(eng.seqs[sid].generated_ids) == 6
    outs3 = eng.step()         # 剩余预算 2 < m=5 → 降级 decode:每步 1 token
    assert [o.token_id for o in outs3] == [e]          # decode 腿单发
    rest = []
    while eng.has_pending():
        rest += list(eng.step())   # 第二步补到 8
    assert [o.token_id for o in rest] == [e]
    assert rest[-1].finished and eng.seqs[sid].stop_reason == "length"
    assert eng.seqs[sid].generated_ids == [e] * 8
    eng.drop(sid)
    while eng.has_pending():
        eng.step()
    eng.assert_balanced()
    model.to("cpu")
    torch.cuda.empty_cache()

    # --- eos 截断:spy 捕获真 logits 后改写 argmax 链,KV/视图状态真实 ---
    model2 = _make_model()
    eng2 = _make_engine(model2, spec=True)
    p = [7, 8, 5, 9, 1, 2, 7, 8]   # 末 2 位 (7,8) 的后继 = 5 → drafts[0]=5(真草稿)
    sid2 = eng2.submit(p, SamplingParams(temperature=0.0), max_tokens=8,
                       eos_ids=frozenset({123}))
    eng2.step()  # prefill
    real = eng2.executor.verify_batch
    calls = {"n": 0}
    seen = {}

    def spy(seqs, views, drafts):
        logits = real(seqs, views, drafts)   # 真前向:KV 写 m 位,lens+len+m
        calls["n"] += 1
        seen["d0"] = drafts[0][0]            # 真实首草稿(propose 含 prefill 位)
        crafted = logits.clone()
        for b in range(len(seqs)):
            for j in range(logits.shape[1]):
                crafted[b, j] = 0.0
                # row0 argmax = drafts[0][0](a=1 必中);row1 = 123 = eos → bonus 即停
                crafted[b, j, drafts[b][0] if j == 0 else (123, 200, 42, 7)[(j - 1) % 4]] = 10.0
        return crafted

    eng2.executor.verify_batch = spy
    outs4 = eng2.step()      # a=1(d0 中),bonus=123=eos → 发射 [d0,123] 即停
    del eng2.executor.verify_batch
    assert calls["n"] == 1
    d0 = seen["d0"]
    assert d0 != 123                      # 首发非 eos(eos 在 bonus 位)
    assert [(o.token_id, o.finished, o.stop_reason) for o in outs4] == [
        (d0, False, None), (123, True, "stop")]
    seq2 = eng2.seqs[sid2]
    assert seq2.generated_ids[-2:] == [d0, 123]
    assert seq2.is_finished
    while eng2.has_pending():
        eng2.step()
    eng2.assert_balanced()
    model2.to("cpu")
    torch.cuda.empty_cache()


# ---------- 4. 连续 verify 步:KV 对拍 + 页表复用 + 泄漏守卫 ----------


@pytest.mark.gpu
def test_spec_dead_slot_reuse_engine():
    """连续 verify 步(a<K → 死槽留置;下步 ensure_n 复用):①allocate 次数
    钉死复用(不复用则每步 +1 页);②锁步 lens + gather 形状 = lens(死槽
    读侧不可见);③双窗口宽度(m=5 / m=3)token 流自洽(读侧正确性);
    ④排空 assert_balanced。"""
    prompts = [_ids(8, 0) + _ids(8, 0), _ids(9, 3)]
    legs = {}
    for spec_num in (4, 2):
        model = _make_model()
        model.attention_backend = "triton"
        eng = Engine(model, None, num_blocks=64, page_size=16, max_running=4,
                     watermark_blocks=0, spec_ngram=2, spec_num=spec_num)
        sids = [eng.submit(p, SamplingParams(temperature=0.0), max_tokens=10)
                for p in prompts]
        streams = {s: [] for s in sids}
        allocs = {"n": 0}
        real_alloc = eng.cache.allocator.allocate

        def spy_alloc(_real=real_alloc):
            allocs["n"] += 1
            return _real()

        eng.cache.allocator.allocate = spy_alloc
        m = spec_num + 1
        checked = 0
        max_table = 0
        while eng.has_pending():
            for o in eng.step():
                streams[o.seq_id].append(o.token_id)
            for s in sids:                      # 逐步 KV 不变式(存活序列)
                seq = eng.seqs.get(s)
                view = eng._views.get(s)
                if seq is None or seq.is_finished or view is None:
                    continue
                lens = set(view._lens.values())
                assert len(lens) == 1           # 层间锁步
                L = lens.pop()
                assert L == seq.num_tokens - 1  # KV 恰含已提交前缀(末位在下步写)
                assert L <= len(view.table) * 16
                g = view._gather(0, eng.cache.k_pages)
                assert g.shape[2] == L          # 读侧截到 lens:死槽不可见
                max_table = max(max_table, len(view.table))
                checked += 1
        # 页表断言:复用 ⇒ 分配次数 ≪ 步数×(m/页);不复用实现每步要 1+ 页
        assert allocs["n"] <= max_table + 2
        assert checked > 0
        legs[spec_num] = ([eng.seqs[s].all_ids for s in sids], streams)
        for s in sids:
            eng.drop(s)
        eng.assert_balanced()
        model.to("cpu")
        torch.cuda.empty_cache()
    # 双窗口宽度自洽:A1 的读侧正确性补强(不同死槽形态 → 同 token 流)
    assert legs[4][0] == legs[2][0]


# ---------- 5. server 透传 + interdicts ----------


def test_server_passthrough_spec(monkeypatch):
    """--spec-ngram/--spec-num 透传 Engine(from_config 与 _build_engine 双
    构造路径);缺省(旧调用方 args 无字段)= 0/4 零触碰;spec × tp /
    spec × cuda-graph / spec × sdpa 构造期 ValueError。"""
    import acorninfer.engine.engine as engine_mod
    import acorninfer.server as server_mod

    created = {}
    monkeypatch.setattr(server_mod, "from_pretrained",
                        lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(server_mod, "Engine",
                        lambda model, tok, **kw: created.update(kw) or "ENGINE")
    monkeypatch.setattr(server_mod, "Tokenizer", lambda path: "TOK")

    base = dict(model_dir="/m", device="cuda:0", num_blocks=8, page_size=16,
                max_running=4, watermark_blocks=2, chunked_budget=None)
    # 旧调用方(args 无 spec 字段):getattr 缺省 = 0/4 = 逐字节旧行为
    assert server_mod._build_engine(types.SimpleNamespace(**base)) == "ENGINE"
    assert created["spec_ngram"] == 0 and created["spec_num"] == 4
    # 显式透传(spec>0 须 triton 后端:EngineConfig 构造期校验)
    assert server_mod._build_engine(types.SimpleNamespace(
        **base, attention_backend="triton",
        spec_ngram=2, spec_num=6)) == "ENGINE"
    assert created["spec_ngram"] == 2 and created["spec_num"] == 6
    # interdicts:tp × spec(server 层)/ cuda-graph × spec / sdpa × spec(config 层)
    with pytest.raises(ValueError, match="tp"):
        server_mod._build_engine(types.SimpleNamespace(
            **base, tp=2, spec_ngram=2))
    with pytest.raises(ValueError, match="decode_graph"):
        server_mod._build_engine(types.SimpleNamespace(
            **base, attention_backend="triton", cuda_graph=True, spec_ngram=2))
    with pytest.raises(ValueError, match="triton"):
        server_mod._build_engine(types.SimpleNamespace(
            **base, spec_ngram=2))
    # from_config 双构造路径同步透传(from_pretrained 桩 = 真 tiny 模型:
    # Engine.__init__ 读 model.cfg/attention_backend;spec>0 须 triton)
    monkeypatch.setattr(engine_mod, "from_pretrained",
                        lambda *a, **k: Qwen3ForCausalLM(
                            Qwen3Config.tiny()).to(dtype=torch.bfloat16))
    eng = engine_mod.Engine.from_config(
        EngineConfig(model_dir="/m", device="cpu", num_blocks=16,
                     attention_backend="triton", spec_ngram=3, spec_num=5),
        tokenizer=None)
    assert eng.spec_ngram == 3 and eng.spec_num == 5


# ---------- 6. 边界:剩余预算 < m 降级 m=1 decode ----------


@pytest.mark.gpu
def test_spec_remaining_budget_degrades_to_decode():
    """剩余生成预算 < m(或非贪心采样)的 seq 该步降级 m=1 纯 decode
    (引擎级,正确性优先):spy 钉两条腿的分流;输出与非 spec 参照逐位一致。"""
    prompts = [_ids(8, 1), _ids(8, 2)]
    # max_tokens=7:prefill 后 remaining 6 ≥ m=5 → verify 腿;接受 ~0 每步
    # 1 token → remaining 5 → verify;4 < 5 → decode 腿。两腿都被钉到。
    off, _, _ = _run_leg(prompts, spec=False, max_tokens=7)   # 参照(max_tokens=7)
    model = _make_model()
    eng = _make_engine(model, spec=True)
    sids = [eng.submit(p, SamplingParams(temperature=0.0), max_tokens=7)
            for p in prompts]
    spy_v = _count_spy(eng, "verify_batch")
    spy_d = _count_spy(eng, "decode_batch")
    while eng.has_pending():
        eng.step()
    on = [eng.seqs[s].all_ids for s in sids]
    assert on == off                       # 降级不影响逐位
    assert spy_v["n"] >= 1                 # 中段确有 verify
    assert spy_d["n"] >= 1                 # 尾段(剩余 < m)确有 decode 腿
    for s in sids:
        eng.drop(s)
    eng.assert_balanced()
    model.to("cpu")
    torch.cuda.empty_cache()

    # 非贪心采样:按 seq 降级 decode 腿(_sample 既有逐行语义,零静默 greedy)
    model2 = _make_model()
    eng2 = _make_engine(model2, spec=True)
    spy_v2 = _count_spy(eng2, "verify_batch")
    sid3 = eng2.submit(_ids(9, 4), SamplingParams(temperature=0.8, top_p=0.9),
                       max_tokens=2)
    while eng2.has_pending():
        eng2.step()
    seq3 = eng2.seqs[sid3]
    assert seq3.is_finished and len(seq3.generated_ids) == 2
    assert spy_v2["n"] == 0                # temp>0 序列从不进 verify 腿
    eng2.drop(sid3)
    eng2.assert_balanced()
    model2.to("cpu")
    torch.cuda.empty_cache()
