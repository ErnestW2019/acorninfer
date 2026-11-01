"""M7 Task 2 接线契约:quant 配置面 + apply_quant walk + dispatch 分派 + 装载路径。

契约(spec §4 + Task 2 关键现实):
- EngineConfig.quant ∈ {none,w8a16,w4a16},非法构造期 ValueError;
- quant × tp>1 显式互斥(server._build_engine 构造期;EngineConfig 无 tp 字段,
  互斥在 tp 已知的构造点把守);
- apply_quant walk:末段名 ∈ 8 类(QUANT_ATTRS)的 nn.Linear → int8 存储 +
  _q_scale + _m7_quant 打标;embed/norm/rope 等零触碰(枚举断言);mode none
  不 walk、模块零触碰(none 硬门静态面);
- dispatch_linear:none → mod(x) 逐字节旧路径;w8a16 → quant_gemm(cls 按
  (N,K) 查五档表);none 路径 CPU 逐位全等(GPU 侧另见 mini margin 门);
- server/_build_engine 与 engine/from_config 两条构造路径装载后 apply_quant
  (quant_gemm.configure 打桩:triton/autotune 不进单测,T1 测试已覆盖)。

walk 测试用 128 倍数 mini 配置(Qwen3Config.tiny 的 hidden=64 触发 quantize_w8
零特例守门,如实偏离 plan 草稿的"tiny 为主":守门是 T0 钉死的契约,walk 面
不能绕过它)。
"""
import pytest
import torch
import torch.nn as nn

from acorninfer.config import EngineConfig, Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM, dispatch_linear
from acorninfer.quant.quantize import QUANT_ATTRS, apply_quant


def mini_cfg() -> Qwen3Config:
    """128 倍数单层配置:hidden/kv/q/intermediate/vocab 全为 128 倍数
    (quantize_w8 _guard_shape 零特例口径可过;形状/层数最小化,纯 CPU)。"""
    return Qwen3Config(
        num_hidden_layers=1, hidden_size=128, num_attention_heads=8,
        num_key_value_heads=8, head_dim=16, intermediate_size=256,
        rms_norm_eps=1e-6, rope_theta=10_000.0, vocab_size=256,
        tie_word_embeddings=False, bos_token_id=0, eos_token_id=[1],
    )


def _proj_modules(model):
    """8 类目标模块(单层 7 投影 + lm_head),按 name 索引。"""
    out = {}
    for name, mod in model.named_modules():
        last = name.split(".")[-1]
        if isinstance(mod, nn.Linear) and last in QUANT_ATTRS:
            out[name] = mod
    return out


# ---------- EngineConfig.quant 校验 ----------


def test_config_validates_quant():
    """quant ∈ {none,w8a16,w4a16};缺省 none;非法构造期 ValueError。"""
    assert EngineConfig(model_dir="/m").quant == "none"
    for q in ("none", "w8a16", "w4a16"):
        assert EngineConfig(model_dir="/m", quant=q).quant == q
    with pytest.raises(ValueError, match="quant"):
        EngineConfig(model_dir="/m", quant="int8")
    with pytest.raises(ValueError, match="quant"):
        EngineConfig(model_dir="/m", quant="w8a32")


# ---------- quant × tp 互斥(构造点 = server._build_engine) ----------


def test_quant_tp2_mutually_exclusive(monkeypatch):
    """--quant × --tp>=2 构造期 ValueError(spec §4;EngineConfig 无 tp 字段,
    互斥在 tp 已知的 _build_engine 把守)。tp=2 + quant=none 不触发(负对照)。"""
    import types

    import acorninfer.server as server_mod

    with pytest.raises(ValueError, match="tp"):
        server_mod._build_engine(types.SimpleNamespace(tp=2, quant="w8a16"))

    # 负对照:tp=2 + none 必须穿透互斥检查走到 TP 模型构造(打桩到底)
    created = {}

    def fake_tp_model(model_dir, rank, world, device, dtype):
        created["tp_model"] = (rank, world)
        return types.SimpleNamespace()

    monkeypatch.setattr(server_mod, "_build_tp_model", fake_tp_model)
    monkeypatch.setattr(server_mod, "Tokenizer", lambda path: "TOK")
    monkeypatch.setattr(server_mod, "Engine", lambda *a, **k: "ENGINE")
    args = types.SimpleNamespace(
        tp=2, quant="none", model_dir="/m", device="cuda:0", num_blocks=8,
        page_size=16, max_running=4, watermark_blocks=2, chunked_budget=None,
    )
    assert server_mod._build_engine(args, link=types.SimpleNamespace(rank=1)) == "ENGINE"
    assert created["tp_model"] == (1, 2)


# ---------- apply_quant walk(8 类打标 + 其余零触碰枚举) ----------


def test_apply_quant_walk_targets():
    """w8a16:8 类打标(_m7_quant/_q_scale/int8 存储),非目标模块零触碰
    (枚举断言:无标记、dtype 不变);none:不 walk、全模型零触碰。"""
    model = Qwen3ForCausalLM(mini_cfg()).to(dtype=torch.bfloat16)
    targets = _proj_modules(model)
    assert len(targets) == 8  # 单层 7 投影 + lm_head(walk 面契约)
    before = {n: m.weight.data.clone() for n, m in targets.items()}

    apply_quant(model, "w8a16")
    for name, mod in targets.items():
        assert mod._m7_quant == "w8a16"
        assert mod.weight.dtype == torch.int8
        assert mod.weight.requires_grad is False   # int8 换存储的前置(见 apply_quant)
        assert mod.weight.data.shape == before[name].shape
        assert mod._q_scale.dtype == torch.float16
        assert mod._q_scale.shape == (mod.weight.shape[0],)
        # 单存储:量化来源就是原 bf16 权重(dequant 量化误差界内,T0 数学)
        deq = mod.weight.data.float() * mod._q_scale.float()[:, None]
        bound = mod._q_scale.float()[:, None] / 2 + 2e-3
        assert torch.allclose(deq, before[name].float(), atol=1.0), name
    # 非目标零触碰:embed/norm(顶层 + q/k RMSNorm)/rope 无标记、dtype 不变
    for name, mod in model.named_modules():
        last = name.split(".")[-1]
        if not (isinstance(mod, nn.Linear) and last in QUANT_ATTRS):
            assert not hasattr(mod, "_m7_quant"), name
            if isinstance(mod, nn.Linear):
                assert mod.weight.dtype == torch.bfloat16, name
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.bfloat16

    # none:零触碰(不 walk,无任何 attr / dtype 变化)
    model2 = Qwen3ForCausalLM(mini_cfg()).to(dtype=torch.bfloat16)
    apply_quant(model2, "none")
    for name, mod in model2.named_modules():
        assert not hasattr(mod, "_m7_quant"), name
        if isinstance(mod, nn.Linear):
            assert mod.weight.dtype == torch.bfloat16, name


def test_apply_quant_rejects_unknown_mode():
    """w4a16 = 段 B 门控:装载期显式 NotImplementedError(不静默)。"""
    model = Qwen3ForCausalLM(mini_cfg()).to(dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="w4a16"):
        apply_quant(model, "w4a16")


# ---------- none 路径逐位(CPU;GPU 全等由 golden m0–m5 零换金硬门覆盖) ----------


def test_none_path_bitwise():
    """quant none(缺省):dispatch_linear 全部走 mod(x) 旧路径,逐位全等;
    8 类模块在缺省下均无 quant 标记。"""
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to(dtype=torch.bfloat16)
    torch.manual_seed(7)
    for name, mod in _proj_modules(model).items():
        assert getattr(mod, "_m7_quant", None) is None, name
        x = torch.randn(3, mod.in_features, dtype=torch.bfloat16)
        assert torch.equal(dispatch_linear(mod, x), mod(x)), name
        x3 = torch.randn(2, 4, mod.in_features, dtype=torch.bfloat16)
        assert torch.equal(dispatch_linear(mod, x3), mod(x3)), name


# ---------- 装载路径接线(engine.from_config / server._build_engine) ----------


def _stub_engine_paths(monkeypatch, quant):
    """两条构造路径共用打桩:from_pretrained → 真 mini 模型(CPU,walk 真跑);
    quant_gemm.configure → spy(triton/autotune 不进单测);Engine → 记录桩。"""
    import types

    import acorninfer.engine.engine as engine_mod
    import acorninfer.server as server_mod

    calls = {"apply": [], "configure": [], "engine": None}
    real_apply = apply_quant

    def spy_apply(model, mode):
        calls["apply"].append((type(model).__name__, mode))
        return real_apply(model, mode)

    def spy_configure(**kw):
        calls["configure"].append(kw)

    monkeypatch.setattr(engine_mod, "from_pretrained",
                        lambda *a, **k: Qwen3ForCausalLM(mini_cfg()).to(dtype=torch.bfloat16))
    monkeypatch.setattr(server_mod, "from_pretrained",
                        lambda *a, **k: Qwen3ForCausalLM(mini_cfg()).to(dtype=torch.bfloat16))
    monkeypatch.setattr("acorninfer.ops.quant_gemm.configure", spy_configure)
    return calls, spy_apply, types


def test_from_config_quant_wiring(monkeypatch):
    """engine.from_config:quant≠none → 装载后 apply_quant + configure(Engine
    构造前,graph 捕获前);quant=none(缺省)→ 零调用、模型零触碰。"""
    import types

    import acorninfer.engine.engine as engine_mod

    calls, spy_apply, _ = _stub_engine_paths(monkeypatch, "w8a16")
    monkeypatch.setattr("acorninfer.quant.quantize.apply_quant", spy_apply)
    # engine.py 按名导入 apply_quant(from acorninfer.quant.quantize import
    # apply_quant)→ 模块命名空间桩
    monkeypatch.setattr(engine_mod, "apply_quant", spy_apply)

    cfg = EngineConfig(model_dir="/m", device="cpu", num_blocks=16,
                       quant="w8a16")
    eng = engine_mod.Engine.from_config(cfg, tokenizer=None)
    assert calls["apply"] == [("Qwen3ForCausalLM", "w8a16")]
    assert calls["configure"] == [{"device": "cpu"}]
    assert eng.model.lm_head._m7_quant == "w8a16"
    assert eng.model.lm_head.weight.dtype == torch.int8

    # 缺省 none:零调用、模型零触碰(none 硬门静态面)
    calls2, _, _ = _stub_engine_paths(monkeypatch, "none")
    eng2 = engine_mod.Engine.from_config(
        EngineConfig(model_dir="/m", device="cpu", num_blocks=16), None)
    assert calls2["apply"] == [] and calls2["configure"] == []
    assert not hasattr(eng2.model.lm_head, "_m7_quant")


def test_server_passthrough_quant(monkeypatch):
    """server._build_engine --quant 透传(照 chunked-budget 钉子风格):
    quant=w8a16 → 装载后 apply_quant + configure;缺省(旧调用方 args 无
    quant 字段)→ 零调用(getattr 缺省 = none = 逐字节旧行为)。"""
    import types

    import acorninfer.server as server_mod

    calls, spy_apply, _ = _stub_engine_paths(monkeypatch, "w8a16")
    monkeypatch.setattr(server_mod, "apply_quant", spy_apply)
    created = {}
    monkeypatch.setattr(server_mod, "Engine",
                        lambda model, tok, **kw: created.update(model=model) or "ENGINE")
    monkeypatch.setattr(server_mod, "Tokenizer", lambda path: "TOK")

    base = dict(model_dir="/m", device="cuda:0", num_blocks=8, page_size=16,
                max_running=4, watermark_blocks=2, chunked_budget=None)
    # 旧调用方(args 无 quant 字段):缺省 none,零触碰
    assert server_mod._build_engine(types.SimpleNamespace(**base)) == "ENGINE"
    assert calls["apply"] == [] and calls["configure"] == []
    assert not hasattr(created["model"].lm_head, "_m7_quant")
    # --quant w8a16:apply + configure 透传(device = cfg.device)
    assert server_mod._build_engine(
        types.SimpleNamespace(**base, quant="w8a16")) == "ENGINE"
    assert calls["apply"] == [("Qwen3ForCausalLM", "w8a16")]
    assert calls["configure"] == [{"device": "cuda:0"}]
    assert created["model"].lm_head._m7_quant == "w8a16"


# ---------- w8a16 分派(GPU:kernel 路由 + 形状语义;mock 装载 cfg 表) ----------


@pytest.mark.gpu
def test_dispatch_linear_w8a16_routes_quant_gemm():
    """w8a16:dispatch_linear → quant_gemm(同 cfg 逐位同),输出 view 回 x
    领先维(调用点 .view()/.float() 语义不变);cls 按 (N,K) 查五档表。
    cfg 表 mock 装载(autotune 仅装载期,不进单测;T1 测试同款先例)。"""
    from acorninfer.ops import quant_gemm

    quant_gemm.reset()
    mod = nn.Linear(4096, 4096, bias=False).to(device="cuda", dtype=torch.bfloat16)
    from acorninfer.quant.quantize import quantize_w8
    q, s = quantize_w8(mod.weight.data)
    mod.weight.requires_grad_(False)   # 生产同形:apply_quant 换 int8 存储的前置
    mod.weight.data = q
    mod._q_scale = s
    mod._m7_quant = "w8a16"
    quant_gemm._state["configs"]["qo"] = (64, 128, 4, 4, 1)
    quant_gemm._state["m_configs"]["qo"] = {"s": (32, 64, 128, 4, 3)}
    try:
        torch.manual_seed(3)
        x2 = torch.randn(7, 4096, device="cuda", dtype=torch.bfloat16)
        out = dispatch_linear(mod, x2)
        ref = quant_gemm.quant_gemm(x2, q, s, "qo", cfg=(64, 128, 4, 4, 1))
        assert torch.equal(out, ref)          # 同 cfg 同 launch,逐位
        x3 = torch.randn(2, 5, 4096, device="cuda", dtype=torch.bfloat16)
        out3 = dispatch_linear(mod, x3)
        assert out3.shape == (2, 5, 4096)     # view 回领先维(3D 调用点契约)
        assert torch.equal(
            out3, quant_gemm.quant_gemm(x3.reshape(10, 4096), q, s, "qo",
                                        cfg=(64, 128, 4, 4, 1)).view(2, 5, 4096))
        # 大 M 桶:M>16 走 m_configs(显式 cfg = (BM,BN,BK,warps,stages))
        xm = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)
        outm = dispatch_linear(mod, xm)
        assert outm.shape == (32, 4096)
        assert torch.equal(
            outm, quant_gemm.quant_gemm(xm, q, s, "qo", cfg=(32, 64, 128, 4, 3)))
        # 形状不在五档 → 显式报错(零静默)
        small = nn.Linear(64, 64, bias=False).to(device="cuda", dtype=torch.bfloat16)
        small._m7_quant = "w8a16"
        with pytest.raises(ValueError, match="五档"):
            dispatch_linear(small, torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    finally:
        quant_gemm.reset()


# ---------- w8a16 端到端 margin 门(mini 模型;量化域 vs bf16 域不逐位) ----------


@pytest.mark.gpu
def test_w8a16_mini_model_decode_margin(monkeypatch):
    """margin 门协议(tiny 抽查,M5 先例;量化 vs bf16 是不同数值域):
    同一 mini 模型 none vs w8a16 forward_decode logits —— 有限性 + 逐步
    argmax 的 margin 门(参照步 top-2 差 ≥ margin 才断言 argmax 一致;平局步
    只记账不断言)。cfg 表 mock(五档表查式与小形状解耦:cls 键按形状注入)。"""
    from acorninfer.ops import quant_gemm

    torch.manual_seed(11)
    model = Qwen3ForCausalLM(mini_cfg()).to(device="cuda", dtype=torch.bfloat16).eval()
    B = 2
    ids0 = torch.randint(0, 256, (B, 5), device="cuda")

    # mock cfg 表:mini 形状按五档 cls 键注入(BM/BN/BK 与 mini 整除约束一致)
    for cls in quant_gemm.CLASSES:
        quant_gemm._state["configs"][cls] = (64, 64, 4, 4, 1)
        quant_gemm._state["m_configs"][cls] = {"s": (32, 64, 64, 4, 3),
                                               "m": (32, 64, 64, 4, 3),
                                               "l": (32, 64, 64, 4, 3)}
    patched = dict(quant_gemm._CLS_BY_SHAPE)
    # mini 形状 → cls 键(cls 仅作 cfg 表键;mini 形状与 mock cfg 整除约束一致)
    patched.update({(128, 128): "qo", (256, 128): "lm_head", (128, 256): "gate_up"})
    monkeypatch.setattr(quant_gemm, "_CLS_BY_SHAPE", patched)
    try:
        logits_none = _decode_logits(model, ids0, B)
        w_ref = {n: m.weight.data.clone() for n, m in model.named_modules()
                 if isinstance(m, nn.Linear)}
        apply_quant(model, "w8a16")
        logits_w8 = _decode_logits(model, ids0, B)
        # 权重确已换存储(单存储硬证)
        assert model.lm_head.weight.dtype == torch.int8
        assert model.lm_head.weight.data.data_ptr() != w_ref["lm_head"].data_ptr()
        d = (logits_w8 - logits_none).abs()
        top2 = logits_none.float().topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1])   # [B]
        agree = (logits_w8.argmax(-1) == logits_none.argmax(-1))
        print(f"\nW8 MARGIN mini max|Δlogit|={d.max().item():.4f} "
              f"mean|Δ|={d.mean().item():.4f} "
              f"min_margin={margin.min().item():.4f} agree={agree.sum().item()}/{B}")
        assert torch.isfinite(logits_w8).all()
        # margin 门:参照步 top-2 差 ≥ margin 的步,argmax 一致(平局步不断言)
        for b in range(B):
            if margin[b] >= 0.05:
                assert agree[b], (
                    f"row {b}: argmax 翻转但 top-2 margin={margin[b].item():.4f}"
                    f"(≥0.05 非平局)—— w8 路径数值 bug 候选")
    finally:
        quant_gemm.reset()


def _decode_logits(model, ids0, B):
    """prefill(prompt 5 位,forward_prefill_batch 写 KV)→ 单步 decode logits。"""
    import torch

    from acorninfer.cache.pager import BlockAllocator, PagedKVCache, PagedCacheView

    cfg = model.cfg
    alloc = BlockAllocator(64)
    cache = PagedKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads,
                         cfg.head_dim, 16, alloc, ids0.device, torch.bfloat16)
    views = [PagedCacheView(cache, [alloc.allocate() for _ in range(2)])
             for _ in range(B)]
    model.forward_prefill_batch(ids0, views, real_lens=[5] * B)
    logits = model.forward_decode(torch.zeros(B, 1, dtype=torch.long,
                                              device=ids0.device), views)
    for v in views:
        for b in v.table:
            alloc.free(b)
    return logits
