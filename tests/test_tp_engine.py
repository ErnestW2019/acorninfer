"""TP=2 graph decode + SPMD engine 双卡对拍(M5 Task 4;单卡环境整模块 skip)。

三个对拍(协议沿 T3:强制输入流 + margin 门;贪婪链逐位断言不允许——
bf16 精确平局会翻 argmax,M4/T3 已证伪;此处以"参照侧 min top-2 gap ≥
margin"作协议健康哨兵,哨兵绿时流逐位断言才有效,哨兵红 = 协议漂移先红):

- test_tp2_graph_matches_eager:TP 模型 decode 过既有 DecodeGraphRunner
  (零改动,prepared 契约 + triton 分页核)vs 同模型 eager(sdpa)双卡对拍。
- test_tp2_engine_spmd_tokens:TPEngine 命令壳 + token 级同步 + 页表哨兵全链;
  rank0 = 测试进程,spawn 单个 rank1;token 流三方逐位 + logits TOL +
  页表快照三方逐位。
- test_tp2_chunked_extend:chunked_budget < prompt 强制切块,SPMD TP vs
  TP=1 chunked 参照;中间块零发射沿用(空输出步存在性断言)。
"""

import json
import multiprocessing as mp
import types

import pytest
import torch
from safetensors.torch import save_file

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM
from acorninfer.sampler import SamplingParams

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="TP=2 需要 2 张 CUDA 卡"
)

FORCED_SEED = 42   # 强制输入流种子(与 T3 同源)
P1_STEPS = 3
P2_STEPS = 40
P3_STEPS = 5

PROMPTS = [[7, 11, 13, 200, 45], [123, 5, 77]]       # SPMD engine 协议
CHUNK_PROMPTS = [[7, 11, 13, 200, 45, 88], [123, 5, 77, 9, 21]]
MAX_TOKENS = 12

# 容差标定(TOL 协议,按实测写死,再触发即红):
# - graph 对拍(bf16)= triton 分页核 vs SDPA(TP 同构)。标定:2026-11-01,
#   `CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python /tmp/opencode/tp_graph_measure/
#   run_measure.py`(复用本文件 _graph_worker):ALL max|diff| = 0.011719
#   (triton 归约序差 + TP 切分噪声,≈1.5 bf16 ulp @|logit|≈2);min gap=0.0
#   (精确平局位 21 个豁免),强 gap 位 67 全部 argmax 一致。
# - SPMD engine 对拍 = TP 分片 vs 全宽接线门。**fp32 协议偏离**(对 bf16 默认):
#   engine 的 decode 链是贪婪链,bf16 tiny 模型 gap 天然落平局带(实测 40 候选
#   prompt 扫描 min gap ≤ 0.0234 < margin 0.08,20/40 出精确平局)——纪律禁放宽
#   门,故接线门升 fp32:SPMD 噪声 ~1e-6 ≪ 全部 gap,平局带消失,调度/采样/
#   页表接线得到强断言;bf16 数值面仍由 T3(eager)与本文件 graph 对拍钉住。
#   SPMD fp32 标定:2026-11-01,`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python
#   /tmp/opencode/spmd_debug/calib.py`(复用本文件 _run_spmd):spmd
#   max|diff|=3.576e-07 min_gap=0.0080;chunked max|diff|=4.768e-07
#   min_gap=0.0357;两侧 token 流均与参照逐位同。
_TP_GRAPH_TOL = dict(rtol=1.6e-2, atol=1.2e-2)    # bf16:实测 0.011719 之上留裕量
_TP_TIE_MARGIN = 8e-2                              # graph 对拍用(≈8 × bf16 实测 max|diff|)
_SPMD_TOL = dict(rtol=1e-5, atol=1e-6)             # fp32:实测 4.768e-07 之上留 2-4× 裕量
_SPMD_TIE_MARGIN = 1e-3                            # fp32 哨兵(≫ 噪声,≪ 观测 min gap 0.008)


def _forced_inputs():
    """预生成输入 token 流([B, steps];两侧逐位一致;同 T3)。"""
    g = torch.Generator().manual_seed(FORCED_SEED)
    vocab = Qwen3Config.tiny().vocab_size
    p1 = torch.randint(0, vocab, (1, P1_STEPS), generator=g)
    p2 = torch.randint(0, vocab, (2, P2_STEPS), generator=g)
    p3 = torch.randint(0, vocab, (1, P3_STEPS), generator=g)
    return p1, p2, p3


def _protocol_phases():
    """三段协议的统一描述:(views 数, 输入流, 步数)。"""
    p1, p2, p3 = _forced_inputs()
    return [(1, p1, P1_STEPS), (2, p2, P2_STEPS), (1, p3, P3_STEPS)]


# ---------- 模型级协议(graph 测试用;runner=None = eager decode)----------


def _model_protocol(model, device, runner=None, *, cache=None, alloc=None):
    """强制流 decode 协议;runner 非 None 时经 DecodeGraphRunner.decode_batch。

    runner 路径 cache/alloc 必须传 runner 构造所用的同一实例 —— 图捕获冻结
    页张量地址,换 cache 实例 = replay 读写错池。
    """
    cfg = model.cfg
    kv_heads = getattr(model, "num_kv_heads_local", cfg.num_key_value_heads)
    if alloc is None:
        alloc = BlockAllocator(64)
    if cache is None:
        cache = PagedKVCache(
            cfg.num_hidden_layers, kv_heads, cfg.head_dim, 4, alloc, device,
            torch.bfloat16,
        )
    views = [PagedCacheView(cache) for _ in range(2)]
    phases = []
    with torch.inference_mode():
        for b_rows, stream, steps in _protocol_phases():
            toks_all: list = []
            logits_all: list = []
            for t in range(steps):
                cur = stream[:, t:t + 1].to(device=device, dtype=torch.long)
                if runner is not None:
                    seqs = [types.SimpleNamespace(generated_ids=[int(x)])
                            for x in cur[:, 0]]
                    toks = runner.decode_batch(seqs, views[:b_rows])
                    logits = runner.last_logits[:b_rows].float().cpu().clone()
                else:
                    logits = model.forward_decode(cur, views[:b_rows]).float().cpu()
                    toks = [int(x) for x in logits.argmax(dim=-1)]
                toks_all.append(toks)
                logits_all.append(logits)
            phases.append({
                "b": b_rows,
                "tokens": toks_all,
                "logits": torch.stack(logits_all),   # [steps, B, vocab]
            })
    return {"phases": phases}


def _graph_worker(rank, world, init_file, ckpt_dir, out_json):
    """子进程:eager(sdpa)腿 + graph(triton runner)腿;JSON 落盘。

    图捕获 + NCCL 收尾会挂(T0 先例):结果落盘后 os._exit(0),不走 destroy。
    两腿共用同一 cache/allocator:runner 捕获冻结页张量地址,协议必须同池。
    """
    import os as _os

    from acorninfer.distributed import setup
    from acorninfer.engine.graph_runner import DecodeGraphRunner
    from acorninfer.models.qwen3_tp import Qwen3TPForCausalLM, build_shard_fn
    from acorninfer.weights import load_safetensors_sharded

    setup(rank, world, f"file://{init_file}")
    cfg = Qwen3Config.tiny()
    shard_fn = build_shard_fn(cfg, rank, world)
    state = load_safetensors_sharded(
        ckpt_dir, shard_fn, rank, world, device=f"cuda:{rank}",
        dtype=torch.bfloat16,
    )
    model = Qwen3TPForCausalLM(cfg, rank, world).to(
        device=f"cuda:{rank}", dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    device = f"cuda:{rank}"

    alloc = BlockAllocator(64)
    cache = PagedKVCache(cfg.num_hidden_layers, model.num_kv_heads_local,
                         cfg.head_dim, 4, alloc, device, torch.bfloat16)

    model.attention_backend = "sdpa"
    eager = _model_protocol(model, device, cache=cache, alloc=alloc)

    model.attention_backend = "triton"   # runner 构造期断言 triton(契约同单卡)
    runner = DecodeGraphRunner(model, cache, alloc)
    graph = _model_protocol(model, device, runner=runner, cache=cache,
                            alloc=alloc)

    with open(out_json, "w") as f:
        json.dump({
            "eager": {"phases": [{**ph, "logits": ph["logits"].tolist()}
                                 for ph in eager["phases"]]},
            "graph": {"phases": [{**ph, "logits": ph["logits"].tolist()}
                                 for ph in graph["phases"]]},
        }, f)
    _os._exit(0)   # NCCL + graph teardown 挂;保持图存活硬退(T0 先例)


# ---------- engine 级协议(SPMD 测试用)----------


class _Rec:
    """前向记录代理:包 model 的四个入口,logits 全序落池(engine 对拍用)。"""

    def __init__(self, inner, pool):
        self._inner = inner
        self._pool = pool

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __call__(self, ids, view, *, num_logits_to_keep=None):
        out = self._inner(ids, view, num_logits_to_keep=num_logits_to_keep)
        self._pool.append(out[:, -1, :].float().cpu())
        return out

    def forward_prefill_batch(self, ids, views, *, real_lens):
        out = self._inner.forward_prefill_batch(ids, views, real_lens=real_lens)
        self._pool.append(out.float().cpu())
        return out

    def forward_decode(self, x, views):
        out = self._inner.forward_decode(x, views)
        self._pool.append(out.float().cpu())
        return out

    def forward_extend(self, ids, views, *, starts, chunk_lens):
        out = self._inner.forward_extend(ids, views, starts=starts,
                                         chunk_lens=chunk_lens)
        self._pool.append(out.float().cpu())
        return out


def _attach_recorder(engine, pool):
    proxy = _Rec(engine.model, pool)
    engine.model = proxy            # engine.step 的 part 分支直呼 model
    engine.executor.model = proxy   # executor 四路前向
    return proxy


def _snap(engine):
    """活跃序列页表快照:{sid: (table, 层长列表)}(退休后键消失,两侧同构)。"""
    out = {}
    for sid, v in engine._views.items():
        lens = v._lens
        keys = sorted(lens)
        out[sid] = (list(v.table), [int(lens[k]) for k in keys])
    return out


def _build_tp_engine(ckpt_dir, rank, budget, device=None):
    """TP engine(两 rank 同构):分片加载 → TP 模型 → Engine(tp=TPLink)。

    dtype = fp32(SPMD 接线门协议,见模块头容差注释的偏离说明)。
    """
    from acorninfer.engine.engine import Engine
    from acorninfer.engine.tp_engine import TPLink
    from acorninfer.models.qwen3_tp import Qwen3TPForCausalLM, build_shard_fn
    from acorninfer.weights import load_safetensors_sharded

    cfg = Qwen3Config.tiny()
    shard_fn = build_shard_fn(cfg, rank, 2)
    state = load_safetensors_sharded(
        ckpt_dir, shard_fn, rank, 2,
        device=device or f"cuda:{rank}", dtype=torch.float32,
    )
    model = Qwen3TPForCausalLM(cfg, rank, 2).to(
        device=device or f"cuda:{rank}", dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    return Engine(model, None, num_blocks=64, page_size=4, max_running=4,
                  watermark_blocks=0, chunked_budget=budget, tp=TPLink(rank))


def _build_ref_engine(budget, device="cuda:0"):
    """TP=1 参照 engine(全宽 tiny,同调度参数;fp32 同协议)。"""
    from acorninfer.engine.engine import Engine

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to(device, torch.float32)
    return Engine(model, None, num_blocks=64, page_size=4, max_running=4,
                  watermark_blocks=0, chunked_budget=budget)


def _spmd_rank1_worker(init_file, ckpt_dir, out_json, budget):
    """rank1:NCCL setup → 同构 TP engine → 命令循环(生产 tp_rank1_loop 同构)。"""
    from acorninfer.distributed import broadcast_object, setup

    setup(1, 2, f"file://{init_file}")
    eng = _build_tp_engine(ckpt_dir, 1, budget)
    snapshots = []
    while True:
        cmd, payload = broadcast_object(None, src=0)
        if cmd == "submit":
            ids, sp, mt, eos = payload
            eng.submit(ids, sp, max_tokens=mt, eos_ids=eos)
        elif cmd == "step":
            eng.step()
            snapshots.append(_snap(eng))
        elif cmd == "pending":
            eng.has_pending()
        elif cmd == "drop":
            eng.drop(payload)
        elif cmd == "balanced":
            eng.assert_balanced()
        elif cmd == "fini":
            break
        else:
            raise ValueError(f"unknown cmd {cmd!r}")
    result = {
        "tokens": {sid: list(seq.generated_ids)
                   for sid, seq in eng.seqs.items()},
        "snapshots": snapshots,
    }
    with open(out_json, "w") as f:
        json.dump(result, f)
    # NCCL teardown 挂(T0 先例):结果落盘后硬退;rank0 已先广播 fini 配平
    import os as _os
    _os._exit(0)


def _run_spmd(budget, prompts, max_tokens, ckpt_dir, tmp_path):
    """SPMD 全链驱动(rank0 = 测试进程)+ TP=1 参照;返回双方全量记录。"""
    from acorninfer.distributed import destroy, setup
    from acorninfer.engine.tp_engine import TPEngine

    ctx = mp.get_context("spawn")
    init_file = tmp_path / "dist_init"
    worker = ctx.Process(target=_spmd_rank1_worker,
                         args=(str(init_file), str(ckpt_dir),
                               str(tmp_path / "rank1.json"), budget))
    worker.start()
    setup(0, 2, f"file://{init_file}")
    try:
        tp_eng = _build_tp_engine(ckpt_dir, 0, budget, device="cuda:0")
        ref_eng = _build_ref_engine(budget)
        tp_pool: list = []
        ref_pool: list = []
        _attach_recorder(tp_eng, tp_pool)
        _attach_recorder(ref_eng, ref_pool)

        shell = TPEngine(tp_eng, tp_eng.tp)
        for p in prompts:
            shell.submit(list(p), SamplingParams(temperature=0.0),
                         max_tokens=max_tokens)
            ref_eng.submit(list(p), SamplingParams(temperature=0.0),
                           max_tokens=max_tokens)
        tp_out: list = []
        ref_out: list = []
        tp_snaps: list = []
        ref_snaps: list = []
        tp_step_out: list = []
        while shell.has_pending() or ref_eng.has_pending():
            if shell.has_pending():
                outs = shell.step()
                tp_out.extend(outs)
                tp_step_out.append(len(outs))
                tp_snaps.append(_snap(tp_eng))
            if ref_eng.has_pending():
                ref_out.extend(ref_eng.step())
                ref_snaps.append(_snap(ref_eng))
        shell.shutdown()
        worker.join(timeout=600)
        assert worker.exitcode == 0, f"rank1 exitcode={worker.exitcode}"
        rank1 = json.load(open(tmp_path / "rank1.json"))
        return {
            "tp_out": tp_out, "ref_out": ref_out,
            "tp_pool": tp_pool, "ref_pool": ref_pool,
            "tp_snaps": tp_snaps, "ref_snaps": ref_snaps, "rank1": rank1,
            "tp_seqs": {sid: list(s.generated_ids)
                        for sid, s in tp_eng.seqs.items()},
            "tp_step_out": tp_step_out,
        }
    finally:
        destroy()


def _norm_snap(snap):
    """JSON 往返归一化:{int(sid): (table list, lens list)}(tuple→list)。"""
    return {int(sid): (list(tab), list(lens)) for sid, (tab, lens) in snap.items()}


def _assert_parity(rec, label):
    """共同断言:logits TOL + 平局哨兵 + token 流三方逐位 + 页表三方逐位。"""
    tp_pool, ref_pool = rec["tp_pool"], rec["ref_pool"]
    assert len(tp_pool) == len(ref_pool), (
        f"{label}: 前向调用数分叉 {len(tp_pool)} vs {len(ref_pool)}"
    )
    for i, (got, want) in enumerate(zip(tp_pool, ref_pool)):
        assert got.shape == want.shape, f"{label}: 前向 #{i} 形状分叉"
        torch.testing.assert_close(got, want, **_SPMD_TOL)
    # 平局哨兵:参照全池 min top-2 gap ≥ margin → 流逐位断言有效
    ref_cat = torch.cat([r.reshape(-1, r.shape[-1]) for r in ref_pool])
    gap = ref_cat.topk(2, dim=-1).values
    min_gap = float((gap[:, 0] - gap[:, 1]).min())
    assert min_gap >= _SPMD_TIE_MARGIN, (
        f"{label}: 协议落进平局带(min gap={min_gap:.4g} < {_SPMD_TIE_MARGIN}),"
        f"流逐位断言无效——需换协议而不是放宽门"
    )
    # token 流:TP(rank0) vs 参照 逐位
    assert len(rec["tp_out"]) == len(rec["ref_out"])
    for a, b in zip(rec["tp_out"], rec["ref_out"]):
        assert (a.seq_id, a.token_id, a.finished, a.stop_reason) == \
               (b.seq_id, b.token_id, b.finished, b.stop_reason)
    # rank1 token 逐位 + 页表三方
    for sid, toks in rec["rank1"]["tokens"].items():
        assert toks == rec["tp_seqs"][int(sid)], f"{label}: rank{sid} 流分叉"
    assert len(rec["tp_snaps"]) == len(rec["ref_snaps"]) == \
        len(rec["rank1"]["snapshots"]), f"{label}: 步数分叉"
    for t, (s0, s1, sr) in enumerate(zip(rec["tp_snaps"],
                                         rec["rank1"]["snapshots"],
                                         rec["ref_snaps"])):
        s0, s1, sr = _norm_snap(s0), _norm_snap(s1), _norm_snap(sr)
        assert s0 == s1, f"{label}: step{t} 页表 rank0/rank1 漂移"
        assert s0 == sr, f"{label}: step{t} 页表 vs 参照漂移"


def test_tp2_graph_matches_eager(ckpt_fixture, tmp_path):
    """TP graph replay vs eager:logits TOL + 非平局步 argmax 逐位。"""
    ctx = mp.get_context("spawn")
    init_file = tmp_path / "dist_init"
    procs = []
    for rank in range(2):
        p = ctx.Process(target=_graph_worker,
                        args=(rank, 2, str(init_file), str(ckpt_fixture),
                              str(tmp_path / f"g{rank}.json")))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=1200)
        assert p.exitcode == 0, f"rank exitcode={p.exitcode}"

    r0 = json.load(open(tmp_path / "g0.json"))
    r1 = json.load(open(tmp_path / "g1.json"))
    eager0 = [torch.tensor(ph["logits"], dtype=torch.float32)
              for ph in r0["eager"]["phases"]]
    eager1 = [torch.tensor(ph["logits"], dtype=torch.float32)
              for ph in r1["eager"]["phases"]]
    graph0 = [torch.tensor(ph["logits"], dtype=torch.float32)
              for ph in r0["graph"]["phases"]]
    # rank 间 eager 逐位(all-gather 后 logits 两 rank 同一张量)
    for a, b in zip(eager0, eager1):
        assert torch.equal(a, b), "rank 间 eager logits 应逐位一致"
    for key in ("eager", "graph"):
        for pa, pb in zip(r0[key]["phases"], r1[key]["phases"]):
            assert pa["tokens"] == pb["tokens"]
    # 平局防护(逐位豁免语义,同 T3):强制流步间独立,平局位翻 argmax
    # 不放大;gap ≥ margin 的位断言 graph argmax == eager argmax,gap <
    # margin 的位只受 logits TOL 约束(计数报告)。
    cat = torch.cat([e.reshape(-1, e.shape[-1]) for e in eager0])
    gap = cat.topk(2, dim=-1).values
    min_gap = float((gap[:, 0] - gap[:, 1]).min())
    # graph vs eager:逐段 logits TOL + 非平局位 argmax 逐位
    strong = weak = n_dis = 0
    for ea, ga, e_ph, g_ph in zip(eager0, graph0,
                                  r0["eager"]["phases"], r0["graph"]["phases"]):
        torch.testing.assert_close(ga, ea, **_TP_GRAPH_TOL)
        gp = ea.topk(2, dim=-1).values
        m = (gp[..., 0] - gp[..., 1]) >= _TP_TIE_MARGIN
        strong += int(m.sum())
        weak += int((~m).sum())
        et = torch.tensor(e_ph["tokens"]); gt = torch.tensor(g_ph["tokens"])
        n_dis += int((et[m] != gt[m]).sum())
    print(f"\n[graph 对拍] min gap={min_gap:.4f};强 gap 位 {strong},豁免 {weak},"
          f"强位分叉 {n_dis}")
    assert n_dis == 0, "非平局位 graph argmax 分叉"


@pytest.fixture
def ckpt_fixture(tmp_path):
    """单文件 checkpoint(全量 seed0 tiny;两测试共享形状)。"""
    torch.manual_seed(0)
    full = Qwen3ForCausalLM(Qwen3Config.tiny())
    sd = {k: v.contiguous() for k, v in full.state_dict().items()}
    path = tmp_path / "model.safetensors"
    save_file(sd, str(path))
    return tmp_path


def test_tp2_engine_spmd_tokens(ckpt_fixture, tmp_path):
    """SPMD submit/step 全链:token 三方逐位 + logits TOL + 页表三方逐位。"""
    rec = _run_spmd(None, PROMPTS, MAX_TOKENS, ckpt_fixture, tmp_path)
    _assert_parity(rec, "spmd")


def test_tp2_chunked_extend(ckpt_fixture, tmp_path):
    """budget=4 < prompt:切块推进 + TP 组合;中间块零发射;末 token 同参照。"""
    rec = _run_spmd(4, CHUNK_PROMPTS, MAX_TOKENS, ckpt_fixture, tmp_path)
    _assert_parity(rec, "chunked")
    # 中间块零发射(沿用):总输出 = 每序列 max_tokens(仅完成行发);
    # 且存在零输出步 —— chunk 协议真实执行过(截断/续读块绝不外泄)
    assert len(rec["tp_out"]) == len(CHUNK_PROMPTS) * MAX_TOKENS
    assert any(n == 0 for n in rec["tp_step_out"]), (
        f"chunk 协议应存在零输出步,实测逐步输出 {rec['tp_step_out']}"
    )

