"""M5 Task 5:TP=2 8B 冒烟(强制流协议;计划修正版,协议以 M4/T3 先例为准)。

为何不做贪婪链逐位断言(计划原文过时):bf16 精确平局(top-2 gap = 0.0)翻
argmax 是结构性现象(T3 实测 tiny 上 172 步 21 平局)——逐位门必假红。改用:
  ①强制输入流:预生成 token 逐步喂入,平局不级联;
  ②逐步 logits TOL:TP all-gather 后全词表 [B, V] fp32 vs TP=1 参照,
    max|diff| 标定写死(标定命令/日期见 SMOKE_TOL_MAXDIFF 注释);
  ③强 gap 步 argmax 逐步断言:参照 top-2 gap >= margin 的位,TP argmax
    必须 token 相等;margin = 实测 max|diff| x 8(与 T3/T4 同口径);
  ④自由贪婪只做分歧记账:平局步数 / 平局翻转向 / 链条分裂,如实报告不设
    零门;**首分叉步 gap >= margin = 疑 bug 信号 → 退出码 2(BLOCKED 停手)**。

阶段:
  A(TP=1 参照,单进程 cuda:0):全量加载 → 强制流(prefill [8,128] +
    32 forced decode,逐步落全词表 logits)+ 自由贪婪链(gap 记账)+
    chunked 引擎参照(budget=64)。
  B(TP=2,模型级):释放 A → spawn rank1 分片加载 → 同强制流逐步对拍
    (TOL + 强 gap argmax + 页表逐位 + rank 间 logits sha256 位级同)。
  D(自由贪婪记账):TP=2 vs TP=1 参照链,32 步 x B=8(首分叉步 gap 门)。
  C(chunked x TP,引擎级):budget=64 强制 2 块,B=4/128/16;中间块零发射、
    首 token 出自 extend 完成行、逐 seq 首 token 门控、页表三方逐步逐位。
粗计时(TP=1 vs TP=2 eager decode 歇止均 / prefill)为**非官方数字**,官方
以 T6 graph/HTTP 为准。

用法(双卡):
  CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/smoke_tp2_8b.py \
      --model-dir /home/ernest/model-weight/Qwen3-8B [--calibrate]
输出:SMOKE <key> <value> 结构化行;归档 /tmp/opencode/m5t5/。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import Qwen3Config
from acorninfer.sampler import SamplingParams

ARCHIVE = Path("/tmp/opencode/m5t5")

# 协议常数(草案固定)
STEPS = 32          # forced decode 步数
BATCH = 8           # B
IN_LEN = 128        # prompt 长度
C_BATCH = 4         # 阶段 C 批
C_OUT = 16          # 阶段 C max_tokens
C_BUDGET = 64       # chunked budget(128 prompt → 强制 2 块)
PAGE = 16
SEED = 7

# 标定常数(TOL 协议:2026-11-01 标定,命令:`CUDA_VISIBLE_DEVICES=0,1
# .venv/bin/python scripts/smoke_tp2_8b.py --model-dir
# /home/ernest/model-weight/Qwen3-8B --calibrate`,归档
# /tmp/opencode/m5t5/{calib.out,summary.json,ref_forced.pt}):
# - logits TOL:8B bf16 TP=2(sdpa eager)vs TP=1 全宽,33 次前向(prefill +
#   32 decode)x 8 行全词表 max|diff| = 0.21875(两次独立跑同值,确定性;
#   ≈1-2 bf16 ulp @|logit|≈32——TP 切分归约序差经 36 层放大,量级远大于
#   tiny 单层(T3 实测 ~1.5 ulp @|logit|≈2),但仍是 ulp 级噪声非 bug);
#   TOL = 0.5 ≈ 2.3x 实测裕量(真错权重会差 O(10))。
# - margin = 实测 max|diff| x 8 = 1.75(T3/T4 口径)。参照全池 top-2 gap:
#   min 0 / p01 0 / median 0.4375——8B bf16 精确平局真实存在(margin 门
#   必要性的直接证据);strong(gap>=1.75)位 17/264,其 argmax 全对。
SMOKE_TOL_MAXDIFF: float | None = 0.5    # 实测 0.21875 之上 ~2.3x 裕量
SMOKE_MARGIN: float | None = 1.75        # = 8 x 实测 0.21875


def _log(key, value):
    print(f"SMOKE {key} {value}", flush=True)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def _gap_row(row: torch.Tensor) -> float:
    """单行 logits 的 top-2 gap(fp32)。"""
    t2 = torch.topk(row, 2).values
    return float(t2[0] - t2[1])


def _make_inputs(cfg):
    """prompt [B, IN_LEN] + forced 流 [B, STEPS](两侧逐位一致,seed 固定)。"""
    g = torch.Generator().manual_seed(SEED)
    prompt = torch.randint(0, cfg.vocab_size, (BATCH, IN_LEN), generator=g)
    forced = torch.randint(0, cfg.vocab_size, (BATCH, STEPS), generator=g)
    return prompt, forced


def _make_cache(model, cfg, device, blocks=128):
    # 128 块:模型级腿 8 seq x (128+32)/16 = 80 页 + 裕量
    kv_heads = getattr(model, "num_kv_heads_local", cfg.num_key_value_heads)
    alloc = BlockAllocator(blocks)
    cache = PagedKVCache(cfg.num_hidden_layers, kv_heads, cfg.head_dim, PAGE,
                         alloc, device, torch.bfloat16)
    return alloc, cache


def _forced_leg(model, device, prompt, forced):
    """模型级强制流:prefill logits + STEPS 步 forced decode logits。

    返回 (logits [1+STEPS, B, V] fp32 cpu, 页表 {row: (table, seq_len)},
    prefill_ms, decode_ms_mean)。
    """
    cfg = model.cfg
    alloc, cache = _make_cache(model, cfg, device)
    views = [PagedCacheView(cache) for _ in range(prompt.shape[0])]
    outs = []
    with torch.inference_mode():
        t0 = time.perf_counter()
        lg = model.forward_prefill_batch(prompt.to(device), views,
                                         real_lens=[IN_LEN] * BATCH)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3
        outs.append(lg.float().cpu())
        t0 = time.perf_counter()
        for t in range(forced.shape[1]):
            cur = forced[:, t:t + 1].to(device=device, dtype=torch.long)
            outs.append(model.forward_decode(cur, views).float().cpu())
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1e3 / forced.shape[1]
    snap = {i: (list(v.table), int(v.seq_len)) for i, v in enumerate(views)}
    return torch.stack(outs), snap, prefill_ms, decode_ms


def _greedy_leg(model, device, prompt):
    """自由贪婪链:prefill → argmax 喂回 STEPS-1 步(共 STEPS 个 token)。

    返回 (tokens [B][STEPS](首 token 出自 prefill logits), gaps [B][STEPS])。
    """
    cfg = model.cfg
    alloc, cache = _make_cache(model, cfg, device)
    views = [PagedCacheView(cache) for _ in range(prompt.shape[0])]
    toks: list[list[int]] = [[] for _ in range(prompt.shape[0])]
    gaps: list[list[float]] = [[] for _ in range(prompt.shape[0])]
    with torch.inference_mode():
        lg = model.forward_prefill_batch(prompt.to(device), views,
                                         real_lens=[IN_LEN] * BATCH).float()
        for t in range(STEPS):
            row_toks = lg.argmax(dim=-1)
            for i in range(lg.shape[0]):
                toks[i].append(int(row_toks[i]))
                gaps[i].append(_gap_row(lg[i]))
            if t + 1 == STEPS:
                break
            lg = model.forward_decode(row_toks.unsqueeze(1).to(device),
                                      views).float()
    return toks, gaps


def _norm_snap(snap):
    """引擎快照 JSON 往返归一:{int(sid): [table list, lens list]}。"""
    return {int(sid): [list(t), [int(x) for x in lens]]
            for sid, (t, lens) in snap.items()}


def _norm_tables(tables):
    """模型级腿页表 JSON 往返归一:{int(row): [table list, seq_len int]}。"""
    return {int(i): [list(t), int(n)] for i, (t, n) in tables.items()}


# ---------- 阶段 C:引擎级 chunked(参照与 TP 同构驱动)----------


class _Rec:
    """前向记录代理(同 tests/test_tp_engine):logits 全序落池。"""

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


class _EmissionProbe:
    """包 executor 四路:每次前向的 (kind, sid, token, gap) 全序落表。

    gap 取 pool 末条对应行(每 executor 方法恰一次模型前向);chunk 首块走
    engine.model 直调不经 executor → 零发射(冒烟断言依赖此口径)。
    """

    def __init__(self, inner, emissions, pool):
        self._inner = inner
        self._em = emissions
        self._pool = pool

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def prefill(self, seq, view):
        tok = self._inner.prefill(seq, view)
        self._em.append(("prefill", seq.seq_id, tok,
                         _gap_row(self._pool[-1][0])))
        return tok

    def prefill_batch(self, seqs, views):
        toks = self._inner.prefill_batch(seqs, views)
        rows = self._pool[-1]
        for i, (s, t) in enumerate(zip(seqs, toks)):
            self._em.append(("prefill_batch", s.seq_id, t, _gap_row(rows[i])))
        return toks

    def extend(self, seqs, views, chunks):
        toks = self._inner.extend(seqs, views, chunks)
        rows = self._pool[-1]
        for i, (s, t) in enumerate(zip(seqs, toks)):
            if t is not None:
                self._em.append(("extend", s.seq_id, t, _gap_row(rows[i])))
        return toks

    def decode_batch(self, seqs, views):
        toks = self._inner.decode_batch(seqs, views)
        rows = self._pool[-1]
        for i, (s, t) in enumerate(zip(seqs, toks)):
            self._em.append(("decode", s.seq_id, t, _gap_row(rows[i])))
        return toks


def _attach_probes(engine, pool, emissions):
    proxy = _Rec(engine.model, pool)
    engine.model = proxy            # chunk 首块的 engine.model 直调也入池
    engine.executor.model = proxy
    engine.executor = _EmissionProbe(engine.executor, emissions, pool)


def _snap(engine):
    """活跃序列页表快照:{sid: (table, 层长列表)}(同 tests)。"""
    out = {}
    for sid, v in engine._views.items():
        lens = v._lens
        keys = sorted(lens)
        out[sid] = (list(v.table), [int(lens[k]) for k in keys])
    return out


def _chunked_ref_leg(model, prompts, device):
    """TP=1 chunked 参照:budget=C_BUDGET,C_BATCH 序 x C_OUT token。

    返回 (emissions [(kind, sid, tok, gap)], snaps [每 step])。
    """
    from acorninfer.engine.engine import Engine

    eng = Engine(model, None, num_blocks=256, page_size=PAGE, max_running=4,
                 watermark_blocks=8, chunked_budget=C_BUDGET)
    pool: list = []
    emissions: list = []
    _attach_probes(eng, pool, emissions)
    for p in prompts:
        eng.submit(list(p), SamplingParams(temperature=0.0), max_tokens=C_OUT)
    snaps = []
    while eng.has_pending():
        eng.step()
        snaps.append(_snap(eng))
    return emissions, snaps


# ---------- rank1 worker ----------


def _shard_fn8b(cfg, rank: int, world: int):
    """HF 命名归一分片闭包:8B safetensors 键带 `model.` 前缀
    (model.embed_tokens.weight / model.norm.weight),tiny 自存 ckpt 无前缀;
    load_safetensors_sharded 原样喂文件键 → 此处剥离后再交 build_shard_fn
    (脚本侧包装,weights.py 零改动)。"""
    from acorninfer.models.qwen3_tp import build_shard_fn

    inner = build_shard_fn(cfg, rank, world)

    def shard_fn(name, t, r, w):
        return inner(name.removeprefix("model."), t, r, w)

    return shard_fn


def _rank1_worker(init_file, model_dir, out_json):
    """rank1:setup → 分片加载(计时)→ B/D 腿 → C 命令循环 → 落盘硬退。"""
    import acorninfer.distributed as dist_tp
    from acorninfer.engine.engine import Engine
    from acorninfer.engine.tp_engine import TPLink, install_parent_guard
    from acorninfer.models.qwen3_tp import Qwen3TPForCausalLM
    from acorninfer.weights import load_safetensors_sharded

    install_parent_guard()   # rank0 死 → 本进程必死(T6 孤儿防护)
    dist_tp.setup(1, 2, f"file://{init_file}")
    try:
        _rank1_body(init_file, model_dir, out_json)
    except BaseException:
        # 异常必须可见可诊断:traceback 落盘 + 非零退出(rank0 侧集合会挂到
        # watchdog——这是设计好的停世界失败,绝不能用 os._exit(0) 掩盖)
        import traceback
        with open(out_json + ".err", "w") as f:
            f.write(traceback.format_exc())
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        os._exit(1)
    # 成功路径:destroy 限 10s + 硬退兜底(生产 rank1 同款;NCCL teardown 会挂)
    import threading as _th

    def _hard_exit():
        time.sleep(10)
        os._exit(0)

    _th.Thread(target=_hard_exit, daemon=True).start()
    try:
        dist_tp.destroy()
    except Exception:
        pass
    os._exit(0)


def _rank1_body(init_file, model_dir, out_json):
    """rank1 主体(setup 之后):加载 → B/D 腿 → C 命令循环 → JSON 落盘。"""
    from acorninfer.engine.engine import Engine
    from acorninfer.engine.tp_engine import TPLink
    from acorninfer.models.qwen3_tp import Qwen3TPForCausalLM
    from acorninfer.weights import load_safetensors_sharded

    result = {}
    cfg = Qwen3Config.from_json(f"{model_dir}/config.json")
    t0 = time.perf_counter()
    state = load_safetensors_sharded(model_dir, _shard_fn8b(cfg, 1, 2),
                                     1, 2, device="cuda:1",
                                     dtype=torch.bfloat16)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    result["load_shard_s"] = time.perf_counter() - t0
    model = Qwen3TPForCausalLM(cfg, 1, 2).to(device="cuda:1",
                                             dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    del state
    model.attention_backend = "sdpa"

    prompt, forced = _make_inputs(cfg)
    # B 腿(集合操作与 rank0 同序同数,天然配对)
    logits, bsnap, _, _ = _forced_leg(model, "cuda:1", prompt, forced)
    result["b_logits_sha"] = _sha(logits)
    result["b_tables"] = _norm_tables(bsnap)
    # D 腿(自由贪婪;SPMD 两 rank 链应位级同)
    toks, _gaps = _greedy_leg(model, "cuda:1", prompt)
    result["d_chain_sha"] = _sha(torch.tensor(toks))

    # C:同构 engine(tp=TPLink(1)),进生产命令循环(每步页表快照)
    eng = Engine(model, None, num_blocks=256, page_size=PAGE,
                 max_running=4, watermark_blocks=8,
                 chunked_budget=C_BUDGET, tp=TPLink(1))
    snapshots = []
    while True:
        cmd, payload = eng.tp.command(None)
        if cmd == "submit":
            ids, sp, mt, eos = payload
            eng.submit(ids, sp, max_tokens=mt, eos_ids=eos)
        elif cmd == "step":
            eng.step()
            snapshots.append(_snap(eng))
        elif cmd == "drop":
            eng.drop(payload)
        elif cmd == "fini":
            break
        else:
            raise ValueError(f"unknown TP command {cmd!r}")
    result["c_tokens"] = {str(sid): list(seq.generated_ids)
                          for sid, seq in eng.seqs.items()}
    result["c_snapshots"] = [_norm_snap(s) for s in snapshots]
    with open(out_json, "w") as f:
        json.dump(result, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/ernest/model-weight/Qwen3-8B")
    ap.add_argument("--calibrate", action="store_true",
                    help="只测不断:打印实测 max|diff| 与 gap 分布(标定用)")
    args = ap.parse_args()
    tol = SMOKE_TOL_MAXDIFF
    margin = SMOKE_MARGIN
    if not args.calibrate and (tol is None or margin is None):
        print("SMOKE status NOT_CALIBRATED(先 --calibrate,写死常数再正式跑)")
        sys.exit(3)

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    cfg = Qwen3Config.from_json(f"{args.model_dir}/config.json")
    prompt, forced = _make_inputs(cfg)
    g = torch.Generator().manual_seed(SEED + 1)
    c_prompts = torch.randint(0, cfg.vocab_size, (C_BATCH, IN_LEN), generator=g)
    summary: dict = {}

    # ---------- 阶段 A:TP=1 参照(单进程 cuda:0,全量加载)----------
    from acorninfer.models.qwen3 import from_pretrained

    _log("phase", "A TP=1 reference load begin")
    t0 = time.perf_counter()
    ref_model = from_pretrained(args.model_dir, "cuda:0", torch.bfloat16)
    ref_model.attention_backend = "sdpa"
    _log("load_full_s", round(time.perf_counter() - t0, 1))

    a_logits, a_tables, a_prefill_ms, a_decode_ms = _forced_leg(
        ref_model, "cuda:0", prompt, forced)
    a_toks, a_gaps = _greedy_leg(ref_model, "cuda:0", prompt)
    c_emissions, c_ref_snaps = _chunked_ref_leg(ref_model, c_prompts, "cuda:0")

    ref_gap = (a_logits.reshape(-1, a_logits.shape[-1])
               .topk(2, dim=-1).values)
    ref_gap = ref_gap[:, 0] - ref_gap[:, 1]          # 全池 [33*8]
    ref_gap_b = ref_gap.view(a_logits.shape[0], BATCH)
    _log("ref_gap_forced_min", f"{float(ref_gap.min()):.6g}")
    _log("ref_gap_forced_p01", f"{float(ref_gap.quantile(0.01)):.6g}")
    _log("ref_gap_forced_median", f"{float(ref_gap.median()):.6g}")
    torch.save(a_logits, ARCHIVE / "ref_forced.pt")
    del ref_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    _log("phase", "A done; VRAM freed")

    # ---------- 阶段 B/C/D:TP=2(spawn rank1,file init)----------
    from acorninfer.distributed import destroy, setup
    from acorninfer.engine.engine import Engine
    from acorninfer.engine.tp_engine import TPEngine, TPLink
    from acorninfer.models.qwen3_tp import Qwen3TPForCausalLM
    from acorninfer.weights import load_safetensors_sharded

    ctx = mp.get_context("spawn")
    init_file = ARCHIVE / "dist_init"
    worker = ctx.Process(target=_rank1_worker,
                         args=(str(init_file), args.model_dir,
                               str(ARCHIVE / "rank1.json")))
    worker.start()
    setup(0, 2, f"file://{init_file}")
    ok = False
    try:
        t0 = time.perf_counter()
        state = load_safetensors_sharded(args.model_dir,
                                         _shard_fn8b(cfg, 0, 2), 0, 2,
                                         device="cuda:0",
                                         dtype=torch.bfloat16)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        _log("load_shard_rank0_s", round(time.perf_counter() - t0, 1))
        tp_model = Qwen3TPForCausalLM(cfg, 0, 2).to(device="cuda:0",
                                                    dtype=torch.bfloat16)
        tp_model.load_state_dict(state, strict=True)
        del state
        tp_model.attention_backend = "sdpa"

        # ---- B:模型级强制流对拍 ----
        b_logits, b_tables, b_prefill_ms, b_decode_ms = _forced_leg(
            tp_model, "cuda:0", prompt, forced)
        d_toks, d_gaps = _greedy_leg(tp_model, "cuda:0", prompt)

        n_calls = b_logits.shape[0]
        per_call = (b_logits - a_logits).abs().amax(dim=(1, 2))
        maxdiff = float(per_call.max())
        _log("maxdiff_overall", f"{maxdiff:.6g}")
        for i, v in enumerate(per_call.tolist()):
            _log("maxdiff_call", f"{i} {v:.6g}")
        b_arg = b_logits.argmax(dim=-1)
        a_arg = a_logits.argmax(dim=-1)
        # calibrate 模式用"本次实测 x8"作候选 margin 只出统计;正式跑用写死值
        margin_eff = margin if margin is not None else maxdiff * 8
        _log("margin_eff", f"{margin_eff:.6g}")
        tie = ref_gap_b < margin_eff
        strong = ref_gap_b >= margin_eff
        arg_bad = int(((b_arg != a_arg) & strong).sum())
        _log("strong_gap_positions", int(strong.sum()))
        _log("strong_gap_argmax_mismatch", arg_bad)
        _log("tie_positions_forced", int(tie.sum()))
        if not args.calibrate:
            assert maxdiff <= tol, f"logits TOL 爆:max|diff|={maxdiff} > {tol}"
            assert arg_bad == 0, f"强 gap 步 argmax 分叉 {arg_bad} 位"
        for i in range(BATCH):
            assert b_tables[i][0] == a_tables[i][0], f"B 页表分叉 row{i}"
            assert b_tables[i][1] == a_tables[i][1] == IN_LEN + STEPS, (
                f"B 长度分叉 row{i}")
        _log("b_tables", "OK bitwise vs reference")
        _log("b_rank0_logits_sha", _sha(b_logits))

        # ---- D:自由贪婪记账(首分叉步 gap 门;分叉后两链解耦不再逐位比)----
        tie_steps = 0
        tie_flips = 0          # 首分叉步 gap < margin 的行数(结构性平局翻转)
        splits = 0             # 发生过分叉的行数
        strong_splits = 0      # 首分叉步 gap >= margin(疑 bug 信号)
        for r in range(BATCH):
            tie_steps += sum(1 for t in range(STEPS) if a_gaps[r][t] < margin_eff)
            div = next((t for t in range(STEPS)
                        if d_toks[r][t] != a_toks[r][t]), None)
            if div is None:
                continue
            splits += 1
            gap_at = a_gaps[r][div]
            if gap_at < margin_eff:
                tie_flips += 1
            else:
                strong_splits += 1
            tail = all(d_toks[r][t] == a_toks[r][t]
                       for t in range(div + 1, STEPS))
            _log("split_row", f"row={r} first_at={div} gap={gap_at:.6g} "
                              f"rejoined={tail}")
        _log("d_tie_steps", tie_steps)
        _log("d_tie_flips", tie_flips)
        _log("d_chain_splits", splits)
        _log("d_strong_gap_splits", strong_splits)
        _log("d_rank0_chain_sha", _sha(torch.tensor(d_toks)))
        if strong_splits > 0:
            print("SMOKE status BLOCKED:强 gap 步链条分裂(疑 bug,停手)")
            sys.exit(2)

        # ---- C:引擎级 chunked x TP(budget=64 强制 2 块)----
        tp_eng = Engine(tp_model, None, num_blocks=256, page_size=PAGE,
                        max_running=4, watermark_blocks=8,
                        chunked_budget=C_BUDGET, tp=TPLink(0))
        tp_pool: list = []
        tp_emissions: list = []
        _attach_probes(tp_eng, tp_pool, tp_emissions)
        shell = TPEngine(tp_eng, tp_eng.tp)
        for p in c_prompts:
            shell.submit(list(p), SamplingParams(temperature=0.0),
                         max_tokens=C_OUT)
        tp_snaps = []
        while shell.has_pending():
            shell.step()
            tp_snaps.append(_snap(tp_eng))
        shell.shutdown()

        # C 对拍:发射结构对齐 + 逐 seq 首 token gap 门控 + 页表三方逐步逐位
        assert len(tp_emissions) == len(c_emissions) == C_BATCH * C_OUT, (
            f"C 发射数分叉 tp={len(tp_emissions)} ref={len(c_emissions)}")
        assert ([e[1] for e in tp_emissions]
                == [e[1] for e in c_emissions]), "C 发射 (sid) 序分叉"
        ref_by_sid: dict = defaultdict(list)
        for k, sid, t, gp in c_emissions:
            ref_by_sid[sid].append((t, gp))
        tp_by_sid: dict = defaultdict(list)
        for k, sid, t, gp in tp_emissions:
            tp_by_sid[sid].append(t)
        first_kind = {}
        for k, sid, _t, _g in tp_emissions:
            first_kind.setdefault(sid, k)
        assert all(k == "extend" for k in first_kind.values()), (
            f"chunked 首 token 出自非 extend 块:{first_kind}")
        c_tie_flips = 0
        c_strong_bad = 0
        for sid, rt in ref_by_sid.items():
            tt = tp_by_sid[sid]
            assert len(tt) == len(rt) == C_OUT
            div = next((j for j in range(C_OUT) if rt[j][0] != tt[j]), None)
            if div is None:
                continue
            if rt[div][1] >= margin_eff:
                c_strong_bad += 1
                _log("c_strong_div", f"sid={sid} at={div} "
                                     f"gap={rt[div][1]:.6g}")
            else:
                c_tie_flips += 1
                _log("c_tie_div", f"sid={sid} at={div} "
                                  f"gap={rt[div][1]:.6g}")
        if not args.calibrate:
            assert c_strong_bad == 0, f"C 强 gap 分叉 {c_strong_bad} 处"
        _log("c_emissions", len(tp_emissions))
        _log("c_tie_flips", c_tie_flips)
        _log("c_first_token_kind", "extend(all)")
        assert len(tp_snaps) == len(c_ref_snaps), "C 步数分叉"
        for i, (ts, rs) in enumerate(zip(tp_snaps, c_ref_snaps)):
            assert ts == rs, f"C 页表逐步分叉 @step{i}"
        _log("c_tables", "OK bitwise tp_rank0 vs reference per-step")

        worker.join(timeout=600)
        assert worker.exitcode == 0, f"rank1 exitcode={worker.exitcode}"
        rank1 = json.load(open(ARCHIVE / "rank1.json"))
        _log("load_shard_rank1_s", round(rank1["load_shard_s"], 1))
        # rank 间位级(SPMD):logits sha / 链 sha / B 页表 / C token+页表
        assert rank1["b_logits_sha"] == _sha(b_logits), "B logits rank 间分叉"
        assert rank1["d_chain_sha"] == _sha(torch.tensor(d_toks)), (
            "D 链 rank 间分叉")
        assert rank1["b_tables"] == {str(k): v for k, v in
                                     _norm_tables(b_tables).items()}, (
            "B 页表 rank 间分叉")
        r0_by_sid: dict = defaultdict(list)
        for _k, sid, t, _g in tp_emissions:
            r0_by_sid[sid].append(t)
        r1_stream = [(int(sid), t) for sid in sorted(rank1["c_tokens"],
                                                     key=int)
                     for t in rank1["c_tokens"][sid]]
        r0_stream = [(sid, t) for sid in sorted(r0_by_sid)
                     for t in r0_by_sid[sid]]
        assert r1_stream == r0_stream, "C token 流 rank 间分叉"
        assert rank1["c_snapshots"] == [{str(k): v for k, v in
                                         _norm_snap(s).items()}
                                        for s in tp_snaps], (
            "C 页表 rank 间分叉")
        _log("rank_parity", "OK (sha256 logits/chain + tokens + tables)")

        # ---- 粗计时(非官方数字,官方以 T6 graph/HTTP 为准)----
        _log("time_tp1_eager_decode_ms", f"{a_decode_ms:.1f}")
        _log("time_tp2_eager_decode_ms", f"{b_decode_ms:.1f}")
        _log("time_tp1_eager_prefill_ms", f"{a_prefill_ms:.1f}")
        _log("time_tp2_eager_prefill_ms", f"{b_prefill_ms:.1f}")
        _log("speedup_decode", f"{a_decode_ms / b_decode_ms:.2f}x")

        summary.update(
            maxdiff=maxdiff,
            strong_positions=int(strong.sum()),
            arg_bad=arg_bad,
            tie_positions_forced=int(tie.sum()),
            ref_gap_forced_min=float(ref_gap.min()),
            d_tie_steps=tie_steps, d_tie_flips=tie_flips,
            d_splits=splits, d_strong_splits=strong_splits,
            c_tie_flips=c_tie_flips, c_strong_bad=c_strong_bad,
            times_ms={"tp1_decode": a_decode_ms, "tp2_decode": b_decode_ms,
                      "tp1_prefill": a_prefill_ms, "tp2_prefill": b_prefill_ms},
        )
        with open(ARCHIVE / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        ok = True
        _log("status", "OK")
    finally:
        destroy()
        worker.join(timeout=30)
        if worker.is_alive():
            os.kill(worker.pid, 9)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
