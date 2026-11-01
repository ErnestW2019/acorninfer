"""M6 段 0 尖刀:四腿测量(腿 1 in-situ 拆账 / 腿 2 调参面 / 腿 3 布局面 / 腿 4 步开销)。

用法(单卡,CUDA_VISIBLE_DEVICES=1,跑前 nvidia-smi 应 ~4MiB;前台 timeout ≥ 1800000):
  .venv/bin/python scripts/m6_spike.py insitu  [--steps 20]      # 腿 1 + dmon 同窗时钟
  .venv/bin/python scripts/m6_spike.py knobs   [--combos 0-11]  # 腿 2(子进程注 env)
  .venv/bin/python scripts/m6_spike.py layout                                # 腿 3 孤立打分
  .venv/bin/python scripts/m6_spike.py layout --engine-retest <t|pad8|pad16> # 腿 3 引擎内复测
  .venv/bin/python scripts/m6_spike.py overhead [--steps 50]    # 腿 4(wrapper+profiler+py-spy)

引擎/模型源码零触碰:腿 3 引擎内复测走 m6_ab.py --layout 的 harness 内
monkey-patch;腿 4 逐项计时用逐语句镜像 graph_runner.decode_batch 的
harness 内插桩函数(对齐 graph_runner.py @ c6b311b,仅插 perf_counter)。
输出行前缀:INSITU / CHECK / TOPKERN / DMON / KNOB / LAYOUT / OHEAD / PYSPY
(stdout 逐字进 docs;只写实测,异常数据是尖刀的目的,不修到好看)。

腿 1 口径:torch.profiler 包 decode loop(C=1/C=8 × graph/eager,各 steps 步,
协议同 scripts/profile_decode.py:该 seed 的 C=8 有 1 条序列在稳定期 EOS 退役,
计时窗内 batch 恒 7 / graph bucket 8 + 1 padding 行)。GPU kernel 事件经
chrome trace 解析(实例级 ts 序):每层 GEMM 发射序固定 q,k,v,o,gate,up,down
—— GEMM 类 kernel 主实例子序列的第 i 个(i<252)按 i%7 ∈{0,1,2}→qkv /
3→o / {4,5}→gate_up / 6→down 归类,第 253 个 = lm_head;splitKreduce 辅核
附着前一实例不占槽位。结构自证:主实例 count/step 必须 = 252+1、逐类有效
带宽必须落物理带;破例打印 INSITU_WARN,四类映射作废仅 gemm_all 有效(如实)。
冷口径参照:M5 段 0 补充二 full=15.64ms(M=7 四类投影;**不含 lm_head**,
故 lm_head 单独列行、对照线只比投影四类)。
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUTDIR = "/tmp/opencode/m6"
MODEL_DIR = "/home/ernest/model-weight/Qwen3-8B"

# ---- 腿 1:kernel 名 pattern(本栈 probe 实测:GEMM = cutlass_80_wmma...gemm
# (M=7/8)/ gemvx(M=1)+ splitKreduce 辅核;triton 核 _paged_attn_*_kernel /
# _fused_add_rmsnorm_kernel / _rope_*_kernel)----
GEMM_PAT = ("gemm", "gemvx", "splitKreduce", "nvjet", "cutlass")
ATTN_PAT = ("paged_attn", "attention")
NORM_PAT = ("rmsnorm",)
ROPE_PAT = ("rope",)
COPY_PAT = ("copy", "CatArray")
CLASS_ORDER = ("qkv", "qkv", "qkv", "o", "gate_up", "gate_up", "down")
BYTES_PER_STEP = {"qkv": (4096 * 4096 + 2 * 4096 * 1024) * 2 * 36,
                  "o": 4096 * 4096 * 2 * 36,
                  "gate_up": 2 * 4096 * 12288 * 2 * 36,
                  "down": 12288 * 4096 * 2 * 36,
                  "lm_head": 4096 * 151936 * 2}
CALLS_PER_STEP = {"qkv": 3 * 36, "o": 36, "gate_up": 2 * 36, "down": 36,
                  "lm_head": 1}
COLD_FULL_MS = 15.64   # M5 段 0 补充二:四类投影冷口径(M=7,不含 lm_head)

DMON_LINES: list[str] = []


# ================= 腿 1:insitu =================

def _build_engine(model, mode):
    model.attention_backend = "triton"
    model.fused_ops = True
    # decode_graph 必须在 Engine 构造前置位(graph runner 构造期捕获各桶)
    model.decode_graph = mode == "graph"
    from acorninfer.engine.engine import Engine
    return Engine(model, None, num_blocks=1024, page_size=16, max_running=8)


def _prepare_window(engine, model, c, steps):
    """prefill → 3 步稳定 → sids(协议镜像 scripts/profile_decode.py)。"""
    from acorninfer.bench import _prompts
    from acorninfer.sampler import SamplingParams
    vocab = model.cfg.vocab_size
    prompts = _prompts(c, 128, vocab, 20261027)
    sids = [engine.submit(p, SamplingParams(temperature=0.0),
                          max_tokens=steps + 16) for p in prompts]
    engine.step()
    while any(len(engine.seqs[i].generated_ids) == 0 for i in sids):
        engine.step()
    for _ in range(3):
        engine.step()
    alive = {i for i in sids if not engine.seqs[i].is_finished}
    if not alive:
        raise RuntimeError(f"C={c} 稳定期后无 alive 序列")
    return sids, alive


def _classify_gemm(kernels, steps):
    """GEMM kernel 实例按发射序映射四类 + lm_head(机制见模块 docstring)。"""
    gemm = [(ts, dur, name) for (name, ts, dur) in kernels
            if any(p in name for p in GEMM_PAT)]
    gemm.sort(key=lambda t: t[0])
    mains = [g for g in gemm if "splitKreduce" not in g[2]]
    per_step = len(mains) / steps
    expected = 36 * 7 + 1
    ok = abs(per_step - expected) < 0.01
    classes = {k: [] for k in ("qkv", "o", "gate_up", "down", "lm_head")}
    if ok:
        # 每步恰 252 投影 + 1 lm_head:投影相位每步归零(lm_head 后 reset),
        # 否则 253 ≢ 0 (mod 7) 相位跨步漂移(首跑实录 bug)
        proj_idx, last = 0, None
        for ts, dur, name in gemm:
            if "splitKreduce" in name:
                if last is not None:      # 辅核附着前一实例(同一次 mm 的收尾)
                    classes[last].append(dur)
                continue
            if proj_idx >= 36 * 7:
                classes["lm_head"].append(dur)
                last, proj_idx = "lm_head", 0
            else:
                last = CLASS_ORDER[proj_idx % 7]
                classes[last].append(dur)
                proj_idx += 1
    return classes, ok, per_step, expected, len(gemm) - len(mains)


def _report_insitu(trace, c, mode, n, wall, win_start, win_end) -> None:
    tr = json.load(open(trace))
    kernels, memcpies = [], []
    for e in tr["traceEvents"]:
        if e.get("cat") == "kernel":
            kernels.append((e["name"], e["ts"], float(e.get("dur", 0))))
        elif e.get("cat") == "gpu_memcpy":
            memcpies.append((e["name"], float(e.get("dur", 0))))

    buckets = {"attention": [], "norm": [], "rope": [], "copy": [], "other": []}
    for name, ts, dur in kernels:
        if any(p in name for p in GEMM_PAT):
            continue
        if any(p in name for p in ATTN_PAT):
            buckets["attention"].append(dur)
        elif any(p in name for p in NORM_PAT):
            buckets["norm"].append(dur)
        elif any(p in name for p in ROPE_PAT):
            buckets["rope"].append(dur)
        elif any(p in name for p in COPY_PAT):
            buckets["copy"].append(dur)
        else:
            buckets["other"].append(dur)
    memcpy_us = sum(d for _, d in memcpies)

    classes, ok, per_step, expected, n_helper = _classify_gemm(kernels, n)
    gemm_all_us = sum(d for name, _, d in kernels
                      if any(p in name for p in GEMM_PAT))
    gpu_total = sum(d for _, _, d in kernels) + memcpy_us

    print(f"INSITU C={c} mode={mode} wall_ms={wall / n * 1000:.2f} "
          f"gpu_ms={gpu_total / n / 1000:.2f} "
          f"gap_ms={(wall - gpu_total / 1e6) / n * 1000:.2f} "
          f"gemm_all_us_per_step={gemm_all_us / n:.1f} "
          f"cold_ref_ms={COLD_FULL_MS}(M=7 四类投影,不含 lm_head)",
          flush=True)
    if ok:
        proj_us = sum(sum(v) for k, v in classes.items() if k != "lm_head") / n
        lm_us = sum(classes["lm_head"]) / n
        print(f"INSITU C={c} mode={mode} class=gemm_proj us_per_step={proj_us:.1f} "
              f"proj_minus_cold_ms={(proj_us - COLD_FULL_MS * 1000) / 1000:.2f}",
              flush=True)
        print(f"INSITU C={c} mode={mode} class=lm_head us_per_step={lm_us:.1f} "
              f"(冷口径不含此类,单独列)", flush=True)
        for cls in ("qkv", "o", "gate_up", "down", "lm_head"):
            us = sum(classes[cls]) / n
            gb = BYTES_PER_STEP[cls] / (us * 1e-6) / 1e9
            print(f"INSITU C={c} mode={mode} class={cls} us_per_step={us:.1f} "
                  f"n_per_step={CALLS_PER_STEP[cls]} check_gb={gb:.0f}",
                  flush=True)
    else:
        print(f"INSITU_WARN C={c} mode={mode} gemm 主实例 {per_step:.2f}/步 != "
              f"期望 {expected}/步(splitKreduce 辅核 {n_helper / n:.2f}/步)"
              f"→ 四类映射不可证,gemm_all 聚合仍有效", flush=True)
    for cls in ("attention", "norm", "rope", "copy", "other"):
        print(f"INSITU C={c} mode={mode} class={cls} "
              f"us_per_step={sum(buckets[cls]) / n:.1f}", flush=True)
    print(f"INSITU C={c} mode={mode} class=memcpy us_per_step={memcpy_us / n:.1f} "
          f"count_per_step={len(memcpies) / n:.2f}", flush=True)
    proj_bytes = 13.9e9   # 36 层四类权重 ≈ 13.9GB/步(设计文档口径)
    print(f"CHECK C={c} mode={mode} proj_gb=13.9GB/步 "
          f"eff_gb={(proj_bytes / (gemm_all_us / n * 1e-6)) / 1e9:.0f}"
          f"(按 gemm_all 分母,含 lm_head 略偏高);cold 对照 889GB/s",
          flush=True)

    byname: dict[str, list] = {}
    for name, ts, dur in kernels:
        byname.setdefault(name, [0, 0.0])
        byname[name][0] += 1
        byname[name][1] += dur
    top = sorted(byname.items(), key=lambda kv: -kv[1][1])[:15]
    for name, (cnt, dur) in top:
        print(f"TOPKERN C={c} mode={mode} n_per_step={cnt / n:.2f} "
              f"us_per_step={dur / n:.1f} name={name[:90]}", flush=True)

    rows = []
    for line in DMON_LINES:
        parts = line.split()
        if len(parts) < 14 or not parts[13].isdigit():
            continue
        try:
            # dmon TD 实测格式:`20261101 11:30:12 1 33 34 - 0 0 ... 405 210`
            # = Date(YYYYMMDD) Time gpu pwr gtemp mtemp sm% mem% enc% dec%
            # jpg% ofa% mclk pclk → 降频看 pclk(parts[13]),功耗 parts[3]
            ts = datetime.datetime.strptime(parts[0] + " " + parts[1],
                                            "%Y%m%d %H:%M:%S")
            pclk, pwr = int(parts[13]), int(parts[3])
        except (IndexError, ValueError):
            continue
        # 窗口 ±2s 放宽(20 步窗 ~0.4s,1s 采样难命中)
        if win_start <= ts + datetime.timedelta(seconds=2) and \
                ts <= win_end + datetime.timedelta(seconds=2):
            rows.append((pclk, pwr))
    if rows:
        pc = [r[0] for r in rows]
        pwr = [r[1] for r in rows]
        print(f"DMON C={c} mode={mode} win={win_start:%H:%M:%S}-"
              f"{win_end:%H:%M:%S}±2s samples={len(rows)} "
              f"pclk_mhz_min={min(pc)} max={max(pc)} mean={sum(pc) / len(pc):.0f} "
              f"pwr_w_min={min(pwr)} max={max(pwr)}", flush=True)
    else:
        print(f"DMON C={c} mode={mode} win={win_start:%H:%M:%S}-"
              f"{win_end:%H:%M:%S} samples=0 (无覆盖样本;总窗摘要见结尾)",
              flush=True)


def insitu_main(args) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile
    os.makedirs(OUTDIR, exist_ok=True)
    from acorninfer.models.qwen3 import from_pretrained
    model = from_pretrained(MODEL_DIR, "cuda:0", torch.bfloat16)
    n = args.steps
    for mode in ("graph", "eager"):
        engine = _build_engine(model, mode)
        try:
            for c in (1, 8):
                win_start = datetime.datetime.now()
                sids, alive = _prepare_window(engine, model, c, n)
                # CUPTI 一次性开销落独立预热窗(同 profile_decode.py)
                with profile(activities=[ProfilerActivity.CPU,
                                         ProfilerActivity.CUDA]):
                    engine.step()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU,
                                         ProfilerActivity.CUDA]) as prof:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(n):
                        engine.step()
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                if {i for i in sids
                        if not engine.seqs[i].is_finished} != alive:
                    raise RuntimeError("计时窗内提前退役")
                trace = f"{OUTDIR}/insitu_c{c}_{mode}.json"
                prof.export_chrome_trace(trace)
                win_end = datetime.datetime.now()
                _report_insitu(trace, c, mode, n, wall, win_start, win_end)
                while engine.has_pending():
                    engine.step()
                for sid in sids:
                    engine.drop(sid)
                engine.assert_balanced()
        finally:
            # 内层 finally:每模式拆除引擎(KV pool/graph 静态缓冲),
            # 下一模式重建;首跑实录:外层再 del 同名变量会 UnboundLocalError
            del engine
            torch.cuda.empty_cache()


# ================= 腿 2:knobs =================

KNOB_WS = (("def", None), ("64M", "65536"), ("256M", "262144"))     # KiB
KNOB_RED = (("T", None), ("F", "false"))   # torch 缺省 allow_bf16...=True
KNOB_CFG = (("def", None), ("4096x8", ":4096:8"))


def _parse_ab(stdout: str):
    return re.findall(
        r"AB (.+) C=(\d+) mode=(\w+) step_ms=([\d.]+) tpot_ms=([\d.]+)", stdout)


def _parse_cold(stdout: str):
    m = re.search(r"per-step total \(36 layers\): full=([\d.]+)ms "
                  r"half=([\d.]+)ms", stdout)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def _combo_range(spec, total):
    if not spec or spec == "all":
        return 0, total
    if "-" in spec:
        a, b = spec.split("-")
        return int(a), min(int(b) + 1, total)
    return int(spec), int(spec) + 1


def knobs_main(args) -> None:
    combos = list(itertools.product(KNOB_WS, KNOB_RED, KNOB_CFG))
    lo, hi = _combo_range(args.combos, len(combos))
    for idx in range(lo, hi):
        (ws_l, ws_env), (red_l, red_arg), (cfg_l, cfg_env) = combos[idx]
        label = f"ws={ws_l},red={red_l},cfg={cfg_l}"
        env = os.environ.copy()
        for k, v in (("CUBLASLT_WORKSPACE_SIZE", ws_env),
                     ("CUBLAS_WORKSPACE_CONFIG", cfg_env)):
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        cmd = [PY, f"{REPO}/scripts/m6_ab.py", "--label", label]
        if red_arg:
            cmd += ["--bf16-reduction", red_arg]
        print(f"KNOB_RUN [{idx}] {label} "
              f"CUBLASLT_WORKSPACE_SIZE={ws_env or '-'} "
              f"CUBLAS_WORKSPACE_CONFIG={cfg_env or '-'} "
              f"bf16_reduced_precision_reduction={red_arg or 'default(True)'}",
              flush=True)
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=1800, cwd=REPO)
        except subprocess.TimeoutExpired:
            print(f"KNOB_FAIL [{idx}] {label} timeout", flush=True)
            continue
        if r.returncode != 0:
            print(f"KNOB_FAIL [{idx}] {label} rc={r.returncode} "
                  f"stderr_tail={r.stderr[-400:]!r}", flush=True)
            continue
        sys.stdout.write(r.stdout)
        for lbl, c_, mode, step, tpot in _parse_ab(r.stdout):
            print(f"KNOB {lbl} C={c_}_{mode}_step_ms {step}", flush=True)
            print(f"KNOB {lbl} C={c_}_{mode}_tpot_ms {tpot}", flush=True)
        # 孤立微基准口径:同 env 复跑 gemm-cold(冷权重 36 套轮转,env 先于
        # CUDA 初始化生效 —— 子进程隔离满足)
        r2 = subprocess.run(
            [PY, f"{REPO}/scripts/phase0_spike.py", "--gemm-cold"],
            env=env, capture_output=True, text=True, timeout=1800, cwd=REPO)
        if r2.returncode != 0:
            print(f"KNOB_FAIL [{idx}] {label} gemm-cold rc={r2.returncode} "
                  f"stderr_tail={r2.stderr[-400:]!r}", flush=True)
            continue
        full, half = _parse_cold(r2.stdout)
        if full is not None:
            print(f"KNOB {label} gemm_cold_full_perstep_ms {full:.2f}",
                  flush=True)
            print(f"KNOB {label} gemm_cold_half_perstep_ms {half:.2f}",
                  flush=True)


# ================= 腿 3:layout =================

LAYOUT_CLASSES = [("qkv", 6144, 4096), ("o", 4096, 4096),
                  ("gate_up", 24576, 4096), ("down", 4096, 12288)]
LAYOUT_LAYERS = 36
LAYOUT_REPS = 30


def layout_main(args) -> None:
    if args.engine_retest:
        cmd = [PY, f"{REPO}/scripts/m6_ab.py",
               "--label", f"layout_{args.engine_retest}",
               "--layout", args.engine_retest]
        print(f"LAYOUT_ENGINE_RETEST {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           cwd=REPO)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-2000:])
        return
    import torch
    import torch.nn.functional as F
    dev = "cuda:0"
    torch.manual_seed(0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def bench_rot(call, ws):
        for w in ws:
            call(w)
        torch.cuda.synchronize()
        start.record()
        for _ in range(LAYOUT_REPS):
            for w in ws:
                call(w)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / (LAYOUT_REPS * len(ws))

    best: dict = {}
    for m in (7, 8):
        for name, n_dim, k_dim in LAYOUT_CLASSES:
            for variant in ("tn", "nn", "pad8", "pad16"):
                p = {"pad8": 8, "pad16": 16}.get(variant, m)
                wshape, call = None, None
                if variant == "nn":
                    ws = [torch.randn(n_dim, k_dim, dtype=torch.bfloat16,
                                      device=dev).t().contiguous()
                          for _ in range(LAYOUT_LAYERS)]        # [K, N]
                    x = torch.randn(m, k_dim, dtype=torch.bfloat16, device=dev)
                    call = lambda w: torch.mm(x, w)             # noqa: E731
                elif variant in ("pad8", "pad16"):
                    ws = [torch.randn(n_dim, k_dim, dtype=torch.bfloat16,
                                      device=dev)
                          for _ in range(LAYOUT_LAYERS)]        # [N, K]
                    x = torch.zeros(p, k_dim, dtype=torch.bfloat16, device=dev)
                    x[:m] = torch.randn(m, k_dim, dtype=torch.bfloat16,
                                        device=dev)
                    # 切片 [:m] 是 view 不产 kernel,不计时(照 gemm-variants 先例)
                    call = lambda w: F.linear(x, w)[:m]         # noqa: E731
                else:
                    ws = [torch.randn(n_dim, k_dim, dtype=torch.bfloat16,
                                      device=dev)
                          for _ in range(LAYOUT_LAYERS)]        # [N, K] TN 现行
                    x = torch.randn(m, k_dim, dtype=torch.bfloat16, device=dev)
                    call = lambda w: F.linear(x, w)             # noqa: E731
                us = bench_rot(call, ws)
                gb = n_dim * k_dim * 2 / (us * 1e-6) / 1e9
                cur = best.get((m, name))
                if cur is None or us < cur[1]:
                    best[(m, name)] = (variant, us)
                print(f"LAYOUT m={m} {variant} {name} us={us:.2f} gb={gb:.0f}",
                      flush=True)
                del ws, x, wshape, call
                torch.cuda.empty_cache()
    for (m, name), (variant, us) in sorted(best.items()):
        print(f"LAYOUT_BEST m={m} {name} {variant} us={us:.2f}", flush=True)


# ================= 腿 4:overhead =================

def overhead_main(args) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from acorninfer.engine.graph_runner import BUCKETS, DecodeGraphRunner
    from acorninfer.models.qwen3 import from_pretrained
    from acorninfer.ops.paged_attention import build_tables

    n = args.steps
    # EOS 实录(seed=20261027 探针,sid 生成步位):sid7 stop@2(稳定期退役)、
    # sid5 stop@55、sid4 stop@74。窗口总步数(1 prefill + 3 稳定 + 5 预热 +
    # n 计时 + 5 profiler)须 ≤ 54:n=35 → 末端生成 49,对 55 留 6 步余量;
    # 首跑 n=50 → 末端 59 越过 55,sid5 窗内退役 → RuntimeError(如实记录)
    n_prof = 5
    model = from_pretrained(MODEL_DIR, "cuda:0", torch.bfloat16)
    model.attention_backend = "triton"
    model.fused_ops = True
    model.decode_graph = True
    engine = _build_engine(model, "graph")
    T: dict[str, float] = {}
    CNT: dict[str, int] = {}

    def timed(key, fn, *a, **kw):
        t0 = time.perf_counter_ns()
        r = fn(*a, **kw)
        T[key] = T.get(key, 0.0) + (time.perf_counter_ns() - t0) / 1000.0
        CNT[key] = CNT.get(key, 0) + 1
        return r

    # ---- harness 内插桩:逐语句镜像 graph_runner.py decode_batch @ c6b311b,
    # 仅插 perf_counter / cuda-event(分段名后注源行号);语义零改动 ----
    def instrumented(self, seqs, views):
        B = len(views)
        assert 1 <= B <= BUCKETS[-1]
        assert len(seqs) == B, "seqs/views 必须一一对应"
        st = self.states[min(b for b in BUCKETS if b >= B)]
        bk = st.bucket
        targets = timed("db_ensure(:146)",
                        lambda: [v.ensure_append_capacity(0) for v in views])
        pre = [int(v.seq_len) for v in views]
        new_len = [p + 1 for p in pre]

        def _mirror():   # :150-157(pinned 镜像填充,torch.tensor ×5)
            st.pin_long[0:B] = torch.tensor(
                [s.generated_ids[-1] for s in seqs], dtype=torch.long)
            st.pin_long[bk:bk + B] = torch.tensor(pre, dtype=torch.long)
            st.pin_long[2 * bk:2 * bk + B] = torch.tensor(
                [p for p, _ in targets], dtype=torch.long)
            st.pin_long[3 * bk:3 * bk + B] = torch.tensor(
                [s for _, s in targets], dtype=torch.long)
            st.pin_i32[0:B] = torch.tensor(new_len, dtype=torch.int32)
        timed("db_mirror_fill(:150-157)", _mirror)

        def _tables():   # :158-160
            st.pin_tables.zero_()
            build_tables(views, self.page_size, self.device,
                         lens_override=new_len, out=st.pin_tables)
        timed("db_tables_build(:158-160)", _tables)

        def _pad():      # :166-168(padding 常量场压回)
            st.pin_long[2 * bk + B:3 * bk].fill_(self.scratch)
            st.pin_long[3 * bk + B:4 * bk].zero_()
            st.pin_tables[B:, 0] = self.scratch
        timed("db_pad_restore(:166-168)", _pad)

        def _h2d():      # :170-171(每 dtype 一发 H2D)
            st.flat_long.copy_(st.pin_long)
            st.flat_i32.copy_(st.pin_i32)
        timed("db_h2d_2x(:170-171)", _h2d)
        timed("db_commit_lens(:172,36xB)",
              lambda: self._commit_lens(views, new_len))
        timed("db_replay_launch(:173)", lambda: st.graph.replay())
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        self.last_logits = st.logits                        # :175
        t0 = time.perf_counter_ns()
        toks = [int(t) for t in st.logits[:B].argmax(dim=1).tolist()]  # :177
        T["db_d2h_sync(:177)"] = T.get("db_d2h_sync(:177)", 0.0) \
            + (time.perf_counter_ns() - t0) / 1000.0
        CNT["db_d2h_sync(:177)"] = CNT.get("db_d2h_sync(:177)", 0) + 1
        ev1.record()
        torch.cuda.synchronize()
        T["db_gpu_span(ev0->ev1)"] = T.get("db_gpu_span(ev0->ev1)", 0.0) \
            + ev0.elapsed_time(ev1) * 1000.0
        CNT["db_gpu_span(ev0->ev1)"] = CNT.get("db_gpu_span(ev0->ev1)", 0) + 1
        return toks

    runner_orig = DecodeGraphRunner.decode_batch
    DecodeGraphRunner.decode_batch = instrumented   # 类级替换(harness 内)
    sched_cls = type(engine.scheduler)
    sched_orig = sched_cls.schedule

    def sched_timed(self):
        return timed("sched_schedule", sched_orig, self)
    sched_cls.schedule = sched_timed
    engine_cls = type(engine)
    rec_orig = engine_cls._record

    def rec_timed(self, seq, token_id, outputs):
        return timed("eng_record(bookkeeping)", rec_orig, self,
                     seq, token_id, outputs)
    engine_cls._record = rec_timed

    from acorninfer.bench import _prompts
    from acorninfer.sampler import SamplingParams
    vocab = model.cfg.vocab_size
    prompts = _prompts(8, 128, vocab, 20261027)
    sids = [engine.submit(p, SamplingParams(temperature=0.0),
                          max_tokens=n + 25) for p in prompts]
    engine.step()
    while any(len(engine.seqs[i].generated_ids) == 0 for i in sids):
        engine.step()
    for _ in range(3):
        engine.step()
    alive = {i for i in sids if not engine.seqs[i].is_finished}
    b_real = len(alive)
    for _ in range(5):
        engine.step()          # 预热(计时器随 T.clear() 丢弃)
    T.clear()
    CNT.clear()
    t0 = time.perf_counter()
    for _ in range(n):
        engine.step()
    wall = time.perf_counter() - t0
    if {i for i in sids if not engine.seqs[i].is_finished} != alive:
        raise RuntimeError("计时窗内提前退役")

    seg_sum = sum(v for k, v in T.items() if not k.startswith("db_gpu_span"))
    print(f"OHEAD step_wall_total us_per_step={wall / n * 1e6:.1f} "
          f"evidence=perf_counter,c=8,graph,real_B={b_real},steps={n},"
          f"seed=20261027", flush=True)
    for key in ("sched_schedule", "db_ensure(:146)", "db_mirror_fill(:150-157)",
                "db_tables_build(:158-160)", "db_pad_restore(:166-168)",
                "db_h2d_2x(:170-171)", "db_commit_lens(:172,36xB)",
                "db_replay_launch(:173)", "db_d2h_sync(:177)",
                "db_gpu_span(ev0->ev1)", "eng_record(bookkeeping)"):
        print(f"OHEAD {key} us_per_step={T.get(key, 0.0) / n:.1f} "
              f"evidence=wrapper,c=8,graph,bucket=8,real_B={b_real},steps={n}",
              flush=True)
    print(f"OHEAD residual_other us_per_step={wall / n * 1e6 - seg_sum / n:.1f} "
          f"evidence=step_wall−Σ段(engine.step 内 views 查表/num_pages/finally "
          f"退休循环/executor 分派)", flush=True)

    # profiler 口径交叉核(独立 10 步窗;插桩仍在,分段开销已由 wrapper 数字本身呈现)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        engine.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_prof):
            engine.step()
        torch.cuda.synchronize()
        wall2 = time.perf_counter() - t0
    evs = prof.key_averages()

    def _is_gpu(e):
        return str(e.device_type).endswith("CUDA")

    gpu_us = sum(e.self_device_time_total for e in evs if _is_gpu(e))
    print(f"OHEAD profiler_wall us_per_step={wall2 / n_prof * 1e6:.1f} "
          f"evidence=torch.profiler,{n_prof} 步独立窗(插桩仍在)", flush=True)
    print(f"OHEAD profiler_gpu us_per_step={gpu_us / n_prof:.1f} "
          f"evidence=self_device_time_total 只按 DeviceType.CUDA 事件累加"
          f"(profile_decode.py 注记,防 CPU 算子双计)", flush=True)
    print(f"OHEAD profiler_gap us_per_step={wall2 / n_prof * 1e6 - gpu_us / n_prof:.1f} "
          f"evidence=wall−GPU(profiler 口径)", flush=True)
    for key in ("cudaGraphLaunch", "cudaLaunchKernel", "cudaMemcpyAsync",
                "cudaStreamSynchronize", "cudaDeviceSynchronize"):
        hits = [e for e in evs if e.key == key]
        if hits:
            e = hits[0]
            print(f"OHEAD cpu_runtime:{key} us_per_step={e.self_cpu_time_total / n_prof:.1f} "
                  f"count_per_step={e.count / n_prof:.2f} "
                  f"evidence=profiler-cpu-events(cuda_runtime)", flush=True)
    for e in evs:
        if _is_gpu(e) and e.key.startswith("Memcpy"):
            print(f"OHEAD gpu_memcpy:{e.key} us_per_step={e.self_device_time_total / n_prof:.1f} "
                  f"count_per_step={e.count / n_prof:.2f} "
                  f"evidence=profiler-device-events", flush=True)

    # 清理 + 场收尾;py-spy 目标进程需整卡装载 8B,父进程必须先释放 GPU
    # (首跑实录:父进程持模型+池 ~20GB 不放,目标装载 OOM,py-spy rc=1)
    sched_cls.schedule = sched_orig
    engine_cls._record = rec_orig
    DecodeGraphRunner.decode_batch = runner_orig
    while engine.has_pending():
        engine.step()
    for sid in sids:
        engine.drop(sid)
    engine.assert_balanced()
    del engine
    model.to("cpu")
    torch.cuda.empty_cache()

    # py-spy 第三口径:子进程跑 m6_ab.py(graph 腿长窗),父进程 record。
    # 前两次实录(spawn 模式 --):rc=1 ECHILD「No child process」,且
    # py-spy spawn 吞掉子进程 stderr,子进程死因(OOM?)无从取证;
    # 平凡子进程 spawn 测试 rc=0 → 机制未坏,嫌疑在目标子进程早夭。
    # 改为父进程自起目标(可见 stderr 落盘)+ 就绪轮询 + attach(-p PID);
    # ptrace_scope=0 允许同 uid attach(2026-11-01 实查)。
    pyspy_bin = "/home/userlist/zxs/python_venv/bin/py-spy"
    raw = f"{OUTDIR}/pyspy_overhead.raw"
    os.makedirs(OUTDIR, exist_ok=True)

    def _gpu1_mi() -> int:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "1"],
            capture_output=True, text=True)
        try:
            return int(q.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            return -1

    print(f"PYSPY_PRE parent_teardown gpu1_used_mi={_gpu1_mi()} "
          f"evidence=nvidia-smi(装 target 前父进程余留)", flush=True)
    tgt_log = open(f"{OUTDIR}/pyspy_target.stderr.log", "w")
    child = subprocess.Popen([PY, f"{REPO}/scripts/m6_ab.py",
                              "--label", "pyspy_target", "--steps", "30"],
                             cwd=REPO, stdout=tgt_log, stderr=tgt_log)
    gpu_mi = -1
    for _ in range(300):
        time.sleep(1)
        gpu_mi = _gpu1_mi()
        if gpu_mi > 15000:      # 模型+池装载完成即视为就绪
            break
    print(f"PYSPY_PRE target_pid={child.pid} gpu1_used_mi={gpu_mi} "
          f"evidence=nvidia-smi 1s 轮询(>15000 就绪,上限 300s)", flush=True)
    r = None
    if gpu_mi > 15000:
        cmd = [pyspy_bin, "record", "--format", "raw", "-d", "120", "-r", "200",
               "-o", raw, "-p", str(child.pid)]
        print(f"PYSPY_RUN {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(f"PYSPY rc={r.returncode} stdout_tail={r.stdout[-200:]!r}", flush=True)
        if r.returncode != 0:
            print(f"PYSPY_FAIL rc={r.returncode} stderr_tail={r.stderr[-500:]!r}",
                  flush=True)
    else:
        print("PYSPY_SKIP target 300s 未就绪( attach 放弃)", flush=True)
    try:
        child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    tgt_log.close()
    with open(f"{OUTDIR}/pyspy_target.stderr.log") as f:
        tgt_tail = f.read()[-500:]
    print(f"PYSPY_TARGET rc={child.returncode} stderr_tail={tgt_tail!r}", flush=True)
    if r is not None and r.returncode == 0:
        step_samp = db_samp = tot_samp = 0
        frames: dict[str, int] = {}
        with open(raw) as f:
            for line in f:
                try:
                    stack, cnt = line.rsplit(" ", 1)
                except ValueError:
                    continue
                cnt = int(cnt)
                tot_samp += cnt
                # py-spy raw 帧格式为「func (path:line)」,帧间以 ';' 连接;
                # 插桩副本代码对象名为 instrumented(m6_spike.py 内定义)
                fl = [fr.split(" (")[0] for fr in stack.split(";")]
                if "step" in fl:
                    step_samp += cnt
                if "decode_batch" in fl or "instrumented" in fl:
                    db_samp += cnt
                    for fn in ("_commit_lens", "ensure_append_capacity",
                               "build_tables", "schedule", "timed"):
                        if fn in fl:
                            frames[fn] = frames.get(fn, 0) + cnt
        print(f"PYSPY denom_engine_step_samples={step_samp} "
              f"decode_batch_samples={db_samp} total_samples={tot_samp} "
              f"eff_hz={tot_samp / 120:.1f} "
              f"evidence=collapsed-stacks,attach=pid,req_rate=200Hz,120s"
              f"(m6_ab 各档 warmup10+计30 长窗 ×4 组合;attach 采样有税,"
              f"目标 AB 偏慢,与 knobs 同档对照为准)", flush=True)
        if db_samp:
            print(f"PYSPY decode_batch pct_of_engine_step="
                  f"{db_samp / max(step_samp, 1) * 100:.2f}", flush=True)
            for fn, cnt in sorted(frames.items(), key=lambda kv: -kv[1]):
                print(f"PYSPY {fn} pct_of_decode_batch={cnt / db_samp * 100:.2f} "
                      f"samples={cnt}", flush=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("insitu")
    p1.add_argument("--steps", type=int, default=20)
    p2 = sub.add_parser("knobs")
    p2.add_argument("--combos", default="all", help="如 0-5 / 6-11 / all")
    p3 = sub.add_parser("layout")
    p3.add_argument("--engine-retest", default=None,
                    choices=["t", "pad8", "pad16"])
    p4 = sub.add_parser("overhead")
    p4.add_argument("--steps", type=int, default=35)
    args = ap.parse_args()

    if args.cmd == "insitu":
        proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "puc", "-i", "1", "-d", "1", "-o", "TD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        import threading

        def _pump():
            for line in proc.stdout:
                DMON_LINES.append(line.rstrip("\n"))
        th = threading.Thread(target=_pump, daemon=True)
        th.start()
        try:
            insitu_main(args)
        finally:
            proc.terminate()
            th.join(timeout=10)
            for line in DMON_LINES[:8]:
                print(f"DMON_RAW {line}", flush=True)
            with open(f"{OUTDIR}/dmon_insitu.log", "w") as f:
                f.write("\n".join(DMON_LINES) + "\n")
            print(f"DMON_RAW ... 全文 {len(DMON_LINES)} 行存 "
                  f"{OUTDIR}/dmon_insitu.log", flush=True)
            pclk = [int(l.split()[13]) for l in DMON_LINES
                    if len(l.split()) > 13 and l.split()[13].isdigit()]
            if pclk:
                print(f"DMON overall pclk_mhz min={min(pclk)} max={max(pclk)} "
                      f"mean={sum(pclk) / len(pclk):.0f} samples={len(pclk)} "
                      f"(全程,含装载/构建;4090 boost ~2520MHz,pclk 显著 "
                      f"< 2400 才是降频证据)", flush=True)
        return
    if args.cmd == "knobs":
        knobs_main(args)
        return
    if args.cmd == "layout":
        layout_main(args)
        return
    if args.cmd == "overhead":
        overhead_main(args)
        return


if __name__ == "__main__":
    main()
