"""M6 A/B 计时协议(段 1 各项的度量衡):decode loop 8B,C=1/C=8 × graph/eager 各一遍。

用法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_ab.py --label <名字>
输出行:AB <label> C=<c> mode=<graph|eager> step_ms=<v> tpot_ms=<v>
  - step_ms = 计时窗(warmup 10 + 计 30,perf_counter)墙钟/步;
  - tpot_ms = bench 同口径((last_token_at - first_token_at)/(len-1) 对窗内
    alive 序列取均值;跨度含预热步,作 step_ms 的引擎记账交叉核,非主口径)。
同 seed(20261027)同输入;结构照 scripts/profile_decode.py 的引擎构造
(from_pretrained → backend 旗标前置 → Engine;decode_graph 必须构造前置位)。
该 seed 的 C=8 档有 1 条请求在第 2 个生成 token 命中 EOS、于 3 步稳定期退役,
计时窗内 batch 恒 7(graph 走 bucket 8 + 1 padding 行)——与 M4.1 锚账目
(wall 21.50 / GPU 19.81 / gap 1.70 ms,graph C=8)同协议可比。窗口内任何
提前退役 → RuntimeError 立即失败(测得的数据必须 batch 恒定)。

--layout <none|t|down_t|pad8|pad16>(缺省 none = 逐字节旧行为):harness 内
monkey-patch 投影模块(零仓库改动,仅脚本内),段 0 腿 3 的引擎内复测入口:
  t     权重装载后预变换 .t().contiguous() 存 [K,N](原 [N,K] 存储随之
        释放,显存不翻倍)+ 替换 forward 为 torch.mm(x2d, w_t);
  down_t 同 t 的变换与 mm 前向,但仅打 down_proj(T0 裁决:全局 NN 出局
        于 C=1 +0.64 回归,down-only NN 是唯一胜者;输出行 RETEST down_t;
        2026-11-01 引擎内复测:C=8 graph −0.27~−0.34 ✓ 但 C=1 graph
        +0.06~+0.08 / C=1 eager +0.17~+0.51 ✗ —— 双 C 门未过,生产接线
        不落地,账目见 docs/benchmarks.md M6 段 1 布局账);
  pad8/pad16  权重不动,x2d 零填充行切片(F.linear(pad, W)[:M]);
  四类投影(q/k/v/o + gate/up/down)+ lm_head 全patch;embed/norm 不动。
    数值 = 同一数学的不同 cuBLAS 路径,bf16 下不保逐位(本脚本只计时;
    逐位等价门在 Task 1 落地时按 TOL/margin 协议另行钉)。
每模式独立构造 Engine(graph 先、eager 后);模式间 del engine + empty_cache。
"""

from __future__ import annotations

import argparse
import time
import types

import torch

from acorninfer.bench import DEFAULT_MODEL, _prompts
from acorninfer.engine.engine import Engine
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.sampler import SamplingParams

SEED = 20261027


def apply_layout_patch(model, layout: str) -> None:
    """投影模块 harness 内 monkey-patch(见模块 docstring;零仓库改动)。"""
    if layout == "none":
        return
    targets = []
    for layer in model.layers:
        sa, mlp = layer.self_attn, layer.mlp
        targets += [sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj,
                    mlp.gate_proj, mlp.up_proj, mlp.down_proj]
    targets.append(model.lm_head)
    if layout == "down_t":
        # T0 裁决形态:仅 down_proj 进 NN(m6_ab.py --layout t 的 down-only 腿)
        targets = [layer.mlp.down_proj for layer in model.layers]

    if layout in ("t", "down_t"):
        def make_forward(w_t, n):
            def forward(x):
                m = x.numel() // x.shape[-1]
                return torch.mm(x.reshape(m, x.shape[-1]), w_t).view(
                    *x.shape[:-1], n)
            return forward
        for mod in targets:
            w_t = mod.weight.data.t().contiguous()   # [K, N]
            mod.weight.data = w_t                    # 释放原 [N, K] 存储
            mod.forward = make_forward(w_t, w_t.shape[1])
    elif layout in ("pad8", "pad16"):
        p = int(layout[3:])

        def make_forward(w, p):
            def forward(x):
                k = x.shape[-1]
                x2 = x.reshape(-1, k)
                m = x2.shape[0]
                if m == p:
                    return torch.nn.functional.linear(x2, w).view(
                        *x.shape[:-1], w.shape[0])
                pad = torch.zeros(p, k, dtype=x.dtype, device=x.device)
                pad[:m] = x2
                return torch.nn.functional.linear(pad, w)[:m].view(
                    *x.shape[:-1], w.shape[0])
            return forward
        for mod in targets:
            mod.forward = make_forward(mod.weight.data, p)
    else:
        raise ValueError(
            f"layout must be none|t|down_t|pad8|pad16, got {layout!r}")


def prefill_account(engine: Engine, model, c: int, label: str, quant: str,
                    in_tokens: int = 128) -> None:
    """in-process prefill 计时(M7 Task 2 spec §6 R1 TTFT 账):C=c 一次 step
    全准入(max_running=8),128 in_tokens/条;首步预热(JIT/池首触)后计一步,
    sync 前后夹墙钟。输出行:ABPREFILL <label> C=<c> quant=<q> in_tokens=<n>
    prefill_ms=<v>(none 腿 = 基线,w8a16 腿照实入账;prefill 回退允许不粉饰)。"""
    prompts = _prompts(c, in_tokens, model.cfg.vocab_size, SEED)
    for _ in range(2):          # 第 1 轮预热,第 2 轮计量
        sids = [engine.submit(p, SamplingParams(temperature=0.0), max_tokens=1)
                for p in prompts]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step()           # 全准入 prefill 步
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        for sid in sids:
            engine.drop(sid)
    print(f"ABPREFILL {label} C={c} quant={quant} in_tokens={in_tokens} "
          f"prefill_ms={ms:.2f}", flush=True)
    engine.assert_balanced()


def run_combo(engine: Engine, model, c: int, warmup: int, steps: int,
              label: str, mode: str, prefix: str = "AB") -> None:
    """C=c 档:prefill → 3 步稳定 → warmup → 计 30 步 → AB 行。"""
    vocab = model.cfg.vocab_size
    prompts = _prompts(c, 128, vocab, SEED)
    # max_tokens 留足余量:计时窗口内不得有序列因 length 提前退役
    sids = [
        engine.submit(p, SamplingParams(temperature=0.0),
                      max_tokens=warmup + steps + 16)
        for p in prompts
    ]
    engine.step()  # prefill 步:max_running=8 一次全准入,各产首 token
    while any(len(engine.seqs[i].generated_ids) == 0 for i in sids):
        engine.step()
    for _ in range(3):        # 稳定期:该 seed 的 EOS 命中者在此退役
        engine.step()
    alive_before = {i for i in sids if not engine.seqs[i].is_finished}
    if not alive_before:
        raise RuntimeError(f"C={c} 稳定期后无 alive 序列,计时窗无意义")
    for _ in range(warmup):
        engine.step()
    if {i for i in sids if not engine.seqs[i].is_finished} != alive_before:
        raise RuntimeError("预热窗内有序列提前退役")
    t0 = time.perf_counter()
    for _ in range(steps):
        engine.step()
    wall = time.perf_counter() - t0
    if {i for i in sids if not engine.seqs[i].is_finished} != alive_before:
        raise RuntimeError("计时窗内有序列提前退役(账目失真)")
    # bench 同口径 TPOT(含预热步跨度):交叉核,非主口径
    tpots = [
        (engine.seqs[i].last_token_at - engine.seqs[i].first_token_at)
        / (len(engine.seqs[i].generated_ids) - 1)
        for i in alive_before
        if len(engine.seqs[i].generated_ids) > 1
    ]
    tpot_ms = sum(tpots) / len(tpots) * 1000
    print(f"{prefix} {label} C={c} mode={mode} step_ms={wall / steps * 1000:.2f} "
          f"tpot_ms={tpot_ms:.2f}", flush=True)
    # 收尾:驱动到全部完成(按 length),drop 清场;下一档零残留
    while engine.has_pending():
        engine.step()
    for sid in sids:
        engine.drop(sid)
    engine.assert_balanced()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--layout", default="none",
                    choices=["none", "t", "down_t", "pad8", "pad16"])
    ap.add_argument("--quant", default="none", choices=["none", "w8a16"],
                    help="M7 Task 2:生产同构量化腿(装载后 apply_quant + "
                         "quant_gemm.configure,与 engine/server 接线同码)")
    ap.add_argument("--bf16-reduction", default=None, choices=["true", "false"],
                    help="allow_bf16_reduced_precision_reduction;缺省 = torch 默认")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.bf16_reduction is not None:
        # 旋钮腿:Python 侧旋钮在首个 GEMM 前置位即可;env 类旋钮由 m6_spike
        # knobs 经子进程 env 注入(CUDA 初始化前生效)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            args.bf16_reduction == "true")

    print(f"loading {DEFAULT_MODEL} -> {args.device} (bf16) "
          f"label={args.label} layout={args.layout} quant={args.quant} "
          f"bf16_reduction={args.bf16_reduction or 'torch-default'} ...",
          flush=True)
    t_load = time.perf_counter()
    model = from_pretrained(DEFAULT_MODEL, args.device, torch.bfloat16)
    load_s = time.perf_counter() - t_load
    t_apply = time.perf_counter()
    apply_s = configure_s = 0.0
    if args.quant != "none":
        # 生产同构:与 engine.from_config / server._build_engine 接线同码
        from acorninfer.ops import quant_gemm
        from acorninfer.quant.quantize import apply_quant
        apply_quant(model, args.quant)
        apply_s = time.perf_counter() - t_apply
        t_cfg = time.perf_counter()
        quant_gemm.configure(device=args.device)
        configure_s = time.perf_counter() - t_cfg
    print(f"LOADTIME label={args.label} quant={args.quant} "
          f"load_s={load_s:.1f} apply_s={apply_s:.1f} configure_s={configure_s:.1f}",
          flush=True)
    model.attention_backend = "triton"
    model.fused_ops = True
    apply_layout_patch(model, args.layout)
    for mode in ("graph", "eager"):
        # decode_graph 必须在 Engine 构造前置位:graph runner 在 Engine.__init__
        # 捕获各桶,事后置位不重建 runner
        model.decode_graph = mode == "graph"
        engine = Engine(model, None, num_blocks=1024, page_size=16, max_running=8)
        try:
            # down_t 腿(T0 裁决复测)输出行换 RETEST down_t 前缀,便于 grep 归档
            prefix = "RETEST" if args.layout == "down_t" else "AB"
            shown = "down_t" if args.layout == "down_t" else args.label
            if mode == "eager":
                # prefill 账与 decode A/B 同进程同模型(prefill 无 graph 面,
                # 双 mode 数值同;取 eager 腿一次,免重复)
                for c in (1, 8):
                    prefill_account(engine, model, c, shown, args.quant)
            for c in (1, 8):
                run_combo(engine, model, c, args.warmup, args.steps,
                          shown, mode, prefix)
        finally:
            # 失败安全:del engine 先释放 KV pool / graph runner 引用,
            # empty_cache 回收池子;权重驻留 GPU,下个模式(eager)复用,
            # 进程退出时由驱动统一回收
            try:
                del engine
            except Exception:
                pass
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
