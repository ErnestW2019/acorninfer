"""M8 Task 3 计测脚本:8B A1 无损性硬门 + spec 计测(GPU 1,W8)。

腿定义(一次装载,共享模型;W8 = apply_quant("w8a16") + quant_gemm.configure,
与 server._build_engine 生产接线同码):
  ① A1 无损性:2 wiki 型 + 2 短 chat 型 prompt,greedy,spec on/off 双腿
     (双腿均 eager:spec × decode_graph 互斥;decode_graph ≡ eager 逐位
     已由 M3/M5 硬门)全量生成 token 流逐位 diff。任一 token 不一致 =
     FAIL(如实打印首个分歧位置,零粉饰)。
  ② 稳态步耗时:4×128 随机 prompt(bench._prompts 同源),prefill → 3 步
     稳定 → 计时 24 步:decode eager(C=4)vs spec(m=5,接受 ~0,
     每步 1 发射,窗口成本可比)。tok_s = C / mean_step_s。
  ③ 接受率账:② 的 spec 腿 spy verify_batch(drafts + argmax 链),
     greedy_accept 逐位重算:mean a / mean emitted / 多 token 步占比。
  ④ 草稿延迟:NgramProposer(512 位上下文)propose() ×2000,mean/p95/max µs
     (目标 <100µs)。

输出逐行可入账(docs 引 stdout 原样);退出码 0 = A1 PASS。
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from acorninfer.bench import DEFAULT_MODEL, _prompts
from acorninfer.engine.engine import Engine
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.quant.quantize import apply_quant
from acorninfer.sampler import SamplingParams
from acorninfer.spec.ngram import NgramProposer
from acorninfer.tokenizer import Tokenizer

WIKI1 = ("Qwen3 is a family of large language models developed by Alibaba Cloud. "
         "The architecture uses grouped query attention with rotary positional "
         "embeddings, root mean square layer normalization, and SwiGLU feed-forward "
         "blocks. Models range from 0.6 billion to 235 billion parameters.")
WIKI2 = ("光合作用是植物、藻类和蓝菌利用光能将二氧化碳和水转化为有机物并释放氧气的过程。"
         "它是地球上最重要的生化反应之一,为几乎所有生命形式提供能量与物质基础。")
CHAT1 = "请用三句话介绍你自己。"
CHAT2 = "Write a haiku about autumn rain."


def build_engine(model, *, spec: bool) -> Engine:
    # 两腿均 eager(与 spec 互斥约束对称);graph ≡ eager 已有独立硬门
    model.decode_graph = False
    return Engine(model, None, num_blocks=1024, page_size=16, max_running=8,
                  spec_ngram=2 if spec else 0, spec_num=4)


def run_streams(model, prompts, *, spec: bool, max_tokens: int):
    """A1 腿:全量生成,返回 {sid: [generated]},引擎级 spy 附带接受账。"""
    eng = build_engine(model, spec=spec)
    sids = [eng.submit(p, SamplingParams(temperature=0.0), max_tokens=max_tokens)
            for p in prompts]
    streams = {s: [] for s in sids}
    while eng.has_pending():
        for o in eng.step():
            streams[o.seq_id].append(o.token_id)
    for s in sids:
        eng.drop(s)
    return streams


def steady_window(model, *, spec: bool, warmup: int = 3, steps: int = 24):
    """② 稳态计时 + ③ 接受账(spec 腿)。4×128 随机 prompt,greedy。"""
    vocab = model.cfg.vocab_size
    prompts = _prompts(4, 128, vocab, 20261102)
    eng = build_engine(model, spec=spec)
    sids = [eng.submit(p, SamplingParams(temperature=0.0),
                       max_tokens=warmup + steps + 16) for p in prompts]

    # ③ spy:真前向后逐位重算接受(spec 腿)
    acc = {"a": [], "emit": [], "verify_calls": 0}
    if spec:
        real = eng.executor.verify_batch
        from acorninfer.spec.ngram import greedy_accept

        def spy(seqs, views, drafts):
            logits = real(seqs, views, drafts)
            acc["verify_calls"] += 1
            for i, (s, d) in enumerate(zip(seqs, drafts)):
                nxt = logits[i].argmax(dim=-1).tolist()
                a, bonus = greedy_accept(nxt, d)
                acc["a"].append(a)
                acc["emit"].append(len(nxt[:a] + [bonus]))
            return logits

        eng.executor.verify_batch = spy

    # prefill → 稳定 → 计时窗
    step_ms = []
    torch.cuda.synchronize()
    try:
        for t in range(warmup + steps):
            t0 = time.perf_counter()
            eng.step()
            torch.cuda.synchronize()
            if t >= warmup and not all(eng.seqs[s].is_finished for s in sids):
                step_ms.append((time.perf_counter() - t0) * 1000)
    finally:
        if spec:
            eng.executor.verify_batch = real
    alive = [s for s in sids if not eng.seqs[s].is_finished]
    mean_ms = statistics.mean(step_ms) if step_ms else float("nan")
    p50_ms = statistics.median(step_ms) if step_ms else float("nan")
    tok_s = 4 / (mean_ms / 1000) if step_ms else float("nan")
    print(f"STEADY leg={'spec' if spec else 'decode'} C=4 steps={len(step_ms)} "
          f"mean_step_ms={mean_ms:.2f} p50_step_ms={p50_ms:.2f} tok_s={tok_s:.1f}")
    if spec and acc["a"]:
        n = len(acc["a"])
        multi = sum(1 for e in acc["emit"] if e >= 2)
        print(f"ACCEPT verify_calls={acc['verify_calls']} steps_scored={n} "
              f"mean_a={statistics.mean(acc['a']):.3f} "
              f"mean_emitted={statistics.mean(acc['emit']):.3f} "
              f"multi_token_step_ratio={multi / n:.3f}")
    while eng.has_pending():
        eng.step()
    for s in sids:
        if s in eng.seqs:
            eng.drop(s)
    return mean_ms


def draft_latency():
    """④ 草稿延迟:512 位上下文 propose() ×2000(纯 CPU,O(1) 查表)。"""
    g = torch.Generator().manual_seed(7)
    ctx = torch.randint(0, 150000, (1, 512), generator=g)[0].tolist()
    p = NgramProposer(ctx, 2, 4)
    ts = []
    for _ in range(2000):
        t0 = time.perf_counter()
        p.propose()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    print(f"DRAFT_LATENCY n=512 iters=2000 mean_us={statistics.mean(ts):.1f} "
          f"p95_us={ts[int(0.95 * len(ts))]:.1f} max_us={ts[-1]:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default="w8a16", choices=["none", "w8a16"])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"loading {args.model_dir} -> {args.device} (bf16, quant={args.quant})")
    model = from_pretrained(args.model_dir, args.device, torch.bfloat16)
    model.attention_backend = "triton"
    model.fused_ops = True
    if args.quant != "none":
        apply_quant(model, args.quant)
        from acorninfer.ops import quant_gemm
        quant_gemm.configure(device=args.device)

    import os
    tk = Tokenizer(os.path.join(args.model_dir, "tokenizer.json"))
    prompts = [tk.encode(t) for t in (WIKI1, WIKI2, CHAT1, CHAT2)]
    kinds = ("wiki1", "wiki2", "chat1", "chat2")
    for k, p in zip(kinds, prompts):
        print(f"PROMPT {k}: {len(p)} tokens")

    # ① A1 无损性(硬门)
    off = run_streams(model, prompts, spec=False, max_tokens=args.max_tokens)
    on = run_streams(model, prompts, spec=True, max_tokens=args.max_tokens)
    ok = True
    for sid_off, sid_on, k in zip(off, on, kinds):
        same = off[sid_off] == on[sid_on]
        ok &= same
        if same:
            print(f"A1 {k}: IDENTICAL ({len(off[sid_off])} tokens, greedy, "
                  f"{args.quant} both legs)")
        else:
            a, b = off[sid_off], on[sid_on]
            i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            print(f"A1 {k}: *** DIVERGE at gen[{i}] off={a[i]} on={b[i]} "
                  f"(len off={len(a)} on={len(b)}) ***")
            ok = False
    print(f"A1_RESULT {'PASS' if ok else 'FAIL'} (bitwise spec on/off, "
          f"{args.quant}, {len(prompts)} prompts x {args.max_tokens} tokens)")

    # ② 稳态步耗时(decode eager vs spec m=5)
    steady_window(model, spec=False)
    steady_window(model, spec=True)

    # ④ 草稿延迟
    draft_latency()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
