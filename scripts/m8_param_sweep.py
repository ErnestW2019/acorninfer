"""M8 Task 4 参数扫 harness(GPU 1,in-process,W8 生产同构接线)。

口径(任务派单 + spec §1 R1):N∈{1,2,3} × K∈{2,4,6} × 双 workload
(wiki 续写语料 = make_spec_prompts.py 产物 / chat = bench 缺省随机语料)
C=1,每组合出(接受率 E[emitted/step], tok_s);选定缺省配置 = wiki tok_s
最高,并列取 K 小者(docs 记录全表,锚在选定配置上判,仪式不许换锚)。

每行 = 独立 Engine(spec 配置,decode_graph=False 与 spec 互斥同构):
  warmup seq(8-token prompt × out=8 ≥ m=K+1 → 强制 ≥1 verify 步,
  triton verify kernel 进程内 JIT 预热;eager 参照行同跑,不计入)→
  计测 seq(out=64 wiki / 32 chat,greedy)→ tok_s(bench 同款引擎记账
  口径:gen / (last_token_at − created_at))+ 引擎侧接受账(engine.
  spec_stats 步级增量:E_acc1 = accepted/verify_steps + 1;mean_committed
  = 实际入流 token/verify 步;hist = committed/步分布)。
首行 = spec off(eager W8 参照,同口径新鲜测,仅供扫表内相对读数;
  A2 分母以仪式 HTTP wiki 腿新鲜测为准)。

模型装载一次(W8 = apply_quant + quant_gemm.configure,生产接线同码);
engine 按行重建,行间 drop + assert_balanced 零残留。
用法:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m8_param_sweep.py \
        [--prompts-file data/spec_prompts_wiki.json] [--out-wiki 64 --out-chat 32]
"""

from __future__ import annotations

import argparse
import json

import torch

from acorninfer.bench import DEFAULT_MODEL, _prompts
from acorninfer.engine.engine import Engine
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.quant.quantize import apply_quant
from acorninfer.sampler import SamplingParams


def run_one(model, ids: list[int], out_tokens: int, *, n: int, k: int,
            workload: str) -> None:
    """单组合计测:独立 Engine → warmup(JIT)→ 计测 → 增量接受账。"""
    eng = Engine(model, None, num_blocks=1024, page_size=16, max_running=8,
                 spec_ngram=n, spec_num=k)
    # warmup seq:out=8 ≥ m=K+1(K≤6 → m≤7)强制 verify 腿,JIT 预热不计入
    w = eng.submit(ids[:8], SamplingParams(temperature=0.0), max_tokens=8)
    while eng.has_pending():
        eng.step()
    eng.drop(w)
    st0 = {key: dict(v) if isinstance(v, dict) else v
           for key, v in eng.spec_stats.items()}
    sid = eng.submit(ids, SamplingParams(temperature=0.0), max_tokens=out_tokens)
    while eng.has_pending():
        eng.step()
    seq = eng.seqs[sid]
    gen = len(seq.generated_ids)
    tok_s = gen / (seq.last_token_at - seq.created_at)
    tpot_ms = ((seq.last_token_at - seq.first_token_at) / (gen - 1)) * 1000 \
        if gen > 1 else float("nan")
    st = eng.spec_stats
    vs = st["verify_steps"] - st0["verify_steps"]
    acc = st["accepted"] - st0["accepted"]
    com = st["committed"] - st0["committed"]
    hist = {}
    for h in set(st0["hist"]) | set(st["hist"]):
        d = st["hist"].get(h, 0) - st0["hist"].get(h, 0)
        if d:
            hist[h] = d
    e_acc1 = f"{acc / vs + 1:.3f}" if vs else "n/a"
    mean_com = f"{com / vs:.3f}" if vs else "n/a"
    print(f"SWEEP workload={workload} N={n} K={k} gen={gen} "
          f"tok_s={tok_s:.2f} tpot_ms={tpot_ms:.2f} verify_steps={vs} "
          f"E_acc1={e_acc1} mean_committed={mean_com} "
          f"hist={dict(sorted(hist.items()))}", flush=True)
    eng.drop(sid)
    eng.assert_balanced()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompts-file", default="data/spec_prompts_wiki.json")
    ap.add_argument("--out-wiki", type=int, default=64)
    ap.add_argument("--out-chat", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()

    print(f"loading {args.model_dir} -> {args.device} (bf16, quant=w8a16)")
    model = from_pretrained(args.model_dir, args.device, torch.bfloat16)
    model.attention_backend = "triton"
    model.fused_ops = True
    model.decode_graph = False  # spec × decode_graph 互斥;eager 参照行同模式
    apply_quant(model, "w8a16")
    from acorninfer.ops import quant_gemm
    quant_gemm.configure(device=args.device)

    with open(args.prompts_file, encoding="utf-8") as f:
        wiki = json.load(f)["prompts"]
    chat = _prompts(8, 128, model.cfg.vocab_size, args.seed)

    for workload, prompts, out in (("wiki", wiki, args.out_wiki),
                                   ("chat", chat, args.out_chat)):
        print(f"== sweep workload={workload} C=1 out={out} "
              f"(in={len(prompts[0])} tok, prompt0 首条;N=0 行 = eager 参照)")
        run_one(model, prompts[0], out, n=0, k=0, workload=workload)
        for n in (1, 2, 3):
            for k in (2, 4, 6):
                run_one(model, prompts[0], out, n=n, k=k, workload=workload)

    model.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
