"""自研困惑度 harness(M7 Task 3,spec §3):wikitext-2-raw-v1 test split 滑窗 NLL。

运行时禁 transformers(纪律自查:本文件与 import 链 acorninfer.tokenizer /
models.qwen3 / quant.quantize / ops.quant_gemm 均零 transformers)——分词用自家
`Tokenizer(f"{model_dir}/tokenizer.json")`,模型直建
`qwen3.from_pretrained`(cache=None 无 KV 全前向,logits [B,S,vocab] fp32)。
量化腿与生产同构:装载后 `apply_quant(model, "w8a16")` +
`quant_gemm.configure(device)`(window=512 → lm_head M=4096 走大 M 桶,
m_configs 表必须先配置,与 engine/server 接线同码,m6_ab 先例)。

滑窗 stride 法(window=512 stride=256 缺省):窗口 i 覆盖 token
[i·stride, i·stride+window);logits 行 r(输入位 s+r)预测 token s+r+1。
- 窗口 0 计全部 window−1 个位置(行 0..window−2 → 目标 1..window−1);
- 后续窗口只计"新暴露"的末 stride 个目标(行 window−stride−1..window−2,
  与此前所有窗口的目标集严格不相交 → 每个 token 恰计一次);
- 尾部(最后一个整窗覆盖面之外的 token)不计。
counted = (window−1) + (windows−1)·stride;ppl = exp(nll_sum / counted) 可复算。

内存纪律:fp32 logits 全量物化 [batch=8,512,151936] ≈ 2.5GB/批,可接受;
CE 逐批累加(批内再逐窗口行求和),绝不跨批拼接 logits。

数据(R7 双源,取活者):首选 Smerity zip(2026-11-02 实测活,命令原样入
docs/benchmarks.md),备选 HF parquet;缓存 data/wikitext-2-raw/(.gitignore)。
文本组装:wiki.test.raw 按 "\n" 切行保序、"\n".join 拼接(空行保留原始
语义;与原文件逐字节等价,含尾行),Tokenizer 全文 encode 一次性。

输出:PPL 行(逐字)+ PPLPOS(counted 可复算行)+ LOADTIME(装载/apply/
configure 耗时,R5 账);--dump-windows 落每窗 NLL json(超标归因用:
双腿对照取 top-10 恶化窗口)。

用法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/perplexity.py \
      [--model-dir /home/ernest/model-weight/Qwen3-8B] [--quant none|w8a16] \
      [--window 512 --stride 256 --batch 8] [--dump-windows out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from acorninfer.models.qwen3 import from_pretrained
from acorninfer.tokenizer import Tokenizer

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"
DEFAULT_DATA = "data/wikitext-2-raw/wiki.test.raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="wikitext-2-raw test split 本地缓存路径")
    ap.add_argument("--quant", default="none", choices=["none", "w8a16"])
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump-windows", default=None,
                    help="每窗 NLL json 输出路径(双腿对照归因用)")
    args = ap.parse_args()
    if args.stride < 1 or args.stride > args.window:
        raise ValueError(f"--stride 须 1 ≤ stride ≤ window,got {args.stride}")
    if args.batch < 1:
        raise ValueError("--batch 须 ≥ 1")

    # ---- 文本组装 + 分词(自家 Tokenizer,全文一次性 encode)----
    with open(args.data, encoding="utf-8") as f:
        lines = f.read().split("\n")
    text = "\n".join(lines)  # 逐行保序拼接;空行原样保留
    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    ids = tok.encode(text)
    n = len(ids)
    if n < args.window + 1:
        raise ValueError(f"token 数 {n} 不足以开一个 window={args.window} 窗")
    tokens = torch.tensor(ids, dtype=torch.long, device=args.device)
    print(f"DATA path={args.data} lines={len(lines)} tokens={n}", flush=True)

    # ---- 装载(直建,无引擎)+ 量化腿生产同构接线 ----
    print(f"loading {args.model_dir} -> {args.device} (bf16) "
          f"quant={args.quant} window={args.window} stride={args.stride} "
          f"batch={args.batch} ...", flush=True)
    t_load = time.perf_counter()
    model = from_pretrained(args.model_dir, args.device, torch.bfloat16)
    load_s = time.perf_counter() - t_load
    apply_s = configure_s = 0.0
    if args.quant != "none":
        t_apply = time.perf_counter()
        from acorninfer.ops import quant_gemm
        from acorninfer.quant.quantize import apply_quant
        apply_quant(model, args.quant)
        apply_s = time.perf_counter() - t_apply
        t_cfg = time.perf_counter()
        quant_gemm.configure(device=args.device)
        configure_s = time.perf_counter() - t_cfg
    print(f"LOADTIME quant={args.quant} load_s={load_s:.1f} "
          f"apply_s={apply_s:.1f} configure_s={configure_s:.1f}", flush=True)

    # ---- 滑窗 NLL(stride 法,见模块 docstring;窗口定长,尾部不计)----
    n_windows = 1 + (n - args.window) // args.stride
    counted = (args.window - 1) + (n_windows - 1) * args.stride
    nll_sum = 0.0
    win_nll: list[float] = []
    t_eval = time.perf_counter()
    with torch.inference_mode():
        for b0 in range(0, n_windows, args.batch):
            idx = range(b0, min(b0 + args.batch, n_windows))  # 尾批截断
            batch_ids = torch.stack([
                tokens[i * args.stride: i * args.stride + args.window]
                for i in idx])
            # cache=None 全前向 → fp32 logits [B, window, vocab] 全量物化
            logits = model(batch_ids)
            for r, i in enumerate(idx):
                # 本窗口计数行:[lo, hi](含端点);行 r' 预测 token s+r'+1;
                # 末行恒为 window−2(行 window−1 预测窗外 token,不可计)
                if i == 0:
                    lo, hi = 0, args.window - 2          # 全部 window−1 位
                else:
                    lo, hi = (args.window - args.stride - 1,
                              args.window - 2)           # 新暴露末 stride 位
                tgt = batch_ids[r, lo + 1: hi + 2]
                ce = F.cross_entropy(logits[r, lo: hi + 1], tgt,
                                     reduction="sum")
                w = float(ce.double())
                nll_sum += w
                win_nll.append(w)
            del logits
    eval_s = time.perf_counter() - t_eval

    ppl = math.exp(nll_sum / counted)
    print(f"PPL quant={args.quant} tokens={n} nll_sum={nll_sum:.2f} "
          f"ppl={ppl:.4f} windows={n_windows}", flush=True)
    print(f"PPLPOS quant={args.quant} counted={counted} "
          f"mean_nll={nll_sum / counted:.6f} eval_s={eval_s:.1f} "
          f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}",
          flush=True)
    if args.dump_windows:
        with open(args.dump_windows, "w", encoding="utf-8") as f:
            json.dump({"quant": args.quant, "window": args.window,
                       "stride": args.stride, "tokens": n,
                       "windows": n_windows, "nll_sum": nll_sum,
                       "ppl": ppl, "win_nll": win_nll}, f)
        print(f"DUMP {args.dump_windows}", flush=True)


if __name__ == "__main__":
    main()
