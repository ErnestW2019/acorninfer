"""M8 Task 4 语料:wiki.test.raw → 128-token 块 → data/spec_prompts_wiki.json。

口径(spec §6):全文按行保序拼接(perplexity.py 同款拼接语义,与原文件
逐字节等价)→ 自家 Tokenizer 全文 encode 一次性 → 连续 128-token 块
→ seeded 无放回抽样 count 块(确定性:random.Random(seed),样本索引升序,
可复现)→ prompt 内 EOS id 按 bench._prompts 同款置换(eos → (eos+1)%vocab;
引擎只对生成位判 EOS,prompt 位置换属协议防御,与 bench 语料同口径)
→ JSON(prompts = token-id 数组,`bench_remote.py --prompts-file` 直投
/v1/completions,不做重分词)。

用法(纯 CPU,零 GPU;运行时禁 import transformers):
    .venv/bin/python scripts/make_spec_prompts.py
    # 可选:--count 64 --block 128 --seed 20261102
    #       --out data/spec_prompts_wiki.json
"""

from __future__ import annotations

import argparse
import json
import random

from acorninfer.bench import DEFAULT_MODEL
from acorninfer.config import QWEN3_EOS_IDS, Qwen3Config
from acorninfer.tokenizer import Tokenizer

DEFAULT_DATA = "data/wikitext-2-raw/wiki.test.raw"
DEFAULT_OUT = "data/spec_prompts_wiki.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261102)
    args = ap.parse_args()

    # 全文按行保序拼接(空行原样保留;perplexity.py 同款)
    with open(args.data, encoding="utf-8") as f:
        lines = f.read().split("\n")
    text = "\n".join(lines)
    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    ids = tok.encode(text)

    n_blocks = len(ids) // args.block
    if args.count > n_blocks:
        raise ValueError(f"count={args.count} > blocks={n_blocks}")
    # seeded 无放回抽样,索引升序(确定性;跨机/跨次可复现)
    picked = sorted(random.Random(args.seed).sample(range(n_blocks), args.count))
    vocab = Qwen3Config.from_json(f"{args.model_dir}/config.json").vocab_size

    prompts, eos_replaced = [], 0
    for b in picked:
        block = ids[b * args.block : (b + 1) * args.block]
        # EOS 置换(bench._prompts 同款):引擎只对生成位判 EOS,此为协议防御
        eos_replaced += sum(1 for t in block if t in QWEN3_EOS_IDS)
        block = [(t + 1) % vocab if t in QWEN3_EOS_IDS else t for t in block]
        prompts.append(block)

    doc = {
        "source": args.data,
        "block_tokens": args.block,
        "seed": args.seed,
        "count": len(prompts),
        "total_tokens": len(ids),
        "blocks_total": n_blocks,
        "eos_replaced": eos_replaced,
        "prompts": prompts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    print(f"CORPUS source={args.data} total_tokens={len(ids)} "
          f"blocks={n_blocks} block={args.block} count={len(prompts)} "
          f"seed={args.seed} eos_replaced={eos_replaced} out={args.out}")


if __name__ == "__main__":
    main()
