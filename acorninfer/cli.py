"""`python -m acorninfer` command-line entry."""

from __future__ import annotations

import argparse
import time

from acorninfer.generate import generate_cached
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.sampler import Sampler, SamplingParams
from acorninfer.tokenizer import Tokenizer

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"


def main() -> None:
    parser = argparse.ArgumentParser(prog="acorninfer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="greedy or sampled generation")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--model-dir", default=DEFAULT_MODEL)
    gen.add_argument("--max-new-tokens", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = from_pretrained(args.model_dir)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("prompt encoded to zero tokens; pass a non-empty prompt")

    sampler = None
    if args.temperature > 0:
        sampler = Sampler(
            SamplingParams(
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
            )
        )

    t0 = time.perf_counter()
    ids, _ = generate_cached(
        model, prompt_ids, max_new_tokens=args.max_new_tokens, sampler=sampler
    )
    elapsed = time.perf_counter() - t0
    new_tokens = len(ids) - len(prompt_ids)
    print(f"\n{tok.decode(ids)}")
    print(f"\n--- {new_tokens} tokens in {elapsed:.1f}s ({new_tokens / elapsed:.2f} tok/s, KV cache)")
