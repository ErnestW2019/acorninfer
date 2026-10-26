"""`python -m suiren` command-line entry."""

from __future__ import annotations

import argparse
import time

import torch

from suiren.config import Qwen3Config
from suiren.generate import generate_token_ids
from suiren.models.qwen3 import Qwen3ForCausalLM, map_hf_state_dict
from suiren.tokenizer import Tokenizer
from suiren.weights import load_safetensors

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"


def main() -> None:
    parser = argparse.ArgumentParser(prog="suiren")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="greedy generation (M0)")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--model-dir", default=DEFAULT_MODEL)
    gen.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = "cuda:0"
    t0 = time.perf_counter()
    cfg = Qwen3Config.from_json(f"{args.model_dir}/config.json")
    model = Qwen3ForCausalLM(cfg)  # CPU 构建 → assign → 上卡,单份显存峰值
    mapped = map_hf_state_dict(load_safetensors(args.model_dir, device="cpu"))
    model.load_state_dict(mapped, strict=True, assign=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    t0 = time.perf_counter()
    ids = generate_token_ids(
        model, tokenizer=tok, prompt=args.prompt, max_new_tokens=args.max_new_tokens
    )
    elapsed = time.perf_counter() - t0
    prompt_len = len(tok.encode(args.prompt))
    new_tokens = len(ids) - prompt_len
    print(f"\n{tok.decode(ids)}")
    print(
        f"\n--- {new_tokens} tokens in {elapsed:.1f}s "
        f"({new_tokens / elapsed:.2f} tok/s, 无 cache 全量重算)"
    )


if __name__ == "__main__":
    main()
