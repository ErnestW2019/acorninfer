"""vLLM 基线压测:与 acorninfer.bench 同协议打 OpenAI /v1/completions。

协议对齐:同一批随机 prompt id(acorninfer.bench._prompts 同 seed)以 token-id
数组直投;temperature=0、max_tokens=32、自然 EOS 停止。指标公式与
acorninfer.bench._report 一致;差异(入档 docs):HTTP 往返计入两端延迟
(vLLM 侧含其调度开销),这是产品真实路径,70% 线按此口径。

运行(vLLM 已在 GPU 0 :8100 起好,见 Task 1):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import torch

from acorninfer.bench import _prompts

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"


def _run_one(client, base, ids, out_tokens, out):
    out["t0"] = time.perf_counter()
    out["first"] = None
    out["last"] = out["t0"]
    out["n"] = 0
    with client.stream(
        "POST", f"{base}/v1/completions",
        json={"prompt": ids, "max_tokens": out_tokens, "temperature": 0.0,
              "stream": True, "stream_options": {"include_usage": True}},
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            obj = json.loads(payload)
            now = time.perf_counter()
            if out["first"] is None:
                out["first"] = now          # 首个 data 事件 ≈ TTFT
            out["last"] = now
            if obj.get("usage"):
                out["n"] = obj["usage"]["completion_tokens"]


def _report(level: int, outs: list, out_tokens: int) -> None:
    ttfts = [o["first"] - o["t0"] for o in outs]
    tpots = [(o["last"] - o["first"]) / (o["n"] - 1) for o in outs if o["n"] > 1]
    total = sum(o["n"] for o in outs)
    span = max(o["last"] for o in outs) - min(o["t0"] for o in outs)
    print(f"RESULT-VLLM C={level} ttft_p50={statistics.median(ttfts):.4f}s "
          f"ttft_mean={statistics.mean(ttfts):.4f}s "
          f"tpot_mean={statistics.mean(tpots) * 1000:.2f}ms tok_s={total / span:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8100")
    p.add_argument("--model-dir", default=DEFAULT_MODEL)  # 仅取 vocab 大小,与 _prompts 对齐
    p.add_argument("--in-tokens", type=int, default=128)
    p.add_argument("--out-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--seed", type=int, default=20261027)
    args = p.parse_args()

    from acorninfer.config import Qwen3Config
    vocab = Qwen3Config.from_json(f"{args.model_dir}/config.json").vocab_size

    with httpx.Client(timeout=300.0) as client:
        for level in args.concurrency:
            prompts = _prompts(level, args.in_tokens, vocab, args.seed)
            outs = [dict() for _ in prompts]
            with ThreadPoolExecutor(max_workers=level) as pool:
                list(pool.map(lambda t: _run_one(client, args.base, t[0], args.out_tokens, t[1]),
                              zip(prompts, outs)))
            _report(level, outs, args.out_tokens)


if __name__ == "__main__":
    main()
