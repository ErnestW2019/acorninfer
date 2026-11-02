"""vLLM 基线压测:与 acorninfer.bench 同协议打 OpenAI /v1/completions。

协议对齐:同一批随机 prompt id(acorninfer.bench._prompts 同 seed)以 token-id
数组直投;temperature=0、max_tokens=32、自然 EOS 停止。指标公式与
acorninfer.bench._report 一致;差异(入档 docs):HTTP 往返计入两端延迟
(vLLM 侧含其调度开销),这是产品真实路径,70% 线按此口径。

运行(vLLM 起服命令与基线实测数字见 docs/benchmarks.md「M3 基线」小节):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from acorninfer.bench import _prompts

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"


def _run_one(client, base, ids, out_tokens, out):
    out["t0"] = time.perf_counter()
    out["first"] = None
    out["last"] = out["t0"]
    out["n"] = 0
    out["done"] = False
    with client.stream(
        "POST", f"{base}/v1/completions",
        json={"model": DEFAULT_MODEL, "prompt": ids, "max_tokens": out_tokens,
              "temperature": 0.0, "stream": True,
              "stream_options": {"include_usage": True}},
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                out["done"] = True
                break
            obj = json.loads(payload)
            now = time.perf_counter()
            if out["first"] is None:
                out["first"] = now          # 首个 data 事件 ≈ TTFT
            if obj.get("choices"):
                out["last"] = now           # 仅内容 chunk 推进 last(usage 尾包不算)
            if obj.get("usage"):
                out["n"] = obj["usage"]["completion_tokens"]


def _report(level: int, outs: list, out_tokens: int) -> None:
    bad = [i for i, o in enumerate(outs) if not o.get("done") or o["n"] <= 0]
    if bad:
        raise RuntimeError(f"C={level}: {len(bad)} request(s) ended without [DONE]/usage")
    ttfts = [o["first"] - o["t0"] for o in outs]
    tpots = [(o["last"] - o["first"]) / (o["n"] - 1) for o in outs if o["n"] > 1]
    total = sum(o["n"] for o in outs)
    span = max(o["last"] for o in outs) - min(o["t0"] for o in outs)
    print(f"RESULT-VLLM C={level} ttft_p50={statistics.median(ttfts):.4f}s "
          f"ttft_mean={statistics.mean(ttfts):.4f}s "
          f"tpot_mean={statistics.mean(tpots) * 1000:.2f}ms tok_s={total / span:.2f}")


def _spec_report(level: int, client, base: str, prev: dict) -> dict:
    """M8 Task 4:GET /spec_stats(AcornInfer server 专有)打印本档增量接受账。

    E_acc1 = E[接受+1] = accepted/verify_steps + 1(spec §1 定义);
    mean_committed = 实际入流 token/verify 步(eos 截断可 < E_acc1);
    hist = committed/步 分布(本档增量,键 ≤ m)。返回累计快照供下档差分。
    """
    cur = client.get(f"{base}/spec_stats").json()

    def delta(key):
        return cur[key] - prev.get(key, 0)

    vs = delta("verify_steps")
    acc = delta("accepted")
    com = delta("committed")
    hist = {}
    for k in set(prev.get("hist", {})) | set(cur["hist"]):
        d = cur["hist"].get(k, 0) - prev.get("hist", {}).get(k, 0)
        if d:
            hist[k] = d
    e_acc1 = f"{acc / vs + 1:.3f}" if vs else "n/a"
    mean_com = f"{com / vs:.3f}" if vs else "n/a"
    print(f"SPEC-STATS C={level} verify_steps={vs} E_acc1={e_acc1} "
          f"mean_committed={mean_com} hist={dict(sorted(hist.items()))}")
    return cur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8100")
    p.add_argument("--model-dir", default=DEFAULT_MODEL)  # 仅取 vocab 大小,与 _prompts 对齐
    p.add_argument("--in-tokens", type=int, default=128)
    p.add_argument("--out-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--prompts-file", default=None,
                   help="wiki 语料 JSON(make_spec_prompts.py 产物);给出则各档"
                        "用其前 C 条 prompt(token-id 数组直投),缺省随机语料"
                        "行为不变(A4)")
    p.add_argument("--spec-stats", action="store_true",
                   help="各档压测后 GET {base}/spec_stats 打印 verify 步接受账"
                        "(AcornInfer server 专有;缺省不发该请求 = 行为不变)")
    args = p.parse_args()

    from acorninfer.config import Qwen3Config
    vocab = Qwen3Config.from_json(f"{args.model_dir}/config.json").vocab_size

    file_prompts = None
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            file_prompts = json.load(f)["prompts"]

    with httpx.Client(timeout=300.0) as client:
        # warmup:1 条小请求(镜像 acorninfer.bench 协议,不计入任何档位指标)。
        # --prompts-file 腿改用语料首条 + out=min(8, out_tokens):out ≥ m=K+1
        # 强制 ≥1 个 verify 步 → triton verify kernel 在服务端进程内 JIT 预热,
        # 不落进首档计时(spec 腿;缺省语料腿保持 8-token/4-out 原样)
        warm = [dict()]
        if file_prompts is not None:
            warm_ids, warm_out = file_prompts[0], min(8, args.out_tokens)
        else:
            warm_ids, warm_out = _prompts(1, 8, vocab, args.seed)[0], \
                min(4, args.out_tokens)
        _run_one(client, args.base, warm_ids, warm_out, warm[0])
        print("warmup done (1 request, not measured)")

        prev_stats = None
        if args.spec_stats:
            # warmup 的 verify 步(如有)计入基线快照,不落任何档位增量
            prev_stats = _spec_report("warmup", client, args.base, {})
        for level in args.concurrency:
            prompts = file_prompts[:level] if file_prompts is not None else \
                _prompts(level, args.in_tokens, vocab, args.seed)
            outs = [dict() for _ in prompts]
            with ThreadPoolExecutor(max_workers=level) as pool:
                list(pool.map(lambda t: _run_one(client, args.base, t[0], args.out_tokens, t[1]),
                              zip(prompts, outs)))
            _report(level, outs, args.out_tokens)
            if args.spec_stats:
                prev_stats = _spec_report(level, client, args.base, prev_stats)


if __name__ == "__main__":
    main()
