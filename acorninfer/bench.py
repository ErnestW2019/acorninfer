"""吞吐 / TTFT / TPOT 压测 CLI(直连 Engine,不经 HTTP —— 测的是引擎本体)。

每个并发档 C:一次性提交 C 条 in=128 token 的随机 prompt(greedy,out=32),
step() 驱动到全部完成,再从 Sequence 的引擎记账时间戳(created_at /
first_token_at / last_token_at,逐 token 由 engine._record 写入)收集指标:

- TTFT   = first_token_at - created_at(含排队等待 + prefill)
- TPOT   = (last_token_at - first_token_at) / (len(generated_ids) - 1)
- 单请求吞吐 = len(generated_ids) / (last_token_at - created_at)
- 聚合吞吐   = 总生成 token / (max(last_token_at) - min(created_at))

模型跨档复用(只加载一次);每档结束 assert_balanced + 全量 drop,档间零残留。
prompt 由 --seed 决定:每档独立重播种,因此各档第 1 条 prompt 相同,跨档可比。
计时前跑 1 条短 warmup 请求(触发 CUDA kernel / 分配器路径),否则首个 prefill
会吃进一次性初始化开销,污染 C=1 的 TTFT。

运行(GPU 0 被外部服务占用,统一走 GPU 1):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 \
        --attention-backend triton --fused-ops
"""

from __future__ import annotations

import argparse
import statistics

import torch

from acorninfer.config import QWEN3_EOS_IDS
from acorninfer.engine.engine import Engine
from acorninfer.engine.sequence import Sequence
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"


def _prompts(count: int, in_tokens: int, vocab: int, seed: int) -> list[list[int]]:
    """Seeded 随机 prompt,恰为 in_tokens 个 id,且不含 EOS(避免 prompt 提前终止)。"""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (count, in_tokens), generator=gen)
    for eos in QWEN3_EOS_IDS:
        if 0 <= eos < vocab:
            ids[ids == eos] = (eos + 1) % vocab  # 151644/151646 非 EOS,不会互相踩
    return ids.tolist()


def _run_level(
    engine: Engine, prompts: list[list[int]], out_tokens: int
) -> dict[int, "Sequence"]:
    """一次性提交全部请求,驱动到完成,返回 {seq_id: Sequence}。"""
    sids = [
        engine.submit(ids, SamplingParams(temperature=0.0), max_tokens=out_tokens)
        for ids in prompts
    ]
    while engine.has_pending():
        engine.step()
    seqs = {sid: engine.seqs[sid] for sid in sids}
    for sid in sids:
        engine.drop(sid)
    assert not engine.seqs, "drop 之后 seqs 必须为空(跨档零残留)"
    engine.assert_balanced()
    return seqs


def _report(level: int, seqs: dict, out_tokens: int) -> None:
    ttfts = [s.first_token_at - s.created_at for s in seqs.values()]
    tpots = [
        (s.last_token_at - s.first_token_at) / (len(s.generated_ids) - 1)
        for s in seqs.values()
        if len(s.generated_ids) > 1
    ]
    per_req = [
        len(s.generated_ids) / (s.last_token_at - s.created_at) for s in seqs.values()
    ]
    total_tokens = sum(len(s.generated_ids) for s in seqs.values())
    span = max(s.last_token_at for s in seqs.values()) - min(
        s.created_at for s in seqs.values()
    )
    agg = total_tokens / span

    print(f"\n== C={level}  requests={len(seqs)}  out/token-stats: "
          f"min={min(len(s.generated_ids) for s in seqs.values())} "
          f"max={max(len(s.generated_ids) for s in seqs.values())} / {out_tokens}")
    print(f"   TTFT  mean {statistics.mean(ttfts):7.4f}s   p50 {statistics.median(ttfts):7.4f}s")
    if tpots:
        print(f"   TPOT  mean {statistics.mean(tpots) * 1000:7.2f}ms  "
              f"({len(tpots)}/{len(seqs)} seqs with >1 token)")
    print(f"   per-request tok/s: {' '.join(f'{x:.2f}' for x in per_req)}")
    print(f"   aggregate throughput: {agg:.2f} tok/s  "
          f"(total {total_tokens} tokens / span {span:.3f}s)")
    tpot_str = (
        f"{statistics.mean(tpots) * 1000:.2f}ms" if tpots else "n/a"
    )
    print(f"RESULT C={level} ttft_p50={statistics.median(ttfts):.4f}s "
          f"ttft_mean={statistics.mean(ttfts):.4f}s "
          f"tpot_mean={tpot_str} tok_s={agg:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="acorninfer-bench",
        description="AcornInfer 引擎压测:吞吐 / TTFT / TPOT(直连 Engine)",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument("--watermark-blocks", type=int, default=8)
    parser.add_argument("--attention-backend", default="sdpa", choices=["sdpa", "triton"])
    parser.add_argument("--fused-ops", action="store_true")
    parser.add_argument("--in-tokens", type=int, default=128)
    parser.add_argument("--out-tokens", type=int, default=32)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 4, 8],
        help="并发档位列表,如 --concurrency 1 4 8",
    )
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()

    print(f"loading {args.model_dir} -> {args.device} (bf16) ...")
    model = from_pretrained(args.model_dir, args.device, torch.bfloat16)
    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    engine = Engine(
        model,
        tok,
        num_blocks=args.num_blocks,
        page_size=args.page_size,
        max_running=args.max_running,
        watermark_blocks=args.watermark_blocks,
    )
    # M3 后端开关(bench.py 直接构造 Engine,不经 EngineConfig —— 属性赋值与
    # Engine.from_config 同一机制;K1/K2/K3 均在 forward_decode 读属性生效)
    model.attention_backend = args.attention_backend
    model.fused_ops = args.fused_ops
    vocab = model.cfg.vocab_size
    print(f"engine ready: num_blocks={args.num_blocks} page={args.page_size} "
          f"max_running={args.max_running} watermark={args.watermark_blocks} "
          f"backend={model.attention_backend} fused_ops={model.fused_ops}")

    try:
        # warmup:1 条短请求,触发 prefill/decode/paged-cache 的 CUDA 路径,
        # 不计入任何档位的指标(一次性初始化开销不是引擎性能)
        warm = _run_level(
            engine, _prompts(1, 8, vocab, args.seed), min(4, args.out_tokens)
        )
        print(f"warmup done ({len(warm)} request, not measured)")

        for level in args.concurrency:
            prompts = _prompts(level, args.in_tokens, vocab, args.seed)
            seqs = _run_level(engine, prompts, args.out_tokens)
            _report(level, seqs, args.out_tokens)
    finally:
        # 镜像 golden_m0/m1 的清理协议:权重回 CPU,释放 GPU 显存
        model.to("cpu")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
