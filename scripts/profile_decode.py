"""decode 步 profiler 账目(M4 T1):docs 的"推断"升级为实测。

跑法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py
输出:每步墙钟、GPU 端累计(kernel+memcpy)、发射间隙(gap = 墙钟 − GPU 端,
即 CUDA graph 的理论收益上限)、每步 kernel 启动次数与 H2D/D2H 次数、
top-15 kernel 表。
8 序列(in=128,seed=20261027,与 bench 同 seed)prefill 完成后先空跑
3 步出稳定,再计时 + profile N 个 decode 步(默认 20),temp=0。
注意:该 seed 下与 M2/M3 的 C=8 档记录一致,有 1 条请求在第 2 个生成 token
命中 EOS、于预热期退役,计时窗口内 batch 恒为 7;脚本尾部 sanity 行打印
alive 数,若窗口内仍有提前退役(batch 不均匀)则断言失败。

torch 2.13 profiler API 探测结论(脚本按此写,勿再硬猜):
- FunctionEventAvg 上是 self_device_time_total(旧名 self_cuda_time_total
  属性已不存在;table(sort_by=...) 两种键都接受,统一用新名);
- GPU 忙时/启动数必须只按 device_type == DeviceType.CUDA 的事件累加:
  CPU 侧算子事件(如 aten::mm)的 self_device_time_total 会聚合其子 kernel
  的设备时间,全量累加会把同一份 GPU 时间双计一遍;
- 设备侧拷贝事件 key 形如 "Memcpy DtoH (...)",CPU 侧发射是
  "cudaMemcpyAsync",memcpy 计数只数前者,避免双计。
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from acorninfer.bench import DEFAULT_MODEL, _prompts
from acorninfer.engine.engine import Engine
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.sampler import SamplingParams


def _is_gpu(evt) -> bool:
    """GPU 侧事件(kernel / device Memcpy)判定,CPU 侧算子事件返回 False。"""
    return str(evt.device_type).endswith("CUDA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = from_pretrained(DEFAULT_MODEL, args.device, torch.bfloat16)
    model.attention_backend = "triton"
    model.fused_ops = True
    engine = Engine(model, None, num_blocks=1024, page_size=16, max_running=8)
    try:
        vocab = model.cfg.vocab_size
        prompts = _prompts(8, 128, vocab, 20261027)
        # max_tokens 留足余量:计时窗口内不得有序列因 length 提前退役
        sids = [
            engine.submit(p, SamplingParams(temperature=0.0), max_tokens=args.steps + 16)
            for p in prompts
        ]
        engine.step()  # prefill 步:max_running=8 一次全准入,各产首 token
        while any(len(engine.seqs[i].generated_ids) == 0 for i in sids):
            engine.step()
        # 全员进入 decode;先空跑 3 步出稳定性(该 seed 的 EOS 命中者也在此退役)
        for _ in range(3):
            engine.step()

        # profiler 预热窗:CUPTI 首次启用/缓冲分配的一次性开销落在独立窗口,
        # 不进计时也不进账目(key_averages 只取下面的测量窗)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            engine.step()
        torch.cuda.synchronize()

        n = args.steps
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                engine.step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

        evs = prof.key_averages()
        gpu = [e for e in evs if _is_gpu(e)]
        gpu_us = sum(e.self_device_time_total for e in gpu)
        launches = sum(e.count for e in gpu if not e.key.startswith("Memcpy"))
        memcpy = sum(e.count for e in gpu if e.key.startswith("Memcpy"))
        print(f"steps={n} wall/step={wall / n * 1000:.2f}ms "
              f"gpu_kernel+memcpy/step={gpu_us / n / 1000:.2f}ms "
              f"gap/step={(wall - gpu_us / 1e6) / n * 1000:.2f}ms")
        print(f"launches/step={launches / n:.1f} memcpy/step={memcpy / n:.2f}")
        print(evs.table(sort_by="self_device_time_total", row_limit=15))

        # 测量有效性:计时窗口内 batch 必须恒定(提前退役会使各步 batch 不同)
        alive = [i for i in sids if not engine.seqs[i].is_finished]
        gens = sorted({len(engine.seqs[i].generated_ids) for i in alive})
        assert len(gens) == 1, f"计时窗口内仍有序列提前退役,batch 不均匀: {gens}"
        print(f"sanity: alive={len(alive)}/8 generated_per_seq={gens[0]} "
              f"(batch 恒定贯穿计时窗口)")
    finally:
        model.to("cpu")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
