"""prefill 批量前向 profiler 拆账(M5 Task 0):C=8 TTFT 拆成 GPU 侧账目。

跑法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_prefill.py
对象:Qwen3ForCausalLM.forward_prefill_batch(8B,bf16,cuda),[8,128] 随机 ids
(seed=20261027,与 bench/profile_decode 同源)× 10 步。每步新建 views
(forward_prefill_batch 经 view.update 追加 KV,复用会续写 —— 步间零状态:
每步各序列分 8 页(128 tok / page 16),步末归还),前向本体不含调度/采样,
即 M4.1 批量 prefill 的纯模型侧账。prefill 路径不接 attention_backend /
fused_ops 选择器(qwen3.py forward_prefill_batch 契约),保持默认不置位。

输出:每步墙钟、GPU 自用时间(kernel+memcpy)、gap(墙钟 − GPU 端)、
有效 TFLOPS(14.2 TFLOP/次 ÷ 实测;总 FLOP 按 36 层静态 GEMM 估算:
qkv+o 2*D*(D+2KV*D)+MLP 2*3*D*F,×36 层 ×1024 token,attention 与
lm_head 不计 —— 标注"按 36 层静态 FLOP 估算",非实测 FLOP)、top-15
kernel 表(按自用设备时间,归类进 docs)。

profiler 口径沿用 scripts/profile_decode.py 实测结论(torch 2.13):
GPU 忙时只按 device_type=CUDA 事件累加,属性名 self_device_time_total
(旧名 self_cuda_time_total 在 FunctionEventAvg 上不存在);表排序键同名。
"""

from __future__ import annotations

import math
import time

import torch
from torch.profiler import ProfilerActivity, profile

from acorninfer.bench import DEFAULT_MODEL, _prompts
from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.models.qwen3 import from_pretrained

BATCH, IN_TOKENS, SEED = 8, 128, 20261027


def _is_gpu(evt) -> bool:
    """GPU 侧事件(kernel / device Memcpy)判定,CPU 侧算子事件返回 False。"""
    return str(evt.device_type).endswith("CUDA")


def main() -> None:
    device = "cuda:0"
    model = from_pretrained(DEFAULT_MODEL, device, torch.bfloat16)
    cfg = model.cfg
    allocator = BlockAllocator(1024)
    cache = PagedKVCache(
        cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
        16, allocator, device, torch.bfloat16,
    )
    pages_per_seq = math.ceil(IN_TOKENS / 16)

    ids = torch.tensor(
        _prompts(BATCH, IN_TOKENS, cfg.vocab_size, SEED),
        dtype=torch.long, device=device,
    )
    lens = [IN_TOKENS] * BATCH

    def one_step() -> torch.Tensor:
        # engine.step 同款 inference_mode:无 backward 图,激活即用即弃
        # (缺此包装时 autograd 保留整段前向中间量,24GB 卡 OOM,实测)
        with torch.inference_mode():
            views = []
            for _ in range(BATCH):
                table = [allocator.allocate() for _ in range(pages_per_seq)]
                views.append(PagedCacheView(cache, table))
            logits = model.forward_prefill_batch(ids, views, real_lens=lens)
            for v in views:
                for b in list(v.table):
                    allocator.free(b)
                v.table.clear()
            return logits

    # 形状 sanity + 一次性初始化(CUDA context / cuBLAS handle)先出清
    logits = one_step()
    assert tuple(logits.shape) == (BATCH, cfg.vocab_size), logits.shape
    allocator.assert_balanced()

    # profiler 预热窗:CUPTI 首次启用/缓冲分配的一次性开销落在独立窗口,
    # 不进计时也不进账目(同 profile_decode)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        one_step()
    torch.cuda.synchronize()

    n = 10
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            one_step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    evs = prof.key_averages()
    gpu = [e for e in evs if _is_gpu(e)]
    gpu_us = sum(e.self_device_time_total for e in gpu)
    print(f"steps={n} wall/step={wall / n * 1000:.2f}ms "
          f"gpu_kernel+memcpy/step={gpu_us / n / 1000:.2f}ms "
          f"gap/step={(wall - gpu_us / 1e6) / n * 1000:.2f}ms")

    # 有效 TFLOPS:14.2 TFLOP/次 ÷ 实测(墙钟与 GPU 侧各一行);
    # 总 FLOP 按 36 层静态 GEMM 估算(attention QK/AV 与 lm_head 不计)
    d, f = cfg.hidden_size, cfg.intermediate_size
    qkv_o = d * (d + 2 * cfg.num_key_value_heads * cfg.head_dim + d)
    flop = 2.0 * cfg.num_hidden_layers * (qkv_o + 3 * d * f) * BATCH * IN_TOKENS
    tflop = flop / 1e12
    eff_wall = flop / (wall / n) / 1e12
    eff_gpu = flop / (gpu_us / n / 1e6) / 1e12
    print(f"flops={tflop:.2f}TFLOP/次(按 36 层静态 FLOP 估算) "
          f"eff_wall={eff_wall:.2f}TFLOPS eff_gpu={eff_gpu:.2f}TFLOPS")
    print(evs.table(sort_by="self_device_time_total", row_limit=15))

    # 步间零状态核对:所有页归还,无泄漏
    allocator.assert_balanced()
    print(f"sanity: blocks live={allocator.num_live} "
          f"(每步 8 序列 × {pages_per_seq} 页步末全归还)")
    try:
        model.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
