"""段 0 尖刀:NCCL allreduce 与半宽 GEMM 微基准。用法:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 scripts/phase0_spike.py
(本仓 venv 无 torchrun 入口脚本,等价命令:
  CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run
  --nproc-per-node 2 scripts/phase0_spike.py)
输出逐档延迟/带宽/TFLOPS 行(stdout),数字原样进 docs。

口径:
- allreduce:bf16 一维张量,尺寸档 [8KiB, 57KiB, 1MiB, 8MiB](字节);
  每档 eager 侧 warmup 10 + 计 50 次(torch.cuda.Event 计总时/50);
  graph 侧:CUDAGraph 捕获 50 次 allreduce(同张量,测通信不测数据依赖),
  捕获前 warmup 3 次在侧 stream(torch.cuda.graph 惯例)+ 全员 barrier 对齐,
  捕获后 replay 2 次热身再计时 1 次 replay 计总时/50;捕获 RuntimeError →
  打印 GRAPH_CAPTURE_FAIL 行、graph 栏记 FAIL,继续下一档(裁决输入,非退出)。
  bw_eager = size*2/latency(读+写两笔的简化口径,ring 系数不较真 —— 有效带宽(简化))。
- GEMM:bf16 矩阵乘(matmul 预分配 out),warmup 10 + 计 50 次取均值,
  TFLOPS = 2*M*N*K/t;decode 半宽 [7,4096]x[4096,2048]/[4096,3072]、
  全宽对照 [7,4096]x[4096,4096]、prefill 半宽 [1024,4096]x[4096,2048];
  RATIO half/full 仅 decode 档可比(N=2048 半宽 vs N=4096 全宽)。
- 两 rank 输出交错,每行带 [rankN] 前缀;入档以 rank0 为准,rank1 顺手对一致。

实测注记(首跑发现,脚本按此绕行):CUDAGraph 捕获 NCCL allreduce 后,
进程组的收尾路径(barrier/destroy_process_group/解释器退出时 graph 析构)
会挂死在 CudaEventDestroy 上(NCCL watchdog 480s 无进展报警后终止进程)。
故:捕获的 graph 保留到结尾不提前析构;全部数据行 print+flush 后以
os._exit(0) 直接退出,绕过 teardown —— 只影响测量脚本的退出路径,
不影响任何测量数字。
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

ALLOT_SIZES_B = [8 * 1024, 57 * 1024, 1024 * 1024, 8 * 1024 * 1024]
GEMM_SHAPES = [(7, 4096, 2048), (7, 4096, 3072), (7, 4096, 4096), (1024, 4096, 2048)]


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    dev = f"cuda:{local_rank}"

    def p(msg: str) -> None:
        print(f"[rank{rank}] {msg}", flush=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # ---- allreduce 段 ----
    graphs = []  # 首跑实测:捕获的 graph 提前析构会挂死 NCCL teardown,保活到结尾
    for size in ALLOT_SIZES_B:
        t = torch.ones(size // 2, dtype=torch.bfloat16, device=dev)
        for _ in range(10):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dist.barrier()
        start.record()
        for _ in range(50):
            dist.all_reduce(t)
        end.record()
        torch.cuda.synchronize()
        eager_us = start.elapsed_time(end) * 1000 / 50
        bw_gbs = 2 * size / (eager_us * 1e3)  # 简化口径:读+写两笔

        graph_str = "FAIL"
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    dist.all_reduce(t)
            torch.cuda.current_stream().wait_stream(side)
            dist.barrier()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(50):
                    dist.all_reduce(t)
            graphs.append(g)
            g.replay()
            g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            graph_us = start.elapsed_time(end) * 1000 / 50
            graph_str = f"{graph_us:.2f}"
        except RuntimeError:
            p(f"GRAPH_CAPTURE_FAIL {size}")
            torch.cuda.synchronize()
        p(f"ALLOTSIZE={size} eager={eager_us:.2f} graph={graph_str} "
          f"bw_eager={bw_gbs:.1f}")
        del t

    # ---- GEMM 段 ----
    dist.barrier()
    tflops = {}
    for m, k, n in GEMM_SHAPES:
        a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        b = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
        o = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
        for _ in range(10):
            torch.matmul(a, b, out=o)
        torch.cuda.synchronize()
        start.record()
        for _ in range(50):
            torch.matmul(a, b, out=o)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 50
        tf = 2 * m * n * k / (ms * 1e-3) / 1e12
        tflops[(m, k, n)] = tf
        p(f"GEMM M={m} K={k} N={n} tflops={tf:.2f}")
    ratio = tflops[(7, 4096, 2048)] / tflops[(7, 4096, 4096)]
    p(f"RATIO half/full={ratio:.3f}")

    # 首跑实测:末尾 barrier/destroy_process_group 会挂死(CudaEventDestroy,
    # watchdog 480s 报警后杀进程)—— 数据行已全部 flush,os._exit 绕过
    # NCCL/graph teardown,干净退出。测量数字不经此路径。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
