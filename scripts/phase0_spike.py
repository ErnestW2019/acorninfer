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
- --gemm-variants 单卡模式(不需 torchrun;main 按 argv 分流,双卡路径原样):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/phase0_spike.py --gemm-variants
  形状矩阵 M ∈ {7,8,16,64,256,1024} × N ∈ {2048,3072,4096},K=4096,bf16;
  每格四个核选变体取时间最好者(w 按 nn.Linear 布局 [N,K] 连续存放):
    matmul     torch.matmul(a, w.t()) —— .t() 是 view;规格字面 `a @ w` 在
               w [N,K] 下不成形,按模型实际调用路径(.t() 视图,TN 布局)实现
    linear     F.linear(a, w) —— 无 bias 线性层 dispatcher 路径
    pad16      仅 M=7:输入零填充到 16 行 matmul(w.t())后切片(计 16 行 GEMM
               总时;填充缓冲步外预填零,切片为 view 不产 kernel)
    transposed w.t().contiguous() → [K,N](预处理不计入)后 a @ w_t —— NN 布局
  每变体 warmup 10 + 计 50(cuda.Event 均值);输出行
  `GEMMV M=<m> N=<n> best=<variant> us=<v> effbw=<GB/s>`,
  effbw = 权重字节(N×K×2B)/时间;M≥256 档为算术+访存混合,带宽仅供参考。
  末尾汇总表 + `RATIO' half/full=M=7 最好变体口径`(0.5×us_full/us_half,
  与 Task 0 TFLOPS 口径同构可比)+ prefill 全宽对照 TFLOPS 行。
- --gemm-cold 单卡模式(不需 torchrun;argv 分流,既有两模式零改动):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/phase0_spike.py --gemm-cold
  冷权重裁决口径:decode 真实四类 GEMM(qkv/o/gate_up/down,N/K 切半同引擎
  TP 切法)每类 36 套独立权重轮转(>72MB L2,强制 HBM),F.linear bf16,
  M=7,warmup 一整轮 + 计 30 轮 ÷(30×36);同格输出单套复用 hot 对照列
  (复现旧微基准的 L2 伪象——热复用 ≤72MB 权重根本不走 HBM,Task 0 的
  RATIO 0.373 即此伪象,作废)。输出行
  `GEMMCOLD <class> full=<us> hot_full=<us> half=<us> hot_half=<us> ratio=<t_half/t_full>`
  + 每步 GEMM 总账(36 层)+ TP=2 投影行(假设注明:非 GEMM 不变、
  allreduce 73×17.84µs 串行、锚 M4.1 单卡 graph GPU 19.81 ms/步)。
  模式收编自控制器裁决测量脚本 /tmp/opencode/gemm_cold.py(逻辑照搬,
  输出行格式向本脚本既有风格对齐)。

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
import torch.nn.functional as F

ALLOT_SIZES_B = [8 * 1024, 57 * 1024, 1024 * 1024, 8 * 1024 * 1024]
GEMM_SHAPES = [(7, 4096, 2048), (7, 4096, 3072), (7, 4096, 4096), (1024, 4096, 2048)]
GEMMV_MS = [7, 8, 16, 64, 256, 1024]
GEMMV_NS = [2048, 3072, 4096]
GEMMV_K = 4096

# 冷权重裁决口径:引擎真实四类 GEMM 形状,(类名, N_full, K_full, N_half, K_half)
# —— F.linear(x[M,K], W[N,K]);qkv/gate_up 输出列切(N 半),o/down 行并行输入切(K 半)
GEMMCOLD_CLASSES = [
    ("qkv", 6144, 4096, 3072, 4096),
    ("o", 4096, 4096, 4096, 2048),
    ("gate_up", 24576, 4096, 12288, 4096),
    ("down", 4096, 12288, 4096, 6144),
]
GEMMCOLD_LAYERS = 36   # 36 套独立权重轮转 = 36 层,任何一套本轮内不重访
GEMMCOLD_REPS = 30     # 计时 30 轮 × 36 次,总时 ÷ (30×36)


def gemm_variants_main() -> None:
    """--gemm-variants 单卡模式:同形状多核选变体取最好(段 0 补充)。

    动机:Task 0 的 decode RATIO 0.373 未分离「切半固有损失 vs cuBLAS
    对该形状选核差」—— M=7 decode GEMM 是权重流搬(内存受限),变体
    矩阵给核选留修复空间。变体语义见模块 docstring。"""
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 下即物理 GPU 1
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_us(fn, warmup=10, iters=50) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / iters  # µs

    results = {}
    for m in GEMMV_MS:
        for n in GEMMV_NS:
            a = torch.randn(m, GEMMV_K, dtype=torch.bfloat16, device=dev)
            w = torch.randn(n, GEMMV_K, dtype=torch.bfloat16, device=dev)
            w_t = w.t().contiguous()  # transposed 变体的预处理,不计入
            variants = {
                "matmul": lambda: torch.matmul(a, w.t()),
                "linear": lambda: F.linear(a, w),
                "transposed": lambda: torch.matmul(a, w_t),
            }
            if m == 7:
                a_pad = torch.zeros(16, GEMMV_K, dtype=torch.bfloat16, device=dev)
                a_pad[:m] = a
                variants["pad16"] = lambda: torch.matmul(a_pad, w.t())[:m]
            times = {name: time_us(fn) for name, fn in variants.items()}
            best = min(times, key=times.get)
            us = times[best]
            bw = n * GEMMV_K * 2 / (us * 1e-6) / 1e9  # 权重字节口径(us→s)
            results[(m, n)] = (best, us, bw)
            print(f"GEMMV M={m} N={n} best={best} us={us:.2f} effbw={bw:.0f}",
                  flush=True)

    print("\nSUMMARY bf16 best-variant, cell = us | effbw GB/s(权重字节口径;"
          "M>=256 为算术+访存混合,带宽仅供参考)", flush=True)
    print("M\\N      " + "".join(f"{'N=%d' % n:>28}" for n in GEMMV_NS), flush=True)
    for m in GEMMV_MS:
        cells = []
        for n in GEMMV_NS:
            best, us, bw = results[(m, n)]
            cells.append(f"{us:8.2f}/{bw:6.0f}/{best:<10}")
        print(f"{m:>5}    " + "".join(f"{c:>28}" for c in cells), flush=True)

    half_us = results[(7, 2048)][1]
    full_us = results[(7, 4096)][1]
    print(f"RATIO' half/full=0.5*{full_us:.2f}/{half_us:.2f}="
          f"{0.5 * full_us / half_us:.3f}(M=7 最好变体,TFLOPS 口径与 Task 0 同构)",
          flush=True)
    tf_half = 2.0 * 1024 * GEMMV_K * 2048 / (results[(1024, 2048)][1] * 1e-6) / 1e12
    tf_full = 2.0 * 1024 * GEMMV_K * 4096 / (results[(1024, 4096)][1] * 1e-6) / 1e12
    print(f"CONTROL prefill M=1024 half(2048)tflops={tf_half:.2f} "
          f"full(4096)tflops={tf_full:.2f}(补 Task 0 prefill 全宽对照)",
          flush=True)


def gemm_cold_main() -> None:
    """--gemm-cold 单卡模式:冷权重(36 套轮转,>72MB L2)半宽 vs 全宽 GEMM。

    解 --gemm-variants/双卡 GEMM 段的热 L2 伪象:微基准热复用单套权重,
    ≤72MB 根本没走 HBM,带宽虚高、半宽塌方被夸大(Task 0 RATIO 0.373 即
    此伪象,作废)。收编自控制器裁决测量脚本 /tmp/opencode/gemm_cold.py
    (逻辑照搬):每类形状预分配 36 套独立权重轮流使用(本轮内不重访),
    F.linear(引擎真实路径)bf16,M=7,warmup 一整轮 + torch.cuda.Event
    计 30 轮 ÷ (30×36);hot 对照 = 同形状单套复用。TP 投影假设:非 GEMM
    GPU 时间不变、allreduce 73×17.84µs 串行(--gemm-variants 轮 57KiB graph
    档)、锚 M4.1 段一 profiler 单卡 graph GPU 19.81 ms/步。"""
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 下即物理 GPU 1
    torch.manual_seed(0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def bench_linear(x, ws):
        """ws 轮流 F.linear:warmup 一整轮(也把 L2 置成"刚用过最后一张"
        状态)+ 计 GEMMCOLD_REPS 轮,返回 µs/次(总时 ÷ (reps×len(ws)))。"""
        for w in ws:
            F.linear(x, w)
        torch.cuda.synchronize()
        start.record()
        for _ in range(GEMMCOLD_REPS):
            for w in ws:
                F.linear(x, w)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / (GEMMCOLD_REPS * len(ws))

    m = 7
    print(f"GEMMCOLD M={m} bf16 F.linear; cold={GEMMCOLD_LAYERS} 权重轮转"
          f"(>72MB L2) hot=单套复用; REPS={GEMMCOLD_REPS}", flush=True)
    tot = {"full": 0.0, "half": 0.0}
    bws = []
    for name, nf, kf, nh, kh in GEMMCOLD_CLASSES:
        row = {}
        for tag, n, k in (("full", nf, kf), ("half", nh, kh)):
            x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
            ws = [torch.randn(n, k, dtype=torch.bfloat16, device=dev)
                  for _ in range(GEMMCOLD_LAYERS)]
            cold = bench_linear(x, ws)
            hot = bench_linear(x, ws[:1])
            row[tag] = (cold, hot)
            tot[tag] += cold
            bws.append((name, tag, n * k * 2 / (cold * 1e-6) / 1e9))
            del ws
            torch.cuda.empty_cache()
        print(f"GEMMCOLD {name} full={row['full'][0]:.2f} "
              f"hot_full={row['full'][1]:.2f} half={row['half'][0]:.2f} "
              f"hot_half={row['half'][1]:.2f} "
              f"ratio={row['half'][0] / row['full'][0]:.3f}", flush=True)
    print("GEMMCOLD bw(GB/s,权重字节/冷时间): "
          + " ".join(f"{name}-{tag}={gb:.0f}" for name, tag, gb in bws), flush=True)
    full_ms = tot["full"] * GEMMCOLD_LAYERS / 1000
    half_ms = tot["half"] * GEMMCOLD_LAYERS / 1000
    print(f"GEMMCOLD per-step total (36 layers): full={full_ms:.2f}ms "
          f"half={half_ms:.2f}ms ratio={half_ms / full_ms:.3f}", flush=True)
    single_gpu = 19.81
    allreduce_ms = 73 * 17.84e-3
    gemm_share = full_ms / single_gpu
    non_gemm = single_gpu * (1 - gemm_share)
    tp_gpu = half_ms + allreduce_ms + non_gemm
    print(f"GEMMCOLD projection: single_gpu={single_gpu}ms "
          f"gemm_share~{gemm_share * 100:.0f}% allreduce~{allreduce_ms:.2f}ms "
          f"-> TP2 decode GPU ~{tp_gpu:.2f}ms/step "
          f"(speedup ~{single_gpu / tp_gpu:.2f}x on GPU time)", flush=True)


def main() -> None:
    if "--gemm-variants" in sys.argv:
        gemm_variants_main()
        return
    if "--gemm-cold" in sys.argv:
        gemm_cold_main()
        return
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
