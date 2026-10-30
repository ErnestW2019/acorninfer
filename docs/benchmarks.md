# 性能基线

记录每个里程碑的吞吐数字,测速均在 RTX 4090 上(M0–M2 物理 GPU 1;M3 基线物理 GPU 0,同型号)。

| 里程碑 | 配置 | 吞吐 | 备注 |
|---|---|---|---|
| M0 | Qwen3-8B bf16,greedy,无 cache 全量重算,batch=1 | 32.90 tok/s | O(n²) oracle 路径,M1 的 KV cache 是对照基线 |
| M1 | Qwen3-8B bf16,greedy,KV cache(prefill+decode),batch=1 | 36.32 tok/s | 对比 M0 的 32.90 tok/s;上下文越长优势越大(M0 是 O(n²)) |
| M2 | Qwen3-8B bf16,greedy,in=128/out=32,分页 KV + 连续批处理 | 并发1: 36.99 tok/s;并发4: 89.06 tok/s;并发8: 110.68 tok/s | TTFT/TPOT 见下表;M1 单流 36.32 tok/s 为对照 |

测量协议(M0):CLI 报告的平均值;prompt = "The capital of France is"(6 token),`--max-new-tokens 32`;吞吐只计生成阶段(不含加载)。M1 对比时必须沿用同一协议。
M1 同协议(prompt 6 token,max-new-tokens 32,greedy,KV cache 路径)。
注意:M0/M1 两行均为**单请求 batch=1**(无并发批处理),与 M2 的并发档对比时只看并发1 档。

## M2 实测(2026-10-27,`python -m acorninfer.bench`,真机输出)

| 并发 | TTFT p50 | TPOT mean | 聚合吞吐 |
|---|---|---|---|
| 1 | 0.0346 s | 26.79 ms | 36.99 tok/s |
| 4 | 0.0802 s | 43.77 ms | 89.06 tok/s |
| 8 | 0.1441 s | 61.50 ms | 110.68 tok/s |

协议:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8`(in=128/out=32,greedy,bf16,num_blocks=1024,page=16,max_running=8,watermark=8,单张 RTX 4090,seed=20261027);每档一次性提交全部请求,TTFT=提交→首 token(含排队+prefill),TPOT=(末 token−首 token)/(n−1),聚合吞吐=总生成 token/(max(last_token_at)−min(created_at));计时前有 1 条 warmup 请求(不计入)。并发8 档有一条请求在第 2 token 命中 EOS 提前终止(总生成 226 而非 256 token),属贪心真实行为。趋势符合预期:TTFT 随并发上涨(prefill 在步内串行排队),聚合吞吐持续上升(batch-8 解摊单步前向),并发1 档 36.99 与 M1 单流 36.32 同量级互证(M0/M1 的 6-token 协议下 prefill 占比更小,128-token 之下仍同量级,说明解码吞吐没有因长 prompt 劣化)。

## M3 基线:vLLM 对照(2026-10-30,`scripts/bench_remote.py`,真机输出)

M3 验收线(引擎聚合吞吐 ≥ vLLM 的 70%,同型号卡、同权重)的对照基线。vLLM 0.32.0(pip,独立 venv `/home/ernest/vllm-baseline`,torch 2.13.0),服务命令:`CUDA_VISIBLE_DEVICES=0 vllm serve /home/ernest/model-weight/Qwen3-8B --dtype bfloat16 --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90`(物理 GPU 0,RTX 4090 24GB,与引擎测试卡同型号)。

| 并发 | TTFT p50 | TTFT mean | TPOT mean | 聚合吞吐 |
|---|---|---|---|---|
| 1 | 0.0289 s | 0.0289 s | 16.92 ms | 57.81 tok/s |
| 4 | 0.0689 s | 0.0581 s | 17.86 ms | 207.39 tok/s |
| 8 | 0.0819 s | 0.0749 s | 17.84 ms | 354.91 tok/s |

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8`(压测端为纯 HTTP 客户端,不占 GPU;in=128/out=32,temperature=0,seed=20261027)。

协议(与引擎侧 `acorninfer.bench` 对齐):同一批随机 prompt id(`_prompts` 同 seed)以 token-id 数组直投 OpenAI `/v1/completions`(completions API 原生支持,绕开文本分词往返),temperature=0、max_tokens=32、自然 EOS 停止;TTFT/TPOT/聚合吞吐公式与 `acorninfer.bench._report` 一致;计时前有 1 条 smoke 请求暖机(不计入;该 warmup 已固化进 `scripts/bench_remote.py`——1 条 8-token prompt / min(4,out) 输出的小请求,镜像引擎侧 warmup 协议,复测无需手工预热)。两条口径注记:

1. **HTTP 往返计入两端延迟**:TTFT 含客户端→vLLM 的 HTTP 上行与 vLLM 调度开销→首 token 回传全程,即产品真实路径,70% 线按此口径(对引擎侧是偏严的口径)。
2. **completion_tokens 与引擎 generated_ids 同口径(含 EOS 终止 token)**:实测 C=8 档同 seed 下同样恰有一条请求在第 2 token 命中 EOS 提前终止(总生成 226 而非 256;vLLM 侧 `finish_reason=stop`、`completion_tokens=2`),与引擎 M2 实测逐项一致(总 token 数、提前终止位置)——同权重同贪心路径的交叉验证,两侧 token 计数可直接相比。

70% 验收线(基线聚合吞吐 × 0.7,逐档;TTFT/TPOT 越低越好,不设 ×0.7 线,仅作参考):

| 并发 | vLLM 基线 | 70% 线(M3 引擎聚合吞吐须 ≥) | 引擎 M2 实测(参考) |
|---|---|---|---|
| 1 | 57.81 tok/s | 40.47 tok/s | 36.99 tok/s |
| 4 | 207.39 tok/s | 145.17 tok/s | 89.06 tok/s |
| 8 | 354.91 tok/s | 248.44 tok/s | 110.68 tok/s |

## M3 实测:四方对比与验收判定(2026-10-30,`python -m acorninfer.bench`,真机输出)

数字为 Task 9 发射合批(见下节)落地后的当日复测;vLLM 列为上文 M3 基线小节的 Task 1 实测引用,非当日重跑:

| 并发 | vLLM(引用 Task 1) | sdpa(M2 路径) | fused(K1/K2,sdpa attention) | triton+fused(K1/K2/K3,全开) | 70% 线 | triton 判定 |
|---|---|---|---|---|---|---|
| 1 | 57.81 | 35.65 | 37.42 | 50.39 | 40.47 | **✓ 线的 124.5%** |
| 4 | 207.39 | 85.66 | 88.23 | 163.91 | 145.17 | **✓ 线的 112.9%** |
| 8 | 354.91 | 109.14 | 110.04 | 246.37 | 248.44 | **✗ 线的 99.2%** |

sdpa/fused 列为优化后当日同卡实测。改动只在 triton 路径(两配置执行的 GPU 算子与改动前逐条相同,步首 prepare 仅在 triton 后端触发),其相对 Task 8 归档的波动——C=1 档 ±1% 内、C=4/C=8 档 −2.5~−3.3%——经查非代码回归:sdpa 复跑两次(85.66/109.14 与 85.65/110.93)自身稳定;同幅度的下行同样出现在改动前旧代码的当日复跑中(stash A/B,见下节 before 列:C=8 triton 157.50 vs 归档 161.60)。判定为当日机时状态整体慢 ~2-3%(C≥4 档),本表 triton 数字与 before 对照同环境实测,增益口径不受影响(如以归档环境计还略偏保守)。

引擎侧 TTFT p50 / TPOT mean(同上三配置 + vLLM 引用;格式 `TTFT / TPOT`):

| 并发 | vLLM(引用 Task 1) | sdpa | fused | triton+fused |
|---|---|---|---|---|
| 1 | 0.0289 s / 16.92 ms | 0.0352 s / 27.82 ms | 0.0353 s / 26.45 ms | 0.0375 s / 19.28 ms |
| 4 | 0.0689 s / 17.86 ms | 0.0824 s / 45.54 ms | 0.0815 s / 44.17 ms | 0.0806 s / 22.59 ms |
| 8 | 0.0819 s / 17.84 ms | 0.1473 s / 62.66 ms | 0.1508 s / 62.43 ms | 0.1455 s / 24.93 ms |

协议注记:三配置与 M2 实测/vLLM 基线同协议(in=128/out=32,greedy,bf16,num_blocks=1024,page=16,max_running=8,watermark=8,seed=20261027,计时前 1 条 warmup 不计入),统一物理 GPU 1(RTX 4090),每条命令跑前 `nvidia-smi` 确认无残留进程;命令依次为 `--concurrency 1 4 8`、`--concurrency 1 4 8 --fused-ops`、`--concurrency 1 4 8 --attention-backend triton --fused-ops`。C=8 档三配置均复现「一条请求第 2 token 命中 EOS 提前终止(总 226/256 token)」,与 M2/vLLM 两侧既有记录逐项一致。triton 配置复跑一次验证稳定性:50.58/160.61/245.83(C=1/C=4 ±2% 内,C=8 ±0.2%),数字非单次涨落。

**验收判定:C=1/C=4 两档过线;C=8 距线 0.8%,两次复跑均未过(246.37/245.83 vs 248.44),M3 性能门仍未全过。** 按预案不放宽门槛、不为达标改码;归因(吞吐/TPOT 差为实测;成因标注为推断处见文中):

- triton+fused 相对 sdpa 增益进一步拉大:吞吐 +41.3%(C=1)/+91.3%(C=4)/+125.7%(C=8);TPOT −30.7%/−50.4%/−60.2%。增益主体来自 K3 + Task 9 发射合批(机制见下节,before/after 已同环境实测量化);
- C=8 档剩余缺口:vLLM TPOT 17.84ms vs 本引擎 24.93ms,差 ~7ms/步。原首要归因「`build_tables` 每层每步重建 + 逐序列 write 发射」已被 Task 9 消除并量化(下节 before/after);剩余部分按代码审查推断为:decode 步仍无 CUDA graph(28 层 × 每层 ~十几个算子的逐 kernel 发射间隙)、K3 kernel 本体相对 vLLM paged attention 的效率差、以及 GEMV 小批量形状的利用率——均未单独计时(推断,未单独计时;量化归因需 profiler,属下一里程碑);
- 预案参考口径(相对 M2 gather 版 ≥2×):C=8 triton 246.37 vs M2 gather 版 110.68 = **2.23×,达标**。

### Task 9 发射合批(2026-10-30,decode 步 CPU 发射开销削减)

改动(triton 路径;torch 路径 byte-identical,pager 仅新增 API):

- `PagedCacheView.ensure_append_capacity()`:步首对每个 view 一次容量确保——跨页序列(seq_len % page_size == 0)在此分新页追加进块表,否则 no-op;返回各序列追加位 (page_id, slot_id)。BlockExhausted 语义与 write 一致(已分配页留表、_lens 未提交,单测钉住);
- `PagedKVCache.scatter_decode()`:每层单次高级索引 scatter(k/v 各一次发射)+ 逐 view `_lens` 提交;位值语义 ≡ 逐序列 `write()`(单测逐位对拍钉住);
- `forward_decode` 顶层步首一次构建块表/长度(lens = pre-write + 1,与旧层内 `build_tables` 在 layer-0 写后取值逐位相同)经 kwargs 穿透 Block → Attention;attention triton 分支消费传入值,None 时回退自建(独立可用性保持)。

消除的固定开销:原每层每步 28×(`build_tables` CPU 构造 + H2D + 同步,加 B 次逐序列 write 小算子),改为每步 1 次 prepare + 每层 2 次发射。

优化前后同配置对照(triton+fused;before 为改动前代码当日 stash 复跑,同机同环境;after 为上表主跑;均实测):

| 并发 | before 吞吐 | after 吞吐 | Δ 吞吐 | before TPOT | after TPOT | Δ TPOT |
|---|---|---|---|---|---|---|
| 1 | 40.53 | 50.39 | +24.3% | 24.35 ms | 19.28 ms | −20.8% |
| 4 | 115.39 | 163.91 | +42.1% | 33.17 ms | 22.59 ms | −31.9% |
| 8 | 157.50 | 246.37 | +56.4% | 41.75 ms | 24.93 ms | −40.3% |

before 复跑与 Task 8 归档(40.35/116.30/161.60)一致性:C=1/C=4 ±1% 内,C=8 −2.5%,与本节 sdpa/fused 漂移同因(机时状态)。
