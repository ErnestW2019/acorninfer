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
- C=8 档剩余缺口:vLLM TPOT 17.84ms vs 本引擎 24.93ms,差 ~7ms/步。原首要归因「`build_tables` 每层每步重建 + 逐序列 write 发射」已被 Task 9 消除并量化(下节 before/after);剩余部分按代码审查推断为:decode 步仍无 CUDA graph(36 层 × 每层 ~十几个算子的逐 kernel 发射间隙)、K3 kernel 本体相对 vLLM paged attention 的效率差、以及 GEMV 小批量形状的利用率——均未单独计时(推断,未单独计时;量化归因需 profiler,属下一里程碑);
- 预案参考口径(相对 M2 gather 版 ≥2×):C=8 triton 246.37 vs M2 gather 版 110.68 = **2.23×,达标**。

### Task 9 发射合批(2026-10-30,decode 步 CPU 发射开销削减)

改动(triton 路径;torch 路径 byte-identical,pager 仅新增 API):

- `PagedCacheView.ensure_append_capacity()`:步首对每个 view 一次容量确保——跨页序列(seq_len % page_size == 0)在此分新页追加进块表,否则 no-op;返回各序列追加位 (page_id, slot_id)。BlockExhausted 语义与 write 一致(已分配页留表、_lens 未提交,单测钉住);
- `PagedKVCache.scatter_decode()`:每层单次高级索引 scatter(k/v 各一次发射)+ 逐 view `_lens` 提交;位值语义 ≡ 逐序列 `write()`(单测逐位对拍钉住);
- `forward_decode` 顶层步首一次构建块表/长度(lens = pre-write + 1,与旧层内 `build_tables` 在 layer-0 写后取值逐位相同)经 kwargs 穿透 Block → Attention;attention triton 分支消费传入值,None 时回退自建(独立可用性保持)。

消除的固定开销:原每步 36 层 ×(build_tables CPU 构造 + H2D + 同步,加 B 次逐序列 write 小算子),改为每步 1 次 prepare + 每层 2 次发射。

优化前后同配置对照(triton+fused;before 为改动前代码当日 stash 复跑,同机同环境;after 为上表主跑;均实测):

| 并发 | before 吞吐 | after 吞吐 | Δ 吞吐 | before TPOT | after TPOT | Δ TPOT |
|---|---|---|---|---|---|---|
| 1 | 40.53 | 50.39 | +24.3% | 24.35 ms | 19.28 ms | −20.8% |
| 4 | 115.39 | 163.91 | +42.1% | 33.17 ms | 22.59 ms | −31.9% |
| 8 | 157.50 | 246.37 | +56.4% | 41.75 ms | 24.93 ms | −40.3% |

before 复跑与 Task 8 归档(40.35/116.30/161.60)一致性:C=1/C=4 ±1% 内,C=8 −2.5%,与本节 sdpa/fused 漂移同因(机时状态)。

### 验收判定(2026-10-30,用户裁定收官)

M3 收官:C=1 ✓(线的 124.5%)、C=4 ✓(线的 112.9%)、C=8 以线的 99.2%(246.37 vs 248.44)带缺口收官;spec §3 预案参考「相对 M2 gather 版 ≥2×」达成(2.23×)。C=8 剩余缺口首要归因指向无 CUDA graph 的逐 kernel 发射间隙(推断,未单独计时),列为 M4 首项;量化归因需 profiler,随 M4 一并处理。

## M4 实测:decode 步 profiler 账目(2026-10-30,`scripts/profile_decode.py`,真机输出)

M4(CUDA graph 消发射间隙)的开工基线:8 序列(in=128,seed=20261027,与 bench 同 seed)prefill 完成后空跑 3 步出稳定,再以 torch.profiler(CPU+CUDA activity,独立预热窗吸收 CUPTI 初始化)连测 20 个 decode 步(temp=0,triton+fused 后端,num_blocks=1024,page=16,物理 GPU 1,RTX 4090)。命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py --steps 20`。

**decode 步的时间去向自此为实测**(上节 M3 收官中「无 CUDA graph 的逐 kernel 发射间隙」一项由推断升级为下表数字):

| 指标 | 实测值 |
|---|---|
| 每步墙钟 wall/step | 27.37 ms |
| GPU 端累计(kernel+memcpy)/step | 19.99 ms |
| 发射间隙 gap/step(wall − GPU 端) | 7.38 ms |
| kernel 启动次数/step | 1430.0 |
| H2D/D2H memcpy 次数/step | 13.00 |

脚本原始输出行(逐字):`steps=20 wall/step=27.37ms gpu_kernel+memcpy/step=19.99ms gap/step=7.38ms`;`launches/step=1430.0 memcpy/step=13.00`。gap 占每步墙钟的 27.0%(7.38/27.37)——即 CUDA graph 对 decode 步墙钟的理论收益上限(实测占比)。

top kernel(按设备时间,20 步累计 Self CUDA / 折合每步 / 占 GPU 忙时占比,数字取自 profiler 表):

| kernel | Self CUDA(20 步) | 每步 | 次数/步 | 占比 |
|---|---|---|---|---|
| cutlass_80_wmma_tensorop_bf16(GEMM) | 354.041 ms | 17.70 ms | 253 | 88.55% |
| _paged_attn_split_kernel | 11.796 ms | 0.59 ms | 36 | 2.95% |
| unrolled_elementwise_kernel | 5.649 ms | 0.28 ms | 183 | 1.41% |
| index_elementwise_kernel | 4.468 ms | 0.22 ms | 72 | 1.12% |
| elementwise_kernel<128, 2> | 3.987 ms | 0.20 ms | 146 | 1.00% |
| reduce_kernel | 2.987 ms | 0.15 ms | 73 | 0.75% |
| _fused_add_rmsnorm_kernel | 2.473 ms | 0.12 ms | 72 | 0.62% |
| _rope_inplace_kernel | 1.790 ms | 0.09 ms | 72 | 0.45% |

其余为长尾小 elementwise/reduce kernel,单项占比均 <1%;GPU 忙时合计 399.810 ms(20 步)。GEMM 一项独占 GPU 忙时的 88.55%(每步 253 次 = 36 层 × 7 次小矩阵乘 + lm_head),decode 是典型 GEMV 形状、带宽受限——graph 化消除的是 27.0% 的发射间隙,GEMM 本体的利用率是 M4 之后的下一瓶颈(此句为对实测数字的解读,非新增实测)。

口径注记(均为脚本行为/实测事实):

1. **GPU 忙时只按 device_type=CUDA 事件累加**(torch 2.13:`key_averages()` 事件的属性名为 `self_device_time_total`,旧名 `self_cuda_time_total` 在 FunctionEventAvg 上已不存在,实测 AttributeError;`table(sort_by=...)` 统一用新名),与 profiler 表格 footer「Self CUDA time total: 399.810ms」同口径;表格中 CPU 算子行(如 aten::mm 行 Self CUDA 355.728ms)是子 kernel 时间归因,与 kernel 行双计,不入账。
2. **batch=7 非 8**:同 seed 下与 M2/M3 C=8 档既有记录一致,1 条请求在第 2 个生成 token 命中 EOS,于预热期退役;计时窗口内 batch 恒为 7(脚本 sanity 行逐字:`sanity: alive=7/8 generated_per_seq=25 (batch 恒定贯穿计时窗口)`;窗口内若再有提前退役,脚本断言直接失败)。
3. **27.37 ms/步含 profiler 逐 kernel 采样开销**(对照:M3 无 profiler 同配置 C=8 TPOT 24.93 ms,见上节引用);本节数字用于 kernel/gap 结构归因,绝对吞吐以 M3 节为准。

## M4 T6 判定:kernel 调参按触发条件跳过(2026-10-31,`scripts/profile_decode.py --cuda-graph`,真机账目)

T6(kernel 微调)预设双触发条件,须同时成立方开工:① decode GPU 忙时中 kernel 占比 ≥20%;② 引擎距 vLLM 100% 线 ≤10%。当日实测:① 成立(kernel 占比 73%,当日 profiler 账目);② 不成立(今晨 preview C=8 档 254.55 vs vLLM 354.91,距 100% 线 28.3% > 10%)→ **按计划跳过 kernel 调参,不为达标改码**。

graph 路径 profiler 账目(2026-10-31,同上节协议,物理 GPU 1),与 eager(上节 2026-10-30 账目引用,非当日重跑)对照:

| 指标 | eager(10-30) | graph(当日) |
|---|---|---|
| 每步墙钟 wall/step | 27.37 ms | 21.64 ms |
| GPU 端累计(kernel+memcpy)/step | 19.99 ms | 19.84 ms |
| 发射间隙 gap/step(wall − GPU 端) | 7.38 ms | 1.80 ms |

gap 由 7.38 ms 降至 1.80 ms,graph 回收发射间隙的 75.6%(残差占 graph 墙钟的 8.3%)。结论:decode GPU 侧 ≈ 平价(graph 19.84 ms vs eager 19.99 ms),GPU 时间由 cuBLAS GEMM 主导(占 89%,上节账目)——graph 化只回收发射间隙,不改 GPU 侧工作量;当日残差归因图外 CPU(残余 gap ~1.8 ms/步)与 prefill 串行化(C=8 档本引擎 TTFT p50 0.1467 s vs 当日配对 vLLM 0.0817 s,调度器 prefill 单序列推进所致,推断,见下节)。

## M4 实测:当日配对 A/B 验收(2026-10-31,`scripts/bench_remote.py` + `python -m acorninfer.bench`,真机输出)

今晨中间实测(preview,graph on,物理 GPU 1,分母为 2026-10-30 归档 vLLM 线):C=1 53.97 tok/s(93.4%)、C=4 172.18 tok/s(83.0%)、C=8 254.55 tok/s(71.7%)。验收不用归档分母,以下为同日配对复测——vLLM 起服、三轮压测、关服在同一工作时段内完成(01:24:44 起服,01:25:23 健康探活 200,末轮压测完成 01:27:12 后随即关服),消机时漂移。聚合吞吐(tok/s):

| 并发 | vLLM(当日配对) | 本引擎 graph | 本引擎 eager | graph vs eager | 达标率(graph ÷ 当日 vLLM) |
|---|---|---|---|---|---|
| 1 | 58.06 | 53.93 | 50.69 | +6.4% | **92.9%,未过 100%** |
| 4 | 208.15 | 172.77 | 163.79 | +5.5% | **83.0%,未过 100%** |
| 8 | 354.92 | 256.09 | 246.59 | +3.9% | **72.2%,未过 100%** |

TTFT p50 / TPOT mean(格式 `TTFT / TPOT`):

| 并发 | vLLM(当日配对) | graph | eager |
|---|---|---|---|
| 1 | 0.0270 s / 16.91 ms | 0.0355 s / 18.00 ms | 0.0348 s / 19.24 ms |
| 4 | 0.0661 s / 17.81 ms | 0.0822 s / 21.25 ms | 0.0812 s / 22.59 ms |
| 8 | 0.0817 s / 17.89 ms | 0.1467 s / 23.95 ms | 0.1457 s / 24.89 ms |

命令(照 M3 基线格式)。服务(物理 GPU 0,与 M3 基线记录逐字一致):`CUDA_VISIBLE_DEVICES=0 /home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --dtype bfloat16 --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90`(nohup 后台,前台轮询 `/health` 至 200)。压测三连(物理 GPU 1,每条前台执行;seed 默认 20261027,与 M3 同):

- vLLM:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8`
- graph:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 --attention-backend triton --fused-ops --cuda-graph`
- eager:同上,去掉 `--cuda-graph`

RESULT 行逐字(stdout):

- vLLM:`RESULT-VLLM C=1 ttft_p50=0.0270s ttft_mean=0.0270s tpot_mean=16.91ms tok_s=58.06` / `RESULT-VLLM C=4 ttft_p50=0.0661s ttft_mean=0.0570s tpot_mean=17.81ms tok_s=208.15` / `RESULT-VLLM C=8 ttft_p50=0.0817s ttft_mean=0.0749s tpot_mean=17.89ms tok_s=354.92`
- graph:`RESULT C=1 ttft_p50=0.0355s ttft_mean=0.0355s tpot_mean=18.00ms tok_s=53.93` / `RESULT C=4 ttft_p50=0.0822s ttft_mean=0.0821s tpot_mean=21.25ms tok_s=172.77` / `RESULT C=8 ttft_p50=0.1467s ttft_mean=0.1467s tpot_mean=23.95ms tok_s=256.09`
- eager:`RESULT C=1 ttft_p50=0.0348s ttft_mean=0.0348s tpot_mean=19.24ms tok_s=50.69` / `RESULT C=4 ttft_p50=0.0812s ttft_mean=0.0811s tpot_mean=22.59ms tok_s=163.79` / `RESULT C=8 ttft_p50=0.1457s ttft_mean=0.1458s tpot_mean=24.89ms tok_s=246.59`

口径注记(均为实测事实):

1. **机时漂移核对**:当日 vLLM 对 10-30 归档漂移 +0.4%/+0.4%/+0.0%(58.06/57.81、208.15/207.39、354.92/354.91),远小于 M3 节记录的 ±2-3% 机时漂移带,配对口径成立;本引擎 graph 当日数字与今晨 preview 差 ≤0.6%(53.93/53.97、172.77/172.18、256.09/254.55)。
2. **C=8 档提前 EOS 复现**:本引擎 graph/eager 两列 out/token-stats 均 `min=2 max=32 / 32`(总 226/256 token,一条请求第 2 token 命中 EOS),与 M2/M3/vLLM 侧既有记录逐项一致;TPOT 按 n>1 的 8/8 序计算。
3. **GPU 分工与清理**:起服前双卡 ~4 MiB / 0%;vLLM 独占物理 GPU 0、引擎压测独占物理 GPU 1;关服后 GPU 0 回落 4 MiB,`nvidia-smi --query-compute-apps` 为空(无残留计算进程)。

**验收判定(逐档 ≥100% 当日 vLLM 与否,如实):三档均未过——C=1 线的 92.9%、C=4 线的 83.0%、C=8 线的 72.2%。** graph 相对 eager 的当日净增益为 +6.4%/+5.5%/+3.9%,与 T6 账目(回收 75.6% 发射间隙、GPU 侧平价)相互印证;C=8 残余缺口主体仍在 GPU 侧 GEMM 利用率与 prefill 串行化(TTFT 0.1467 vs 0.0817 s)——此句为对实测数字的归因推断。

## M4.1 段一(图外残差)中间账(2026-10-31,`python -m acorninfer.bench` + `scripts/profile_decode.py`,真机输出)

段一改动(HEAD = eb7442b):decode 步 argmax 合批——B 次逐序列 D2H 同步 → 1 次(M4.1 Task 1);步首输入扁平缓冲 + pinned CPU 镜像——每 dtype 单发 H2D,共 2 次 H2D/步(M4.1 Task 2)。本节为两改动落地后的当日复测(物理 GPU 1,RTX 4090;每条命令跑前 `nvidia-smi` 确认无残留进程;seed 与 M2–M4 既有记录同)。

### bench 双跑(graph + eager,C=1 4 8)

命令(照上节 M4 当日配对 A/B 验收配方):

- graph:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 --attention-backend triton --fused-ops --cuda-graph`
- eager:同上,去掉 `--cuda-graph`

聚合吞吐(tok/s)当日 vs M4 验收当日(2026-10-31 配对)对照:

| 并发 | graph 当日 | graph(10-31 验收) | eager 当日 | eager(10-31 验收) | graph vs eager(当日) |
|---|---|---|---|---|---|
| 1 | 54.12 | 53.93 | 50.62 | 50.69 | +6.9% |
| 4 | 173.60 | 172.77 | 162.54 | 163.79 | +6.8% |
| 8 | 257.75 | 256.09 | 245.89 | 246.59 | +4.8% |

graph 三档较验收当日 +0.35%/+0.48%/+0.65%;eager 三档 −0.14%/−0.76%/−0.28%,在 M3 节记录的 ±2-3% 机时漂移带内,原样记录不复跑择优。RESULT 行逐字(stdout):

- graph:`RESULT C=1 ttft_p50=0.0349s ttft_mean=0.0349s tpot_mean=17.95ms tok_s=54.12` / `RESULT C=4 ttft_p50=0.0822s ttft_mean=0.0821s tpot_mean=21.13ms tok_s=173.60` / `RESULT C=8 ttft_p50=0.1469s ttft_mean=0.1468s tpot_mean=23.68ms tok_s=257.75`
- eager:`RESULT C=1 ttft_p50=0.0347s ttft_mean=0.0347s tpot_mean=19.27ms tok_s=50.62` / `RESULT C=4 ttft_p50=0.0827s ttft_mean=0.0830s tpot_mean=22.73ms tok_s=162.54` / `RESULT C=8 ttft_p50=0.1488s ttft_mean=0.1488s tpot_mean=24.88ms tok_s=245.89`

C=8 档两配置 out/token-stats 均 `min=2 max=32 / 32`(总 226/256 token,一条请求第 2 token 命中 EOS),与 M2/M3/M4 侧既有记录逐项一致。

### profiler 双账(同 M4/T6 协议,--steps 20)

命令(照 M4 节/T6 节配方):

- graph:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py --steps 20 --cuda-graph`
- eager:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py --steps 20`

| 指标 | eager(10-30 归档) | eager(当日) | graph(10-31 T6) | graph(当日) |
|---|---|---|---|---|
| 每步墙钟 wall/step | 27.37 ms | 26.61 ms | 21.64 ms | 21.50 ms |
| GPU 端累计(kernel+memcpy)/step | 19.99 ms | 19.97 ms | 19.84 ms | 19.81 ms |
| 发射间隙 gap/step(wall − GPU 端) | 7.38 ms | 6.64 ms | 1.80 ms | 1.70 ms |
| kernel 事件次数/step | 1430.0 | 1418.0 | 未归档 | 1418.0 |
| memcpy 次数/step | 13.00 | 7.00 | 未归档 | 3.00 |

注:表内 kernel/memcpy 两行标签较 M4 表改「事件」,以反映 GPU 侧计数口径(M4 表原标签「kernel 启动次数」「H2D/D2H memcpy 次数」)。

脚本原始输出行(逐字):graph `steps=20 wall/step=21.50ms gpu_kernel+memcpy/step=19.81ms gap/step=1.70ms`;`launches/step=1418.0 memcpy/step=3.00`。eager `steps=20 wall/step=26.61ms gpu_kernel+memcpy/step=19.97ms gap/step=6.64ms`;`launches/step=1418.0 memcpy/step=7.00`。两轮 sanity 行均逐字 `sanity: alive=7/8 generated_per_seq=25 (batch 恒定贯穿计时窗口)`(batch=7 非 8,同 M4 口径注记 2);GPU 忙时合计(20 步,Self CUDA time total)graph 396.130 ms / eager 399.331 ms。launches/memcpy 按 GPU 侧事件计,graph 模式下不反映 CPU 发射数——图内 kernel 仍逐条入账,计数与 eager 同源。

段一改动在当日账目上的实测映照(计数对照):

- eager memcpy/step 13.00 → 7.00(−6.00):与 argmax 合批的 D2H 数目变化吻合(B=7 逐序列 D2H → 1 次);
- eager kernel 事件/step 1430.0 → 1418.0(−12.0):与 argmax 逐序列 → 单次合批的 kernel 次数精简方向一致;
- graph memcpy/step = 3.00:与段一改造后的每步传输组成(2 H2D + 1 D2H)逐项一致;
- graph 与 eager 当日 kernel 事件数同为 1418.0:graph 回放的 kernel 集与 eager 发射的相同。

### 残差判定(graph gap)

**gap 实测 1.70 ms/步,段一目标 ≤1 ms(M4.1 计划预设门)未达**(对照 T6 账 1.80 ms:−0.10 ms;占当日 graph 墙钟的 7.9%(1.70/21.50);当日 graph 相对 eager 回收发射间隙的 74.4%((6.64−1.70)/6.64))。残差构成,实测锚点与推断分别标注:

1. 实测:每步 GPU 侧 memcpy 恰 3 次(2 H2D + 1 D2H);argmax 合批后的 1 次 D2H(`logits.argmax().tolist()`)是 decode_batch 路径内唯一的设备同步点——同步返回后 GPU 队列为空,至下一次 `graph.replay()` 之间的全部图外 CPU 工作直接暴露为 gap;
2. 实测:同日 eager gap 6.64 ms → graph 1.70 ms,差值 4.94 ms 为逐 kernel CPU 发射路径被 graph 消除的部分(eager 当日账内 cuLaunchKernelEx Self CPU 13.561 ms/20 步 = 0.678 ms/步、216 次/步,为该路径的实测组成之一);
3. **推断(未单独计时)**:图外 CPU 每步工作清单(`graph_runner.decode_batch` 代码路径可数):`ensure_append_capacity` 逐 view 循环;CPU 镜像填充(4+1 次 `torch.tensor` 构造 + pinned 切片赋值,`pin_tables.zero_()` + `build_tables` CPU 直填,padding 复制压回至 3 次);`_commit_lens` 逐 view × 逐层 python 循环(B=7 × 36 层 = 252 次赋值/步);调度 python。何者主导为推断。

口径注记(均为实测事实):

1. eager wall/step 含 profiler 逐 kernel 采样开销(M4 口径注记 3 沿用);eager gap 7.38 → 6.64 ms(−0.74 ms)与 argmax 同步减少方向一致,但采样事件数同变(1430.0 → 1418.0),不作精确归因。
2. graph 模式 wall/step 21.50 ms 与 T6 账 21.64 ms 基本持平,段一两改动(同步 7→1、镜像单发 H2D)在 gap 上的当日净变化为 1.80 → 1.70 ms;bench 侧当日 graph 三档净增益仍为正(见上表)。

段二(批量 prefill)与最终当日配对账见下方「M4.1 段二(批量 prefill)+ 最终当日配对」节。

## M4.1 段二(批量 prefill)+ 最终当日配对(2026-10-31,`scripts/bench_remote.py` + `python -m acorninfer.bench`,真机输出)

段二改动(HEAD = 3f98e1b):模型层 `forward_prefill_batch`——右 padding 合批、逐行 attention(M4.1 Task 4)+ `Executor.prefill_batch`——[B, Lmax] 一次前向、逐行 `_sample`(采样语义与单序列不变)+ Engine `step()` N>1 分流:同批多序列一次批量 prefill;N=1 仍走原 `prefill` 路径,byte-identical(零回归面)。本节为段二落地后的最终当日配对验收:vLLM 起服、三腿压测、关服在同一工作时段内完成(健康探活 200 于 15:14:00,vLLM 腿压测 15:14:03–15:14:06,graph 腿 15:14:15–15:15:05,eager 腿 15:15:19–15:16:09,末轮完成随即关服),消机时漂移。聚合吞吐(tok/s):

| 并发 | vLLM(当日配对) | 本引擎 graph | 本引擎 eager | graph vs eager | 达标率(graph ÷ 当日 vLLM) | M4 验收达标率 | 较 M4 验收 |
|---|---|---|---|---|---|---|---|
| 1 | 58.01 | 54.17 | 50.84 | +6.6% | **93.4%,未过 100%** | 92.9% | +0.5 pp |
| 4 | 207.43 | 187.13 | 176.38 | +6.1% | **90.2%,未过 100%** | 83.0% | +7.2 pp |
| 8 | 355.39 | 296.76 | 283.16 | +4.8% | **83.5%,未过 100%** | 72.2% | +11.3 pp |

达标率算术(graph ÷ 当日 vLLM):C=1 54.17/58.01 = 93.4%;C=4 187.13/207.43 = 90.2%;C=8 296.76/355.39 = 83.5%。较 M4 验收(2026-10-31 配对,92.9%/83.0%/72.2%):+0.5 / +7.2 / +11.3 pp,逐档距 100% 线尚余 6.6% / 9.8% / 16.5%。

TTFT p50 / TPOT mean(格式 `TTFT / TPOT`):

| 并发 | vLLM(当日配对) | graph | eager |
|---|---|---|---|
| 1 | 0.0273 s / 16.92 ms | 0.0352 s / 17.92 ms | 0.0347 s / 19.18 ms |
| 4 | 0.0684 s / 17.87 ms | 0.0776 s / 19.56 ms | 0.0766 s / 20.94 ms |
| 8 | 0.0809 s / 17.83 ms | 0.1454 s / 19.95 ms | 0.1440 s / 21.12 ms |

**C=8 TTFT 对照(批量 prefill 是否兑现,如实)**:graph TTFT p50 0.1454 s vs 当日 vLLM 0.0809 s——M4 当日为 0.1467 vs 0.0817 s,今值 −1.3 ms(0.1454−0.1467),基本持平,仍为 vLLM 的 1.80×(0.1454/0.0809;M4 当日同为 1.80×)。**判读:批量 prefill 在 TTFT p50 维度未兑现**(实测依据:graph C=8 TTFT p50 段一/M4 验收/当日三账 0.1469/0.1467/0.1454 s 基本一线)。兑现处不在 TTFT 而在 TPOT(当日实测):graph C=8 TPOT 19.95 ms vs M4 验收日 23.95(−4.00 ms)、段一日 23.68(−3.73 ms);eager C=8 21.12 vs 24.89/24.88(−3.77/−3.76 ms);C=4 graph 21.25 → 19.56 ms。机制判读(推断,实测锚点标注):旧路径一步内逐序列串行 prefill(段二前 engine 代码路径可数),早 prefill 序列的首 token 提前落账、其第 2 token 却要等串行 prefill 全部结束,TPOT 被该等待垫高;今 C=8 TPOT mean 19.95 ms 已收敛到段一节 profiler 账纯 decode GPU 侧 19.8 ms/步的同量级(23.95(M4)/23.68(段一) → 19.95),旧路径 TPOT 高出纯 decode GPU 时 ~4 ms 的垫高消失;C=8 档当日吞吐 +15.1%(296.76/257.75)的主体即来自此项。

RESULT 行逐字(stdout):

- vLLM:`RESULT-VLLM C=1 ttft_p50=0.0273s ttft_mean=0.0273s tpot_mean=16.92ms tok_s=58.01` / `RESULT-VLLM C=4 ttft_p50=0.0684s ttft_mean=0.0578s tpot_mean=17.87ms tok_s=207.43` / `RESULT-VLLM C=8 ttft_p50=0.0809s ttft_mean=0.0742s tpot_mean=17.83ms tok_s=355.39`
- graph:`RESULT C=1 ttft_p50=0.0352s ttft_mean=0.0352s tpot_mean=17.92ms tok_s=54.17` / `RESULT C=4 ttft_p50=0.0776s ttft_mean=0.0776s tpot_mean=19.56ms tok_s=187.13` / `RESULT C=8 ttft_p50=0.1454s ttft_mean=0.1454s tpot_mean=19.95ms tok_s=296.76`
- eager:`RESULT C=1 ttft_p50=0.0347s ttft_mean=0.0347s tpot_mean=19.18ms tok_s=50.84` / `RESULT C=4 ttft_p50=0.0766s ttft_mean=0.0766s tpot_mean=20.94ms tok_s=176.38` / `RESULT C=8 ttft_p50=0.1440s ttft_mean=0.1440s tpot_mean=21.12ms tok_s=283.16`

口径注记(均为实测事实):

1. **机时漂移核对**:当日 vLLM 对 M4 验收当日漂移 −0.1%/−0.3%/+0.1%(58.01/58.06、207.43/208.15、355.39/354.92),在 M3 节记录的 ±2-3% 机时漂移带内,配对口径成立;本引擎 graph 当日对段一日 +0.1%/+7.8%/+15.1%(54.17/54.12、187.13/173.60、296.76/257.75)——C=1 在漂移带内,C=4/C=8 为段二改动的当日净变化,且 eager 同幅受益(+8.5%/+15.2%:176.38/162.54、283.16/245.89),收益来自 prefill/调度路径而非 graph 本身(判读)。
2. **C=8 档提前 EOS 复现**:graph/eager 两列 out/token-stats 均 `min=2 max=32 / 32`(总 226/256 token,一条请求第 2 token 命中 EOS),与 M2/M3/M4/段一侧既有记录逐项一致;TPOT 按 n>1 的 8/8 序计算。
3. **GPU 分工与清理**:起服前双卡 ~4 MiB / 0 进程;vLLM 独占物理 GPU 0、引擎压测独占物理 GPU 1;关服后 GPU 0 回落 4 MiB,`nvidia-smi --query-compute-apps` 为空(无残留计算进程)。收服过程注记:起服 `$!` 记到的是 nohup 包装进程,kill 后实际 serve 进程(PID 2507157,ps 反查)仍驻留至二次定向 kill——此后 GPU 0 即回落,无残留。

**验收判定(逐档 ≥100% 当日 vLLM 与否,如实):三档均未过——C=1 线的 93.4%、C=4 线的 90.2%、C=8 线的 83.5%。** 段二(批量 prefill)当日兑现 C=4/C=8 两档(较 M4 验收 +7.2/+11.3 pp),C=1 持平(+0.5 pp);graph 相对 eager 的当日净增益 +6.6%/+6.1%/+4.8%,与段一日(+6.9%/+6.8%/+4.8%)同带;C=8 残余缺口主体仍在 GPU 侧 GEMM 利用率与 TTFT 维度(0.1454 vs 0.0809 s,上节 profiler 账 GEMM 占 GPU 忙时 88.55%)——此句为对实测数字的归因推断。北极星「逐档 ≥100%」维持挂账,`m4-acceptance` tag 本日不打。

回归(当日,HEAD = 3f98e1b,任务仅改 docs 零代码变动):`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -m slow -q` → `17 passed, 106 deselected in 501.66s`,含 test_golden_m0–m4 全部 golden,零换题;fast 套件(默认 `-m 'not slow'`)当日复跑 `106 passed`(首跑出现 1 例 `test_server.py::test_cancelled_request_cleans_up` 失败,单测隔离复跑通过、全套件复跑全绿,判定偶发,如实记)。

命令(照 M4 节当日配对 A/B 验收配方)。服务(物理 GPU 0,与 M3/M4 基线记录逐字一致):`CUDA_VISIBLE_DEVICES=0 /home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --dtype bfloat16 --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90`(nohup 后台,前台轮询 `/health` 至 200)。压测三连(物理 GPU 1,每条前台执行;seed 默认 20261027,与 M3/M4 同):

- vLLM:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8`
- graph:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 --attention-backend triton --fused-ops --cuda-graph`
- eager:同上,去掉 `--cuda-graph`

## M5 段 0 尖刀:NCCL allreduce / 半宽 GEMM / prefill 拆账(2026-10-31,`scripts/phase0_spike.py` + `scripts/profile_prefill.py`,真机输出)

M5(chunked prefill + TP=2)开工前三决定点的实测。拓扑背景:本机 GPU0-GPU1 = NODE 拓扑(同 NUMA 走 PCIe 主桥,无 NVLink);torch 2.13.0+cu130(NCCL 随 torch 内置)。当日每腿跑前 `nvidia-smi` 确认无残留进程;HEAD = 96218e0。裁决输入仅为数据,判定行不在本节(留给控制器)。

### allreduce 微基准(双卡,CUDA graph 捕获)

命令:`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --nproc-per-node 2 scripts/phase0_spike.py`(本仓 venv 无 torchrun 入口脚本,`-m torch.distributed.run` 为其模块入口等价;跑前双卡 4 MiB / 0 进程)。nccl backend,bf16 一维张量;每档 eager warmup 10 + 计 50 次取均值,graph 侧 CUDAGraph 捕获 50 次 allreduce(侧 stream warmup 3 次 + 全员 barrier 对齐),replay 2 次热身后计时 1 次 replay ÷50。数字为 rank0(rank1 同档差 <1%,顺手核对一致)。

| 尺寸(B) | eager(µs/次) | graph(µs/次) | 有效带宽 eager(GB/s,简化口径) |
|---|---|---|---|
| 8 192 | 18.14 | 11.53 | 0.9 |
| 58 368(57 KiB) | 18.66 | 17.84 | 6.3 |
| 1 048 576(1 MiB) | 92.32 | 94.97 | 22.7 |
| 8 388 608(8 MiB) | 567.70 | 570.55 | 29.6 |

stdout 行逐字(rank0):`ALLOTSIZE=8192 eager=18.14 graph=11.53 bw_eager=0.9` / `ALLOTSIZE=58368 eager=18.66 graph=17.84 bw_eager=6.3` / `ALLOTSIZE=1048576 eager=92.32 graph=94.97 bw_eager=22.7` / `ALLOTSIZE=8388608 eager=567.70 graph=570.55 bw_eager=29.6`。

口径注记(均为实测事实):

1. **四档 graph 捕获全部成功,零 GRAPH_CAPTURE_FAIL**(M4 decode graph 捕获模型前向、TP=2 allreduce 须能入 graph 的前提,当日实测成立)。
2. 有效带宽 = size×2/延迟(读+写两笔,简化口径;ring 系数不较真)。计划参考值「8MB 在 ~25GB/s 下 ~0.3ms/次」为推算非实测,当日 8 MiB 档实测 0.568 ms(eager)/ 0.571 ms(graph)。
3. 稳定性:同日首跑(收尾挂死前数据已全部打印)四档 eager/graph 19.15/10.52、16.61/17.88、93.88/93.75、569.47/572.39,两轮差 <2%;57 KiB eager 档例外(16.61→18.66,±12%,小尺寸档噪声带)。
4. **首跑发现:graph 捕获后 NCCL 进程组收尾路径(barrier/destroy_process_group)挂死在 CudaEventDestroy**(watchdog 480 s 无进展报警后终止进程,两 rank 同;错误行逐字:`ProcessGroupNCCL's watchdog got stuck for 480 seconds without making progress in monitoring enqueued collectives`、`Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang`)。脚本已绕行(捕获图保活到结尾 + 数据行 flush 后 `os._exit(0)`),复跑干净退出。此为环境事实,TP server 生命周期(M5 后续任务)设计时须知。
5. 位置参考(算术,非实测):decode 每步 72 次 allreduce(计划背景数字),若全为 57 KiB 档且全串行,按 graph 档 17.84 µs/次 ≈ 1.28 ms/步。

### 半宽 GEMM 效率(双卡同测,rank0 为准)

| 形状(M×K×N) | 档位 | TFLOPS |
|---|---|---|
| 7×4096×2048 | decode 半宽 | 4.28 |
| 7×4096×3072 | decode 半宽 | 3.44 |
| 7×4096×4096 | decode 全宽对照 | 11.46 |
| 1024×4096×2048 | prefill 半宽 | 169.78 |

stdout 行逐字(rank0):`GEMM M=7 K=4096 N=2048 tflops=4.28` / `GEMM M=7 K=4096 N=3072 tflops=3.44` / `GEMM M=7 K=4096 N=4096 tflops=11.46` / `GEMM M=1024 K=4096 N=2048 tflops=169.78` / `RATIO half/full=0.373`。

注(实测事实 + 标注判读):RATIO = decode 半宽 N=2048 ÷ 全宽 N=4096 = 0.373(首跑 0.371;rank1 两轮 0.372/0.372,一致)。prefill 半宽(M=1024)169.78 TFLOPS 无塌方;decode 半宽塌至全宽的 37% —— 判读:塌方主体与 M=7 的 GEMV 形状相伴(N=2048 与 N=3072 两档同塌),非切半单独所致(推断)。

### prefill 拆账(单卡物理 GPU 1,torch.profiler)

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_prefill.py`(对象 `forward_prefill_batch`,8B bf16,[8,128] 随机 ids,seed=20261027 与 bench 同源,× 10 步;每步 8 序列 × 8 页、步末全归还;跑前 GPU 1 = 4 MiB)。

脚本原始输出行(逐字):`steps=10 wall/step=188.67ms gpu_kernel+memcpy/step=105.18ms gap/step=83.49ms`;`flops=14.22TFLOP/次(按 36 层静态 FLOP 估算) eff_wall=75.40TFLOPS eff_gpu=135.24TFLOPS`;`sanity: blocks live=0 (每步 8 序列 × 8 页步末全归还)`。

| 指标 | 实测值 |
|---|---|
| 每步墙钟 wall/step(含 profiler 采样开销) | 188.67 ms |
| GPU 自用时间(kernel+memcpy)/step | 105.18 ms |
| gap/step(墙钟 − GPU 端) | 83.49 ms |
| 有效 TFLOPS(墙钟口径) | 75.40(按 36 层静态 FLOP 估算 14.22 TFLOP/次) |
| 有效 TFLOPS(GPU 侧口径) | 135.24(同上) |

GPU 自用时间拆账(op 聚合,Self CUDA = 各自 kernel,行间不双计;10 步累计 → 折合每步;GEMM/attention/elementwise 归类):

| 归类 | op(聚合) | Self CUDA(10 步) | 每步 | 占 GPU 忙时 | 次数/步 |
|---|---|---|---|---|---|
| GEMM | aten::mm | 844.969 ms | 84.50 ms | 80.33% | 253 |
| elementwise(mul) | aten::mul | 53.689 ms | 5.37 ms | 5.10% | 471 |
| 拷贝(归因为逐行 view.update/_repeat_kv 等,推断) | aten::copy_ | 50.544 ms | 5.05 ms | 4.81% | 2501 |
| 行索引写入 | aten::_index_put_impl_ | 18.729 ms | 1.87 ms | 1.78% | 576 |
| elementwise(残差 add) | aten::add | 18.161 ms | 1.82 ms | 1.73% | 289 |
| attention(SDPA→flash) | aten::_flash_attention_forward | 16.673 ms | 1.67 ms | 1.59% | 288 |

六项合计 95.3%,余为长尾 elementwise/reduce/memcpy。top 设备 kernel 为 4 个 cutlass bf16 GEMM:`ampere_bf16_s1688gemm_bf16_64x128...` 432.853 ms(41.15%,720 次)/ `..._128x64...` 220.697 ms(20.98%,360 次)/ `..._128x128...` 140.672 ms(13.37%,720 次)/ `ampere_bf16_s16816gemm_bf16_128x64...` 36.798 ms(3.50%,720 次),GEMM kernel 合计 831.02 ms(79.0%);Self CUDA time total: 1.052 s(10 步)。

口径注记:wall/step 含 profiler 逐 kernel 采样开销(照 M4 口径注记 3;CPU 侧 Self CPU total 1.520 s,其中 aten::copy_ CPU total 866.939 ms —— 2501 次/步的小拷贝发射是 CPU 侧大头,CPU 算子行与 kernel 行双计、不入账),绝对吞吐以 bench 口径为准。与 M4.1 段二 in-process C=8 TTFT p50 0.1454 s 的算术对照(判读):批量 prefill 前向 GPU 自用 105.18 ms + gap 83.49 ms(profiler 含采样开销口径),与 145.4 ms 同量级;差值部分为步内调度/采样/传输等图外工作(推断,未单独计时)。

### HTTP 方法学对齐腿(单卡物理 GPU 1)—— 当日 BLOCKED,无数据

计划命令:起服 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.server --port 8200`(全默认:model-dir 8B、device cuda:0、num_blocks 1024、page 16、max_running 8;EngineConfig 默认 sdpa、无 fused、无 graph —— server CLI 无后端开关,照现状),压测 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --base http://127.0.0.1:8200 --concurrency 1 4 8`(实参 `--base`,脚本无 `--port`,照现状)。

现场(均为实测事实):

1. 起服成功:前台轮询 `/health` 200(约 40 s),GPU 1 驻留 18 532 MiB。
2. 压测零档出数:warmup 请求即失败,客户端错误行逐字 `httpx.HTTPStatusError: Client error '422 Unprocessable Entity' for url 'http://127.0.0.1:8200/v1/completions'`;服务端日志逐字 `INFO:     127.0.0.1:34242 - "POST /v1/completions HTTP/1.1" 422 Unprocessable Entity`。**无任何 RESULT 行可录**。
3. 根因(协议不匹配,非 CLI 参数拼写):`bench_remote` 以 token-id 数组直投 `prompt`(M3 基线节明言「绕开文本分词往返」,vLLM affordance);本引擎 `PromptCompletionRequest.prompt: str` 仅收文本,tokenizer 编码在服务端边界(`server.py` prompt 字段契约)。随机 id 序列经 str 传递再重编码不可保证等值(BPE 边界效应),无诚实绕行。
4. 旁证:str-prompt smoke(1 条,`"The capital of France is"`,max_tokens=4,stream)流式正常产出(`Paris`/`.`/` The`/` capital`)→ server 本体可用,不匹配纯在客户端协议。二次定向 kill 实际 serve 进程(nohup 包装 PID 失效,M4.1 节口径注记 3 同情形)后 GPU 1 回落 4 MiB、`nvidia-smi --query-compute-apps` 为空。
5. 后果:HTTP vs in-process 的 TTFT/tok/s 方法学差当日**无实测数字**;M4.1 遗留的 C=8 TTFT 0.1454 s vs vLLM 0.0809 s 重新归因无从量化,维持 M4.1 节记录原状。修复需动 server(补 token-id 输入契约)或 bench_remote(换文本协议)——均在本任务文件范围(2 脚本 + docs)之外,留裁决。

### HTTP 方法学对齐腿补账(token-id 契约落地后,2026-10-31,`scripts/bench_remote.py` → 本引擎 server,真机输出)

上节第 5 点的「留裁决」已过期:server token-id prompt 契约当日落地(commit 67a6224,`prompt: str | list[int]`——token-id 数组直进引擎不重分词 + 流式 include_usage 尾包补齐,`bench_remote` 零改动直投),该腿全档出数,补档如下。

命令(与上节计划命令逐字一致):服务 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.server --port 8200`(全默认;EngineConfig 默认 sdpa、无 fused、无 graph),压测 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --base http://127.0.0.1:8200 --concurrency 1 4 8`。

RESULT 行逐字(stdout;`RESULT-VLLM` 前缀为 `bench_remote` 固定标签,此处指向本引擎 server):

- `RESULT-VLLM C=1 ttft_p50=0.0380s ttft_mean=0.0380s tpot_mean=27.99ms tok_s=35.33`
- `RESULT-VLLM C=4 ttft_p50=0.1244s ttft_mean=0.1024s tpot_mean=43.86ms tok_s=86.92`
- `RESULT-VLLM C=8 ttft_p50=0.1947s ttft_mean=0.1755s tpot_mean=60.83ms tok_s=109.83`

HTTP 腿(本引擎,eager,8200 端口)与 in-process eager 腿(M4.1 段二节当日归档引用)对照——只并列陈述(比值列为算术),归因判读留控制器:

| 并发 | HTTP tok/s | in-process eager tok/s | in-process ÷ HTTP | HTTP TTFT p50 | in-process TTFT p50 | HTTP TPOT mean | in-process TPOT mean |
|---|---|---|---|---|---|---|---|
| 1 | 35.33 | 50.84 | 1.4× | 0.0380 s | 0.0347 s | 27.99 ms | 19.18 ms |
| 4 | 86.92 | 176.38 | 2.0× | 0.1244 s | 0.0766 s | 43.86 ms | 20.94 ms |
| 8 | 109.83 | 283.16 | **2.6×** | 0.1947 s | 0.1440 s | 60.83 ms | 21.12 ms |

口径注记(照 M4.1 节同款):同日同机(物理 GPU 1,RTX 4090);HTTP 腿 TTFT 含客户端→服务端上行与 SSE 流式首包回传全程(`bench_remote._run_one` 口径,产品真实路径);prompt 为同一批随机 token ids(`_prompts` 同 seed=20261027,token-id 数组直投、两端零分词差);temperature=0、max_tokens=32,计时前 1 条 warmup 不计入。

复测稳定性:审查复跑三档 tok_s 34.58 / 87.83 / 114.44,对主跑 35.33 / 86.92 / 109.83 差 −2.1% / +1.0% / +4.2%(**审查复跑 ±4.2% 带内**)。

### 段 0 补充:半宽 GEMM 核选矩阵(2026-10-31,`scripts/phase0_spike.py --gemm-variants`,真机输出)

动机:上节 decode RATIO 0.373 未分离「切半固有损失 vs cuBLAS 对该形状选核差」——decode M=7 GEMM 是权重流搬(内存受限),Task 0 实测全宽 33.5 MB / 20.8 µs ≈ 1.6 TB/s、半宽 16.8 MB / 27.4 µs ≈ 0.61 TB/s,纯权重流模型下两者带宽应相同、时间应随 N 减半。本矩阵对每格取四个核选变体的时间最好者,给「核选可修」留实证空间。

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/phase0_spike.py --gemm-variants`(单卡模式,无 torchrun;跑前 GPU 1 = 4 MiB / 0 进程)。形状矩阵 M ∈ {7,8,16,64,256,1024} × N ∈ {2048,3072,4096},K=4096,bf16;每变体 warmup 10 + 计 50(cuda.Event 均值)。

变体定义(脚本按此实现,如实):权重 w 均按模型 nn.Linear 布局连续存 [N,K];`matmul` = `torch.matmul(a, w.t())`(.t() 为 view;规格字面 `a @ w` 在 w [N,K] 下不成形,按模型实际调用路径实现)、`linear` = `F.linear(a, w)`(无 bias)、`pad16` = 仅 M=7,输入零填充到 16 行 matmul 后切片(计 16 行 GEMM 总时)、`transposed` = `w.t().contiguous()` → [K,N](预处理不计入)后 `a @ w_t`。effbw = 权重字节(N×K×2B)/时间;M≥256 档为算术+访存混合,带宽仅供参考。

best 变体逐格(µs | effbw GB/s | 变体;当日第二轮,与前一轮完整跑时间差 <1%):

| M\N | N=2048 | N=3072 | N=4096 |
|---|---|---|---|
| 7 | 18.74 / 895 / linear | 19.74 / 1275 / linear | 19.97 / 1680 / linear |
| 8 | 18.70 / 897 / matmul | 19.80 / 1271 / linear | 20.03 / 1675 / linear |
| 16 | 19.54 / 858 / linear | 20.66 / 1218 / matmul | 21.38 / 1569 / linear |
| 64 | 13.00 / 1290 / matmul | 16.92 / 1488 / transposed | 19.54 / 1717 / transposed |
| 256 | 33.44 / 502* / transposed | 54.78 / 459* / transposed | 56.71 / 592* / transposed |
| 1024 | 109.34 / 153* / linear | 164.17 / 153* / transposed | 197.61 / 170* / transposed |

\* 混合口径,仅供参考。M=7 档 pad16(填充到 16 行)三档均未胜出。

汇总行逐字(stdout):`RATIO' half/full=0.5*19.97/18.74=0.533(M=7 最好变体,TFLOPS 口径与 Task 0 同构)`;`CONTROL prefill M=1024 half(2048)tflops=157.12 full(4096)tflops=173.88(补 Task 0 prefill 全宽对照)`。

判读(实测数字 + 标注推断):

1. **塌方主体为核选可修:RATIO 0.373 → RATIO' 0.533**(当日两轮 0.537/0.533)。Task 0 GEMM 段的 B 为连续 [K,N] 布局,同卡同形状实测折合 27.4 µs(算术换算自 4.28 TFLOPS);本轮 TN 路径(linear = 权重 [N,K] + .t() 视图,即引擎 F.linear 实际布局)M=7 半宽 18.74 µs / 895 GB/s —— 布局/选核一项 ≈9 µs(−31%)(两数跨脚本对照,判读)。
2. **但 RATIO' 0.533 仍远离纯权重流理想(该口径下理想 = 1.0)**:半宽时间 18.74 µs 仅比全宽 19.97 µs 省 6%(权重字节省 50%);895 GB/s 仅为同格全宽 1680 GB/s 的 53%。剩余缺口归因 M=7 小 N GEMV 的每调用固定开销/小形状利用率地板(推断,未单独计时)。
3. M 增大后核选空间兑现:M=64 档半宽 effbw 1290 GB/s(N=2048)/1488 GB/s(N=3072,接近计划背景的 ~1.5 TB/s 参考线),M=7→64 半宽时间 18.74 → 13.00 µs 不升反降;M=7 达标与否(能否经 custom kernel 到 ~1.5 TB/s)留裁决。
4. prefill 全宽对照(补 Task 0 RATIO 分母):M=1024 N=4096 = 173.88 TFLOPS,与 Task 0 半宽 169.78 TFLOPS 同带 —— prefill 大 M 档切半无塌方(与上节 RATIO 注一致)。
