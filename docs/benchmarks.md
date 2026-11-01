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

复测稳定性:审查复跑三档 tok_s 34.58 / 87.83 / 114.44,对主跑 35.33 / 86.92 / 109.83 差 −2.1% / +1.0% / +4.2%(**审查复跑差均在个位数百分比带内**)。

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

### 段 0 补充二:冷权重 GEMM 裁决口径(L2 伪象排除)(2026-10-31,`scripts/phase0_spike.py --gemm-cold`,真机输出)

**上两节的热口径作废原因(实测伪象)**:微基准热复用单套权重,而 decode 四类 GEMM 单套权重 16.8–201 MB,≤72MB(4090 L2)者根本没走 HBM——带宽虚高、半宽塌方被夸大,Task 0 的 RATIO 0.373 与上节 RATIO' 0.533 均为热 L2 口径。冷口径:每类形状预分配 36 套独立权重轮流使用(= 36 层,本轮内不重访,轮转集 > L2,强制 HBM),F.linear(引擎真实路径)bf16、M=7,warmup 一整轮 + 计 30 轮 ÷(30×36);同格带单套复用 hot 对照。测量由控制器执行并裁决,本节脚本为其测量脚本(`/tmp/opencode/gemm_cold.py`)收编进 `phase0_spike.py --gemm-cold` 的复跑归档(来源透明;收编后复跑与控制器数字同带,见下)。

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/phase0_spike.py --gemm-cold`(单卡,无 torchrun;跑前 GPU 1 = 4 MiB / 0 进程)。四类形状按引擎 TP 切法:qkv/gate_up 输出列切(N 半)、o/down 行并行输入切(K 半)。

冷口径逐类(stdout 行逐字):

- `GEMMCOLD qkv full=56.27 hot_full=22.15 half=31.25 hot_half=19.69 ratio=0.555`
- `GEMMCOLD o full=38.56 hot_full=18.23 half=20.46 hot_half=10.41 ratio=0.531`
- `GEMMCOLD gate_up full=216.52 hot_full=216.44 half=117.17 hot_half=116.01 ratio=0.541`
- `GEMMCOLD down full=123.25 hot_full=124.04 half=57.85 hot_half=26.11 ratio=0.469`
- `GEMMCOLD bw(GB/s,权重字节/冷时间): qkv-full=895 qkv-half=805 o-full=870 o-half=820 gate_up-full=930 gate_up-half=859 down-full=817 down-half=870`

| 类 | 切法 | full 冷 µs(hot) | half 冷 µs(hot) | ratio t_half/t_full | 冷 GB/s full→half |
|---|---|---|---|---|---|
| qkv | N 6144→3072 | 56.27(22.15) | 31.25(19.69) | 0.555 | 895→805 |
| o | K 4096→2048 | 38.56(18.23) | 20.46(10.41) | 0.531 | 870→820 |
| gate_up | N 24576→12288 | 216.52(216.44) | 117.17(116.01) | 0.541 | 930→859 |
| down | K 12288→6144 | 123.25(124.04) | 57.85(26.11) | 0.469 | 817→870 |

内证(hot vs cold 分化精确跟随权重字节 vs 72MB L2,均为实测):gate_up 单套 201 MB、down-full 单套 101 MB **>** L2 → hot ≈ cold(216.44/216.52、124.04/123.25,伪象消失);qkv 50 MB、o 34 MB、down-half 50 MB **<** L2 → hot 快 1.5–2.2×(22.15/56.27、10.41/20.46、26.11/57.85)。Task 0 尖刀两形状([7,4096]×[4096,2048] = 16.8 MB、×4096,4096 = 33.6 MB)均在 L2 内,其 0.373 系伪象。冷口径下半宽/全宽带宽同带(805–870 GB/s 两侧互见),切半时间比 ≈ 字节比——权重流模型在 HBM 口径下成立。

每步 GEMM 总账 + TP=2 投影(stdout 行逐字;投影为算术非实测,假设注明:非 GEMM GPU 时间不变、allreduce 73×17.84 µs 串行、锚 M4.1 段一单卡 graph GPU 19.81 ms/步):

- `GEMMCOLD per-step total (36 layers): full=15.65ms half=8.16ms ratio=0.522`
- `GEMMCOLD projection: single_gpu=19.81ms gemm_share~79% allreduce~1.30ms -> TP2 decode GPU ~13.63ms/step (speedup ~1.45x on GPU time)`

冷口径 GEMM 占单卡 decode GPU 的 79%(15.65/19.81),与 M4 profiler 账目「GEMM 主导 decode GPU」一致(M4 口径 88.55%,含 profiler 采样差)。

**复跑 vs 控制器测量(收编验证,同带)**:四类 ratio 0.555/0.531/0.541/0.469 对控制器 0.553/0.530/0.541/0.469;总账 15.65/8.16 ms vs 15.64/8.15 ms(ratio 0.522 vs 0.521);投影 13.63 ms / 1.45× vs ~13.6 ms / ~1.45×——比值同位,绝对值差 <0.7%。

**判定数据小结(裁决行,控制器已裁)**:① decode allreduce 57 KiB 图内 17.84 µs,< 50 µs 参考门;② NCCL allreduce CUDA graph 捕获四档全成(零 GRAPH_CAPTURE_FAIL);③ 冷权重半宽/全宽 GEMM ratio 0.522(四类 0.469–0.555)≥ 40% 门。**三线全过,控制器判定:TP=2 GO。**

### 段 1:chunked prefill 中间账(HTTP 对照,2026-10-31,`acorninfer.server --chunked-budget` + `scripts/bench_remote.py` + `python -m acorninfer.bench`,真机输出)

背景:Task 1 已落地 chunked prefill(Engine 参数 `chunked_budget`,None = 每步全量 prefill、引擎行为与 M4.1 逐字节一致,fast 122 + slow 17 全绿零改动)。本段在 HTTP 产品路径上量化三件事:CLI 默认路径对 Task 0 基线的带内性(A)、预算大到不触发切块的哨兵(B)、强制切块的主数据(C);并以 in-process 回归线证明默认路径吞吐零回退。**方法学边界(如实,不粉饰)**:bench prompt 固定 128 tokens、同刻批量到达——chunking 的核心收益「连续到达请求不被长 prefill 阻塞」在本协议下不触发,本轮测的是**切块本身的代价/中性性**;收益侧数据留给后续连续到达协议。

CLI 接线:server `--chunked-budget`(int,缺省 None)经 `_build_engine` 透传 `Engine(chunked_budget=...)`(测试 `test_build_engine_chunked_budget_passthrough` 钉住;EngineConfig 不加字段,engine.py 本任务不动)。服务每跑 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.server --port 8200 [--chunked-budget N]`(其余全默认:sdpa、无 fused、无 graph——与 Task 0 基线服务口径逐字一致),压测每跑 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --base http://127.0.0.1:8200 --concurrency 1 4 8`(seed 默认 20261027,prompt 128 tokens 同批直投);每跑之间定向 kill 服务真实 PID(`ss -tlnp` 取 PID,**包装壳陷阱实录**:跑 A 首轮 `$!` 捕到的是后台包装子壳(607627),kill 后真身(607630,父进程已转 1)仍占端口持 18.7 GB——docs 段 0 注记 4 场景复现,按注记以 `ss` 定向 kill 真身,kill 后确认端口空 + GPU 回落 4 MiB 才起下一跑;每跑前 GPU 1 均 4 MiB / 0 进程)。

RESULT 行逐字(stdout;`RESULT-VLLM` 前缀为 `bench_remote` 固定标签):

- 跑 A(None,CLI 默认路径;首跑):`RESULT-VLLM C=1 ttft_p50=0.0364s ttft_mean=0.0364s tpot_mean=27.48ms tok_s=36.03`
- （A 续)`RESULT-VLLM C=4 ttft_p50=0.0884s ttft_mean=0.0885s tpot_mean=42.92ms tok_s=89.08`
- （A 续)`RESULT-VLLM C=8 ttft_p50=0.1927s ttft_mean=0.1733s tpot_mean=57.11ms tok_s=116.77`
- 跑 A′(None,同协议复跑):`RESULT-VLLM C=1 ttft_p50=0.0375s ttft_mean=0.0375s tpot_mean=28.17ms tok_s=35.13`
- （A′ 续)`RESULT-VLLM C=4 ttft_p50=0.1230s ttft_mean=0.1012s tpot_mean=42.87ms tok_s=88.85`
- （A′ 续)`RESULT-VLLM C=8 ttft_p50=0.1915s ttft_mean=0.1725s tpot_mean=58.93ms tok_s=113.11`
- 跑 B(`--chunked-budget 512`,哨兵:128-token prompt 不触发切块):`RESULT-VLLM C=1 ttft_p50=0.0415s ttft_mean=0.0415s tpot_mean=27.84ms tok_s=35.37`
- （B 续)`RESULT-VLLM C=4 ttft_p50=0.1247s ttft_mean=0.1027s tpot_mean=43.60ms tok_s=87.37`
- （B 续)`RESULT-VLLM C=8 ttft_p50=0.1427s ttft_mean=0.1693s tpot_mean=58.84ms tok_s=112.95`
- 跑 C(`--chunked-budget 64`,强制 2 块/序列):`RESULT-VLLM C=1 ttft_p50=0.0829s ttft_mean=0.0829s tpot_mean=27.31ms tok_s=34.43`
- （C 续)`RESULT-VLLM C=4 ttft_p50=0.2440s ttft_mean=0.2497s tpot_mean=44.10ms tok_s=74.80`
- （C 续)`RESULT-VLLM C=8 ttft_p50=0.5200s ttft_mean=0.5455s tpot_mean=64.48ms tok_s=85.47`

HTTP 四行对照(Task 0 基线为当日同机同协议归档引用,不重跑;差值列为算术):

| 跑 | 预算 | C=1 tok/s | C=4 tok/s | C=8 tok/s | TTFT p50 C=1/4/8 (s) | TPOT mean C=1/4/8 (ms) |
|---|---|---|---|---|---|---|
| Task 0 基线 | None(无该参数) | 35.33 | 86.92 | 109.83 | 0.0380 / 0.1244 / 0.1947 | 27.99 / 43.86 / 60.83 |
| A 首跑 | None | 36.03(+2.0%) | 89.08(+2.5%) | 116.77(+6.3%) | 0.0364 / 0.0884 / 0.1927 | 27.48 / 42.92 / 57.11 |
| A′ 复跑 | None | 35.13(−0.6%) | 88.85(+2.2%) | 113.11(+3.0%) | 0.0375 / 0.1230 / 0.1915 | 28.17 / 42.87 / 58.93 |
| B | 512 | 35.37(+0.1% vs A′) | 87.37(−1.7% vs A′) | 112.95(−0.1% vs A′) | 0.0415 / 0.1247 / 0.1427 | 27.84 / 43.60 / 58.84 |
| C | 64 | 34.43(−2.0% vs A′) | 74.80(−15.8% vs A′) | 85.47(−24.4% vs A′) | 0.0829 / 0.2440 / 0.5200 | 27.31 / 44.10 / 64.48 |

chunked 判读(如实,只陈述差值与机制,裁决留控制器):

- **A vs 基线**:A′ 三档 tok_s −0.6%/+2.2%/+3.0%,±5% 带内,TTFT p50 逐档 ≤2% 吻合——CLI 默认路径带内成立。A 首跑 C=8 +6.3% 出带、C=4 TTFT p50 0.0884 对基线 0.1244 偏离(同跑 ttft_mean 0.0885 对基线 0.1024),A′ 与 B 的 C=4 TTFT(0.1230/0.1247)复刻基线轮廓(0.1244)——首跑为方差样本;C=8 档历史带本就宽(Task 0 审查复跑 +4.2%,本日 A 首跑 +6.3%,同向),两跑均逐字归档不作挑选。
- **B vs A′**:tok_s 差 +0.7%/−1.7%/−0.1%——budget=512 下 128-token prompt 走单条全量 chunk(start=0 且 length=len(all_ids)),路由回既有 prefill_batch 原路径,与 None 行为同带,哨兵成立。
- **C vs A′(主数据)**:tok_s −2.0%/−15.8%/−24.4%;TTFT p50 全档上升 +121%/+98%/+171%;TPOT −3.1%/+2.9%/+9.4%(decode 慢腿几乎不动)。机制(与 Task 1 调度契约逐条对应,非推测的仅两点):budget=64 < prompt 128 → 每序列切 2 块;且 FCFS 预算下一步只够一条序列的首块(64 恰耗尽预算)→ 准入串行化,TTFT 含排队步数放大(C=8 尾序列等 8 步才完成 prefill)。切块本身让 [N,128] 单次批量 prefill 变成两轮更小的前向,步界开销与更小 GEMM 的代价即为吞吐差——**同步批量到达协议下 chunking 只表现出代价;「连续到达不被长 prefill 阻塞」的收益本协议不覆盖**(见节首方法学边界)。

in-process 回归线(Task 1 零回退证明;默认 None 路径,引擎直连,与 HTTP 无关;M4.1 段二收官数为当日归档引用):

- graph:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --in-tokens 128 --out-tokens 32 --concurrency 1 4 8 --attention-backend triton --fused-ops --cuda-graph`
- eager:同上,去掉 `--cuda-graph`

RESULT 行逐字:

- `RESULT C=1 ttft_p50=0.0352s ttft_mean=0.0352s tpot_mean=17.98ms tok_s=54.00`
- `RESULT C=4 ttft_p50=0.0797s ttft_mean=0.0797s tpot_mean=19.63ms tok_s=185.99`
- `RESULT C=8 ttft_p50=0.1493s ttft_mean=0.1493s tpot_mean=20.00ms tok_s=294.59`
- `RESULT C=1 ttft_p50=0.0357s ttft_mean=0.0357s tpot_mean=19.30ms tok_s=50.48`
- `RESULT C=4 ttft_p50=0.0766s ttft_mean=0.0766s tpot_mean=20.96ms tok_s=176.20`
- `RESULT C=8 ttft_p50=0.1432s ttft_mean=0.1432s tpot_mean=21.15ms tok_s=283.20`

(前 3 行 graph,后 3 行 eager。)

| 腿 | M4.1 收官 tok/s (C=1/4/8) | 本日 tok/s | 差 | ±8% 判定 |
|---|---|---|---|---|
| graph | 54.17 / 187.13 / 296.76 | 54.00 / 185.99 / 294.59 | −0.3% / −0.6% / −0.7% | 带内 |
| eager(triton+fused) | 50.84 / 176.38 / 283.16 | 50.48 / 176.20 / 283.20 | −0.7% / −0.1% / +0.0% | 带内 |

两腿全部 ±1% 以内(远窄于 ±8% 门),TPOT 逐位同带(graph 17.98/19.63/20.00,eager 19.30/20.96/21.15,对归档 19.18/20.94/21.12)——Task 1 对默认路径零回退成立。

辅助对照(同日实测,非 M4.1 配方):纯默认 bench(= HTTP server 同口径,sdpa、无 fused、无 graph)`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --in-tokens 128 --out-tokens 32` → C=1/4/8 = 36.62 / 91.07 / 117.71 tok_s,TTFT p50 0.0348/0.0762/0.1444 s。与 HTTP A′(35.13/88.85/113.11)差 +4.2%/+2.5%/+4.1%——HTTP 腿对 in-process 同口径的差价即客户端↔服务端+SSE 全程开销;亦解释 Task 0 节 HTTP(2.6×)对照的口径差:该表 in-process 列为 triton+fused 后端,HTTP server 为 sdpa 默认,后端不同(彼节归因本就留控制器,此处只并列本日同口径数)。

口径注记(照 M4.1/Task 0 节同款):同日同机(物理 GPU 1,RTX 4090);HTTP TTFT 含客户端→服务端上行与 SSE 流式首包回传全程(`bench_remote._run_one` 口径);prompt 同一批随机 token ids(`_prompts` seed=20261027,token-id 数组直投、两端零分词差);temperature=0、max_tokens=32,计时前 1 条 warmup 不计入;四跑服务配置仅 `--chunked-budget` 一处不同,其余逐字一致;每跑之间服务收服验证(端口空 + GPU 4 MiB)。

### 段 2:TP=2 8B 冒烟(强制流协议,2026-11-01,`scripts/smoke_tp2_8b.py`,真机输出)

8B(Qwen3-8B,bf16)TP=2 对拍冒烟。**协议修正(相对计划原文,以 M4/T3 先例为准):不做贪婪链逐位断言**——bf16 精确平局翻 argmax 是结构性现象,本日 8B 实测再度证实:强制流全池 top-2 gap **min = 0、p01 = 0**(264 位中精确平局真实存在),median 0.4375。改用:①强制输入流(预生成 token 喂入,平局不级联);②逐步 logits TOL;③强 gap 步 argmax 逐步断言;④自由贪婪只做分歧记账(不设零门;**强 gap 步分裂 = 疑 bug 信号 → BLOCKED**)。

标定(TOL 协议,先 `--calibrate` 取实测再写死进脚本;命令:`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/smoke_tp2_8b.py --model-dir /home/ernest/model-weight/Qwen3-8B --calibrate`,归档 /tmp/opencode/m5t5/):

- 33 次前向(prefill [8,128] + 32 forced decode)x 8 行全词表 logits(fp32)vs TP=1 参照:**max|diff| = 0.21875**(两次独立跑同值,确定性)。量级判读:≈1-2 bf16 ulp @|logit|≈32——TP 切分归约序差经 36 层逐层放大,远大于 tiny 单层的 ~1.5 ulp(T3),但仍是 ulp 级噪声非 bug。
- **TOL = 0.5**(≈2.3x 实测;真错权重会差 O(10));**margin = 1.75 = 8 x 实测噪声**(T3/T4 口径)。物理翻转向阈值 ≈ 2 x max|diff| ≈ 0.44,margin 再留 4x。

正式跑(`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/smoke_tp2_8b.py --model-dir /home/ernest/model-weight/Qwen3-8B`,EXIT=0,SMOKE status OK;跑前双卡 4 MiB / 收尾双卡 4 MiB)。注:输出中夹杂 NCCL monitor 诊断行(`TORCH_NCCL_ENABLE_MONITORING` 周期性集合通信统计)——非致命诊断噪声,与任何判定项无关:

| 阶段 | 断言面 | 结果 |
|---|---|---|
| B(模型级强制流) | 逐步 max\|diff\| ≤ 0.5 | 0.21875 ✓(逐行 0.094-0.219 带) |
| B | 强 gap 位(gap≥1.75)argmax 逐步 | 17/17 全对,0 分叉 ✓ |
| B | 页表 vs TP=1 参照逐位 + 长度 = 160 | 逐位同 ✓ |
| B/D | rank 间位级(logits sha256 / 链 sha256 / 页表 / C 流) | 全同 ✓(SPMD) |
| D(自由贪婪记账) | 平局步(gap<1.75)/ 分裂行 / 强 gap 分裂 | 76/256 步平局带;4 行首分叉(gap 0.25/0/0.125/0 全在噪声解释带内);**强 gap 分裂 0** ✓ |
| C(chunked×TP 引擎级) | 中间块零发射、首 token 出自 extend 完成行 | 64 发射,首发射 kind=extend(4/4)✓ |
| C | 逐 seq 首 token gap 门控(ref gap≥1.75 必须 token 同) | 强分叉 0;平局翻转向 2(均 gap=0.0)✓ |
| C | 页表逐步逐位(rank0 vs 参照;rank0 vs rank1) | 逐步逐位同 ✓ |

分叉记账明细(D 腿,4 行如实,不设零门):row4 首分叉@17 gap=0.25 不回归;row5@29 gap=0 不回归;row6@13 gap=0.125 **回归**;row7@6 gap=0 不回归。C 腿平局翻转向 2 处(sid=1@5、sid=2@2,均 gap=0.0)。全部落在平局解释带(≤0.44),无一强 gap 分叉。

分装备忘:全量加载(阶段 A,from_pretrained)35.7 s;分片加载(load_safetensors_sharded,热页缓存)rank0 1.6 s / rank1 2.1 s。

粗计时(**非官方数字,官方以 T6 graph/HTTP 为准**;eager sdpa,B=8,RTX 4090 x2 PCIe NODE 无 NVLink):

| 腿 | decode 步均(ms) | prefill [8,128] 一步(ms) |
|---|---|---|
| TP=1 eager | 64.9 | 321.0 |
| TP=2 eager | 65.8 | 3148.1 |
| 比值 | 0.99x | — |

判读(与段 0 尖刀数字自洽,标推断):decode 无收益——每步 74 次集合(allreduce x72 + all-gather + embed)在 PCIe NODE 上(~18-19 µs/次 ≈ 1.3-1.4 ms)叠加段 0 实测的 decode 半宽 GEMM 塌方(37% 效率),吃满 2x 权重读取的节省。TP=2 prefill 3148 ms 为**冷启动伪象**(该腿是 TP 进程首个前向:NCCL communicator 建立 + 两 rank cublas 首初始化都记在这一步;decode 均值亦含首步 warmup),不作为稳态结论。

### 双轨配对终账(2026-11-01,`scripts/bench_remote.py` + `python -m acorninfer.bench`,真机输出;轨 2 经修复补测收官,原 BLOCKED 现场如实保留)

M5 收官仪式:同日窗口、串行执行、每引擎切换间 `nvidia-smi` 验净(实录见口径注记)。**仪式前提核实现场(先于任何起服)**:`acorninfer/server.py` CLI 旗标全集为 `--model-dir/--host/--port/--device/--num-blocks/--page-size/--max-running/--watermark-blocks/--chunked-budget/--tp`——**无 `--attention-backend/--fused-ops/--cuda-graph`**;`_build_engine` 两条路径(tp==1 与 tp≥2)均以 `EngineConfig` 缺省值(sdpa、无 fused、无 graph)构造。server.py 属保护文件禁改 → **「AcornInfer HTTP 腿开满后端配方」server 不支持,按仪式规则记 BLOCKED(硬前提不成立,别绕)**;T2/段 0 补账已证缺省 sdpa 腿对 in-process 满配方捐 1.4-2.6×(下文有同日修正归因),故缺省配方 HTTP 腿数字只作补充口径,不进验收算术。轨 1 的 AcornInfer 侧以 in-process 满配方(bench CLI 有全套旗标,M4.1 口径延续)为验收腿。

命令原样(每段跑前验净,`ss -tlnp` 取真身 PID 定向 kill):

- vLLM TP=1:`CUDA_VISIBLE_DEVICES=0 /home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90 --dtype bfloat16`
- vLLM TP=2:同上,`CUDA_VISIBLE_DEVICES=0,1` + `--tensor-parallel-size 2`(就绪 75 s,含 compile)
- AcornInfer HTTP 缺省配方(补充口径):`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.server --port 8200`;TP=2 满配方(修复后补测):`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m acorninfer.server --model-dir /home/ernest/model-weight/Qwen3-8B --port 8200 --tp 2 --attention-backend triton --fused-ops --cuda-graph`(原 `--tp 2` 缺省配方装载即崩,见轨 2 BLOCKED 段)
- in-process:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --in-tokens 128 --out-tokens 32 --attention-backend triton --fused-ops --cuda-graph`(graph);同去 `--cuda-graph`(eager)
- 压测:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --base http://127.0.0.1:{8100|8200} --concurrency 1 4 8`

RESULT 行逐字(stdout;`RESULT-VLLM` 前缀为 `bench_remote` 固定标签,四组各自独立起服):

- vLLM TP=1:`RESULT-VLLM C=1 ttft_p50=0.0269s ttft_mean=0.0269s tpot_mean=16.91ms tok_s=58.07`
- (续)`RESULT-VLLM C=4 ttft_p50=0.0681s ttft_mean=0.0575s tpot_mean=17.86ms tok_s=207.63`
- (续)`RESULT-VLLM C=8 ttft_p50=0.0830s ttft_mean=0.0762s tpot_mean=17.85ms tok_s=354.32`
- AcornInfer TP=1 in-process graph:`RESULT C=1 ttft_p50=0.0398s ttft_mean=0.0398s tpot_mean=17.95ms tok_s=53.66`
- (续)`RESULT C=4 ttft_p50=0.0781s ttft_mean=0.0781s tpot_mean=19.59ms tok_s=186.81`
- (续)`RESULT C=8 ttft_p50=0.1450s ttft_mean=0.1450s tpot_mean=19.96ms tok_s=296.71`
- AcornInfer TP=1 in-process eager:`RESULT C=1 ttft_p50=0.0350s ttft_mean=0.0350s tpot_mean=19.26ms tok_s=50.63`
- (续)`RESULT C=4 ttft_p50=0.0774s ttft_mean=0.0774s tpot_mean=21.02ms tok_s=175.55`
- (续)`RESULT C=8 ttft_p50=0.1453s ttft_mean=0.1453s tpot_mean=21.22ms tok_s=282.01`
- AcornInfer TP=1 HTTP 缺省配方(补充口径):`RESULT-VLLM C=1 ttft_p50=0.0386s ttft_mean=0.0386s tpot_mean=27.57ms tok_s=35.82`
- (续)`RESULT-VLLM C=4 ttft_p50=0.1248s ttft_mean=0.1028s tpot_mean=43.24ms tok_s=88.06`
- (续)`RESULT-VLLM C=8 ttft_p50=0.1931s ttft_mean=0.1576s tpot_mean=58.15ms tok_s=115.59`
- vLLM TP=2:`RESULT-VLLM C=1 ttft_p50=0.0266s ttft_mean=0.0266s tpot_mean=9.52ms tok_s=99.44`
- (续)`RESULT-VLLM C=4 ttft_p50=0.0750s ttft_mean=0.0612s tpot_mean=10.66ms tok_s=321.79`
- (续)`RESULT-VLLM C=8 ttft_p50=0.0811s ttft_mean=0.0733s tpot_mean=10.98ms tok_s=541.94`
- AcornInfer TP=2 满配方 HTTP(修复后补测,见轨 2 BLOCKED 段的修复注):`RESULT-VLLM C=1 ttft_p50=0.0359s ttft_mean=0.0359s tpot_mean=13.00ms tok_s=72.92`
- (续)`RESULT-VLLM C=4 ttft_p50=0.1117s ttft_mean=0.1116s tpot_mean=13.97ms tok_s=233.09`
- (续)`RESULT-VLLM C=8 ttft_p50=0.1789s ttft_mean=0.1613s tpot_mean=15.55ms tok_s=349.40`

graph 稳定性注记(如实):graph 腿当日首跑因取尾管道截去 C=1 行,可见部分 C=4 186.58 / C=8 285.40,与第二跑(上表,全量)差 +0.1%/+3.8%,在段 2 记录的 C=8 历史带宽内;两跑均归档不作挑选。

**轨 1 表(TP=1;vLLM HTTP vs AcornInfer in-process 满配方,验收口径 = M4.1 延续;HTTP 缺省配方行仅并列参考):**

| 腿 | C=1 tok/s | C=4 tok/s | C=8 tok/s | TTFT p50 C=1/4/8 (s) | TPOT mean C=1/4/8 (ms) |
|---|---|---|---|---|---|
| vLLM TP=1 HTTP | 58.07 | 207.63 | 354.32 | 0.0269 / 0.0681 / 0.0830 | 16.91 / 17.86 / 17.85 |
| AcornInfer in-process graph(满配方,验收腿) | 53.66 | 186.81 | 296.71 | 0.0398 / 0.0781 / 0.1450 | 17.95 / 19.59 / 19.96 |
| AcornInfer in-process eager(triton+fused) | 50.63 | 175.55 | 282.01 | 0.0350 / 0.0774 / 0.1453 | 19.26 / 21.02 / 21.22 |
| AcornInfer HTTP 缺省配方 sdpa(**非验收口径**) | 35.82 | 88.06 | 115.59 | 0.0386 / 0.1248 / 0.1931 | 27.57 / 43.24 / 58.15 |

轨 1 达标率算术(逐档,AcornInfer ÷ vLLM):graph 腿 C=1 53.66/58.07 = **92.4%**、C=4 186.81/207.63 = **90.0%**、C=8 296.71/354.32 = **83.7%**(对 M4.1 收官 93.4%/90.2%/83.5%:−1.0/−0.2/+0.2 pp,机时漂移带内;距 100% 线余 7.6%/10.0%/16.3%)。eager 腿 87.2%/84.5%/79.6%。HTTP 缺省配方腿 61.7%/42.4%/32.6%(含已知配方差,非验收口径)。

**轨 2 表(TP=2;AcornInfer 腿为修复后补测,见下):**

| 腿 | C=1 tok/s | C=4 tok/s | C=8 tok/s | TTFT p50 C=1/4/8 (s) | TPOT mean C=1/4/8 (ms) |
|---|---|---|---|---|---|
| vLLM TP=2 HTTP | 99.44 | 321.79 | 541.94 | 0.0266 / 0.0750 / 0.0811 | 9.52 / 10.66 / 10.98 |
| AcornInfer TP=2 HTTP(满配方,修复后) | 72.92 | 233.09 | 349.40 | 0.0359 / 0.1117 / 0.1789 | 13.00 / 13.97 / 15.55 |

轨 2 BLOCKED 现场(两重,均如实采证;**当段补测已修复,见注**):

1. **满配方缺失**(仪式硬前提,轨 1 同源):server 无后端配方旗标,`_build_engine` tp≥2 路径同样硬编码缺省;TP=2 下 rank0/rank1 各自独立进程构造,即使外部改 rank0 也与 rank1 SPMD 配方分裂,无合法开法。
2. **生产装载线崩溃**(缺省配方即崩,先于配方问题):`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m acorninfer.server --port 8200 --tp 2` 两 rank 双双 `KeyError: 未知的 TP 分片目标: model.embed_tokens.weight`(qwen3_tp.py `shard_fn` 对 embed/lm_head 用精确名匹配 `"embed_tokens.weight"/"lm_head.weight"`,8B HF 权重键带 `model.` 前缀而 fall-through 到 KeyError;其余层走 `endswith` 不受累)。rank1(SpawnProcess-1)先崩、rank0 随后同错。**Task 5 冒烟能过是因为脚本自带 `_shard_fn8b` 包装(`name.removeprefix("model.")` 再交 `build_shard_fn`,smoke_tp2_8b.py:294 有注)——该剥前缀包装未进生产 `server._build_tp_model` 装载线**。崩溃后现场:端口闭、无孤儿进程、双卡回落 4 MiB。

**修复与轨 2 补测(同日,控制器授权后)**:①装载边界键归一落 `server.py::_prefix_normalized_shard_fn`(生产 shard_fn 调用点;剥离 `model.` 前缀后转发 `build_shard_fn`,返回 state 复用单卡 `map_hf_state_dict` 同口径归一;**qwen3_tp.py 零改动**;红→绿:`test_tp_shard_prefix_normalization` 两套键型逐名同分片 + e2e 参数化 `test_tp2_server_e2e_http[hf-full-recipe]` 双卡过);②server CLI 加 `--attention-backend/--fused-ops/--cuda-graph`(bench 同名旗标,缺省 = 旧行为逐字节,`EngineConfig.__post_init__` 校验 graph 须 triton),两条 `_build_engine` 路径透传(钉子:`test_build_engine_backend_recipe_passthrough` / `..._tp2_backend_recipe_passthrough`;fast 138 全绿)。补测命令:`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m acorninfer.server --model-dir /home/ernest/model-weight/Qwen3-8B --port 8200 --tp 2 --attention-backend triton --fused-ops --cuda-graph`(就绪 ~5 s,双卡各 10126 MiB;TP=2 + graph 引擎级数值对拍由既有 `test_tp_engine::test_tp2_graph_matches_eager` 钉,不重复造);收服:真身 PID 1653464 定向 kill,FINI 广播 rank1 优雅退出,双卡回落 4 MiB。

轨 2 达标率算术(逐档,AcornInfer ÷ vLLM):C=1 72.92/99.44 = **73.3%**、C=4 233.09/321.79 = **72.4%**、C=8 349.40/541.94 = **64.5%**(TTFT p50 比 1.35×/1.49×/2.21×;TPOT 比 1.37×/1.31×/1.42×)——逐档未过 100%。

判读(如实,推断处标注):

1. **TP=2 decode 增益是否兑现——谱系闭合(四分)**:段 0 投影 1.45×(allreduce 17.84 µs + 半宽 GEMM 不塌方假设)→ 段 2 自家 eager 实测 0.99×(半宽 GEMM 37% 效率塌方 + 74 次集合开销吃满)→ 本日 vLLM TP=2 实测 TPOT 16.91→9.52 ms(**1.78×** decode 提速,吞吐 +71.2%/+55.0%/+53.0%)→ **本日自家满配方 TP=2 TPOT 17.95→13.00 ms(1.38×,投影的 95%)**。修复后自家增益兑现:TP=1 graph 把每步发射开销回收后,TP=2 的集合通信代价不再被发射噪声淹没(段 2 eager 0.99× 与本日 1.38× 不矛盾——eager 腿下发射开销主导两者;推断)。自家吞吐对自身 TP=1:C=1 72.92/53.66 = **+35.9%**(对自家缺方 HTTP 腿 35.82 则 +104%,配方+并行复合)。与 vLLM 残余差距(TPOT 1.37×@C=1)锚定在 kernel/通信栈效率差,非结构性不可为。
2. **TTFT/TPOT**:轨 1 C=1 差价拆解——TTFT 0.0398 vs 0.0269 s(Acorn prefill 慢 1.48×,贡献 ~13 ms),TPOT 17.95 vs 16.91 ms(差 +6%,贡献 ~32 ms/31 步):吞吐差主体在 prefill 与发射侧,decode kernel 已接近。vLLM TP=2 TTFT 逐档与自身 TP=1 持平(0.0266/0.0750/0.0811 vs 0.0269/0.0681/0.0830)——prefill 侧 TP=2 无增益(PCIe 通信吃掉切分收益;与段 2 prefill 观察同向,推断)。AcornInfer HTTP 缺省腿 TPOT 27.57/43.24/58.15 ms 对同配方 in-process 缺省(段 1 辅助线 19.18-21.15 ms 带外大涨)为排队/发射合批差异,C≥4 档 TTFT 0.1248/0.1931 s 复刻段 1 A′ 轮廓(0.1230/0.1915)。
3. **chunked 判定(引段 1,不重裁)**:同步批量到达协议下 chunking 只表现代价(段 1 C vs A′:tok_s −2.0%/−15.8%/−24.4%,TTFT +121%/+98%/+171%),「连续到达不被长 prefill 阻塞」的收益场景不在本协议内。本仪式四腿均为同步到达,该判定原样延续。
4. **2.6× 后端配方差的修正归因(同日同卡实测闭环)**:当日 in-process 满配方 ÷ HTTP 缺省配方 = C=1 53.66/35.82 = **1.50×**、C=4 **2.12×**、C=8 **2.57×**;而 HTTP 全程开销本身极小——段 1 同配方对照(HTTP 缺省 35.13 vs in-process 缺省 36.62)= 0.96×,本日 HTTP 缺省(35.82)对段 1 in-process 缺省(36.62)= 0.98×(跨日引用,注机时)。**修正归因:2.6× ≈ 后端配方差(×1.5~2.6,主体,C≥4 档近全额)× HTTP/SSE 全程开销(×~0.96-1.0,小修正项)**——T2「缺省 sdpa 腿捐 2.5×」由配方差解释,非 HTTP 协议本身贵;这也把段 0 补账「后端不同(彼节归因本就留控制器)」的悬案收口。
5. **收服实录(每引擎切换验净)**:vLLM TP=1 起服前双卡 4 MiB → 压测毕 `ss -tlnp` 取真身 PID(1232431)kill,与 AcornInfer TP=1(PID 1268808)同收,双卡 4 MiB 端口闭;vLLM TP=2 起服前双卡 4 MiB → 就绪 75 s → 压测毕真身 PID 1283772 kill,双卡 4 MiB;AcornInfer TP=2 装载崩溃自净,双卡 4 MiB、无计算进程。全程无 pkill 模糊匹配,包装壳陷阱(段 1 实录)以 `ss -tlnp` 定位真身规避。

**判定行(如实,对照 spec 验收锚「双轨配对 + TP=2 逐档 ≥100%」):轨 1 双腿齐——92.4%/90.0%/83.7%,逐档未过 100%(与 M4.1 挂账同带延续,缺口的 prefill/发射侧归因见判读 2);轨 2 修复后双腿齐——73.3%/72.4%/64.5%,逐档未过 100%,**TP=2 逐档 ≥100% 北极星维持挂账**;但自家 TP=2 decode 增益已兑现(1.38×,谱系闭合,判读 1),与 vLLM 的差距从「无数据/不可判」变为可归因的工程差距(kernel/通信栈,判读 1/2)。双轨配对仪式数据面收官;M5 不打 tag、不 push(留用户)。**

口径注记(照前节同款):同日同机(RTX 4090 ×2,GPU0-GPU1 PCIe NODE 无 NVLink);六组腿全部 seed=20261027、in=128/out=32、temperature=0、max_tokens=32、计时前 1 条 warmup 不计入;HTTP TTFT 含客户端→服务端上行与 SSE 流式首包回传全程(`bench_remote._run_one` 口径);轨内各腿同窗口串行,轨间(vLLM TP=1 → vLLM TP=2 → AcornInfer TP=2 补测)切换以双卡 4 MiB 验净为界;轨 2 补测为满配方旗标 + HF 前缀归一修复后的同一产品路径(修复与旗标落点见轨 2 BLOCKED 段修复注);vLLM 0.32.0,基线环境 `/home/ernest/vllm-baseline`;补测时点树 = 0e35482 + 本 commit 修复(未提交),修复与补测数据同 commit 入库(历史序如实)。

## M6 段 0 尖刀:decode 四腿拆账 + 可回收清单(2026-11-01,`scripts/m6_spike.py` + `scripts/m6_ab.py`,真机输出)

背景:锚继 M4 挂账「C=8 需 TPOT 净省 ~3.3ms/步」(M4 期 21.12 → 锚 ~17.82);M5 收官 graph C=8 TPOT 19.96(已收 ~1.2ms),本日基线带 19.89–19.93,**今日残余缺口 ≈ 2.0–2.1ms**(对 vLLM 17.85 同带)。裁决门(spec §4):**Σ(腿 2/3/4 可回收)< 2.31ms(= 3.3 × 70%)且腿 1 定因显示缺口主体在 GEMM 面 → 段 2 triton 立项;否则休眠**。四腿全部 harness 内完成(`--layout` monkey-patch + 逐语句镜像 `graph_runner.decode_batch` 的插桩副本),引擎/模型源码零触碰;每天每腿跑前 GPU 1 验净(4 MiB)、前台运行。本节为四腿数据 + RECOVER 清单;**判定行留控制器**。

### 基线带(m6_ab,当日双样本带内确认)

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_ab.py`(warmup 10 + 计 30,同 seed 同输入;下表「首跑」为当日首次基线,「knobs[0]」为腿 2 组合 0(全缺省 env,同协议复跑):

| 样本 | C=1 graph | C=8 graph | C=1 eager | C=8 eager |
|---|---|---|---|---|
| 首跑 | 17.91 | 19.92 | 19.83 | 21.15 |
| knobs[0](缺省 env) | 17.92 | 19.89 | 19.16 | 21.28 |

### 腿 1:引擎内 GEMM 拆账(insitu)

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_spike.py insitu`(torch.profiler 包 decode loop,C=1/C=8 × graph/eager;chrome trace 实例序映射:每层发射序 q,k,v,o,gate,up,down × 36 + lm_head = 252+1,splitKreduce 辅核附着前实例不占槽;结构自证 count/step 逐类核对通过,零 INSITU_WARN;dmon 同窗)。

四模式总账(stdout 逐字值):

| 口径 | wall_ms | gpu_ms | gap_ms | gemm_proj µs/步 | proj−cold(ms) | lm_head µs/步(GB/s) |
|---|---|---|---|---|---|---|
| C=1 graph | 19.27 | 17.93 | 1.34 | 15168.5 | −0.47 | 1298.5(958) |
| C=8 graph | 21.52 | 19.80 | 1.71 | 16454.8 | **+0.81** | 1320.5(943) |
| C=1 eager | 25.64 | 18.08 | 7.56 | 15184.7 | −0.46 | 1298.6(958) |
| C=8 eager | 26.01 | 19.96 | 6.05 | 16473.0 | +0.83 | 1316.4(945) |

C=8 graph 逐类(stdout 逐字;冷口径参照 = M5 段 0 补充二 full=15.64ms,M=7 四类投影不含 lm_head):

- `INSITU C=8 mode=graph wall_ms=21.52 gpu_ms=19.80 gap_ms=1.71 gemm_all_us_per_step=17775.3 cold_ref_ms=15.64(M=7 四类投影,不含 lm_head)`
- `INSITU C=8 mode=graph class=gemm_proj us_per_step=16454.8 proj_minus_cold_ms=0.81`
- `INSITU C=8 mode=graph class=qkv us_per_step=2262.4 n_per_step=108 check_gb=801`
- `INSITU C=8 mode=graph class=o us_per_step=1404.3 n_per_step=36 check_gb=860`
- `INSITU C=8 mode=graph class=gate_up us_per_step=8348.6 n_per_step=72 check_gb=868`
- `INSITU C=8 mode=graph class=down us_per_step=4439.5 n_per_step=36 check_gb=816`
- `INSITU C=8 mode=graph class=lm_head us_per_step=1320.5 n_per_step=1 check_gb=943`
- `INSITU C=8 mode=graph class=attention us_per_step=646.3`
- `INSITU C=8 mode=graph class=norm us_per_step=110.8`
- `INSITU C=8 mode=graph class=rope us_per_step=86.1`
- `INSITU C=8 mode=graph class=copy us_per_step=332.1`
- `INSITU C=8 mode=graph class=other us_per_step=848.1`
- `INSITU C=8 mode=graph class=memcpy us_per_step=4.4 count_per_step=3.00`
- `CHECK C=8 mode=graph proj_gb=13.9GB/步 eff_gb=782(按 gemm_all 分母,含 lm_head 略偏高);cold 对照 889GB/s`

主 kernel 分类学(TOPKERN,M=7/8 全部四类投影同一 cutlass kernel 名——按名映射不可能,实例序映射为腿 1 前提):`TOPKERN C=8 mode=graph n_per_step=253.00 us_per_step=17694.6 name=…cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`;C=1 侧 M=1 全部走 `internal::gemvx::kernel`(108/步 11634.5µs + 145/步 4832.6µs 两实例组)。`cublasLt::splitKreduce_kernel` 72/步 80.7µs(C=8 graph)。

腿 1 判读(如实):

- **GEMM-vs-cold(C=8)= +0.81ms,是缺口(~2.0ms)的 ~40%,非主体**;C=1 侧 −0.47 为 M=1(gemvx)对冷口径 M=7(cutlass)的跨 M 对照,只作方向参考(口径差如实注记)。
- lm_head 1320.5µs @943GB/s = 峰值 93.5%,已近屋顶;四类投影 782–868GB/s(峰值 78–86%)。
- 非投影面合计 2026µs/步 = attention 646.3 + other 848.1 + copy 332.1 + norm 110.8 + rope 86.1 + memcpy 4.4;`other` 内含未融合 q/k-norm RMSNorm 链(TOPKERN 实测 reduce_kernel 117.6 + rsqrt 70.7 + pow 68.1 + bf16 copy 65.5 + add 65.3 ≈ 387µs/步,非 GEMM 融合候选)。
- **降频排除**:dmon 计算窗(C=8 graph)pclk 2790MHz 全平、pwr ≤192W;全程 pclk min=210 为装载/构建空闲段(全文 54 行 `/tmp/opencode/m6/dmon_insitu.log`)。归因候选(同名 kernel 时长不同/L2 污染/插桩稀释/降频)中降频出局,余者不改变 +0.81 的量级归属。

### 腿 2:调参面扫(knobs)

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_spike.py knobs`(12 组合 = {CUBLASLT_WORKSPACE_SIZE ∈ 缺省/64M/256M} × {bf16_reduced_precision_reduction ∈ T/F} × {CUBLAS_WORKSPACE_CONFIG ∈ 缺省/:4096:8};env 在 CUDA 初始化前经子进程注入,各组合跑 m6_ab 四档 + 孤立冷微基准)。

12 组合逐字值(step_ms):

| # | ws | red | cfg | C=1 graph | C=8 graph | C=1 eager | C=8 eager | cold full |
|---|---|---|---|---|---|---|---|---|
| 0 | def | T | def | 17.92 | 19.89 | 19.16 | 21.28 | 15.63 |
| 1 | def | T | 4096x8 | 17.96 | 19.89 | 19.73 | 21.14 | 15.63 |
| 2 | def | F | def | 17.93 | 19.92 | 19.35 | 21.19 | 15.64 |
| 3 | def | F | 4096x8 | 17.94 | 19.92 | 19.15 | 21.20 | 15.64 |
| 4 | 64M | T | def | 17.92 | 19.90 | 19.15 | 21.18 | 15.64 |
| 5 | 64M | T | 4096x8 | 17.95 | 19.90 | 19.15 | 21.18 | 15.63 |
| 6 | 64M | F | def | 17.92 | 19.91 | 19.17 | 21.16 | 15.64 |
| 7 | 64M | F | 4096x8 | 17.93 | 19.92 | 19.16 | 21.19 | 15.63 |
| 8 | 256M | T | def | 17.93 | 19.90 | 19.20 | 21.15 | 15.64 |
| 9 | 256M | T | 4096x8 | 17.92 | 19.93 | 19.24 | 22.24* | 15.64 |
| 10 | 256M | F | def | 17.95 | 19.92 | 19.16 | 21.17 | 15.63 |
| 11 | 256M | F | 4096x8 | 17.92 | 19.91 | 19.13 | 21.17 | 15.64 |

带:C=1 graph 17.92–17.96(极差 0.04)、C=8 graph **19.89–19.93(极差 0.04)**、cold full 15.63–15.64(极差 0.01)——**死平,R1 实证排除**。*离带样本如实归档:组合 9 eager C=8 22.24(eager C=8 历史带宽本宽,余 11 组合 21.14–21.28);cold half 全程 8.14–8.15。输出格式样例(stdout 逐字,组合 0):`KNOB ws=def,red=T,cfg=def C=8_graph_step_ms 19.89` / `KNOB ws=def,red=T,cfg=def gemm_cold_full_perstep_ms 15.63`。

### 腿 3:布局面扫(layout)

孤立打分命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_spike.py layout`(冷口径 36 套权重轮转;变体:tn=现行 F.linear / nn=预变换 mm(x, w_t) / pad8 / pad16)。逐字值(us / GB/s):

M=7:

| 类 | tn | nn | pad8 | pad16 |
|---|---|---|---|---|
| qkv | 56.32/894 | 58.63/859 | 56.11/897 | 56.13/897 |
| o | 39.00/860 | 39.58/848 | 38.98/861 | 39.02/860 |
| gate_up | 216.73/929 | 217.39/926 | 217.77/924 | 219.27/918 |
| down | 124.39/809 | **113.22/889** | 123.26/817 | 123.90/812 |

M=8:

| 类 | tn | nn | pad8 | pad16 |
|---|---|---|---|---|
| qkv | 56.14/897 | 56.54/890 | 56.11/897 | 56.14/896 |
| o | 39.00/860 | 40.69/825 | 39.00/860 | 39.04/859 |
| gate_up | 216.98/928 | 217.50/926 | 217.76/925 | 219.27/918 |
| down | 124.55/808 | **113.51/887** | 123.24/817 | 123.85/813 |

- `LAYOUT_BEST m=7 down nn us=113.22`、`LAYOUT_BEST m=8 down nn us=113.51`(其余三类 best 落 tn/pad8,差 ≤0.2µs = 噪声带)。**NN 只在 down 赢**(−11.0~−11.2µs/层,808→889GB/s),pad8/pad16 ≈ TN。down-only 投影:11.0~11.2µs × 36 层 = **−0.40~0.41ms/步**(C=8 口径)。

引擎内复测命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_spike.py layout --engine-retest t`(全局 NN;harness monkey-patch 接线,当日第三样本,stdout 逐字):

- `AB layout_t C=1 mode=graph step_ms=18.60 tpot_ms=18.62`
- `AB layout_t C=8 mode=graph step_ms=19.31 tpot_ms=19.33`
- `AB layout_t C=1 mode=eager step_ms=19.95 tpot_ms=20.06`
- `AB layout_t C=8 mode=eager step_ms=20.60 tpot_ms=20.63`

三样本带:graph C=8 19.31/19.32/19.31(**−0.58~−0.62**,增益方向与量级超出孤立 down-only 投影 0.40——引擎内交互效应如实记录)、**graph C=1 18.59/18.60/18.60(+0.64~+0.68 回归)**、eager C=8 20.55–20.62(−0.52~−0.73,对主带 21.14–21.28)、eager C=1 19.91–20.04(+0.6~+0.9,对主带 19.13–19.35)。**全局 NN 触 no-regression 门(C=1 回归)→ 出局(R2 实录)**;M=1(gemvx)面 NN 未做孤立打分,引擎 C=1 回归与之相容(推断)。down-only hybrid 引擎内未复测,Task 1 候选(C=1 面同样须过 A/B 门)。

### 腿 4:步开销拆账(overhead)

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_spike.py overhead`(wrapper 逐语句镜像 decode_batch @ c6b311b + torch.profiler 交叉窗 + py-spy 第三口径;C=8 graph,n=35,EOS 窗约束:探针实测 sid7 stop@2 / sid5 stop@55 / sid4 stop@74 → 全窗 ≤54 步,n=35 末端 49 留 6 步余量)。stdout 逐字(当日第 4 样本,前三样本同带):

- `OHEAD step_wall_total us_per_step=19926.1 evidence=perf_counter,c=8,graph,real_B=7,steps=35,seed=20261027`
- `OHEAD sched_schedule us_per_step=4.8 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_ensure(:146) us_per_step=2.3 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_mirror_fill(:150-157) us_per_step=28.1 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_tables_build(:158-160) us_per_step=45.3 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_pad_restore(:166-168) us_per_step=8.8 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_h2d_2x(:170-171) us_per_step=19.6 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_commit_lens(:172,36xB) us_per_step=5.3 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_replay_launch(:173) us_per_step=7.9 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_d2h_sync(:177) us_per_step=19736.6 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD db_gpu_span(ev0->ev1) us_per_step=37.8 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD eng_record(bookkeeping) us_per_step=4.8 evidence=wrapper,c=8,graph,bucket=8,real_B=7,steps=35`
- `OHEAD residual_other us_per_step=62.6 evidence=step_wall−Σ段(engine.step 内 views 查表/num_pages/finally 退休循环/executor 分派)`
- `OHEAD profiler_wall us_per_step=21501.0 evidence=torch.profiler,5 步独立窗(插桩仍在)`
- `OHEAD profiler_gpu us_per_step=19902.9 evidence=self_device_time_total 只按 DeviceType.CUDA 事件累加(profile_decode.py 注记,防 CPU 算子双计)`
- `OHEAD profiler_gap us_per_step=1598.0 evidence=wall−GPU(profiler 口径)`
- `OHEAD cpu_runtime:cudaGraphLaunch us_per_step=1180.3 count_per_step=1.00 evidence=profiler-cpu-events(cuda_runtime)`
- `OHEAD cpu_runtime:cudaMemcpyAsync us_per_step=19857.4 count_per_step=3.00 evidence=profiler-cpu-events(cuda_runtime)`
- `PYSPY denom_engine_step_samples=984 decode_batch_samples=924 total_samples=1431 eff_hz=11.9 evidence=collapsed-stacks,attach=pid,req_rate=200Hz,120s(m6_ab 各档 warmup10+计30 长窗 ×4 组合;attach 采样有税,目标 AB 偏慢,与 knobs 同档对照为准)`
- `PYSPY decode_batch pct_of_engine_step=93.90`
- `PYSPY build_tables pct_of_decode_batch=0.22 samples=2`

三口径互证(如实):

- **真实 CPU 侧步内开销 = wall − d2h_sync = 19926.1 − 19736.6 = 189.5µs/步**(四样本 172–190µs);其余 19.74ms 全部是 graph 等待(GPU 执行)。d2h_sync 内 argmax 只占 ~38µs(gpu_span ev0→ev1)。
- **M4.1 段一「wall−GPU 1.70ms」修正归因:~90% 是 profiler 税**(gap 1598–1716µs,其中 cudaGraphLaunch 一项 1123–1254µs = 74%,余为 runtime 采样开销);真实 CPU prep 从来只有 ~0.17–0.19ms/步。开销面不存在 1.7ms 的可回收空间——旧账修正入档。
- py-spy 独立口径:decode_batch 占 engine.step 采样 93.9%,全部 CPU 段只有 build_tables 捕到 2 个采样(0.22%)——CPU prep 各段低于采样分辨率,与 wrapper 数字互证。
- **py-spy 方法实录(如实)**:spawn 模式(`py-spy record -- <cmd>`)两次 rc=1 `Error: No child process (os error 10)`,且 spawn 吞掉目标 stderr 无法取证;平凡子进程 spawn 测试 rc=0 → 机制未坏,嫌疑在目标进程早夭不可见。改父进程自起目标(stderr 落盘)+ nvidia-smi 轮询就绪 + **attach(-p PID)** 后成功:目标 rc=0 完整跑完四档(父进程收尾后 GPU 1 余留 642MiB,目标装载 16803MiB——首跑「父进程持 ~20GB 致 OOM」假设经 teardown 重排后实测排除)。attach 采样有税:目标 AB 对 knobs 同档偏慢(graph +0.2ms 带内,eager +1.4~3.7ms 出带),结构读法不受影响。

### 可回收清单(RECOVER,C=8 口径)+ 裁决门数据

- `RECOVER knobs_tuning_face ms_c8=0.00 confidence=H risk=R1`(12 组合死平 0.04ms 带宽,实证排除)
- `RECOVER layout_down_nn_hybrid ms_c8=0.40 confidence=M risk=R2`(孤立实测 −11.0~−11.2µs × 36 层;引擎内仅全局 NN 复测过——C=8 赢 −0.6 但 C=1 回归 +0.68 出局;down-only hybrid 引擎内未复测,C=1 面风险未排)
- `RECOVER overhead_step_prep ms_c8=0.10 confidence=M risk=R4`(实测可压面 96.5µs = mirror 28.1 + tables 45.3 + pad 8.8 + h2d 19.6(non_blocking 仅捕获外,R3 约束),按 60–80% 压缩率 + residual 部分压缩 ~20µs 估 0.09–0.10ms;改动落在 graph_runner.py 保护文件,须最小侵入 + token 逐位硬门)
- `RECOVER commit_lens_write ms_c8=0.005 confidence=H risk=R4`(5.3µs 实测;压缩价值趋零,如实:不值得做)

**Σ(RECOVER)= 0.00 + 0.40 + 0.10 + 0.005 ≈ 0.51ms**;该清单同时是段 1 施工顺序候选表(layout down-NN hybrid 0.40 → overhead prep 0.10 → commit_lens 弃)。

裁决门两条件(数据面,判定行留控制器):

1. **Σ = 0.51ms < 2.31ms → 条件 1 成立**。
2. **缺口主体是否在 GEMM 面——两种读法如实呈现**:读法 A(操作口径,vs 孤立冷基准 15.64):GEMM in-situ penalty +0.81ms 占缺口(~2.0–2.1ms)的 **~40%,非主体**;其余 ~1.2ms 在非 GEMM 面(attention 0.646、未融合 norm 链 ~0.39、copy 0.332、other 余量)→ 条件 2 不成立 → **休眠读法**。读法 B(物理屋顶口径):cold 本身 889GB/s = 峰值 88%,cold-vs-roofline ≈ 1.85ms 若计入 GEMM 面,则 GEMM 面合计 ~2.66ms 占步-vs-屋顶距离的 ~65% → 条件 2 成立 → **立项读法**(但段 2 triton 须在 M=7/8 四类形状上先打赢 cuBLAS 冷口径,腿 3 未提供该证据,R5 风险未动)。

**判定行:留控制器**(两条件 + 两读法所需数据已齐,本节不裁)。

口径注记(照前节同款):同日同机(物理 GPU 1,RTX 4090,24GB;每腿跑前 nvidia-smi 验净 4 MiB / 0 进程,前台运行 timeout ≥ 1800s);全部腿 seed=20261027 同批 prompt(token ids 直投),C=8 计时窗内 batch 恒 7(sid7 稳定期 EOS 退役,graph bucket 8 + 1 padding 行);m6_ab 协议 warmup 10 + 计 30;腿 1 冷口径参照 15.64ms 引 M5 段 0 补充二(M=7 四类投影,不含 lm_head);腿 4 wrapper 插桩为逐语句镜像(graph_runner.py @ c6b311b),语义零改动;全部 harness monkey-patch,引擎/模型源码零触碰;py-spy 目标为 `m6_ab.py --steps 30` 全四档长窗;证据归档 `/tmp/opencode/m6/`(四腿 stdout 全文 `*_official.log`/`layout_engine_t.log`、chrome trace `insitu_c{1,8}_{graph,eager}.json`、dmon 全文 `dmon_insitu.log`、py-spy raw `pyspy_overhead.raw` + 目标 stderr `pyspy_target.stderr.log`);锚 3.3ms 与门 2.31ms 引 spec §4 原文,今日残余缺口 ~2.0–2.1ms 为算术注明。

### 段 1·布局账:down-NN hybrid 引擎内复测 —— 双 C 门未过,no-op 收尾(2026-11-01,`scripts/m6_ab.py --layout down_t`,真机输出)

T0 裁决形态 = down-only NN hybrid(全局 NN 出局于 C=1 +0.64 回归;孤立 M=7/8 down NN −11.0~−11.2µs/层 = −0.40ms/步 C=8 口径,唯一胜者),本段 = M6 Task 1 的 Step 1 欠账复测(**先测后写码**:harness `--layout down_t` monkey-patch 仅 down_proj 进 NN,引擎/模型源码零触碰;双 C 门通过才进生产接线)。当日基线跑双样本、RETEST 跑三样本(交错执行控漂移;每跑前 nvidia-smi 验净 4 MiB):

| 样本(step_ms) | C=1 graph | C=8 graph | C=1 eager | C=8 eager |
|---|---|---|---|---|
| baseline ×2(当日) | 17.91 / 17.91 | 19.87 / 19.89 | 19.11 / 19.18 | 21.11 / 21.17 |
| RETEST down_t ×3 | 17.99 / 17.97 / 17.98 | 19.60 / 19.61 / 19.55 | 19.34 / 19.69 / 19.35 | 20.93 / 20.87 / 20.91 |
| Δ(同日配对) | **+0.06 ~ +0.08** | **−0.27 ~ −0.34** | **+0.17 ~ +0.51** | −0.18 ~ −0.30 |

RETEST stdout 逐字(三样本,`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_ab.py --layout down_t --label down_t`):

- `RETEST down_t C=1 mode=graph step_ms=17.99 tpot_ms=18.01` / `C=8 graph 19.60` / `C=1 eager 19.34` / `C=8 eager 20.93`(样本 1)
- `RETEST down_t C=1 mode=graph step_ms=17.97 tpot_ms=17.99` / `C=8 graph 19.61` / `C=1 eager 19.69` / `C=8 eager 20.87`(样本 2)
- `RETEST down_t C=1 mode=graph step_ms=17.98 tpot_ms=18.00` / `C=8 graph 19.55` / `C=1 eager 19.35` / `C=8 eager 20.91`(样本 3)

M=1 归因微测(补 T0 腿 3 盲区「M=1 gemvx 面 NN 未做孤立打分」;`/tmp/opencode/m6t1/attr_m1_down.py`,36 套权重冷口径轮转,down [N=4096, K=12288] bf16,M=1):`ATTR down m=1 variant=tn us=110.15 gb=914` / `variant=nn us=111.43 gb=904`(复跑 tn2=109.74 / nn2=111.08 同向)——**M=1 下 NN 比 TN 慢 +1.3µs/层 × 36 层 ≈ +0.05ms/步,与 C=1 graph 实测 +0.06~+0.08 定量吻合**:C=1 流量的 down 跑在 M=1,NN 布局在该 M 略输 gemvx 对 [N,K] 行主布局的访存;C=8 流量跑 M=7/8(cutlass TN tile),NN 的 [K,N] 布局赢 −11µs/层。同一布局在两类 M 上方向相反,是无条件 hybrid 的结构性矛盾。

判定(双 C 门,如实):C=8 graph 收益 −0.27~−0.34 ≥ 0.2ms ✓、C=8 eager −0.18~−0.30 ✓;但 **C=1 graph 三样本 +0.06~+0.08 全出带(基线带极差 0.04,历史带顶 17.96)且 C=1 eager 三样本全劣于同日基线(+0.17~+0.51)→ no-regression 门 FAIL → 按派单停手:生产接线(qwen3 调用点分派 / EngineConfig / server CLI)不落地,本任务如实 no-op 收尾**。无 revert 面(生产零改动;fast 138 passed, 6 skipped, 17 deselected / slow 17 passed 全套件零改动复核通过)。T0 腿 3 的「孤立赢、引擎内输 = R2 实录」在 down-only 形态上兑现:RECOVER 清单 `layout_down_nn_hybrid ms_c8=0.40` 按引擎内实测修正为 **ms_c8=0.27~0.34 且 C=1 侧 −0.05ms 反向账,门未过不上线**。

候选留存(裁决留控制器,本任务不越权落地):M 条件分派(如 batch≥2 才对 down 走 NN,M=1 维持 TN)两头可通吃——M=8 NN 胜 −11.2µs/层、M=1 TN 胜 +1.3µs/层,均为冷口径实测;但形态变更(布局选择器进 dispatch 面)超出本次裁决的「down-only NN hybrid」,需新裁决与同款双 C 门复测。

### 段 1·布局账 T1b:M 条件分派 down_t_m2(单存储 + 按批 M)—— C=1 门仍未过,revert + no-op 收尾(2026-11-01,生产接线 TDD 全程 + `scripts/m6_ab.py --layout down_t_m2`,真机输出)

上段候选留存按新裁决落地复测:down_proj 权重存 NN 连续 [K,N](`.t().contiguous()` 换 `weight.data`,原 [N,K] 存储随赋值释放——**单存储、稳态零额外显存**;变换期瞬时峰值 = 单层一份权重的 2×,36 层串行即时回收),`linear_dispatch` 按展平行数 M 分派:M≥2 → `torch.mm(x2d, w)`(NN 快径),M<2 → 经 `w.t()` 非连续 view 的 mm;生产接线(qwen3 四类投影 + lm_head 调用点、`EngineConfig.gemm_layout` ∈ {none, down_t_m2}、from_config/_build_engine 装载后 apply、server `--gemm-layout` 透传)TDD 全程,none 硬门经全套件零改动复核(fast **144 passed, 6 skipped, 17 deselected** = 138 基线 + 新增 6 / slow **17 passed**)。

**形态探针(先于实现,/tmp/opencode/m6t1b/,8B down [N=4096, K=12288] bf16)**:派单字面「M<2 走 `mm(x, w.t())`」在 [K,N] 单存储下维度不可行([M,K]@[N,K]);gemvx TN 核仅能由 [N,K] 行主存储到达(F.linear 与 `mv(W, x0)` 均 hit gemvx 且逐位同),单存储 [K,N] 的全部 M=1 形态(`mm(x,w)` / `mm(w.t(),x.t()).t()` / `mv(w.t(),x)` / `F.linear(x,w.t())`)均落 cutlass NN 核(109.9~110.9µs hot)。即 **TN 原径自单存储不可达,获批分派的 C=1 救活机制(M<2 → TN)在存储物理上不可实现**;落地取「走 w.t() view 的 mm」最近合法实现,A/B 按预注册双 C 门裁决。

A/B(交错执行 B/T/B/T/B/T 控漂移;每跑前 nvidia-smi 验净 4 MiB;warmup 10 + 计 30;down_t_m2 腿 = 生产同构,harness 零 forward patch):

| step_ms | C=1 graph | C=8 graph | C=1 eager | C=8 eager |
|---|---|---|---|---|
| baseline ×3 | 17.94 / 17.96 / 17.93 | 19.89 / 19.90 / 19.89 | (21.98 冷启动弃样)/ 19.16 / 19.14 | 21.19 / 21.14 / 21.19 |
| down_t_m2 ×3 | 17.97 / 17.96 / 17.96 | 19.61 / 19.55 / 19.56 | 19.27 / 19.27 / 19.76 | 20.84 / 20.85 / 20.84 |
| Δ(配对) | +0.00 ~ +0.03 | **−0.28 ~ −0.35** | **+0.11 ~ +0.13(2/3;s3 +0.62 更差)** | −0.29 ~ −0.35 |

判定(双 C 门,如实):C=8 收益 −0.28~−0.35 ✓(复现 T1 的 −0.27~−0.34,NN 快径收益与分派结构解耦成立);**但 C=1 eager +0.11~+0.13(2/3 系统性,T1b 双样本 19.27 齐落 T1 基线带顶 19.18 之上)→ no-regression 门 FAIL → 按预注册协议 revert:生产接线(gemm_dispatch / qwen3 调用点 / EngineConfig / server CLI / m6_ab 腿)全部回退,本任务 docs no-op 收尾**。C=1 graph +0.00~+0.03(带顶持平)不构成独立通过项(双 mode 条件按「任一」计)。

C=1 eager 劣化拆账(`/tmp/opencode/m6t1b/dispatch_overhead.py`,10k 次/形态 per-call wall):dispatch M<2 分支较 module 调用 **+1.39µs/调用 × 36 层 ≈ +0.050ms/步**(python 分派面:分支多 reshape/t/t/view 四个 view 算子),叠加 cutlass-NN-vs-gemvx 核面差(T1 冷口径 +1.3µs/层 ≈ +0.047ms)≈ +0.10ms,与实测 +0.11~+0.13 定量吻合;none 分派面 −0.09µs/调用(≈0,硬门零损)。

数值等价账(标定命令 `/tmp/opencode/m6t1b/probe_tiny_candA.py`,2026-11-01):tiny 形状 M=1 径逐位(8 seeds max|diff|=0.0)、M≥2 径 max|diff|=0.0039(TOL_BF16 内);8B 实形状 M=1 max|diff|=0.0156、M∈{2,7,8} ≤0.0625(「M=1 逐位」预期在 8B 规模不成立,TOL 协议覆盖;tiny 引擎级双 bucket parity 在任务期单测全绿,随 revert 一并撤下)。

账目修正:RECOVER 清单 `layout_down_nn_hybrid` 的「M 条件分派两头通吃」候选**证伪入档**——C=8 侧 −0.28~−0.35 真实且无需分派(无条件 down_t 已复现),C=1 侧的救活以「TN 原径可达」为前提,而该前提被存储物理证伪([K,N] 单存储不可达 gemvx);除非接受双存储(零双存储红线,禁)或 M=1 侧改非 cuBLAS 径(段 2 triton 范围),该面在 cuBLAS 布局维度已收干。

### 段 1·开销账:步开销面收尾(mirror 批量 + commit_lens + H2D/D2H)—— 施工面全在禁改文件、解禁面捕获期影子,no-op 收尾(2026-11-01,M6 Task 2-4 合并执行,真机输出)

T0 合并裁决(RECOVER 三面):overhead_step_prep 0.10ms(M)+ commit_lens_write 0.005ms(H)+ H2D/D2H 未上榜,合计 ~0.105ms 低于原单任务门 0.2ms → 合并一个任务,门降为 **wall-step ≥ 0.05ms(C=8 graph,n=3 交错)**;并预设观察:「host 侧改善不进 step 时间(graph 模式 host 隐于 GPU 影内)→ 如实 no-op 收尾(也算完成)」。本节逐子项交代(禁改条款:graph_runner.py 自 M5 起禁改;pager.py 本任务有限解禁——仅 `_commit_lens` 相关批量提交,签名与语义不变):

- **镜像/padding/tables 扁平构建**(`graph_runner.py:149-168`,RECOVER 实测面 28.1+45.3+8.8µs):施工面全部位于禁改文件 → **no-op**(裁决条款明示「若施工需触碰 graph_runner,该子项直接 no-op」;build_tables 的 pin_tables 直填路径同此)。
- **`_commit_lens` 批量提交**(RECOVER 实测 5.3µs;调用 `graph_runner.py:172`、**函数体 :179-184 在禁改文件内**):按裁决条款「函数体在哪个文件以现状为准——若在 graph_runner 内则同样禁改 → no-op」→ **no-op**(T0 已注「压缩价值趋零,不值得做」;252 次 dict 赋值/步的批量改法无可改落点,executor.py:87 仅薄分派,调用链整体在禁改方法体内,harness 式 monkey-patch 属规则规避非施工,不做)。
- **H2D/D2H 清账**(RECOVER 榜未上榜,如实复核):CPU 侧 copy_ 面属 overhead_step_prep 同文件(:170-171,禁改);传输本体 GPU 侧仅 **3.4µs**(Memcpy HtoD Pinned→Device,count 2)+ **1.0µs**(DtoH,count 1)——本日腿 4 复跑 profiler 设备事件实测,「未上榜」获实证,无回收面 → **no-op**。
- **pager `scatter_decode` 逐 view 提交**(`pager.py:112-113`,解禁条款内):graph_runner docstring :6-8 契约——图内 python(含 scatter_decode 的 `_lens` 提交)**只在捕获期执行一次,重放不执行**,per-step 提交由 runner `_commit_lens` 代行 → graph 主口径(门口径)下该面零 per-step 存在,批量重构不可能进 wall-step → **不上手**(施工必然不达门 → revert,纯 churn;签名/语义不变条款下亦无收益形态)。

当日实测锚(证据在档):腿 4 复跑(`m6_spike.py overhead`,与 T0 同带)——step_wall 19956.8µs、mirror 29.9 / tables 49.8 / pad 10.8 / h2d 20.6 / commit_lens 5.7 / d2h_sync 19738.8(T0:19926.1 / 28.1 / 45.3 / 8.8 / 19.6 / 5.3 / 19736.6),真实 CPU 侧 218.0µs/步(插桩税内,T0 带 172–190),其余 19.74ms 全为 graph 等待;`m6_ab.py --label ohead_batch_pre{,2}` 基线双样本 **C=8 graph 19.91 / 19.92**(T0 带 19.89–19.93 内)、C=1 graph 17.95/17.92、C=8 eager 21.13/21.23。

判定(如实,no-op):**施工面为空集** —— 三面 + 未上榜面共四个子项,三个在禁改文件内、唯一解禁面(pager)在门口径下捕获期影子化;零施工 → 改后 ≡ 改前(A/B 退化为 A/A),门 wall-step ≥ 0.05ms 不可能被任何合规改动满足 → 按裁决「不达 → revert + no-op」收尾。无 revert 面(生产零改动);等价钉测试(批量 vs 逐赋值 `_lens` 全等)因无施工落点不落(零语义面可钉,避免无生产缝的自指测试);全套件零改动复核通过(fast **138 passed, 6 skipped, 17 deselected** / slow **17 passed**,byte-identical 硬门自然成立:token 流逐位、`_lens` 逐 dict 现状保持)。开销面在当前禁改边界内已收干;若后续解禁 graph_runner.py,候选形态照 T0(RECOVER `overhead_step_prep`:mirror 批量构建 + padding 压回合并,估 0.09–0.10ms)直接立项,无需重拆账。
