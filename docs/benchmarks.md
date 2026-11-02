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

### 段 2·布局账:decode GEMM triton kernel(条件立项,限范围)—— 两类均未过 ≥5% 接入阀,no-op 收尾(2026-11-01,`acorninfer/ops/decode_gemm.py` 装载期 `configure()` 实测 + 六轮探针迭代,真机输出)

范围(限范围裁决):仅两类 —— **gate_up**(引擎真实 per-call 形状 N=12288, K=4096;gate_proj / up_proj 各一次,段 0 spike 的 N=24576 为融合口径)与 **down**(N=4096, K=12288);qkv/o 段 0 冷口径已 88–92% roofline、lm_head 93.5% 峰值,均不接。**偏差注记(预注册)**:接线形态取 per-call 逐调用分派(每层 2× [12288,4096])而非融合 [24576,4096] 单 GEMM —— 零权重存储变更(段 1 T1b 双存储红线延续),qwen3 MLP 结构不动(分派仅换单算子),阀逐调用生产形状评估;融合口径仅作探针参照(见下物理上限账)。接入阀(spec §6 R5 延续):某类 triton 冷口径须超该类 cuBLAS 最优变体(tn / nn / pad16 取最小)≥5% 才接入,逐类独立、允许部分接入;全部未过 → 诚实 no-op 收尾(本节即此结局)。

**kernel 设计要点**(`acorninfer/ops/decode_gemm.py`,新文件,全量单测):①BLOCK_M=16 张量核路径,M∈(1,16] pad 到 16(`tl.dot` 下限);W 侧零存储变换(nn.Linear [N,K] 行主,行内 K 连续 → 瓦片流式纯权重读,免 mask),M 侧 mask(瓦片 [BN,BK]×stages 共享内存预滤 ≤96KB)。②split-K(down:naive grid 仅 64 programs vs 128 SM)= **fp32 partial buffer + 确定性 second-pass 归约**(非 atomic_add):逐位确定性(graph 重放与独立 launch 输出逐位同,parity margin 稳定)、无 atomic 争用、reduce 读 SPLIT_K×16×N fp32(~1MB)≈1–2µs 可忽略;workspace 装载期预分配常驻(地址冻结,重放安全)。③CUDA-graph 兼容硬约束:launch 路径零 host 同步(无 .item()/synchronize,grid 全 host 侧 int;单测以 `set_sync_debug_mode` 钉),autotune/表实测全部装载期(Engine 构造 = 捕获前),**graph 捕获内含 triton launch(split-K=4)捕获成功 + 重放确定性(torch.equal)单测在档**。④dispatch 表逐 (类, M-bucket) 装载期实测,M buckets = graph_runner.BUCKETS(禁改,(1,2,4,8),测试钉住一致);layout="none"(缺省)直通 F.linear 逐字节硬门。

**探针迭代账(六轮;证据 `/tmp/opencode/m6t5/iter{3,4,5,6}_probe*.log`,冷口径 8 套轮转)**:iter1 初版网格(BN{64,128,256}×BK{64,128}×warps{4,8}×stages{2,3,4}+split)即 +1.1~+2.6% 两类未过;iter2 扩网格(BK 256、warps 2、两类开放 split、TRANS_B 直载 / CACHE .cg+evict_first 访存变体)→ 变体全部噪声级(±0.5%)且 warm 装载涨到 3.22s 超预算 → 按探针证据回削;iter3 判别探针:**同瓦片纯读上限 gate_up 919–920GB/s = GEMM 920–923 本身**(kernel 已在该访存模式天花板),torch `w.sum()` 归约读 909(低于本 kernel),PAD_X 免 mask、stages 5/6 深流水、BK=64 全更差;down split=1 纯读仅 752–754(64 CTA 半 GPU 空转)→ **CTA 并行度 = down 主瓶颈**;iter4/5:BN=32(384 CTA)纯读 934 / GEMM 926(gate_up 新高),down split {3,4,6}(128→384 CTA)实测 859–907 全不敌 split2(902;reduce 流量随 SPLIT_K 涨吞掉并行收益),gate_up split ≤907 全败;iter6 SWAP 操作数序(W 进 A 操作数免 trans)925–927 / 894–906 噪声级(正确性 maxabs=0.0)。**融合口径物理上限参照**:triton 在 fused 24576×4096(768 CTA)纯读 944 / GEMM 937(cuBLAS fused 段 0 实测 947)—— 即便并行度翻倍,triton 渐近线 ~944 仍低于 per-call 阀线(下表),**阀不可达为形状物理,非工程欠账**。

装载期 autotune(终版网格;iter2 宽网格证据保留不入网格):段1 BN{64,128,256}×BK{64,128,256}×warps{4,8}×stages{2,3,4}(shmem 预滤,24 活配置)+ 段2 top-2×SPLIT_K{2,4}(整除滤);**wall:cold 6.06s(含编译)/ warm 1.16s(<2s 目标达成;iter1 5.26/1.12、iter2 10.94/3.22 演进如实入档)**。胜出 config:gate_up `(BLOCK_N=64, BLOCK_K=128, warps=8, stages=2, split=1)`(192 programs)、down `(64, 128, 4, 4, 2)`(128 programs = 整 1 波)。逐字 stdout(终版 canonical,`configure_final.log`):

```
T5 AUTOTUNE_BEST gate_up cfg=(64, 128, 8, 2, 1) us=110.23 gb=913 skipped=0
T5 AUTOTUNE_BEST down cfg=(64, 128, 4, 4, 2) us=112.97 gb=891 skipped=0
T5 CONFIGURE cold wall_s=6.06
T5 CONFIGURE warm wall_s=1.16
```

**冷口径对照表(36 套轮转 >72MB L2 强制 HBM;两类单套权重 100.66MB 本就超 L2,hot≈cold 双重成立;吞吐按 N·K·2B=100.66MB 计)**:

| 类 | M | triton µs (GB/s) | tn µs | nn µs | pad16 µs | best cuBLAS | win vs best | 阀(≤0.95×best) | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| gate_up | 1 | 108.38 (929) | 107.76 | 109.03 | 118.62 | 107.76 | −0.6% | 102.37 | ✗ |
| gate_up | 2 | 108.40 (929) | 111.84 | 111.10 | 118.70 | 111.10 | +2.5% | 105.55 | ✗ |
| gate_up | 4 | 108.62 (927) | 115.27 | 111.00 | 116.72 | 111.00 | +2.2% | 105.45 | ✗ |
| gate_up | 8 | 108.78 (925) | 115.71 | 111.16 | 116.61 | 111.16 | +2.2% | 105.60 | ✗ |
| down | 1 | 110.46 (911) | 109.91 | 116.07 | 123.89 | 109.91 | −0.5% | 104.41 | ✗ |
| down | 2 | 110.74 (909) | 122.78 | 112.13 | 124.02 | 112.13 | +1.3% | 106.52 | ✗ |
| down | 4 | 111.16 (906) | 122.98 | 112.43 | 123.94 | 112.43 | +1.1% | 106.81 | ✗ |
| down | 8 | 111.63 (902) | 123.34 | 113.84 | 124.01 | 113.84 | +2.0% | 108.15 | ✗ |

```
T5 VALVE gate_up adopted=False active=[]
T5 VALVE down adopted=False active=[]
```

**判定(如实,no-op):两类均未过 ≥5% 阀 —— triton 不接入生产,cuBLAS TN 继续站岗。**三层证据:①逐桶最优 +2.2~+2.5%(gate_up)/ +1.1~+2.0%(down),距 5% 差 2–3 个百分点;M=1 桶两类皆负(gemvx TN 107.76 / 109.91µs ≈ 934/915GB/s,BLOCK_M=16 张量核路径对 M=1 无算力优势,预期内);②**统计学注记**:cuBLAS nn 变体逐轮漂移 ±5%(gate_up nn m=2 同脚本两遍 111.10 → 113.92 → 110.87;down nn m=1 116.07/115.46/118.45),+2% 级边缘优势在 cuBLAS 自身方差内,阀值 5% 正是对此类收益的过滤——按 +2% 接入属统计上不可靠决策;③**物理上限**:同瓦片纯读天花板(gate_up 920–934、down ≤904)、triton 融合口径渐近线 937–944(cuBLAS 融合 947)均低于阀线对应吞吐(gate_up ≥953、down ≥944),kernel 已在其访存模式的 99–100%(GEMM 923 ≈ 纯读 920)。段 0 R5「须先打赢 cuBLAS 冷口径」的立项读法就此证伪入档:triton 能赢 tn 变体(gate_up m≥2 +3.8~+6.3%)但赢不了最优变体 5%。

TOL 标定账(`tests/test_decode_gemm.py` 模块头,协议 `tests/triton_utils.py` 只用不改):`TOL_BF16_GEMM = dict(rtol=2e-2, atol=1e-2)` 写死(2026-11-01,标定命令 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/opencode/m6t5/calib_tol.py`,gate_up/down × M∈{1,7,8,16} × tuned cfg,seed=11):实测 maxrel(|ref|>1) = 7.752e-3(≈2 bf16 ulp)、maxabs(|ref|≤1) = 0.0039、max|diff| = 2.0(|ref|≈258)——大 K 归约序漂移在 bf16 输出上为绝对差随幅值走,故 rtol+atol 组合(纯绝对差口径在此不适用,如实注记);(1e-2,1e-2) 下最紧比 0.758(余量 1.3×)→ 2.6× 余量取 (2e-2,1e-2),复算最紧比 ≈0.39;真错 O(10) 起,不可漏网。

测试账(`tests/test_decode_gemm.py`,17 项全绿;fast 合计 **155 passed, 6 skipped, 17 deselected** = 基线 138 + 新 17 / slow **17 passed**):两类正确性对拍 F.linear × M{1,7,8,16}(标定 TOL)、split-K 强制 SPLIT_K=4(含 M=1)、逐位确定性(两次独立 launch torch.equal)、bucket 归桶 + 与 graph_runner.BUCKETS 一致性钉、dispatch 路由 mock(none 硬门 / 桶级分派)、阀逻辑 mock ×2(过阀接入 / 全败不接)、零 host 同步(set_sync_mode)、**graph 捕获含 triton launch + 重放逐位**、cuBLAS 回退桶。EngineConfig gemm_layout 校验测试随字段回退一并撤下(偏差注记见下)。

生产接线判定:**零接线** —— qwen3.py / server.py / m6_ab.py 全未触碰,`EngineConfig.gemm_layout` 字段已回退(config.py 还原至 240f0cc;无消费者不设死旋钮),`linear_dispatch` 无生产调用点,生产行为与改前逐字节同(硬门自然成立)。模块 + 测试保留存档(kernel / 实测仪 / 阀机制全绿可复跑,后续若融合存储(T1b 前提)或存储布局变更重开此面,`configure()` 直接可用)。RECOVER 清单更新:`layout_down_nn_hybrid`(段 1 证伪)与本次 triton 面合并结论 —— **GEMM 布局/实现面在 cuBLAS+triton 双支柱下已收干,段 0 可回收面仅剩 knobs(证伪)与 overhead(禁改影子)均 no-op,缺口须由段 3+ 其它面承担**。

偏差清单(如实):①gate_up 按 per-call [12288,4096]×2 接线评估(预注册决策;融合口径仅作探针参照,理由零存储变更);②iter2 曾加宽 autotune 网格致 warm 3.22s 超 <2s 目标 → 按探针证据回削至终版(warm 1.16s 达标),宽网格数据留档不入产;③TRANS_B/CACHE 访存变体、PAD_X、SWAP、split>2、stages 5/6、BN=32 七类变体探针证伪(噪声级或更差),kernel 内仅保留 TRANS_B/CACHE constexpr(缺省 (0,0) 与初版逐位同,变体账在 docstring);④EngineConfig.gemm_layout 字段施工后回退 + 对应测试撤下(17 项而非 18);⑤iter4 探针脚本曾绕过整除断言触发 illegal access(gate_up split3/6 非整除),为探针脚本缺陷非 kernel 缺陷(`_launch` 断言在档),已在新进程修复复测。

口径注记(照前节同款):同日同机(物理 GPU 1,RTX 4090,24GB;每轮跑前 nvidia-smi 验净 4 MiB / 0 进程,前台 timeout ≥1800s);冷口径协议 = 段 0 `--gemm-cold` 同款(36 套独立权重轮转,warmup 一整轮 + 计 5 轮 ÷ 轮×36,CUDA event 计时;nn 变体 8 套轮转);autotune 仪 = 8 套轮转同协议;证据归档 `/tmp/opencode/m6t5/`(cold_cublas.py、calib_tol.py、run_configure.py、iter3/4/5/6 探针脚本 + 全部 stdout log、configure_cold.log / configure_iter2.log / configure_final.log)。

## M6 终账(2026-11-01,收官仪式:零漂移硬证 + 双轨仪式轨 1 主验 + 判定行落笔,真机输出)

**裁决记录(承段 0/控制器)**:段 0 四腿拆账数据面收口——RECOVER 清单 Σ ≈ **0.51ms**(knobs 0.00(H)+ layout down-NN hybrid 0.40(M)+ overhead prep 0.10(M)+ commit_lens 0.005(H)),对残余缺口 ~2.0–2.1ms;裁决门条件 1(Σ < 2.31ms)成立,缺口主体两读法并存下**控制器取读法 B(物理屋顶口径)段 2 立项,限范围**(仅 gate_up [12288,4096]×2/call 与 down [4096,12288] 两类;qkv/o 冷口径已 88–92% roofline、lm_head 93.5% 峰值不接;接入阀 = triton 冷口径超该类 cuBLAS 最优变体 ≥5%)。段 2 结局见前节:两类 +1.1~+2.5% 全部未过阀,**兑现 0ms**。

**计划偏离(控制器批准,如实入档)**:双轨仪式的轨 2(TP=2)**免跑**——M6 五个施工任务(T1/T1b/T2-4/T5)全部如实 no-op、生产净变化 = 零(零漂移硬证见下),TP=2 外溢量问题在零 delta 下 moot,轨 2 终账数字延续 M5 收官(vLLM TP=2 99.44/321.79/541.94 vs AcornInfer 满配方 72.92/233.09/349.40,达标率 73.3%/72.4%/64.5%)。本仪式 = **轨 1 主验**。

### 终账·零漂移硬证(先行,生产净变化 = 零)

`git diff 7a20a12..HEAD --stat -- acorninfer/`(原样;逐行核对):

```
 acorninfer/ops/decode_gemm.py | 493 ++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 493 insertions(+)
```

唯一运行时树差异 = M6 Task 5 新增的 `acorninfer/ops/decode_gemm.py`;grep 全仓核实**零运行时 import**(仅 `tests/test_decode_gemm.py` 引用;engine/models/server 运行时链只 import paged_attention/rope/rmsnorm,config.py 还原至 240f0cc)——生产行为与 7a20a12 逐字节同,硬门自然成立。

全量回归三尾行(零换金;跑前双卡各 4 MiB 验净):

| 套件 | 命令 | 尾行(逐字) | 判定 |
|---|---|---|---|
| fast | `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` | `155 passed, 6 skipped, 17 deselected, 1 warning in 11.11s` | = 预期(T5 后基线),零改动 ✓ |
| slow | `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q -m slow` | `17 passed, 161 deselected in 398.53s (0:06:38)` | golden 原样,零换金 ✓ |
| 双卡 tp 全套 | `CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest -q tests/test_tp_eager.py tests/test_tp_engine.py tests/test_tp_server.py` | `13 passed in 18.90s`;跑后双卡回落 4 MiB 双查,无计算进程 | 全绿 ✓ |

tp 计数注记(如实):计划头基线行写「双卡 tp 全套 **16 passed**」;三个 TP 测试文件(M5 期 20e8059→7a20a12 全部提交)在任意提交点收集数最多为 **13 项**(eager 1 + engine 3 + server 8 函数含 e2e 参数化 ×2 = 13),16 从未与文件收集吻合——计数入档以实跑尾行为准,文件与 7a20a12 基线逐字节同(零换金结构成立),差异本身如实记录不抹平。

### 终账·in-process 终bench 零漂移(对照 M5/M6 收官带)

命令(每跑前 GPU 1 验净 4 MiB,三样本串行):`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_ab.py --label m6_final`(warmup 10 + 计 30,stdout 逐字归档 `/tmp/opencode/m6t6/m6_final_s{1,2,3}.log`):

| 样本(step_ms) | C=1 graph | C=8 graph | C=1 eager | C=8 eager |
|---|---|---|---|---|
| m6_final s1 | 18.00 | 19.91 | 19.16 | 21.19 |
| m6_final s2 | 17.93 | 19.92 | 19.14 | 21.37 |
| m6_final s3 | 17.94 | 19.92 | 19.23 | 21.23 |
| 当日带(M6 段 0–段 2 基线带) | 17.91–17.96 | 19.87–19.93 | 19.13–19.35 | 21.11–21.28 |

判定(如实):graph 两档 12 样本全部贴当日带(C=1 graph 17.93–18.00,s1 18.00 出历史带顶 17.96 +0.04——基线带极差本 0.04–0.05ms 级噪声;C=8 graph 19.91–19.92 与 T2-4 收官锚 19.91/19.92 逐位同带)——**引擎级零漂移成立**;eager C=8 s2 21.37 出带顶 +0.09(该档历史带宽本宽,T0 有 22.24 离带先例如实入档),其余在带。零生产 delta 与零漂移互证。

### 终账·轨 1 仪式(GPU 0 同卡对照;照 M5 配方,AcornInfer 腿升级满配方)

配方注记(照抄 + 偏离如实):命令/端口/协议照 docs M5 节原样(seed=20261027、in=128/out=32、temperature=0、max_tokens=32、warmup 1 条不计入;`ss -tlnp` 取真身 PID 定向 kill,每腿切换双卡 4 MiB 验净)。**两处口径升级如实入档**:①M5 轨 1 验收腿为 in-process 满配方(server 无配方旗标所致);M5 修复后 server 支持 `--attention-backend/--fused-ops/--cuda-graph`,本仪式**主验腿 = AcornInfer HTTP 满配方 8200**(HTTP 对 HTTP,产品真实路径);②GPU 归属按本仪式派单**全腿 GPU 0 同卡**(M5:vLLM GPU 0 / AcornInfer GPU 1;双 4090 同型,in-process 数字与 M5 GPU 1 带零漂移互证双卡等价)。vLLM 就绪 ~40 s;每腿起跑前 `nvidia-smi` 双卡 4 MiB 实录在档。

命令原样(四腿串行):

- vLLM TP=1:`CUDA_VISIBLE_DEVICES=0 /home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90 --dtype bfloat16`
- AcornInfer HTTP 满配方(主验):`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.server --model-dir /home/ernest/model-weight/Qwen3-8B --port 8200 --attention-backend triton --fused-ops --cuda-graph`
- AcornInfer in-process graph(M5 验收口径延续腿):`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.bench --in-tokens 128 --out-tokens 32 --attention-backend triton --fused-ops --cuda-graph`;eager:同去 `--cuda-graph`
- AcornInfer HTTP 缺省配方(补充口径):`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.server --port 8200`
- 压测:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --base http://127.0.0.1:{8100|8200} --concurrency 1 4 8`

RESULT 行逐字(stdout 原样归档 `/tmp/opencode/m6t6/leg{A..E}_*.log`;`RESULT-VLLM` 前缀为 `bench_remote` 固定标签):

- vLLM TP=1 HTTP 8100:`RESULT-VLLM C=1 ttft_p50=0.0270s ttft_mean=0.0270s tpot_mean=16.89ms tok_s=58.12`
- (续)`RESULT-VLLM C=4 ttft_p50=0.0682s ttft_mean=0.0576s tpot_mean=17.86ms tok_s=207.49`
- (续)`RESULT-VLLM C=8 ttft_p50=0.0810s ttft_mean=0.0740s tpot_mean=17.81ms tok_s=355.91`
- AcornInfer HTTP 满配方 8200(主验):`RESULT-VLLM C=1 ttft_p50=0.0379s ttft_mean=0.0379s tpot_mean=18.31ms tok_s=52.85`
- (续)`RESULT-VLLM C=4 ttft_p50=0.0814s ttft_mean=0.0814s tpot_mean=20.72ms tok_s=174.26`
- (续)`RESULT-VLLM C=8 ttft_p50=0.1846s ttft_mean=0.1665s tpot_mean=20.96ms tok_s=276.55`
- AcornInfer in-process graph(满配方,M5 验收口径):`RESULT C=1 ttft_p50=0.0393s ttft_mean=0.0393s tpot_mean=17.96ms tok_s=53.68`
- (续)`RESULT C=4 ttft_p50=0.0777s ttft_mean=0.0777s tpot_mean=19.54ms tok_s=187.27`
- (续)`RESULT C=8 ttft_p50=0.1446s ttft_mean=0.1446s tpot_mean=19.97ms tok_s=296.88`
- AcornInfer in-process eager(triton+fused):`RESULT C=1 ttft_p50=0.0360s ttft_mean=0.0360s tpot_mean=19.61ms tok_s=49.70`
- (续)`RESULT C=4 ttft_p50=0.0845s ttft_mean=0.0845s tpot_mean=21.03ms tok_s=173.81`
- (续)`RESULT C=8 ttft_p50=0.1465s ttft_mean=0.1465s tpot_mean=21.17ms tok_s=281.80`
- AcornInfer HTTP 缺省配方 sdpa(补充口径):`RESULT-VLLM C=1 ttft_p50=0.0364s ttft_mean=0.0364s tpot_mean=26.98ms tok_s=36.67`
- (续)`RESULT-VLLM C=4 ttft_p50=0.1237s ttft_mean=0.1019s tpot_mean=44.85ms tok_s=85.12`
- (续)`RESULT-VLLM C=8 ttft_p50=0.2061s ttft_mean=0.1857s tpot_mean=60.82ms tok_s=111.91`

**轨 1 表(M6 终账;vLLM HTTP vs AcornInfer 满配方两口径 + eager/缺省补充):**

| 腿 | C=1 tok/s | C=4 tok/s | C=8 tok/s | TTFT p50 C=1/4/8 (s) | TPOT mean C=1/4/8 (ms) |
|---|---|---|---|---|---|
| vLLM TP=1 HTTP | 58.12 | 207.49 | 355.91 | 0.0270 / 0.0682 / 0.0810 | 16.89 / 17.86 / 17.81 |
| AcornInfer HTTP 满配方(**主验腿,M6 口径**) | 52.85 | 174.26 | 276.55 | 0.0379 / 0.0814 / 0.1846 | 18.31 / 20.72 / 20.96 |
| AcornInfer in-process graph(M5 验收口径,延续对照) | 53.68 | 187.27 | 296.88 | 0.0393 / 0.0777 / 0.1446 | 17.96 / 19.54 / 19.97 |
| AcornInfer in-process eager(triton+fused) | 49.70 | 173.81 | 281.80 | 0.0360 / 0.0845 / 0.1465 | 19.61 / 21.03 / 21.17 |
| AcornInfer HTTP 缺省配方 sdpa(**非验收口径**) | 36.67 | 85.12 | 111.91 | 0.0364 / 0.1237 / 0.2061 | 26.98 / 44.85 / 60.82 |

达标率算术(逐档,AcornInfer ÷ vLLM):

- **主验(HTTP 满配方)**:C=1 52.85/58.12 = **90.9%**、C=4 174.26/207.49 = **84.0%**、C=8 276.55/355.91 = **77.7%**;
- M5 验收口径(in-process graph):C=1 53.68/58.12 = **92.4%**、C=4 187.27/207.49 = **90.3%**、C=8 296.88/355.91 = **83.4%**——对 M5 收官带 92.4%/90.0%/83.7% = **+0.0 / +0.3 / −0.3 pp,机时漂移带内,轨 1 验收数字零漂移**;
- eager 85.5%/83.8%/79.2%(M5 带 87.2/84.5/79.6);HTTP 缺省 63.1%/41.0%/31.4%(M5 带 61.7/42.4/32.6,含已知配方差,非验收口径)。

判读(如实):①满配方 HTTP 主验对 M5 in-process 验收口径的差(C=8 296.88→276.55,×0.931)= SSE 流式全程开销在满配方引擎上的占比,与 M5 判读 4 的 HTTP 开销小修正项口径(×0.96–1.0 于缺省配方)同向偏大——C=8 档 TTFT 0.1846s 复刻 M5 in-process 同档 0.1450s + 发射合批排队轮廓(与 M5 HTTP 缺省腿 0.1931s 同型),如实记为主验口径的方法学代价,不进引擎归因;②vLLM 三档对 M5 收官 +0.1%/−0.1%/+0.5%,基线侧同样零漂移。

**收服实录(每腿切换验净)**:vLLM 真身 PID 4105695(`ss -tlnp` 定位)定向 kill → 端口闭、双卡 4 MiB;AcornInfer 满配方 PID 4120684、缺省 PID 4193056 同法收服,均 SIGTERM 优雅退出、端口闭、双卡 4 MiB、无计算进程;全程无 pkill 模糊匹配,无包装壳误伤。

### 终账·逐任务结局表(M6 五施工任务,全部如实 no-op,生产净变化 = 零)

| 任务 | 面 | 结局 | 关键定量证据 |
|---|---|---|---|
| T1 | 布局面:down-only NN hybrid | **no-op**(双 C 门 FAIL) | C=8 graph −0.27~−0.34 ✓ / eager −0.18~−0.30 ✓,但 **C=1 graph 三样本 +0.06~+0.08 全出带**、C=1 eager +0.17~+0.51;归因微测定量闭环:M=1 下 NN 比 TN 慢 +1.3µs/层 × 36 ≈ +0.05ms/步,与实测 +0.06~+0.08 吻合——同一布局在 M=1(gemvx 行主)与 M=7/8(cutlass tile)方向相反,结构性 |
| T1b | M 条件分派 down_t_m2(单存储 + 按批 M) | **revert + no-op**(C=1 门仍未过) | 形态探针:**[K,N] 单存储下 TN 原径(gemvx)不可达**(全部 M=1 形态落 cutlass NN 核 109.9~110.9µs hot)——获批分派的 C=1 救活机制被存储物理证伪;C=8 收益 −0.28~−0.35 复现 ✓,C=1 eager +0.11~+0.13(2/3 系统性,dispatch 面 +1.39µs/调用 + 核面差定量吻合)→ no-regression 门 FAIL 按预注册协议回退;双存储形态(+3.6GB 显存换 ~0.3ms)违零双存储红线,判不值,留台账 |
| T2-4 | 开销面:mirror 批量 + commit_lens + H2D/D2H + pager scatter | **no-op**(施工面空集) | 四子项三个在禁改文件内(graph_runner.py :149-184),唯一解禁面(pager `_lens`)在 graph 门口径下**捕获期执行一次、重放不执行** = host 隐于 GPU 影;腿 4 三口径互证:真实 CPU 侧步内开销仅 172–218µs/步(M4.1 旧账「wall−GPU 1.70ms」修正归因 ~90% profiler 税),开销面不存在 1.7ms 可回收空间;零施工 → 改后 ≡ 改前,门 wall-step ≥ 0.05ms 不可能被合规改动满足 |
| T5 | 段 2:decode GEMM triton kernel(限范围:gate_up + down) | **no-op**(零接线,cuBLAS TN 继续站岗) | 六轮探针迭代后两类逐桶最优仍仅 **+1.1~+2.5% < 接入阀 5%**(且 M=1 桶两类皆负);三层证据:①cuBLAS nn 变体逐轮漂移 ±5%,+2% 级优势统计不可靠;②**同瓦片纯读天花板 gate_up 920–934 / down ≤904 GB/s,GEMM 923/920 ≈ 100%**——kernel 已在其访问模式读上限;③triton 融合口径渐近线 937–944 仍低于阀线(cuBLAS 融合 947)——**阀不可达为形状物理,非工程欠账** |

### 终账·可回收 → 兑现对照表(RECOVER Σ≈0.51ms 预估 vs 0ms 兑现,逐项死因)

| RECOVER 项 | 预估 ms_c8(conf) | 兑现 | 死因(如实) |
|---|---|---|---|
| knobs_tuning_face | 0.00(H) | 0 | 12 组合死平(极差 0.04ms),实证排除;本就无施工面 |
| layout_down_nn_hybrid | 0.40(M) | 0 | C=8 收益真实(引擎内复现 −0.27~−0.35)但 C=1 结构性回归(T1 微测定量闭环);M 条件分派救活被存储物理证伪(T1b:[K,N] 单存储 TN 不可达)→ 双 C 门 FAIL 不上线;布局面在 cuBLAS 维度收干 |
| overhead_step_prep | 0.10(M) | 0 | 施工面全在禁改文件;解禁面捕获期影子化(graph 重放不执行 host 提交);真实 host prep 仅 ~0.17–0.19ms/步且隐于 GPU 影 → 合规改动不可能进 wall-step |
| commit_lens_write | 0.005(H) | 0 | T0 已判「压缩价值趋零,不值得做」;调用链整体在禁改方法体内 |
| (段 2 立项)triton GEMM | —(读法 B 物理面) | 0 | 两类 +1.1~+2.5% < 阀 5%;同瓦片纯读天花板 = GEMM 实测(99–100%),triton 渐近线 937–944 < cuBLAS 947——形状物理 |
| **Σ** | **≈ 0.51** | **0** | 五面全部实证收干:三面(triton/布局/调参)kernel 与布局物理、一面(开销)禁改影子、一面(commit_lens)价值趋零 |

### 终账·判定行(如实,本里程碑最重要的一行)

**锚「TP=1 graph tok_s 逐档 ≥100%」未达**:主验(HTTP 满配方)90.9%/84.0%/77.7%,M5 验收口径(in-process graph)92.4%/90.3%/83.4%(对 M5 收官带 +0.0/+0.3/−0.3 pp,机时带内零漂移)——距 100% 线余 7.6–9.1 / 9.7–16.0 / 16.6–22.3 pp,逐档未过,**北极星继续挂账**。归因(三面实证闭环):**权重流访存天花板**——四类 GEMM 已在其访问模式的读上限 99–100%(gate_up 923 ≈ 同瓦片纯读 920 GB/s、down 902–911 ≈ 纯读 ≤904;qkv/o 冷口径 88–92% roofline、lm_head 93.5% 峰值),**cuBLAS ≈ 物理上限**;triton(段 2:+1.1~2.5% < 阀 5%,融合渐近线不及 cuBLAS)、布局(T1/T1b:C=1 结构性回归 + TN 单存储不可达)、调参(腿 2:12 组合死平)三面全部实证排除;引擎内 GEMM-vs-cold 罚金 +0.81ms 为 kernel 间 L2 干扰,非单 kernel 工程可回收。**差距定性为硬件访问模式物理,非实现缺陷**:vLLM 同口径有效带宽同带(C=8:13.9GB ÷ TPOT 17.81–17.85ms ≈ **779–780 GB/s**,我们引擎内 GEMM 窗 13.9GB ÷ 17.78ms = 782 GB/s)——**不是 vLLM 有魔法 kernel,是两者都贴着同一堵墙**(孤立冷口径 889 GB/s、4090 峰值 1008 GB/s 之上的系统性差距,只能在访存模式层面重开:更宽 M、权重常驻 L2 的批间复用、或融合存储改访问模式本身)。

### 终账·M7 候选台账(留控制器,本里程碑不立项)

- **triton decode_gemm 模块**(已入库未接线,`acorninfer/ops/decode_gemm.py` + 17 测试全绿):kernel/实测仪/阀机制可复跑,`configure()` 装载期直接可用;重开前提 = 融合存储或访问模式变更(届时阀线重算);
- **双存储 down_t**(+3.6GB 显存换 ~0.3ms C=8,T1/T1b 实测收口):违零双存储红线,判不值;若显存预算放宽可整体重开(M 条件分派形态随存储物理一并重估);
- **连续到达 bench 协议**:chunked 的收益场景(「不被长 prefill 阻塞」)不在同步批量到达协议内(M5 判读 3 挂账),需先建协议再裁 chunked 面价值;
- **半宽 GEMM**(TP=2 半宽 37% 效率塌方,M5 段 0 实测):TP=2 北极星(73.3/72.4/64.5%)的 kernel 面第一候选;
- **custom allreduce**(TP=2 集合通信 74 次/步,M5 段 0 微基准 17.84µs 在档):TP=2 通信栈候选,与半宽 GEMM 同轨;
- **rank0 快失败**(TP 装载崩溃 rank1 先崩、rank0 慢败,M5 轨 2 BLOCKED 实录):工程健壮性项,非性能。

### 终账·口径注记(照前节同款)

同日同机(RTX 4090 ×2;in-process 终bench 物理 GPU 1、轨 1 仪式全腿物理 GPU 0;每腿跑前 `nvidia-smi` 验净 4 MiB / 0 进程,前台 timeout ≥1800s);全部腿 seed=20261027、in=128/out=32、temperature=0、max_tokens=32、计时前 1 条 warmup 不计入;HTTP TTFT 含客户端→服务端上行与 SSE 流式首包回传全程(`bench_remote._run_one` 口径);仪式四腿 + 终bench 三样本同窗口串行,腿间以双卡 4 MiB 验净为界;vLLM 0.32.0,基线环境 `/home/ernest/vllm-baseline`;13.9GB/步 = 段 0 腿 1 CHECK 口径(四类投影 + lm_head,含 lm_head 略偏高);证据归档 `/tmp/opencode/m6t6/`(m6_final 三样本 stdout、五腿 bench/server stdout 全文);本节树 = a13fa46 + 本 commit(docs)。M6 不打 tag、不 push(留用户)。

## M7 段 0:RTN 数学(TDD)+ W8/W4 dequant-GEMM 微基准(2026-11-01,`tests/test_quantize.py` + `scripts/m7_spike.py`,真机输出)

背景:M6 终账定性差距根源 = 每步必须搬运的字节数(bf16 权重流 15.13GB/步)。M7 把权重压到 W8(int8,7.57GB)/ W4(uint4 打包,4.02GB)、dequant 融入 GEMM。本节 = **段 0**:`acorninfer/quant/quantize.py` RTN 数学(TDD,纯 CPU)+ 探索级 dequant-GEMM 变体孤立微基准(η 表 → 锚修正条款复核;**判定留控制器,本节只交数据**)。生产源码零触碰;探索 kernel 全在 spike 脚本内,`ops/decode_gemm.py` 导入复用(`_bench_rot` 冷口径轮转计时 / `_ROT_SETS=36` / `_cfg_shmem_ok` shmem≤96KB 预过滤 / `_workspace` fp32 partials 常驻 / `_splitk_reduce_kernel` 确定性归约,该文件本任务零改动)。

口径(与锚推导同轴,勿用 M6 聚合口径):投影为**独立模块**五档类 —— qo (4096,4096)×2 调用/层、kv (1024,4096)×2、gate_up (12288,4096)×2、down (4096,12288)、lm_head (151936,4096)×1/步(每步调用数 72/72/72/36/1)。gb = **量化权重字节(含元数据)/µs**:W8 = N·K+2N(int8 + fp16 scale)→ 每步 7.571GB;W4 = N·K/2 + 4·N·(K/128)(打包 + g128 scale/zp)→ 每步 4.021GB。roofline = 4090 峰值 1.008TB/s,η = gb/1008。冷口径:`_bench_rot` 36 套轮转(>72MB L2 强制 HBM;warmup 一整轮 + 计 5 轮);lm_head 单套 622MB/311MB,36 套 = 22.4GB 超显存 → **8 套轮转**(5GB,单套 > L2 70×,充分冷,如实注记的口径偏离)。BLOCK_M=16(M≤16 pad;M∈{1,8} 两档);SPLIT_K 仅 kv/down 两 N-不足类照 decode_gemm 先例评估(top-2 × sk{2,4[,8]},fp32 partials + 确定性 second-pass,atomic 零使用)。

### quantize.py 红绿记录(TDD,纯 CPU)

格式契约(spec §2 定案,测试钉死):W8 per-channel 对称(q int8 [N,K] ∈[−127,127],scale fp16 [N]);W4 g128 非对称(q uint8 [N,K] ∈[0,15],scale/zp fp16 [N,K/128]);打包 byte = q_even | q_odd<<4(**低 nibble = 偶数列**,unpack 同约定);形状守门 K/N 非 128 倍数 → ValueError(零特例口径);`dequant_w8/dequant_w4` 为 fp32 参考实现(Task 1 TOL 对拍锚)。

- 红:`.venv/bin/python -m pytest tests/test_quantize.py` → `ModuleNotFoundError: No module named 'acorninfer.quant'`(实现前)。
- 绿:实现后 4 passed(实测输出 `4 passed in 0.14s`):
  - `test_w8_roundtrip_within_theoretical_bound` / `test_w4_roundtrip_within_theoretical_bound`(逐元素 ≤ scale/2 + 表示误差项 + eps)/ `test_pack_w4_roundtrip`(nibble 顺序钉死:byte 0x21 = 列0=1(低 nibble)| 列1=2<<4)/ `test_shape_guards`。
- 对 plan 草稿的两处偏离(逐条记录):①理论界比草稿「scale/2+eps」精确——dequant 吃的是**存储后的 fp16 scale**,界补表示误差项 `127·s·2⁻¹¹`(W8)/`15·s·2⁻¹¹ + |zp|·2⁻¹¹`(W4),草稿界在 fp16 scale 舍入下可被真实超越;②**实测发现 fp16 不可表示 ε=1e-8**(half 最小次正规 ≈5.96e-8)→ 全零行 scale 存储后为 0.0(无损:该行 q≡0,dequant 精确为 0;spec ε 的职责在 fp32 数学域防除零,已成立)。守门额外统一到 quantize_w8(N%128 同守;生产五档全满足,零行为差异)。

### 变体面(实测网格)

- W8:`BLOCK_N{64,128,256} × BLOCK_K{64,128} × warps{4,8} × stages{2,3,4}`(shmem 预过滤 + lm_head N%BN 整除过滤 → 17 活配置);核内 dequant:int8 load → fp32 ×scale(行广播,循环不变提出)→ bf16 → tl.dot fp32 累加。
- W4:同网格 × unpack 两法(**u0** = 位运算移位掩码:偶/奇 nibble 拆半 dequant、x 偶/奇列跨步加载、两次半-K tl.dot,零 shuffle;**u1** = `tl.interleave` 织回 [BN,BK] 单 dot)× 组 scale 两形态(**g0** = 组对齐:k0 整 128,块内组号标量、scale 行广播;**g1** = gather 索引 (k0+i)//128 逐元素)→ 68 活配置。
- SPLIT_K:段2 top-2 × sk{2,4}(kv 加 sk8)整除过滤;g0 附加 K_CHUNK 整 128 过滤。
- 数值门:每 kernel 形态 kv 类对拍 fp32 dequant 参考(spike 级 `rtol=2e-2, atol=1.0`;atol 口径:核内 dequant 后 round bf16 ~2⁻⁹ 相对 × √K 点积 ≈ 0.13σ,见 QSPIKE_CHECK 行;**正式 TOL 留 Task 1 标定写死**)。

### W8 腿(stdout 逐字,2026-11-01)

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m7_spike.py w8`(前台,跑前 GPU 1 验净 4MiB;236 逐变体行 + CHECK/CAP/BEST/SUMMARY 全文):

```
QSPIKE_HEADER leg=w8 date=2026-11-01 torch=2.13.0+cu130 triton=3.7.1 dev=NVIDIA GeForce RTX 4090 rot_sets=36(lm_head=8) block_m=16 ms=(1, 8) roofline_gbps=1008 classes=kv,qo,gate_up,down,lm_head
QSPIKE_CHECK fmt=w8 form=bn128_bk128_w4_s3_sk1 ok=True maxabs=0.9038 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w8 form=bn128_bk64_w4_s3_sk4 ok=True maxabs=0.9038 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w8 form=bn128_bk64_w4_s3_sk8 ok=True maxabs=0.9038 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u0_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u0_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u1_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u1_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u0_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u0_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u1_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u1_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u1_g0_sk2 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE w8 bn64_bk64_w4_s2_sk1 kv M=1 us=34.98 gb=120.0
QSPIKE w8 bn64_bk64_w4_s3_sk1 kv M=1 us=19.98 gb=210.0
QSPIKE w8 bn64_bk64_w4_s4_sk1 kv M=1 us=18.43 gb=227.7
QSPIKE w8 bn64_bk64_w8_s2_sk1 kv M=1 us=34.36 gb=122.1
QSPIKE w8 bn64_bk64_w8_s3_sk1 kv M=1 us=21.23 gb=197.7
QSPIKE w8 bn64_bk64_w8_s4_sk1 kv M=1 us=18.36 gb=228.5
QSPIKE w8 bn64_bk128_w4_s2_sk1 kv M=1 us=24.43 gb=171.8
QSPIKE w8 bn64_bk128_w4_s3_sk1 kv M=1 us=15.13 gb=277.4
QSPIKE w8 bn64_bk128_w4_s4_sk1 kv M=1 us=14.96 gb=280.5
QSPIKE w8 bn64_bk128_w8_s2_sk1 kv M=1 us=23.93 gb=175.3
QSPIKE w8 bn64_bk128_w8_s3_sk1 kv M=1 us=15.73 gb=266.8
QSPIKE w8 bn64_bk128_w8_s4_sk1 kv M=1 us=15.74 gb=266.7
QSPIKE w8 bn128_bk64_w4_s2_sk1 kv M=1 us=45.96 gb=91.3
QSPIKE w8 bn128_bk64_w4_s3_sk1 kv M=1 us=27.48 gb=152.7
QSPIKE w8 bn128_bk64_w4_s4_sk1 kv M=1 us=27.07 gb=155.0
QSPIKE w8 bn128_bk64_w8_s2_sk1 kv M=1 us=43.60 gb=96.2
QSPIKE w8 bn128_bk64_w8_s3_sk1 kv M=1 us=27.21 gb=154.2
QSPIKE w8 bn128_bk64_w8_s4_sk1 kv M=1 us=27.18 gb=154.4
QSPIKE w8 bn128_bk128_w4_s2_sk1 kv M=1 us=35.41 gb=118.5
QSPIKE w8 bn128_bk128_w8_s2_sk1 kv M=1 us=32.27 gb=130.0
QSPIKE w8 bn256_bk64_w4_s2_sk1 kv M=1 us=66.24 gb=63.3
QSPIKE w8 bn256_bk64_w8_s2_sk1 kv M=1 us=60.25 gb=69.6
QSPIKE w8 bn64_bk128_w4_s4_sk2 kv M=1 us=19.22 gb=218.3
QSPIKE w8 bn64_bk128_w4_s4_sk4 kv M=1 us=19.43 gb=215.9
QSPIKE w8 bn64_bk128_w4_s4_sk8 kv M=1 us=20.36 gb=206.1
QSPIKE w8 bn64_bk128_w4_s3_sk2 kv M=1 us=20.64 gb=203.3
QSPIKE w8 bn64_bk128_w4_s3_sk4 kv M=1 us=21.07 gb=199.1
QSPIKE w8 bn64_bk128_w4_s3_sk8 kv M=1 us=20.45 gb=205.2
QSPIKE_BEST fmt=w8 class=kv M=1 us=14.96 gb=280.5 eta=0.278 variant=bn64_bk128_w4_s4_sk1 skipped=0
/home/userlist/zxs/python_venv/lib/python3.12/site-packages/torch/cuda/__init__.py:1333: UserWarning: Synchronization debug mode is a prototype feature and does not yet detect all synchronizing operations (Triggered internally at /__w/pytorch/pytorch/torch/csrc/cuda/Module.cpp:1028.)
  torch._C._cuda_set_sync_debug_mode(debug_mode)
QSPIKE_CAP fmt=w8 class=kv M=1 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.5916)
QSPIKE w8 bn64_bk64_w4_s2_sk1 kv M=8 us=32.93 gb=127.4
QSPIKE w8 bn64_bk64_w4_s3_sk1 kv M=8 us=18.78 gb=223.4
QSPIKE w8 bn64_bk64_w4_s4_sk1 kv M=8 us=16.80 gb=249.8
QSPIKE w8 bn64_bk64_w8_s2_sk1 kv M=8 us=32.64 gb=128.5
QSPIKE w8 bn64_bk64_w8_s3_sk1 kv M=8 us=19.95 gb=210.4
QSPIKE w8 bn64_bk64_w8_s4_sk1 kv M=8 us=16.65 gb=252.1
QSPIKE w8 bn64_bk128_w4_s2_sk1 kv M=8 us=22.58 gb=185.8
QSPIKE w8 bn64_bk128_w4_s3_sk1 kv M=8 us=13.88 gb=302.3
QSPIKE w8 bn64_bk128_w4_s4_sk1 kv M=8 us=13.58 gb=309.0
QSPIKE w8 bn64_bk128_w8_s2_sk1 kv M=8 us=22.10 gb=189.9
QSPIKE w8 bn64_bk128_w8_s3_sk1 kv M=8 us=18.76 gb=223.7
QSPIKE w8 bn64_bk128_w8_s4_sk1 kv M=8 us=14.25 gb=294.5
QSPIKE w8 bn128_bk64_w4_s2_sk1 kv M=8 us=42.54 gb=98.7
QSPIKE w8 bn128_bk64_w4_s3_sk1 kv M=8 us=25.06 gb=167.4
QSPIKE w8 bn128_bk64_w4_s4_sk1 kv M=8 us=24.43 gb=171.8
QSPIKE w8 bn128_bk64_w8_s2_sk1 kv M=8 us=40.93 gb=102.5
QSPIKE w8 bn128_bk64_w8_s3_sk1 kv M=8 us=24.46 gb=171.6
QSPIKE w8 bn128_bk64_w8_s4_sk1 kv M=8 us=24.38 gb=172.1
QSPIKE w8 bn128_bk128_w4_s2_sk1 kv M=8 us=32.20 gb=130.3
QSPIKE w8 bn128_bk128_w8_s2_sk1 kv M=8 us=30.67 gb=136.8
QSPIKE w8 bn256_bk64_w4_s2_sk1 kv M=8 us=66.09 gb=63.5
QSPIKE w8 bn256_bk64_w8_s2_sk1 kv M=8 us=59.59 gb=70.4
QSPIKE w8 bn64_bk128_w4_s4_sk2 kv M=8 us=20.66 gb=203.1
QSPIKE w8 bn64_bk128_w4_s4_sk4 kv M=8 us=20.56 gb=204.2
QSPIKE w8 bn64_bk128_w4_s4_sk8 kv M=8 us=21.00 gb=199.8
QSPIKE w8 bn64_bk128_w4_s3_sk2 kv M=8 us=21.61 gb=194.2
QSPIKE w8 bn64_bk128_w4_s3_sk4 kv M=8 us=21.11 gb=198.8
QSPIKE w8 bn64_bk128_w4_s3_sk8 kv M=8 us=21.44 gb=195.8
QSPIKE_BEST fmt=w8 class=kv M=8 us=13.58 gb=309.0 eta=0.307 variant=bn64_bk128_w4_s4_sk1 skipped=0
QSPIKE_CAP fmt=w8 class=kv M=8 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.7262)
QSPIKE w8 bn64_bk64_w4_s2_sk1 qo M=1 us=46.58 gb=360.4
QSPIKE w8 bn64_bk64_w4_s3_sk1 qo M=1 us=29.92 gb=561.0
QSPIKE w8 bn64_bk64_w4_s4_sk1 qo M=1 us=27.24 gb=616.1
QSPIKE w8 bn64_bk64_w8_s2_sk1 qo M=1 us=46.48 gb=361.1
QSPIKE w8 bn64_bk64_w8_s3_sk1 qo M=1 us=30.89 gb=543.4
QSPIKE w8 bn64_bk64_w8_s4_sk1 qo M=1 us=27.61 gb=607.9
QSPIKE w8 bn64_bk128_w4_s2_sk1 qo M=1 us=26.33 gb=637.4
QSPIKE w8 bn64_bk128_w4_s3_sk1 qo M=1 us=21.11 gb=795.3
QSPIKE w8 bn64_bk128_w4_s4_sk1 qo M=1 us=20.41 gb=822.3
QSPIKE w8 bn64_bk128_w8_s2_sk1 qo M=1 us=26.35 gb=637.1
QSPIKE w8 bn64_bk128_w8_s3_sk1 qo M=1 us=20.90 gb=803.3
QSPIKE w8 bn64_bk128_w8_s4_sk1 qo M=1 us=20.56 gb=816.4
QSPIKE w8 bn128_bk64_w4_s2_sk1 qo M=1 us=49.37 gb=340.0
QSPIKE w8 bn128_bk64_w4_s3_sk1 qo M=1 us=37.27 gb=450.3
QSPIKE w8 bn128_bk64_w4_s4_sk1 qo M=1 us=32.65 gb=514.1
QSPIKE w8 bn128_bk64_w8_s2_sk1 qo M=1 us=48.32 gb=347.4
QSPIKE w8 bn128_bk64_w8_s3_sk1 qo M=1 us=35.69 gb=470.4
QSPIKE w8 bn128_bk64_w8_s4_sk1 qo M=1 us=32.23 gb=520.8
QSPIKE w8 bn128_bk128_w4_s2_sk1 qo M=1 us=33.49 gb=501.2
QSPIKE w8 bn128_bk128_w8_s2_sk1 qo M=1 us=32.42 gb=517.8
QSPIKE w8 bn256_bk64_w4_s2_sk1 qo M=1 us=68.55 gb=244.9
QSPIKE w8 bn256_bk64_w8_s2_sk1 qo M=1 us=62.95 gb=266.6
QSPIKE_BEST fmt=w8 class=qo M=1 us=20.41 gb=822.3 eta=0.816 variant=bn64_bk128_w4_s4_sk1 skipped=0
QSPIKE_CAP fmt=w8 class=qo M=1 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.6749)
QSPIKE w8 bn64_bk64_w4_s2_sk1 qo M=8 us=46.49 gb=361.1
QSPIKE w8 bn64_bk64_w4_s3_sk1 qo M=8 us=29.95 gb=560.5
QSPIKE w8 bn64_bk64_w4_s4_sk1 qo M=8 us=27.23 gb=616.4
QSPIKE w8 bn64_bk64_w8_s2_sk1 qo M=8 us=46.52 gb=360.8
QSPIKE w8 bn64_bk64_w8_s3_sk1 qo M=8 us=30.74 gb=546.1
QSPIKE w8 bn64_bk64_w8_s4_sk1 qo M=8 us=27.49 gb=610.6
QSPIKE w8 bn64_bk128_w4_s2_sk1 qo M=8 us=26.70 gb=628.7
QSPIKE w8 bn64_bk128_w4_s3_sk1 qo M=8 us=21.28 gb=788.9
QSPIKE w8 bn64_bk128_w4_s4_sk1 qo M=8 us=20.42 gb=821.9
QSPIKE w8 bn64_bk128_w8_s2_sk1 qo M=8 us=26.39 gb=636.2
QSPIKE w8 bn64_bk128_w8_s3_sk1 qo M=8 us=21.03 gb=798.3
QSPIKE w8 bn64_bk128_w8_s4_sk1 qo M=8 us=20.57 gb=816.0
QSPIKE w8 bn128_bk64_w4_s2_sk1 qo M=8 us=49.24 gb=340.9
QSPIKE w8 bn128_bk64_w4_s3_sk1 qo M=8 us=37.24 gb=450.7
QSPIKE w8 bn128_bk64_w4_s4_sk1 qo M=8 us=32.71 gb=513.2
QSPIKE w8 bn128_bk64_w8_s2_sk1 qo M=8 us=48.38 gb=347.0
QSPIKE w8 bn128_bk64_w8_s3_sk1 qo M=8 us=35.62 gb=471.2
QSPIKE w8 bn128_bk64_w8_s4_sk1 qo M=8 us=32.22 gb=521.0
QSPIKE w8 bn128_bk128_w4_s2_sk1 qo M=8 us=33.72 gb=497.8
QSPIKE w8 bn128_bk128_w8_s2_sk1 qo M=8 us=32.52 gb=516.1
QSPIKE w8 bn256_bk64_w4_s2_sk1 qo M=8 us=68.94 gb=243.5
QSPIKE w8 bn256_bk64_w8_s2_sk1 qo M=8 us=63.19 gb=265.6
QSPIKE_BEST fmt=w8 class=qo M=8 us=20.42 gb=821.9 eta=0.815 variant=bn64_bk128_w4_s4_sk1 skipped=0
QSPIKE_CAP fmt=w8 class=qo M=8 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.9726)
QSPIKE w8 bn64_bk64_w4_s2_sk1 gate_up M=1 us=113.00 gb=445.6
QSPIKE w8 bn64_bk64_w4_s3_sk1 gate_up M=1 us=98.46 gb=511.4
QSPIKE w8 bn64_bk64_w4_s4_sk1 gate_up M=1 us=95.95 gb=524.8
QSPIKE w8 bn64_bk64_w8_s2_sk1 gate_up M=1 us=112.61 gb=447.2
QSPIKE w8 bn64_bk64_w8_s3_sk1 gate_up M=1 us=96.80 gb=520.2
QSPIKE w8 bn64_bk64_w8_s4_sk1 gate_up M=1 us=92.64 gb=543.5
QSPIKE w8 bn64_bk128_w4_s2_sk1 gate_up M=1 us=59.71 gb=843.3
QSPIKE w8 bn64_bk128_w4_s3_sk1 gate_up M=1 us=57.32 gb=878.6
QSPIKE w8 bn64_bk128_w4_s4_sk1 gate_up M=1 us=56.94 gb=884.4
QSPIKE w8 bn64_bk128_w8_s2_sk1 gate_up M=1 us=59.59 gb=845.1
QSPIKE w8 bn64_bk128_w8_s3_sk1 gate_up M=1 us=57.16 gb=881.0
QSPIKE w8 bn64_bk128_w8_s4_sk1 gate_up M=1 us=57.08 gb=882.3
QSPIKE w8 bn128_bk64_w4_s2_sk1 gate_up M=1 us=113.59 gb=443.3
QSPIKE w8 bn128_bk64_w4_s3_sk1 gate_up M=1 us=102.00 gb=493.7
QSPIKE w8 bn128_bk64_w4_s4_sk1 gate_up M=1 us=99.48 gb=506.2
QSPIKE w8 bn128_bk64_w8_s2_sk1 gate_up M=1 us=112.87 gb=446.1
QSPIKE w8 bn128_bk64_w8_s3_sk1 gate_up M=1 us=101.67 gb=495.3
QSPIKE w8 bn128_bk64_w8_s4_sk1 gate_up M=1 us=99.19 gb=507.7
QSPIKE w8 bn128_bk128_w4_s2_sk1 gate_up M=1 us=60.88 gb=827.1
QSPIKE w8 bn128_bk128_w8_s2_sk1 gate_up M=1 us=60.76 gb=828.8
QSPIKE w8 bn256_bk64_w4_s2_sk1 gate_up M=1 us=113.37 gb=444.2
QSPIKE w8 bn256_bk64_w8_s2_sk1 gate_up M=1 us=112.83 gb=446.3
QSPIKE_BEST fmt=w8 class=gate_up M=1 us=56.94 gb=884.4 eta=0.877 variant=bn64_bk128_w4_s4_sk1 skipped=0
QSPIKE_CAP fmt=w8 class=gate_up M=1 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.7974)
QSPIKE w8 bn64_bk64_w4_s2_sk1 gate_up M=8 us=112.79 gb=446.4
QSPIKE w8 bn64_bk64_w4_s3_sk1 gate_up M=8 us=98.45 gb=511.5
QSPIKE w8 bn64_bk64_w4_s4_sk1 gate_up M=8 us=95.58 gb=526.9
QSPIKE w8 bn64_bk64_w8_s2_sk1 gate_up M=8 us=112.77 gb=446.5
QSPIKE w8 bn64_bk64_w8_s3_sk1 gate_up M=8 us=96.80 gb=520.2
QSPIKE w8 bn64_bk64_w8_s4_sk1 gate_up M=8 us=93.35 gb=539.4
QSPIKE w8 bn64_bk128_w4_s2_sk1 gate_up M=8 us=59.76 gb=842.6
QSPIKE w8 bn64_bk128_w4_s3_sk1 gate_up M=8 us=57.29 gb=879.0
QSPIKE w8 bn64_bk128_w4_s4_sk1 gate_up M=8 us=57.04 gb=882.9
QSPIKE w8 bn64_bk128_w8_s2_sk1 gate_up M=8 us=59.65 gb=844.2
QSPIKE w8 bn64_bk128_w8_s3_sk1 gate_up M=8 us=57.20 gb=880.4
QSPIKE w8 bn64_bk128_w8_s4_sk1 gate_up M=8 us=57.20 gb=880.3
QSPIKE w8 bn128_bk64_w4_s2_sk1 gate_up M=8 us=113.53 gb=443.5
QSPIKE w8 bn128_bk64_w4_s3_sk1 gate_up M=8 us=102.03 gb=493.5
QSPIKE w8 bn128_bk64_w4_s4_sk1 gate_up M=8 us=99.55 gb=505.8
QSPIKE w8 bn128_bk64_w8_s2_sk1 gate_up M=8 us=112.93 gb=445.9
QSPIKE w8 bn128_bk64_w8_s3_sk1 gate_up M=8 us=101.62 gb=495.5
QSPIKE w8 bn128_bk64_w8_s4_sk1 gate_up M=8 us=99.15 gb=507.9
QSPIKE w8 bn128_bk128_w4_s2_sk1 gate_up M=8 us=60.94 gb=826.3
QSPIKE w8 bn128_bk128_w8_s2_sk1 gate_up M=8 us=60.77 gb=828.7
QSPIKE w8 bn256_bk64_w4_s2_sk1 gate_up M=8 us=113.49 gb=443.7
QSPIKE w8 bn256_bk64_w8_s2_sk1 gate_up M=8 us=113.00 gb=445.6
QSPIKE_BEST fmt=w8 class=gate_up M=8 us=57.04 gb=882.9 eta=0.876 variant=bn64_bk128_w4_s4_sk1 skipped=0
QSPIKE_CAP fmt=w8 class=gate_up M=8 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.8608)
QSPIKE w8 bn64_bk64_w4_s2_sk1 down M=1 us=143.76 gb=350.2
QSPIKE w8 bn64_bk64_w4_s3_sk1 down M=1 us=95.36 gb=527.9
QSPIKE w8 bn64_bk64_w4_s4_sk1 down M=1 us=86.32 gb=583.2
QSPIKE w8 bn64_bk64_w8_s2_sk1 down M=1 us=142.73 gb=352.7
QSPIKE w8 bn64_bk64_w8_s3_sk1 down M=1 us=98.08 gb=513.2
QSPIKE w8 bn64_bk64_w8_s4_sk1 down M=1 us=86.73 gb=580.5
QSPIKE w8 bn64_bk128_w4_s2_sk1 down M=1 us=79.27 gb=635.1
QSPIKE w8 bn64_bk128_w4_s3_sk1 down M=1 us=61.80 gb=814.6
QSPIKE w8 bn64_bk128_w4_s4_sk1 down M=1 us=58.68 gb=857.9
QSPIKE w8 bn64_bk128_w8_s2_sk1 down M=1 us=78.90 gb=638.0
QSPIKE w8 bn64_bk128_w8_s3_sk1 down M=1 us=61.37 gb=820.2
QSPIKE w8 bn64_bk128_w8_s4_sk1 down M=1 us=58.79 gb=856.3
QSPIKE w8 bn128_bk64_w4_s2_sk1 down M=1 us=159.18 gb=316.2
QSPIKE w8 bn128_bk64_w4_s3_sk1 down M=1 us=117.73 gb=427.6
QSPIKE w8 bn128_bk64_w4_s4_sk1 down M=1 us=101.09 gb=498.0
QSPIKE w8 bn128_bk64_w8_s2_sk1 down M=1 us=156.84 gb=321.0
QSPIKE w8 bn128_bk64_w8_s3_sk1 down M=1 us=114.31 gb=440.4
QSPIKE w8 bn128_bk64_w8_s4_sk1 down M=1 us=99.61 gb=505.4
QSPIKE w8 bn128_bk128_w4_s2_sk1 down M=1 us=99.12 gb=507.9
QSPIKE w8 bn128_bk128_w8_s2_sk1 down M=1 us=97.05 gb=518.7
QSPIKE w8 bn256_bk64_w4_s2_sk1 down M=1 us=209.89 gb=239.8
QSPIKE w8 bn256_bk64_w8_s2_sk1 down M=1 us=193.64 gb=260.0
QSPIKE w8 bn64_bk128_w4_s4_sk2 down M=1 us=57.87 gb=869.8
QSPIKE w8 bn64_bk128_w4_s4_sk4 down M=1 us=58.27 gb=863.9
QSPIKE w8 bn64_bk128_w8_s4_sk2 down M=1 us=57.95 gb=868.6
QSPIKE w8 bn64_bk128_w8_s4_sk4 down M=1 us=58.44 gb=861.4
QSPIKE_BEST fmt=w8 class=down M=1 us=57.87 gb=869.8 eta=0.863 variant=bn64_bk128_w4_s4_sk2 skipped=0
QSPIKE_CAP fmt=w8 class=down M=1 variant=bn64_bk128_w4_s4_sk2 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=1.1186)
QSPIKE w8 bn64_bk64_w4_s2_sk1 down M=8 us=143.61 gb=350.5
QSPIKE w8 bn64_bk64_w4_s3_sk1 down M=8 us=95.19 gb=528.8
QSPIKE w8 bn64_bk64_w4_s4_sk1 down M=8 us=86.25 gb=583.7
QSPIKE w8 bn64_bk64_w8_s2_sk1 down M=8 us=142.64 gb=352.9
QSPIKE w8 bn64_bk64_w8_s3_sk1 down M=8 us=97.69 gb=515.3
QSPIKE w8 bn64_bk64_w8_s4_sk1 down M=8 us=86.32 gb=583.2
QSPIKE w8 bn64_bk128_w4_s2_sk1 down M=8 us=80.29 gb=627.0
QSPIKE w8 bn64_bk128_w4_s3_sk1 down M=8 us=62.24 gb=808.8
QSPIKE w8 bn64_bk128_w4_s4_sk1 down M=8 us=58.68 gb=857.9
QSPIKE w8 bn64_bk128_w8_s2_sk1 down M=8 us=78.93 gb=637.8
QSPIKE w8 bn64_bk128_w8_s3_sk1 down M=8 us=61.25 gb=821.8
QSPIKE w8 bn64_bk128_w8_s4_sk1 down M=8 us=58.81 gb=855.9
QSPIKE w8 bn128_bk64_w4_s2_sk1 down M=8 us=159.65 gb=315.3
QSPIKE w8 bn128_bk64_w4_s3_sk1 down M=8 us=117.48 gb=428.5
QSPIKE w8 bn128_bk64_w4_s4_sk1 down M=8 us=101.02 gb=498.3
QSPIKE w8 bn128_bk64_w8_s2_sk1 down M=8 us=156.55 gb=321.6
QSPIKE w8 bn128_bk64_w8_s3_sk1 down M=8 us=114.04 gb=441.4
QSPIKE w8 bn128_bk64_w8_s4_sk1 down M=8 us=99.42 gb=506.3
QSPIKE w8 bn128_bk128_w4_s2_sk1 down M=8 us=98.76 gb=509.7
QSPIKE w8 bn128_bk128_w8_s2_sk1 down M=8 us=97.10 gb=518.5
QSPIKE w8 bn256_bk64_w4_s2_sk1 down M=8 us=210.04 gb=239.7
QSPIKE w8 bn256_bk64_w8_s2_sk1 down M=8 us=193.75 gb=259.8
QSPIKE w8 bn64_bk128_w4_s4_sk2 down M=8 us=57.83 gb=870.4
QSPIKE w8 bn64_bk128_w4_s4_sk4 down M=8 us=58.27 gb=863.9
QSPIKE w8 bn64_bk128_w8_s4_sk2 down M=8 us=57.96 gb=868.5
QSPIKE w8 bn64_bk128_w8_s4_sk4 down M=8 us=58.44 gb=861.4
QSPIKE_BEST fmt=w8 class=down M=8 us=57.83 gb=870.4 eta=0.864 variant=bn64_bk128_w4_s4_sk2 skipped=0
QSPIKE_CAP fmt=w8 class=down M=8 variant=bn64_bk128_w4_s4_sk2 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=1.5089)
QSPIKE w8 bn64_bk64_w4_s2_sk1 lm_head M=1 us=1259.60 gb=494.3
QSPIKE w8 bn64_bk64_w4_s3_sk1 lm_head M=1 us=1241.68 gb=501.4
QSPIKE w8 bn64_bk64_w4_s4_sk1 lm_head M=1 us=1244.09 gb=500.5
QSPIKE w8 bn64_bk64_w8_s2_sk1 lm_head M=1 us=1253.66 gb=496.7
QSPIKE w8 bn64_bk64_w8_s3_sk1 lm_head M=1 us=1207.93 gb=515.5
QSPIKE w8 bn64_bk64_w8_s4_sk1 lm_head M=1 us=1245.11 gb=500.1
QSPIKE w8 bn64_bk128_w4_s2_sk1 lm_head M=1 us=680.80 gb=914.6
QSPIKE w8 bn64_bk128_w4_s3_sk1 lm_head M=1 us=672.22 gb=926.2
QSPIKE w8 bn64_bk128_w4_s4_sk1 lm_head M=1 us=661.37 gb=941.4
QSPIKE w8 bn64_bk128_w8_s2_sk1 lm_head M=1 us=681.60 gb=913.5
QSPIKE w8 bn64_bk128_w8_s3_sk1 lm_head M=1 us=672.31 gb=926.1
QSPIKE w8 bn64_bk128_w8_s4_sk1 lm_head M=1 us=662.07 gb=940.4
QSPIKE w8 bn128_bk64_w4_s2_sk1 lm_head M=1 us=1245.69 gb=499.8
QSPIKE w8 bn128_bk64_w4_s3_sk1 lm_head M=1 us=1242.60 gb=501.1
QSPIKE w8 bn128_bk64_w4_s4_sk1 lm_head M=1 us=1250.36 gb=498.0
QSPIKE w8 bn128_bk64_w8_s2_sk1 lm_head M=1 us=1249.20 gb=498.4
QSPIKE w8 bn128_bk64_w8_s3_sk1 lm_head M=1 us=1245.80 gb=499.8
QSPIKE w8 bn128_bk64_w8_s4_sk1 lm_head M=1 us=1253.12 gb=496.9
QSPIKE w8 bn128_bk128_w4_s2_sk1 lm_head M=1 us=681.24 gb=914.0
QSPIKE w8 bn128_bk128_w8_s2_sk1 lm_head M=1 us=680.90 gb=914.4
QSPIKE_BEST fmt=w8 class=lm_head M=1 us=661.37 gb=941.4 eta=0.934 variant=bn64_bk128_w4_s4_sk1 skipped=0
QSPIKE_CAP fmt=w8 class=lm_head M=1 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=skip(lm_head fp32 参考省略,bitwise 钉 + kv CHECK 已覆盖)
QSPIKE w8 bn64_bk64_w4_s2_sk1 lm_head M=8 us=1360.79 gb=457.6
QSPIKE w8 bn64_bk64_w4_s3_sk1 lm_head M=8 us=1348.17 gb=461.8
QSPIKE w8 bn64_bk64_w4_s4_sk1 lm_head M=8 us=1260.85 gb=493.8
QSPIKE w8 bn64_bk64_w8_s2_sk1 lm_head M=8 us=1265.41 gb=492.0
QSPIKE w8 bn64_bk64_w8_s3_sk1 lm_head M=8 us=1224.93 gb=508.3
QSPIKE w8 bn64_bk64_w8_s4_sk1 lm_head M=8 us=1262.28 gb=493.3
QSPIKE w8 bn64_bk128_w4_s2_sk1 lm_head M=8 us=689.20 gb=903.4
QSPIKE w8 bn64_bk128_w4_s3_sk1 lm_head M=8 us=684.42 gb=909.7
QSPIKE w8 bn64_bk128_w4_s4_sk1 lm_head M=8 us=673.84 gb=924.0
QSPIKE w8 bn64_bk128_w8_s2_sk1 lm_head M=8 us=690.74 gb=901.4
QSPIKE w8 bn64_bk128_w8_s3_sk1 lm_head M=8 us=685.03 gb=908.9
QSPIKE w8 bn64_bk128_w8_s4_sk1 lm_head M=8 us=674.30 gb=923.4
QSPIKE w8 bn128_bk64_w4_s2_sk1 lm_head M=8 us=1253.07 gb=496.9
QSPIKE w8 bn128_bk64_w4_s3_sk1 lm_head M=8 us=1253.32 gb=496.8
QSPIKE w8 bn128_bk64_w4_s4_sk1 lm_head M=8 us=1263.38 gb=492.8
QSPIKE w8 bn128_bk64_w8_s2_sk1 lm_head M=8 us=1257.93 gb=495.0
QSPIKE w8 bn128_bk64_w8_s3_sk1 lm_head M=8 us=1256.29 gb=495.6
QSPIKE w8 bn128_bk64_w8_s4_sk1 lm_head M=8 us=1266.48 gb=491.6
QSPIKE w8 bn128_bk128_w4_s2_sk1 lm_head M=8 us=688.18 gb=904.8
QSPIKE w8 bn128_bk128_w8_s2_sk1 lm_head M=8 us=688.94 gb=903.8
QSPIKE_BEST fmt=w8 class=lm_head M=8 us=673.84 gb=924.0 eta=0.917 variant=bn64_bk128_w4_s4_sk1 skipped=0
QSPIKE_CAP fmt=w8 class=lm_head M=8 variant=bn64_bk128_w4_s4_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=skip(lm_head fp32 参考省略,bitwise 钉 + kv CHECK 已覆盖)
QSPIKE_SUMMARY fmt=w8 class=kv M=1 us=14.96 gb=280.5 eta=0.278 variant=bn64_bk128_w4_s4_sk1
QSPIKE_SUMMARY fmt=w8 class=qo M=1 us=20.41 gb=822.3 eta=0.816 variant=bn64_bk128_w4_s4_sk1
QSPIKE_SUMMARY fmt=w8 class=gate_up M=1 us=56.94 gb=884.4 eta=0.877 variant=bn64_bk128_w4_s4_sk1
QSPIKE_SUMMARY fmt=w8 class=down M=1 us=57.87 gb=869.8 eta=0.863 variant=bn64_bk128_w4_s4_sk2
QSPIKE_SUMMARY fmt=w8 class=lm_head M=1 us=661.37 gb=941.4 eta=0.934 variant=bn64_bk128_w4_s4_sk1
QSPIKE_ETA fmt=w8 M=1 weighted_eta=0.800 weighted_gbps=806.2 assume=0.85 dev_pct=-5.9 step_bytes_gb=7.571 step_time_ms=9.391 (roofline=1008GB/s 权重字节口径)
QSPIKE_SUMMARY fmt=w8 class=kv M=8 us=13.58 gb=309.0 eta=0.307 variant=bn64_bk128_w4_s4_sk1
QSPIKE_SUMMARY fmt=w8 class=qo M=8 us=20.42 gb=821.9 eta=0.815 variant=bn64_bk128_w4_s4_sk1
QSPIKE_SUMMARY fmt=w8 class=gate_up M=8 us=57.04 gb=882.9 eta=0.876 variant=bn64_bk128_w4_s4_sk1
QSPIKE_SUMMARY fmt=w8 class=down M=8 us=57.83 gb=870.4 eta=0.864 variant=bn64_bk128_w4_s4_sk2
QSPIKE_SUMMARY fmt=w8 class=lm_head M=8 us=673.84 gb=924.0 eta=0.917 variant=bn64_bk128_w4_s4_sk1
QSPIKE_ETA fmt=w8 M=8 weighted_eta=0.807 weighted_gbps=813.2 assume=0.85 dev_pct=-5.1 step_bytes_gb=7.571 step_time_ms=9.311 (roofline=1008GB/s 权重字节口径)
QSPIKE_ANCHOR fmt=w8 dev_pct=-5.9..-5.1 assume=0.85 threshold=±10% (>10% → 锚修正条款一次;判定留控制器)
```

### W4 腿(stdout 逐字,2026-11-01)

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m7_spike.py w4`(884 逐变体行;首跑在 kv capture 处因 `unpack_w4` 未随输入设备分配输出而中断,修复(`torch.empty(..., device=p.device)`,单测复绿)后**整腿重跑**,本日志为准):

```
QSPIKE_HEADER leg=w4 date=2026-11-01 torch=2.13.0+cu130 triton=3.7.1 dev=NVIDIA GeForce RTX 4090 rot_sets=36(lm_head=8) block_m=16 ms=(1, 8) roofline_gbps=1008 classes=kv,qo,gate_up,down,lm_head
QSPIKE_CHECK fmt=w8 form=bn128_bk128_w4_s3_sk1 ok=True maxabs=0.9038 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w8 form=bn128_bk64_w4_s3_sk4 ok=True maxabs=0.9038 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w8 form=bn128_bk64_w4_s3_sk8 ok=True maxabs=0.9038 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u0_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u0_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u1_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u1_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u0_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u0_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u1_g0_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk128_w4_s3_u1_g1_sk1 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE_CHECK fmt=w4 form=bn128_bk64_w4_s3_u1_g0_sk2 ok=True maxabs=0.6410 tol={'rtol': 0.02, 'atol': 1.0}
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 kv M=1 us=40.94 gb=54.4
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 kv M=1 us=81.78 gb=27.2
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 kv M=1 us=42.95 gb=51.9
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 kv M=1 us=75.38 gb=29.6
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 kv M=1 us=34.28 gb=65.0
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 kv M=1 us=60.56 gb=36.8
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 kv M=1 us=23.49 gb=94.9
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 kv M=1 us=58.79 gb=37.9
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 kv M=1 us=34.24 gb=65.1
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 kv M=1 us=60.52 gb=36.8
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 kv M=1 us=29.02 gb=76.8
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 kv M=1 us=58.73 gb=37.9
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 kv M=1 us=41.24 gb=54.0
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 kv M=1 us=65.66 gb=33.9
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 kv M=1 us=39.40 gb=56.6
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 kv M=1 us=68.02 gb=32.8
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 kv M=1 us=31.71 gb=70.3
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 kv M=1 us=48.22 gb=46.2
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 kv M=1 us=22.46 gb=99.2
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 kv M=1 us=51.07 gb=43.6
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 kv M=1 us=31.56 gb=70.6
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 kv M=1 us=48.13 gb=46.3
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 kv M=1 us=21.42 gb=104.0
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 kv M=1 us=51.24 gb=43.5
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 kv M=1 us=26.65 gb=83.6
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 kv M=1 us=68.02 gb=32.8
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 kv M=1 us=25.87 gb=86.1
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 kv M=1 us=87.10 gb=25.6
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 kv M=1 us=22.11 gb=100.8
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 kv M=1 us=55.60 gb=40.1
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 kv M=1 us=20.08 gb=111.0
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 kv M=1 us=87.81 gb=25.4
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 kv M=1 us=21.98 gb=101.4
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 kv M=1 us=55.49 gb=40.2
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 kv M=1 us=19.70 gb=113.1
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 kv M=1 us=96.69 gb=23.0
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 kv M=1 us=26.18 gb=85.1
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 kv M=1 us=45.47 gb=49.0
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 kv M=1 us=23.34 gb=95.5
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 kv M=1 us=63.34 gb=35.2
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 kv M=1 us=19.78 gb=112.6
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 kv M=1 us=35.49 gb=62.8
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 kv M=1 us=19.19 gb=116.1
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 kv M=1 us=55.59 gb=40.1
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 kv M=1 us=19.72 gb=113.0
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 kv M=1 us=37.72 gb=59.1
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 kv M=1 us=18.89 gb=117.9
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 kv M=1 us=55.77 gb=40.0
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 kv M=1 us=56.64 gb=39.3
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 kv M=1 us=101.25 gb=22.0
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 kv M=1 us=51.38 gb=43.4
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 kv M=1 us=161.66 gb=13.8
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 kv M=1 us=42.91 gb=51.9
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 kv M=1 us=85.96 gb=25.9
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 kv M=1 us=62.79 gb=35.5
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 kv M=1 us=169.44 gb=13.2
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 kv M=1 us=59.04 gb=37.7
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 kv M=1 us=86.00 gb=25.9
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 kv M=1 us=62.44 gb=35.7
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 kv M=1 us=175.32 gb=12.7
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 kv M=1 us=49.55 gb=45.0
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 kv M=1 us=92.08 gb=24.2
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 kv M=1 us=49.27 gb=45.2
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 kv M=1 us=103.86 gb=21.5
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 kv M=1 us=37.92 gb=58.8
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 kv M=1 us=81.99 gb=27.2
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 kv M=1 us=33.20 gb=67.1
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 kv M=1 us=86.39 gb=25.8
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 kv M=1 us=37.79 gb=59.0
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 kv M=1 us=82.01 gb=27.2
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 kv M=1 us=33.23 gb=67.1
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 kv M=1 us=88.47 gb=25.2
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 kv M=1 us=39.39 gb=56.6
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 kv M=1 us=114.63 gb=19.4
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 kv M=1 us=39.75 gb=56.1
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 kv M=1 us=187.36 gb=11.9
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 kv M=1 us=34.47 gb=64.6
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 kv M=1 us=75.37 gb=29.6
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 kv M=1 us=34.67 gb=64.3
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 kv M=1 us=142.67 gb=15.6
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 kv M=1 us=90.13 gb=24.7
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 kv M=1 us=178.41 gb=12.5
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 kv M=1 us=96.80 gb=23.0
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 kv M=1 us=370.27 gb=6.0
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 kv M=1 us=85.73 gb=26.0
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 kv M=1 us=152.67 gb=14.6
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 kv M=1 us=75.50 gb=29.5
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 kv M=1 us=259.18 gb=8.6
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk2 kv M=1 us=20.15 gb=110.6
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk4 kv M=1 us=19.69 gb=113.1
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk8 kv M=1 us=19.65 gb=113.4
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk2 kv M=1 us=19.87 gb=112.2
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk4 kv M=1 us=19.72 gb=113.0
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk8 kv M=1 us=19.69 gb=113.2
QSPIKE_BEST fmt=w4 class=kv M=1 us=18.89 gb=117.9 eta=0.117 variant=bn64_bk128_w8_s4_u1_g0_sk1 skipped=0
/home/userlist/zxs/python_venv/lib/python3.12/site-packages/torch/cuda/__init__.py:1333: UserWarning: Synchronization debug mode is a prototype feature and does not yet detect all synchronizing operations (Triggered internally at /__w/pytorch/pytorch/torch/csrc/cuda/Module.cpp:1028.)
  torch._C._cuda_set_sync_debug_mode(debug_mode)
QSPIKE_CAP fmt=w4 class=kv M=1 variant=bn64_bk128_w8_s4_u1_g0_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.6199)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 kv M=8 us=44.95 gb=49.6
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 kv M=8 us=77.10 gb=28.9
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 kv M=8 us=40.44 gb=55.1
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 kv M=8 us=69.89 gb=31.9
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 kv M=8 us=36.88 gb=60.4
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 kv M=8 us=65.72 gb=33.9
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 kv M=8 us=22.03 gb=101.2
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 kv M=8 us=52.71 gb=42.3
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 kv M=8 us=36.79 gb=60.6
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 kv M=8 us=72.46 gb=30.8
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 kv M=8 us=29.04 gb=76.7
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 kv M=8 us=58.33 gb=38.2
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 kv M=8 us=42.58 gb=52.3
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 kv M=8 us=65.08 gb=34.2
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 kv M=8 us=40.29 gb=55.3
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 kv M=8 us=73.69 gb=30.2
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 kv M=8 us=34.01 gb=65.5
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 kv M=8 us=54.18 gb=41.1
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 kv M=8 us=24.51 gb=90.9
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 kv M=8 us=56.11 gb=39.7
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 kv M=8 us=33.83 gb=65.9
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 kv M=8 us=54.15 gb=41.2
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 kv M=8 us=23.96 gb=93.0
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 kv M=8 us=56.30 gb=39.6
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 kv M=8 us=28.14 gb=79.2
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 kv M=8 us=62.55 gb=35.6
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 kv M=8 us=28.41 gb=78.4
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 kv M=8 us=97.80 gb=22.8
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 kv M=8 us=24.43 gb=91.2
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 kv M=8 us=53.20 gb=41.9
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 kv M=8 us=21.59 gb=103.2
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 kv M=8 us=94.75 gb=23.5
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 kv M=8 us=24.22 gb=92.0
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 kv M=8 us=56.01 gb=39.8
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 kv M=8 us=21.29 gb=104.7
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 kv M=8 us=103.46 gb=21.5
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 kv M=8 us=28.23 gb=78.9
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 kv M=8 us=52.72 gb=42.3
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 kv M=8 us=25.63 gb=86.9
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 kv M=8 us=69.62 gb=32.0
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 kv M=8 us=22.11 gb=100.8
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 kv M=8 us=45.22 gb=49.3
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 kv M=8 us=20.87 gb=106.8
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 kv M=8 us=61.19 gb=36.4
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 kv M=8 us=22.13 gb=100.7
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 kv M=8 us=49.61 gb=44.9
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 kv M=8 us=20.55 gb=108.4
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 kv M=8 us=61.37 gb=36.3
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 kv M=8 us=62.85 gb=35.5
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 kv M=8 us=116.29 gb=19.2
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 kv M=8 us=53.83 gb=41.4
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 kv M=8 us=178.06 gb=12.5
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 kv M=8 us=53.91 gb=41.3
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 kv M=8 us=99.58 gb=22.4
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 kv M=8 us=68.27 gb=32.6
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 kv M=8 us=187.47 gb=11.9
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 kv M=8 us=63.81 gb=34.9
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 kv M=8 us=90.38 gb=24.7
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 kv M=8 us=62.41 gb=35.7
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 kv M=8 us=175.32 gb=12.7
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 kv M=8 us=52.16 gb=42.7
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 kv M=8 us=95.68 gb=23.3
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 kv M=8 us=49.17 gb=45.3
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 kv M=8 us=112.15 gb=19.9
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 kv M=8 us=43.68 gb=51.0
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 kv M=8 us=92.85 gb=24.0
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 kv M=8 us=37.34 gb=59.7
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 kv M=8 us=97.42 gb=22.9
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 kv M=8 us=43.61 gb=51.1
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 kv M=8 us=92.80 gb=24.0
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 kv M=8 us=37.42 gb=59.6
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 kv M=8 us=97.71 gb=22.8
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 kv M=8 us=49.06 gb=45.4
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 kv M=8 us=95.78 gb=23.3
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 kv M=8 us=43.43 gb=51.3
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 kv M=8 us=211.50 gb=10.5
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 kv M=8 us=44.40 gb=50.2
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 kv M=8 us=73.12 gb=30.5
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 kv M=8 us=37.70 gb=59.1
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 kv M=8 us=157.96 gb=14.1
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 kv M=8 us=101.06 gb=22.0
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 kv M=8 us=210.78 gb=10.6
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 kv M=8 us=104.76 gb=21.3
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 kv M=8 us=410.08 gb=5.4
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 kv M=8 us=88.05 gb=25.3
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 kv M=8 us=155.58 gb=14.3
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 kv M=8 us=75.26 gb=29.6
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 kv M=8 us=258.33 gb=8.6
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk2 kv M=8 us=21.73 gb=102.5
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk4 kv M=8 us=20.75 gb=107.4
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk8 kv M=8 us=25.90 gb=86.0
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk2 kv M=8 us=23.32 gb=95.6
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk4 kv M=8 us=23.59 gb=94.4
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk8 kv M=8 us=24.56 gb=90.7
QSPIKE_BEST fmt=w4 class=kv M=8 us=20.55 gb=108.4 eta=0.108 variant=bn64_bk128_w8_s4_u1_g0_sk1 skipped=0
QSPIKE_CAP fmt=w4 class=kv M=8 variant=bn64_bk128_w8_s4_u1_g0_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.7157)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 qo M=1 us=44.34 gb=201.0
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 qo M=1 us=83.78 gb=106.4
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 qo M=1 us=46.88 gb=190.1
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 qo M=1 us=78.46 gb=113.6
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 qo M=1 us=38.76 gb=229.9
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 qo M=1 us=60.94 gb=146.3
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 qo M=1 us=30.47 gb=292.5
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 qo M=1 us=59.22 gb=150.5
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 qo M=1 us=38.63 gb=230.7
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 qo M=1 us=60.98 gb=146.2
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 qo M=1 us=34.70 gb=256.9
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 qo M=1 us=53.74 gb=165.9
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 qo M=1 us=42.25 gb=211.0
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 qo M=1 us=62.86 gb=141.8
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 qo M=1 us=42.02 gb=212.1
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 qo M=1 us=70.58 gb=126.3
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 qo M=1 us=35.40 gb=251.8
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 qo M=1 us=48.87 gb=182.4
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 qo M=1 us=29.37 gb=303.5
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 qo M=1 us=51.48 gb=173.1
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 qo M=1 us=35.32 gb=252.3
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 qo M=1 us=48.79 gb=182.7
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 qo M=1 us=23.64 gb=377.1
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 qo M=1 us=51.69 gb=172.4
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 qo M=1 us=27.85 gb=320.1
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 qo M=1 us=69.48 gb=128.3
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 qo M=1 us=27.56 gb=323.4
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 qo M=1 us=90.16 gb=98.9
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 qo M=1 us=24.78 gb=359.7
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 qo M=1 us=57.17 gb=155.9
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 qo M=1 us=22.81 gb=390.7
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 qo M=1 us=90.87 gb=98.1
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 qo M=1 us=24.95 gb=357.2
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 qo M=1 us=57.34 gb=155.4
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 qo M=1 us=22.72 gb=392.4
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 qo M=1 us=99.08 gb=90.0
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 qo M=1 us=27.75 gb=321.2
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 qo M=1 us=47.33 gb=188.3
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 qo M=1 us=25.78 gb=345.7
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 qo M=1 us=64.97 gb=137.2
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 qo M=1 us=22.41 gb=397.7
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 qo M=1 us=36.06 gb=247.2
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 qo M=1 us=21.66 gb=411.5
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 qo M=1 us=56.38 gb=158.1
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 qo M=1 us=22.49 gb=396.2
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 qo M=1 us=39.35 gb=226.5
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 qo M=1 us=21.59 gb=412.9
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 qo M=1 us=56.72 gb=157.1
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 qo M=1 us=60.09 gb=148.3
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 qo M=1 us=103.27 gb=86.3
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 qo M=1 us=54.48 gb=163.6
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 qo M=1 us=164.77 gb=54.1
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 qo M=1 us=48.30 gb=184.5
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 qo M=1 us=85.87 gb=103.8
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 qo M=1 us=65.76 gb=135.5
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 qo M=1 us=170.96 gb=52.1
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 qo M=1 us=62.31 gb=143.0
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 qo M=1 us=85.96 gb=103.7
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 qo M=1 us=65.65 gb=135.8
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 qo M=1 us=177.06 gb=50.3
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 qo M=1 us=52.92 gb=168.4
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 qo M=1 us=95.18 gb=93.6
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 qo M=1 us=52.88 gb=168.5
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 qo M=1 us=106.75 gb=83.5
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 qo M=1 us=42.22 gb=211.1
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 qo M=1 us=82.61 gb=107.9
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 qo M=1 us=36.08 gb=247.0
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 qo M=1 us=87.38 gb=102.0
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 qo M=1 us=42.32 gb=210.6
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 qo M=1 us=82.68 gb=107.8
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 qo M=1 us=35.85 gb=248.6
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 qo M=1 us=89.77 gb=99.3
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 qo M=1 us=41.71 gb=213.7
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 qo M=1 us=116.91 gb=76.2
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 qo M=1 us=41.79 gb=213.3
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 qo M=1 us=190.19 gb=46.9
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 qo M=1 us=37.81 gb=235.7
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 qo M=1 us=78.01 gb=114.3
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 qo M=1 us=36.73 gb=242.7
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 qo M=1 us=148.76 gb=59.9
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 qo M=1 us=93.82 gb=95.0
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 qo M=1 us=179.51 gb=49.7
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 qo M=1 us=101.90 gb=87.5
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 qo M=1 us=377.14 gb=23.6
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 qo M=1 us=89.51 gb=99.6
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 qo M=1 us=154.70 gb=57.6
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 qo M=1 us=80.52 gb=110.7
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 qo M=1 us=277.01 gb=32.2
QSPIKE_BEST fmt=w4 class=qo M=1 us=21.59 gb=412.9 eta=0.410 variant=bn64_bk128_w8_s4_u1_g0_sk1 skipped=0
QSPIKE_CAP fmt=w4 class=qo M=1 variant=bn64_bk128_w8_s4_u1_g0_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.6005)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 qo M=8 us=49.13 gb=181.4
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 qo M=8 us=79.17 gb=112.6
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 qo M=8 us=44.01 gb=202.5
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 qo M=8 us=71.93 gb=123.9
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 qo M=8 us=40.64 gb=219.3
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 qo M=8 us=66.37 gb=134.3
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 qo M=8 us=29.37 gb=303.5
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 qo M=8 us=53.31 gb=167.2
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 qo M=8 us=40.70 gb=219.0
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 qo M=8 us=66.33 gb=134.4
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 qo M=8 us=34.36 gb=259.4
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 qo M=8 us=53.44 gb=166.8
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 qo M=8 us=43.87 gb=203.2
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 qo M=8 us=62.23 gb=143.2
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 qo M=8 us=41.20 gb=216.3
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 qo M=8 us=70.37 gb=126.7
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 qo M=8 us=36.37 gb=245.0
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 qo M=8 us=49.45 gb=180.2
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 qo M=8 us=29.53 gb=301.8
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 qo M=8 us=51.02 gb=174.7
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 qo M=8 us=36.33 gb=245.3
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 qo M=8 us=49.41 gb=180.4
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 qo M=8 us=24.38 gb=365.6
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 qo M=8 us=51.23 gb=174.0
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 qo M=8 us=28.30 gb=315.0
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 qo M=8 us=58.34 gb=152.8
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 qo M=8 us=28.40 gb=313.8
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 qo M=8 us=92.05 gb=96.8
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 qo M=8 us=22.91 gb=389.1
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 qo M=8 us=49.10 gb=181.5
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 qo M=8 us=28.23 gb=315.7
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 qo M=8 us=88.81 gb=100.4
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 qo M=8 us=23.02 gb=387.1
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 qo M=8 us=53.08 gb=167.9
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 qo M=8 us=22.72 gb=392.3
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 qo M=8 us=97.50 gb=91.4
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 qo M=8 us=28.43 gb=313.5
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 qo M=8 us=50.65 gb=176.0
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 qo M=8 us=25.89 gb=344.3
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 qo M=8 us=65.35 gb=136.4
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 qo M=8 us=23.26 gb=383.2
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 qo M=8 us=41.81 gb=213.2
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 qo M=8 us=21.69 gb=411.0
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 qo M=8 us=56.28 gb=158.4
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 qo M=8 us=23.44 gb=380.2
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 qo M=8 us=46.87 gb=190.2
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 qo M=8 us=21.57 gb=413.2
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 qo M=8 us=56.70 gb=157.2
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 qo M=8 us=61.57 gb=144.8
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 qo M=8 us=108.53 gb=82.1
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 qo M=8 us=54.39 gb=163.9
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 qo M=8 us=164.79 gb=54.1
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 qo M=8 us=54.28 gb=164.2
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 qo M=8 us=91.56 gb=97.3
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 qo M=8 us=65.63 gb=135.8
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 qo M=8 us=171.58 gb=51.9
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 qo M=8 us=67.04 gb=132.9
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 qo M=8 us=91.69 gb=97.2
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 qo M=8 us=65.58 gb=135.9
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 qo M=8 us=177.46 gb=50.2
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 qo M=8 us=56.30 gb=158.3
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 qo M=8 us=98.91 gb=90.1
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 qo M=8 us=52.82 gb=168.7
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 qo M=8 us=104.84 gb=85.0
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 qo M=8 us=45.92 gb=194.1
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 qo M=8 us=85.06 gb=104.8
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 qo M=8 us=36.90 gb=241.5
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 qo M=8 us=89.38 gb=99.7
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 qo M=8 us=45.74 gb=194.9
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 qo M=8 us=85.13 gb=104.7
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 qo M=8 us=36.72 gb=242.8
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 qo M=8 us=90.09 gb=98.9
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 qo M=8 us=47.05 gb=189.4
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 qo M=8 us=88.39 gb=100.8
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 qo M=8 us=42.25 gb=210.9
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 qo M=8 us=194.18 gb=45.9
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 qo M=8 us=42.52 gb=209.6
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 qo M=8 us=68.61 gb=129.9
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 qo M=8 us=36.72 gb=242.7
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 qo M=8 us=150.43 gb=59.3
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 qo M=8 us=95.91 gb=92.9
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 qo M=8 us=194.21 gb=45.9
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 qo M=8 us=101.79 gb=87.6
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 qo M=8 us=379.71 gb=23.5
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 qo M=8 us=91.52 gb=97.4
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 qo M=8 us=159.47 gb=55.9
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 qo M=8 us=80.25 gb=111.1
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 qo M=8 us=275.12 gb=32.4
QSPIKE_BEST fmt=w4 class=qo M=8 us=21.57 gb=413.2 eta=0.410 variant=bn64_bk128_w8_s4_u1_g0_sk1 skipped=0
QSPIKE_CAP fmt=w4 class=qo M=8 variant=bn64_bk128_w8_s4_u1_g0_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.7213)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 gate_up M=1 us=82.91 gb=322.5
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 gate_up M=1 us=97.80 gb=273.4
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 gate_up M=1 us=83.08 gb=321.8
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 gate_up M=1 us=96.64 gb=276.7
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 gate_up M=1 us=80.78 gb=331.0
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 gate_up M=1 us=87.17 gb=306.7
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 gate_up M=1 us=75.92 gb=352.2
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 gate_up M=1 us=92.68 gb=288.5
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 gate_up M=1 us=79.36 gb=336.9
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 gate_up M=1 us=88.17 gb=303.3
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 gate_up M=1 us=78.40 gb=341.1
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 gate_up M=1 us=97.64 gb=273.9
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 gate_up M=1 us=82.64 gb=323.6
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 gate_up M=1 us=90.02 gb=297.0
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 gate_up M=1 us=82.71 gb=323.3
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 gate_up M=1 us=141.06 gb=189.6
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 gate_up M=1 us=81.30 gb=328.9
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 gate_up M=1 us=88.62 gb=301.7
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 gate_up M=1 us=77.40 gb=345.5
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 gate_up M=1 us=104.80 gb=255.1
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 gate_up M=1 us=80.46 gb=332.3
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 gate_up M=1 us=90.79 gb=294.5
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 gate_up M=1 us=75.50 gb=354.2
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 gate_up M=1 us=104.60 gb=255.6
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 gate_up M=1 us=43.74 gb=611.3
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 gate_up M=1 us=90.22 gb=296.4
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 gate_up M=1 us=43.70 gb=611.9
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 gate_up M=1 us=144.95 gb=184.5
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 gate_up M=1 us=43.90 gb=609.1
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 gate_up M=1 us=83.26 gb=321.1
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 gate_up M=1 us=42.84 gb=624.2
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 gate_up M=1 us=142.85 gb=187.2
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 gate_up M=1 us=44.54 gb=600.3
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 gate_up M=1 us=84.03 gb=318.2
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 gate_up M=1 us=43.42 gb=615.8
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 gate_up M=1 us=141.99 gb=188.3
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 gate_up M=1 us=43.75 gb=611.2
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 gate_up M=1 us=64.03 gb=417.6
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 gate_up M=1 us=43.54 gb=614.2
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 gate_up M=1 us=128.53 gb=208.0
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 gate_up M=1 us=44.40 gb=602.3
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 gate_up M=1 us=58.91 gb=453.9
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 gate_up M=1 us=43.06 gb=621.0
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 gate_up M=1 us=112.57 gb=237.5
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 gate_up M=1 us=44.93 gb=595.1
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 gate_up M=1 us=65.10 gb=410.7
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 gate_up M=1 us=43.67 gb=612.3
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 gate_up M=1 us=113.74 gb=235.1
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 gate_up M=1 us=83.67 gb=319.6
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 gate_up M=1 us=108.58 gb=246.2
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 gate_up M=1 us=83.27 gb=321.1
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 gate_up M=1 us=185.27 gb=144.3
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 gate_up M=1 us=84.00 gb=318.3
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 gate_up M=1 us=90.42 gb=295.7
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 gate_up M=1 us=84.81 gb=315.3
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 gate_up M=1 us=192.85 gb=138.6
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 gate_up M=1 us=85.19 gb=313.9
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 gate_up M=1 us=91.61 gb=291.9
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 gate_up M=1 us=85.60 gb=312.4
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 gate_up M=1 us=194.62 gb=137.4
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 gate_up M=1 us=82.88 gb=322.6
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 gate_up M=1 us=101.51 gb=263.4
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 gate_up M=1 us=82.98 gb=322.2
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 gate_up M=1 us=112.12 gb=238.5
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 gate_up M=1 us=83.89 gb=318.7
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 gate_up M=1 us=86.52 gb=309.1
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 gate_up M=1 us=79.68 gb=335.6
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 gate_up M=1 us=90.45 gb=295.6
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 gate_up M=1 us=84.32 gb=317.1
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 gate_up M=1 us=87.62 gb=305.2
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 gate_up M=1 us=80.05 gb=334.0
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 gate_up M=1 us=95.53 gb=279.9
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 gate_up M=1 us=46.83 gb=571.0
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 gate_up M=1 us=119.29 gb=224.2
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 gate_up M=1 us=46.65 gb=573.2
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 gate_up M=1 us=216.81 gb=123.3
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 gate_up M=1 us=44.41 gb=602.1
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 gate_up M=1 us=81.15 gb=329.5
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 gate_up M=1 us=44.40 gb=602.3
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 gate_up M=1 us=183.64 gb=145.6
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 gate_up M=1 us=101.93 gb=262.3
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 gate_up M=1 us=188.18 gb=142.1
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 gate_up M=1 us=107.09 gb=249.7
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 gate_up M=1 us=389.40 gb=68.7
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 gate_up M=1 us=98.30 gb=272.0
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 gate_up M=1 us=161.05 gb=166.0
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 gate_up M=1 us=90.33 gb=296.0
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 gate_up M=1 us=290.07 gb=92.2
QSPIKE_BEST fmt=w4 class=gate_up M=1 us=42.84 gb=624.2 eta=0.619 variant=bn64_bk128_w4_s3_u1_g0_sk1 skipped=0
QSPIKE_CAP fmt=w4 class=gate_up M=1 variant=bn64_bk128_w4_s3_u1_g0_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.7020)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 gate_up M=8 us=85.54 gb=312.6
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 gate_up M=8 us=101.75 gb=262.8
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 gate_up M=8 us=83.19 gb=321.4
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 gate_up M=8 us=97.13 gb=275.3
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 gate_up M=8 us=78.06 gb=342.5
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 gate_up M=8 us=91.62 gb=291.8
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 gate_up M=8 us=76.14 gb=351.2
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 gate_up M=8 us=93.00 gb=287.5
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 gate_up M=8 us=83.85 gb=318.9
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 gate_up M=8 us=92.84 gb=288.0
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 gate_up M=8 us=78.57 gb=340.3
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 gate_up M=8 us=97.89 gb=273.1
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 gate_up M=8 us=82.86 gb=322.7
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 gate_up M=8 us=90.53 gb=295.4
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 gate_up M=8 us=82.71 gb=323.3
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 gate_up M=8 us=140.89 gb=189.8
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 gate_up M=8 us=78.94 gb=338.7
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 gate_up M=8 us=86.81 gb=308.0
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 gate_up M=8 us=78.11 gb=342.3
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 gate_up M=8 us=104.57 gb=255.7
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 gate_up M=8 us=78.28 gb=341.6
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 gate_up M=8 us=87.62 gb=305.2
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 gate_up M=8 us=75.95 gb=352.1
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 gate_up M=8 us=104.32 gb=256.3
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 gate_up M=8 us=43.83 gb=610.0
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 gate_up M=8 us=70.47 gb=379.4
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 gate_up M=8 us=43.90 gb=609.1
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 gate_up M=8 us=146.86 gb=182.1
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 gate_up M=8 us=44.29 gb=603.7
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 gate_up M=8 us=64.45 gb=414.9
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 gate_up M=8 us=42.93 gb=622.9
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 gate_up M=8 us=143.48 gb=186.4
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 gate_up M=8 us=45.14 gb=592.3
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 gate_up M=8 us=72.73 gb=367.6
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 gate_up M=8 us=43.49 gb=614.8
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 gate_up M=8 us=142.87 gb=187.2
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 gate_up M=8 us=43.89 gb=609.2
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 gate_up M=8 us=67.95 gb=393.5
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 gate_up M=8 us=43.66 gb=612.4
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 gate_up M=8 us=129.10 gb=207.1
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 gate_up M=8 us=42.72 gb=625.9
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 gate_up M=8 us=62.70 gb=426.4
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 gate_up M=8 us=43.05 gb=621.1
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 gate_up M=8 us=112.42 gb=237.8
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 gate_up M=8 us=44.69 gb=598.3
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 gate_up M=8 us=72.84 gb=367.1
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 gate_up M=8 us=43.67 gb=612.3
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 gate_up M=8 us=113.40 gb=235.8
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 gate_up M=8 us=83.64 gb=319.7
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 gate_up M=8 us=113.81 gb=234.9
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 gate_up M=8 us=83.37 gb=320.7
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 gate_up M=8 us=185.37 gb=144.2
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 gate_up M=8 us=83.70 gb=319.5
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 gate_up M=8 us=96.79 gb=276.3
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 gate_up M=8 us=84.77 gb=315.4
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 gate_up M=8 us=193.88 gb=137.9
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 gate_up M=8 us=85.87 gb=311.4
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 gate_up M=8 us=98.09 gb=272.6
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 gate_up M=8 us=85.50 gb=312.7
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 gate_up M=8 us=194.87 gb=137.2
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 gate_up M=8 us=82.78 gb=323.0
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 gate_up M=8 us=103.96 gb=257.2
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 gate_up M=8 us=82.86 gb=322.7
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 gate_up M=8 us=110.77 gb=241.4
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 gate_up M=8 us=82.87 gb=322.7
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 gate_up M=8 us=90.41 gb=295.7
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 gate_up M=8 us=78.52 gb=340.5
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 gate_up M=8 us=93.09 gb=287.2
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 gate_up M=8 us=83.29 gb=321.0
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 gate_up M=8 us=91.72 gb=291.5
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 gate_up M=8 us=78.74 gb=339.6
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 gate_up M=8 us=96.64 gb=276.7
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 gate_up M=8 us=51.35 gb=520.7
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 gate_up M=8 us=91.59 gb=292.0
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 gate_up M=8 us=47.22 gb=566.3
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 gate_up M=8 us=219.82 gb=121.6
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 gate_up M=8 us=47.26 gb=565.8
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 gate_up M=8 us=72.15 gb=370.6
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 gate_up M=8 us=44.38 gb=602.4
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 gate_up M=8 us=185.09 gb=144.5
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 gate_up M=8 us=108.04 gb=247.5
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 gate_up M=8 us=199.77 gb=133.8
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 gate_up M=8 us=107.28 gb=249.3
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 gate_up M=8 us=391.58 gb=68.3
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 gate_up M=8 us=103.33 gb=258.8
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 gate_up M=8 us=164.55 gb=162.5
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 gate_up M=8 us=90.08 gb=296.8
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 gate_up M=8 us=289.15 gb=92.5
QSPIKE_BEST fmt=w4 class=gate_up M=8 us=42.72 gb=625.9 eta=0.621 variant=bn64_bk128_w8_s3_u0_g0_sk1 skipped=0
QSPIKE_CAP fmt=w4 class=gate_up M=8 variant=bn64_bk128_w8_s3_u0_g0_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=0.7476)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 down M=1 us=145.54 gb=183.7
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 down M=1 us=237.97 gb=112.4
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 down M=1 us=150.94 gb=177.1
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 down M=1 us=219.82 gb=121.6
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 down M=1 us=136.56 gb=195.8
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 down M=1 us=174.18 gb=153.5
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 down M=1 us=110.84 gb=241.2
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 down M=1 us=167.10 gb=160.0
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 down M=1 us=136.26 gb=196.2
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 down M=1 us=174.41 gb=153.3
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 down M=1 us=142.96 gb=187.0
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 down M=1 us=167.35 gb=159.8
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 down M=1 us=146.25 gb=182.8
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 down M=1 us=193.00 gb=138.5
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 down M=1 us=146.41 gb=182.6
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 down M=1 us=214.40 gb=124.7
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 down M=1 us=134.98 gb=198.1
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 down M=1 us=147.22 gb=181.6
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 down M=1 us=108.78 gb=245.8
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 down M=1 us=156.00 gb=171.4
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 down M=1 us=134.57 gb=198.7
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 down M=1 us=147.81 gb=180.9
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 down M=1 us=100.34 gb=266.5
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 down M=1 us=156.35 gb=171.0
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 down M=1 us=84.99 gb=314.6
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 down M=1 us=246.58 gb=108.4
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 down M=1 us=84.60 gb=316.1
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 down M=1 us=275.91 gb=96.9
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 down M=1 us=78.20 gb=341.9
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 down M=1 us=227.78 gb=117.4
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 down M=1 us=76.48 gb=349.6
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 down M=1 us=279.23 gb=95.8
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 down M=1 us=78.43 gb=340.9
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 down M=1 us=227.76 gb=117.4
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 down M=1 us=76.79 gb=348.2
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 down M=1 us=291.78 gb=91.6
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 down M=1 us=83.18 gb=321.4
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 down M=1 us=140.69 gb=190.1
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 down M=1 us=80.08 gb=333.9
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 down M=1 us=192.69 gb=138.8
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 down M=1 us=71.58 gb=373.6
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 down M=1 us=110.56 gb=241.8
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 down M=1 us=73.49 gb=363.8
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 down M=1 us=165.34 gb=161.7
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 down M=1 us=71.90 gb=371.9
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 down M=1 us=138.87 gb=192.5
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 down M=1 us=73.72 gb=362.7
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 down M=1 us=165.80 gb=161.3
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 down M=1 us=209.06 gb=127.9
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 down M=1 us=320.02 gb=83.6
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 down M=1 us=184.83 gb=144.7
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 down M=1 us=501.99 gb=53.3
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 down M=1 us=179.25 gb=149.2
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 down M=1 us=283.49 gb=94.3
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 down M=1 us=211.50 gb=126.4
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 down M=1 us=511.78 gb=52.2
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 down M=1 us=202.02 gb=132.4
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 down M=1 us=283.78 gb=94.2
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 down M=1 us=211.63 gb=126.3
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 down M=1 us=537.18 gb=49.8
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 down M=1 us=164.18 gb=162.9
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 down M=1 us=297.36 gb=89.9
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 down M=1 us=164.49 gb=162.6
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 down M=1 us=319.66 gb=83.6
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 down M=1 us=146.27 gb=182.8
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 down M=1 us=255.40 gb=104.7
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 down M=1 us=148.74 gb=179.8
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 down M=1 us=259.24 gb=103.1
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 down M=1 us=146.39 gb=182.7
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 down M=1 us=255.70 gb=104.6
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 down M=1 us=148.88 gb=179.6
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 down M=1 us=269.61 gb=99.2
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 down M=1 us=134.18 gb=199.3
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 down M=1 us=450.48 gb=59.4
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 down M=1 us=131.86 gb=202.8
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 down M=1 us=565.89 gb=47.3
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 down M=1 us=109.82 gb=243.5
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 down M=1 us=240.17 gb=111.3
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 down M=1 us=109.51 gb=244.2
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 down M=1 us=441.87 gb=60.5
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 down M=1 us=300.89 gb=88.9
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 down M=1 us=756.31 gb=35.4
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 down M=1 us=313.95 gb=85.2
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 down M=1 us=1145.84 gb=23.3
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 down M=1 us=290.62 gb=92.0
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 down M=1 us=480.40 gb=55.7
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 down M=1 us=257.27 gb=103.9
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 down M=1 us=809.10 gb=33.0
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk2 down M=1 us=55.47 gb=482.1
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk4 down M=1 us=48.73 gb=548.7
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk2 down M=1 us=56.04 gb=477.2
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk4 down M=1 us=50.19 gb=532.7
QSPIKE_BEST fmt=w4 class=down M=1 us=48.73 gb=548.7 eta=0.544 variant=bn64_bk128_w8_s3_u0_g0_sk4 skipped=0
QSPIKE_CAP fmt=w4 class=down M=1 variant=bn64_bk128_w8_s3_u0_g0_sk4 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=1.2027)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 down M=8 us=156.68 gb=170.7
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 down M=8 us=242.26 gb=110.4
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 down M=8 us=150.68 gb=177.5
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 down M=8 us=219.57 gb=121.8
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 down M=8 us=143.79 gb=186.0
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 down M=8 us=217.31 gb=123.0
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 down M=8 us=110.63 gb=241.7
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 down M=8 us=165.68 gb=161.4
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 down M=8 us=144.48 gb=185.1
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 down M=8 us=217.46 gb=123.0
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 down M=8 us=142.99 gb=187.0
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 down M=8 us=166.02 gb=161.1
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 down M=8 us=148.17 gb=180.5
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 down M=8 us=190.74 gb=140.2
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 down M=8 us=146.07 gb=183.1
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 down M=8 us=212.80 gb=125.7
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 down M=8 us=140.54 gb=190.3
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 down M=8 us=160.12 gb=167.0
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 down M=8 us=118.08 gb=226.4
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 down M=8 us=156.68 gb=170.7
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 down M=8 us=140.50 gb=190.3
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 down M=8 us=160.63 gb=166.5
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 down M=8 us=114.72 gb=233.1
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 down M=8 us=157.52 gb=169.7
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 down M=8 us=85.84 gb=311.5
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 down M=8 us=173.41 gb=154.2
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 down M=8 us=85.49 gb=312.8
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 down M=8 us=281.05 gb=95.1
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 down M=8 us=70.40 gb=379.8
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 down M=8 us=153.30 gb=174.4
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 down M=8 us=76.41 gb=349.9
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 down M=8 us=273.35 gb=97.8
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 down M=8 us=70.64 gb=378.5
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 down M=8 us=179.44 gb=149.0
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 down M=8 us=76.73 gb=348.5
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 down M=8 us=287.15 gb=93.1
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 down M=8 us=85.46 gb=312.9
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 down M=8 us=152.17 gb=175.7
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 down M=8 us=80.08 gb=333.9
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 down M=8 us=192.92 gb=138.6
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 down M=8 us=73.07 gb=365.9
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 down M=8 us=133.71 gb=200.0
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 down M=8 us=73.36 gb=364.5
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 down M=8 us=164.82 gb=162.2
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 down M=8 us=73.92 gb=361.7
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 down M=8 us=177.86 gb=150.3
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 down M=8 us=73.68 gb=362.9
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 down M=8 us=165.09 gb=162.0
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 down M=8 us=212.36 gb=125.9
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 down M=8 us=353.12 gb=75.7
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 down M=8 us=184.86 gb=144.6
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 down M=8 us=501.88 gb=53.3
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 down M=8 us=193.75 gb=138.0
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 down M=8 us=302.07 gb=88.5
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 down M=8 us=211.80 gb=126.2
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 down M=8 us=515.18 gb=51.9
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 down M=8 us=219.67 gb=121.7
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 down M=8 us=302.20 gb=88.5
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 down M=8 us=211.71 gb=126.3
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 down M=8 us=538.99 gb=49.6
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 down M=8 us=176.83 gb=151.2
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 down M=8 us=304.53 gb=87.8
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 down M=8 us=164.85 gb=162.2
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 down M=8 us=313.63 gb=85.3
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 down M=8 us=148.87 gb=179.6
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 down M=8 us=269.77 gb=99.1
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 down M=8 us=150.45 gb=177.7
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 down M=8 us=265.82 gb=100.6
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 down M=8 us=149.12 gb=179.3
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 down M=8 us=270.04 gb=99.0
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 down M=8 us=150.56 gb=177.6
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 down M=8 us=272.07 gb=98.3
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 down M=8 us=150.06 gb=178.2
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 down M=8 us=271.95 gb=98.3
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 down M=8 us=133.43 gb=200.4
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 down M=8 us=579.52 gb=46.1
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 down M=8 us=128.40 gb=208.2
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 down M=8 us=203.22 gb=131.6
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 down M=8 us=109.47 gb=244.3
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 down M=8 us=447.51 gb=59.7
QSPIKE w4 bn256_bk64_w4_s2_u0_g0_sk1 down M=8 us=306.60 gb=87.2
QSPIKE w4 bn256_bk64_w4_s2_u0_g1_sk1 down M=8 us=766.13 gb=34.9
QSPIKE w4 bn256_bk64_w4_s2_u1_g0_sk1 down M=8 us=313.98 gb=85.2
QSPIKE w4 bn256_bk64_w4_s2_u1_g1_sk1 down M=8 us=1146.27 gb=23.3
QSPIKE w4 bn256_bk64_w8_s2_u0_g0_sk1 down M=8 us=297.40 gb=89.9
QSPIKE w4 bn256_bk64_w8_s2_u0_g1_sk1 down M=8 us=494.21 gb=54.1
QSPIKE w4 bn256_bk64_w8_s2_u1_g0_sk1 down M=8 us=257.21 gb=104.0
QSPIKE w4 bn256_bk64_w8_s2_u1_g1_sk1 down M=8 us=804.53 gb=33.2
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk2 down M=8 us=56.16 gb=476.2
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk4 down M=8 us=49.69 gb=538.1
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk2 down M=8 us=56.83 gb=470.5
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk4 down M=8 us=50.51 gb=529.4
QSPIKE_BEST fmt=w4 class=down M=8 us=49.69 gb=538.1 eta=0.534 variant=bn64_bk128_w4_s3_u0_g0_sk4 skipped=0
QSPIKE_CAP fmt=w4 class=down M=8 variant=bn64_bk128_w4_s3_u0_g0_sk4 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=ok(maxabs=1.3149)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 lm_head M=1 us=1195.98 gb=276.4
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 lm_head M=1 us=1206.86 gb=273.9
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 lm_head M=1 us=1190.24 gb=277.8
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 lm_head M=1 us=972.42 gb=340.0
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 lm_head M=1 us=1181.41 gb=279.8
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 lm_head M=1 us=1190.07 gb=277.8
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 lm_head M=1 us=1150.59 gb=287.3
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 lm_head M=1 us=1040.42 gb=317.8
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 lm_head M=1 us=1176.46 gb=281.0
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 lm_head M=1 us=1182.26 gb=279.6
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 lm_head M=1 us=1178.11 gb=280.6
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 lm_head M=1 us=1036.24 gb=319.1
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 lm_head M=1 us=1162.68 gb=284.4
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 lm_head M=1 us=1139.81 gb=290.1
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 lm_head M=1 us=1144.15 gb=289.0
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 lm_head M=1 us=1346.66 gb=245.5
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 lm_head M=1 us=1148.16 gb=288.0
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 lm_head M=1 us=1166.46 gb=283.4
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 lm_head M=1 us=1135.89 gb=291.1
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 lm_head M=1 us=984.48 gb=335.8
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 lm_head M=1 us=1135.61 gb=291.1
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 lm_head M=1 us=1158.37 gb=285.4
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 lm_head M=1 us=1123.81 gb=294.2
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 lm_head M=1 us=977.31 gb=338.3
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 lm_head M=1 us=600.04 gb=551.0
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 lm_head M=1 us=640.10 gb=516.5
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 lm_head M=1 us=592.33 gb=558.2
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 lm_head M=1 us=1639.37 gb=201.7
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 lm_head M=1 us=592.95 gb=557.6
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 lm_head M=1 us=643.38 gb=513.9
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 lm_head M=1 us=587.98 gb=562.3
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 lm_head M=1 us=1696.93 gb=194.8
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 lm_head M=1 us=582.35 gb=567.7
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 lm_head M=1 us=645.79 gb=512.0
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 lm_head M=1 us=570.75 gb=579.3
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 lm_head M=1 us=1466.93 gb=225.4
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 lm_head M=1 us=589.54 gb=560.8
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 lm_head M=1 us=532.56 gb=620.8
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 lm_head M=1 us=580.76 gb=569.3
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 lm_head M=1 us=1214.92 gb=272.1
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 lm_head M=1 us=597.53 gb=553.3
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 lm_head M=1 us=517.07 gb=639.4
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 lm_head M=1 us=583.65 gb=566.5
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 lm_head M=1 us=1049.57 gb=315.0
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 lm_head M=1 us=595.89 gb=554.8
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 lm_head M=1 us=594.07 gb=556.5
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 lm_head M=1 us=579.12 gb=570.9
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 lm_head M=1 us=1068.93 gb=309.3
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 lm_head M=1 us=1186.79 gb=278.6
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 lm_head M=1 us=1162.78 gb=284.3
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 lm_head M=1 us=1190.66 gb=277.7
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 lm_head M=1 us=1607.32 gb=205.7
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 lm_head M=1 us=1181.08 gb=279.9
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 lm_head M=1 us=1171.89 gb=282.1
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 lm_head M=1 us=1187.38 gb=278.4
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 lm_head M=1 us=1602.02 gb=206.4
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 lm_head M=1 us=1169.04 gb=282.8
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 lm_head M=1 us=1161.86 gb=284.6
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 lm_head M=1 us=1179.01 gb=280.4
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 lm_head M=1 us=1469.18 gb=225.0
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 lm_head M=1 us=1190.30 gb=277.8
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 lm_head M=1 us=1179.67 gb=280.3
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 lm_head M=1 us=1191.55 gb=277.5
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 lm_head M=1 us=1107.64 gb=298.5
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 lm_head M=1 us=1130.83 gb=292.4
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 lm_head M=1 us=1198.18 gb=275.9
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 lm_head M=1 us=1180.90 gb=280.0
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 lm_head M=1 us=1099.36 gb=300.7
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 lm_head M=1 us=1127.50 gb=293.2
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 lm_head M=1 us=1193.63 gb=277.0
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 lm_head M=1 us=1172.56 gb=282.0
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 lm_head M=1 us=1002.75 gb=329.7
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 lm_head M=1 us=547.40 gb=604.0
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 lm_head M=1 us=706.94 gb=467.7
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 lm_head M=1 us=567.49 gb=582.6
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 lm_head M=1 us=3910.96 gb=84.5
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 lm_head M=1 us=580.33 gb=569.7
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 lm_head M=1 us=566.73 gb=583.4
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 lm_head M=1 us=560.20 gb=590.2
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 lm_head M=1 us=1984.95 gb=166.6
QSPIKE_BEST fmt=w4 class=lm_head M=1 us=517.07 gb=639.4 eta=0.634 variant=bn64_bk128_w8_s3_u0_g1_sk1 skipped=0
QSPIKE_CAP fmt=w4 class=lm_head M=1 variant=bn64_bk128_w8_s3_u0_g1_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=skip(lm_head fp32 参考省略,bitwise 钉 + kv CHECK 已覆盖)
QSPIKE w4 bn64_bk64_w4_s2_u0_g0_sk1 lm_head M=8 us=1295.85 gb=255.1
QSPIKE w4 bn64_bk64_w4_s2_u0_g1_sk1 lm_head M=8 us=1267.71 gb=260.8
QSPIKE w4 bn64_bk64_w4_s2_u1_g0_sk1 lm_head M=8 us=1196.01 gb=276.4
QSPIKE w4 bn64_bk64_w4_s2_u1_g1_sk1 lm_head M=8 us=984.88 gb=335.7
QSPIKE w4 bn64_bk64_w4_s3_u0_g0_sk1 lm_head M=8 us=1191.06 gb=277.6
QSPIKE w4 bn64_bk64_w4_s3_u0_g1_sk1 lm_head M=8 us=1195.46 gb=276.6
QSPIKE w4 bn64_bk64_w4_s3_u1_g0_sk1 lm_head M=8 us=1158.22 gb=285.4
QSPIKE w4 bn64_bk64_w4_s3_u1_g1_sk1 lm_head M=8 us=1056.31 gb=313.0
QSPIKE w4 bn64_bk64_w4_s4_u0_g0_sk1 lm_head M=8 us=1197.47 gb=276.1
QSPIKE w4 bn64_bk64_w4_s4_u0_g1_sk1 lm_head M=8 us=1188.04 gb=278.3
QSPIKE w4 bn64_bk64_w4_s4_u1_g0_sk1 lm_head M=8 us=1193.52 gb=277.0
QSPIKE w4 bn64_bk64_w4_s4_u1_g1_sk1 lm_head M=8 us=1048.39 gb=315.4
QSPIKE w4 bn64_bk64_w8_s2_u0_g0_sk1 lm_head M=8 us=1174.19 gb=281.6
QSPIKE w4 bn64_bk64_w8_s2_u0_g1_sk1 lm_head M=8 us=1151.54 gb=287.1
QSPIKE w4 bn64_bk64_w8_s2_u1_g0_sk1 lm_head M=8 us=1158.81 gb=285.3
QSPIKE w4 bn64_bk64_w8_s2_u1_g1_sk1 lm_head M=8 us=1345.36 gb=245.7
QSPIKE w4 bn64_bk64_w8_s3_u0_g0_sk1 lm_head M=8 us=1160.42 gb=284.9
QSPIKE w4 bn64_bk64_w8_s3_u0_g1_sk1 lm_head M=8 us=1173.81 gb=281.7
QSPIKE w4 bn64_bk64_w8_s3_u1_g0_sk1 lm_head M=8 us=1150.08 gb=287.5
QSPIKE w4 bn64_bk64_w8_s3_u1_g1_sk1 lm_head M=8 us=989.60 gb=334.1
QSPIKE w4 bn64_bk64_w8_s4_u0_g0_sk1 lm_head M=8 us=1150.49 gb=287.4
QSPIKE w4 bn64_bk64_w8_s4_u0_g1_sk1 lm_head M=8 us=1163.08 gb=284.3
QSPIKE w4 bn64_bk64_w8_s4_u1_g0_sk1 lm_head M=8 us=1139.15 gb=290.2
QSPIKE w4 bn64_bk64_w8_s4_u1_g1_sk1 lm_head M=8 us=979.59 gb=337.5
QSPIKE w4 bn64_bk128_w4_s2_u0_g0_sk1 lm_head M=8 us=588.06 gb=562.2
QSPIKE w4 bn64_bk128_w4_s2_u0_g1_sk1 lm_head M=8 us=575.72 gb=574.3
QSPIKE w4 bn64_bk128_w4_s2_u1_g0_sk1 lm_head M=8 us=604.52 gb=546.9
QSPIKE w4 bn64_bk128_w4_s2_u1_g1_sk1 lm_head M=8 us=1679.80 gb=196.8
QSPIKE w4 bn64_bk128_w4_s3_u0_g0_sk1 lm_head M=8 us=608.00 gb=543.8
QSPIKE w4 bn64_bk128_w4_s3_u0_g1_sk1 lm_head M=8 us=555.52 gb=595.1
QSPIKE w4 bn64_bk128_w4_s3_u1_g0_sk1 lm_head M=8 us=598.43 gb=552.5
QSPIKE w4 bn64_bk128_w4_s3_u1_g1_sk1 lm_head M=8 us=1652.79 gb=200.0
QSPIKE w4 bn64_bk128_w4_s4_u0_g0_sk1 lm_head M=8 us=606.77 gb=544.9
QSPIKE w4 bn64_bk128_w4_s4_u0_g1_sk1 lm_head M=8 us=620.36 gb=532.9
QSPIKE w4 bn64_bk128_w4_s4_u1_g0_sk1 lm_head M=8 us=581.61 gb=568.4
QSPIKE w4 bn64_bk128_w4_s4_u1_g1_sk1 lm_head M=8 us=1495.60 gb=221.1
QSPIKE w4 bn64_bk128_w8_s2_u0_g0_sk1 lm_head M=8 us=587.60 gb=562.7
QSPIKE w4 bn64_bk128_w8_s2_u0_g1_sk1 lm_head M=8 us=572.11 gb=577.9
QSPIKE w4 bn64_bk128_w8_s2_u1_g0_sk1 lm_head M=8 us=591.64 gb=558.8
QSPIKE w4 bn64_bk128_w8_s2_u1_g1_sk1 lm_head M=8 us=1218.69 gb=271.3
QSPIKE w4 bn64_bk128_w8_s3_u0_g0_sk1 lm_head M=8 us=589.00 gb=561.3
QSPIKE w4 bn64_bk128_w8_s3_u0_g1_sk1 lm_head M=8 us=561.51 gb=588.8
QSPIKE w4 bn64_bk128_w8_s3_u1_g0_sk1 lm_head M=8 us=593.90 gb=556.7
QSPIKE w4 bn64_bk128_w8_s3_u1_g1_sk1 lm_head M=8 us=1058.41 gb=312.4
QSPIKE w4 bn64_bk128_w8_s4_u0_g0_sk1 lm_head M=8 us=590.87 gb=559.5
QSPIKE w4 bn64_bk128_w8_s4_u0_g1_sk1 lm_head M=8 us=726.07 gb=455.3
QSPIKE w4 bn64_bk128_w8_s4_u1_g0_sk1 lm_head M=8 us=590.05 gb=560.3
QSPIKE w4 bn64_bk128_w8_s4_u1_g1_sk1 lm_head M=8 us=1063.11 gb=311.0
QSPIKE w4 bn128_bk64_w4_s2_u0_g0_sk1 lm_head M=8 us=1184.54 gb=279.1
QSPIKE w4 bn128_bk64_w4_s2_u0_g1_sk1 lm_head M=8 us=1169.92 gb=282.6
QSPIKE w4 bn128_bk64_w4_s2_u1_g0_sk1 lm_head M=8 us=1201.36 gb=275.2
QSPIKE w4 bn128_bk64_w4_s2_u1_g1_sk1 lm_head M=8 us=1623.73 gb=203.6
QSPIKE w4 bn128_bk64_w4_s3_u0_g0_sk1 lm_head M=8 us=1194.57 gb=276.8
QSPIKE w4 bn128_bk64_w4_s3_u0_g1_sk1 lm_head M=8 us=1180.01 gb=280.2
QSPIKE w4 bn128_bk64_w4_s3_u1_g0_sk1 lm_head M=8 us=1201.04 gb=275.3
QSPIKE w4 bn128_bk64_w4_s3_u1_g1_sk1 lm_head M=8 us=1618.71 gb=204.2
QSPIKE w4 bn128_bk64_w4_s4_u0_g0_sk1 lm_head M=8 us=1180.54 gb=280.1
QSPIKE w4 bn128_bk64_w4_s4_u0_g1_sk1 lm_head M=8 us=1169.10 gb=282.8
QSPIKE w4 bn128_bk64_w4_s4_u1_g0_sk1 lm_head M=8 us=1190.14 gb=277.8
QSPIKE w4 bn128_bk64_w4_s4_u1_g1_sk1 lm_head M=8 us=1482.68 gb=223.0
QSPIKE w4 bn128_bk64_w8_s2_u0_g0_sk1 lm_head M=8 us=1180.00 gb=280.2
QSPIKE w4 bn128_bk64_w8_s2_u0_g1_sk1 lm_head M=8 us=1194.70 gb=276.7
QSPIKE w4 bn128_bk64_w8_s2_u1_g0_sk1 lm_head M=8 us=1204.99 gb=274.4
QSPIKE w4 bn128_bk64_w8_s2_u1_g1_sk1 lm_head M=8 us=1101.96 gb=300.0
QSPIKE w4 bn128_bk64_w8_s3_u0_g0_sk1 lm_head M=8 us=1153.74 gb=286.6
QSPIKE w4 bn128_bk64_w8_s3_u0_g1_sk1 lm_head M=8 us=1206.12 gb=274.1
QSPIKE w4 bn128_bk64_w8_s3_u1_g0_sk1 lm_head M=8 us=1194.31 gb=276.8
QSPIKE w4 bn128_bk64_w8_s3_u1_g1_sk1 lm_head M=8 us=1094.82 gb=302.0
QSPIKE w4 bn128_bk64_w8_s4_u0_g0_sk1 lm_head M=8 us=1153.12 gb=286.7
QSPIKE w4 bn128_bk64_w8_s4_u0_g1_sk1 lm_head M=8 us=1198.74 gb=275.8
QSPIKE w4 bn128_bk64_w8_s4_u1_g0_sk1 lm_head M=8 us=1187.66 gb=278.4
QSPIKE w4 bn128_bk64_w8_s4_u1_g1_sk1 lm_head M=8 us=998.78 gb=331.0
QSPIKE w4 bn128_bk128_w4_s2_u0_g0_sk1 lm_head M=8 us=567.32 gb=582.8
QSPIKE w4 bn128_bk128_w4_s2_u0_g1_sk1 lm_head M=8 us=601.29 gb=549.8
QSPIKE w4 bn128_bk128_w4_s2_u1_g0_sk1 lm_head M=8 us=573.59 gb=576.4
QSPIKE w4 bn128_bk128_w4_s2_u1_g1_sk1 lm_head M=8 us=3882.44 gb=85.2
QSPIKE w4 bn128_bk128_w8_s2_u0_g0_sk1 lm_head M=8 us=606.62 gb=545.0
QSPIKE w4 bn128_bk128_w8_s2_u0_g1_sk1 lm_head M=8 us=698.59 gb=473.3
QSPIKE w4 bn128_bk128_w8_s2_u1_g0_sk1 lm_head M=8 us=570.62 gb=579.4
QSPIKE w4 bn128_bk128_w8_s2_u1_g1_sk1 lm_head M=8 us=2002.02 gb=165.1
QSPIKE_BEST fmt=w4 class=lm_head M=8 us=555.52 gb=595.1 eta=0.590 variant=bn64_bk128_w4_s3_u0_g1_sk1 skipped=0
QSPIKE_CAP fmt=w4 class=lm_head M=8 variant=bn64_bk128_w4_s3_u0_g1_sk1 capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok ref=skip(lm_head fp32 参考省略,bitwise 钉 + kv CHECK 已覆盖)
QSPIKE_SUMMARY fmt=w4 class=kv M=1 us=18.89 gb=117.9 eta=0.117 variant=bn64_bk128_w8_s4_u1_g0_sk1
QSPIKE_SUMMARY fmt=w4 class=qo M=1 us=21.59 gb=412.9 eta=0.410 variant=bn64_bk128_w8_s4_u1_g0_sk1
QSPIKE_SUMMARY fmt=w4 class=gate_up M=1 us=42.84 gb=624.2 eta=0.619 variant=bn64_bk128_w4_s3_u1_g0_sk1
QSPIKE_SUMMARY fmt=w4 class=down M=1 us=48.73 gb=548.7 eta=0.544 variant=bn64_bk128_w8_s3_u0_g0_sk4
QSPIKE_SUMMARY fmt=w4 class=lm_head M=1 us=517.07 gb=639.4 eta=0.634 variant=bn64_bk128_w8_s3_u0_g1_sk1
QSPIKE_ETA fmt=w4 M=1 weighted_eta=0.482 weighted_gbps=486.1 assume=0.75 dev_pct=-35.7 step_bytes_gb=4.021 step_time_ms=8.271 (roofline=1008GB/s 权重字节口径)
QSPIKE_SUMMARY fmt=w4 class=kv M=8 us=20.55 gb=108.4 eta=0.108 variant=bn64_bk128_w8_s4_u1_g0_sk1
QSPIKE_SUMMARY fmt=w4 class=qo M=8 us=21.57 gb=413.2 eta=0.410 variant=bn64_bk128_w8_s4_u1_g0_sk1
QSPIKE_SUMMARY fmt=w4 class=gate_up M=8 us=42.72 gb=625.9 eta=0.621 variant=bn64_bk128_w8_s3_u0_g0_sk1
QSPIKE_SUMMARY fmt=w4 class=down M=8 us=49.69 gb=538.1 eta=0.534 variant=bn64_bk128_w4_s3_u0_g0_sk4
QSPIKE_SUMMARY fmt=w4 class=lm_head M=8 us=555.52 gb=595.1 eta=0.590 variant=bn64_bk128_w4_s3_u0_g1_sk1
QSPIKE_ETA fmt=w4 M=8 weighted_eta=0.472 weighted_gbps=475.7 assume=0.75 dev_pct=-37.1 step_bytes_gb=4.021 step_time_ms=8.453 (roofline=1008GB/s 权重字节口径)
QSPIKE_ANCHOR fmt=w4 dev_pct=-37.1..-35.7 assume=0.75 threshold=±10% (>10% → 锚修正条款一次;判定留控制器)
```

### 逐档最优 η 表(从 QSPIKE_BEST/SUMMARY 行汇总;η = gb/1008)

W8(胜出 variant 全记录;每步字节加权合计按调用数 72/72/72/36/1):

| 类 (N,K) | 每步字节占比 | M=1 η (GB/s) | M=8 η (GB/s) | 胜出 variant |
|---|---|---|---|---|
| kv (1024,4096) | 4.0% | 0.278 (280.5) | 0.307 (309.0) | bn64_bk128_w4_s4_sk1 |
| qo (4096,4096) | 16.0% | 0.816 (822.3) | 0.815 (821.9) | bn64_bk128_w4_s4_sk1 |
| gate_up (12288,4096) | 47.9% | 0.877 (884.4) | 0.876 (882.9) | bn64_bk128_w4_s4_sk1 |
| down (4096,12288) | 23.9% | 0.863 (869.8) | 0.864 (870.4) | bn64_bk128_w4_s4_**sk2** |
| lm_head (151936,4096) | 8.2% | 0.934 (941.4) | 0.917 (924.0) | bn64_bk128_w4_s4_sk1 |
| **加权合计** | 100% | **0.800 (806.2)** | **0.807 (813.2)** | GEMM 面 9.391 / 9.311 ms/步 |

W4(同口径):

| 类 (N,K) | 每步字节占比 | M=1 η (GB/s) | M=8 η (GB/s) | 胜出 variant |
|---|---|---|---|---|
| kv (1024,4096) | 4.0% | 0.117 (117.9) | 0.108 (108.4) | bn64_bk128_w8_s4_**u1**_g0_sk1 |
| qo (4096,4096) | 16.0% | 0.410 (412.9) | 0.410 (413.2) | bn64_bk128_w8_s4_**u1**_g0_sk1 |
| gate_up (12288,4096) | 47.9% | 0.619 (624.2) | 0.621 (625.9) | M=1 bn64_bk128_w4_s3_**u1**_g0_sk1 / M=8 bn64_bk128_w8_s3_**u0**_g0_sk1 |
| down (4096,12288) | 23.9% | 0.544 (548.7) | 0.534 (538.1) | bn64_bk128_{w8_s3 / w4_s3}_**u0**_g0_**sk4** |
| lm_head (151936,4096) | 8.2% | 0.634 (639.4) | 0.590 (595.1) | bn64_bk128_{w8_s3 / w4_s3}_**u0**_**g1**_sk1 |
| **加权合计** | 100% | **0.482 (486.1)** | **0.472 (475.7)** | GEMM 面 8.271 / 8.453 ms/步 |

### 变体面判读(形态结论,供 Task 1 段 A 定档 / Task 4 段 B)

- **BLOCK_N=64 全胜**(W8/W4 五档全部胜出配置 bn64):M≤16 小瓦片下 CTA 数主导(64 CTAs@N=4096 vs bn128 32);BK=128 > BK=64 全带,stages 4≥3>2,warps 4≈8 带内(±1%)。
- **down split-K 受益**(W8 sk1 最优 62.24µs/808.8 → sk2 57.83µs/870.4 = +7.6%:64→128 programs 整 1 波;W4 sk1 76.48 → sk4 48.73µs = ×1.57;fp32 partials + 确定性 reduce 代价可忽略)。
- **kv 未受益于 split-K**(sk2–sk8 全劣于 sk1 ~40%):kv kernel 13.6–15.0µs 量级,reduce 第二发 launch 间隔主导,微基准轮转口径下 kv 即 launch/欠填充-bound(η 0.28–0.31 为该口径真实下界;字节占比仅 4%,加权拖累 0.04–0.05;引擎内 graph 相邻发射的真实 kv 成本留 Task 2 引擎内账)。qo 同为 N=4096 欠填充形(64 CTAs,η 0.815),任务口径未评 sk;Task 1 autotune 可按同先例开放。
- **W4 组形态:g0(组对齐,BLOCK_K=128 整除组)为主形态**;g1 逐元素 gather 在低占用类灾难(gate_up bn128_w4_s2_u1:g0 566 → g1 122 GB/s;u1g1 组合全域 85–271 GB/s)——仅 lm_head(594–1187 CTAs 高占用)下 g1≈g0 带内且偶小胜(555.5µs g1 vs 608.0 g0)。**段 B 定档建议:BLOCK_K=128 组对齐,gather 形态出局**。
- **W4 unpack 两法 g0 下平手**:u0(移位掩码拆半双 dot)vs u1(tl.interleave 单 dot)差 <±5%(down bn64_w4_s3_g0:u0 78.20 vs u1 76.48µs;胜出配置两法都有)→ 段 B 两法皆可入 autotune,无定档必要。
- **W4 vs W8 同类开销比 = unpack/组反量化 ALU 税的实测**(R1 风险兑现):gate_up ×0.71、down ×0.62、lm_head ×0.68、qo ×0.50、kv ×0.39。
- 数值门:W8 maxabs 0.59–0.97(kv/qo/gate_up)、down 1.12–1.51(K=12288 点积更长);W4 0.60–0.75 / down 1.20–1.31。atol=1.0 下 down 类靠 rtol 项兜住(Task 1 正式 TOL 标定时 down 类需单独标定,勿复用 1.0)。

### capture 证据(20/20 全过;M6 T5 协议)

逐 (fmt, class, M) 胜出配置:`torch.cuda.CUDAGraph` 捕获(捕获前 side-stream warmup×3,workspace 捕获前就位)→ 重放×2 `torch.equal` 逐位钉 + eager 同输入 launch 逐位对照 + `set_sync_debug_mode(2)` 下 launch 零 host 同步;非 lm_head 类另对 fp32 dequant 参考锚 assert_close(spike 级容差)。QSPIKE_CAP 20 行(上方 stdout 逐字)全部 `capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok`,ref maxabs 0.55–1.51(spike 宽松门内)。split-K 胜出类(down)捕获含主 kernel + reduce 双发射,重放逐位同(确定性 second-pass 复验)。torch 侧注:`set_sync_debug_mode` 为 prototype 特性(首调打 UserWarning,如实记录)。

### 锚修正判定数据(段 0 → 控制器;**本节只交数据与建议,判定行留控制器**)

锚推导假设(spec §1):W8 η=0.85、W4 η=0.75(权重字节口径);修正条款:>10% 偏离允许一次修正。

| fmt | weighted η 实测 (M=1 / M=8) | 假设 | dev | 条款线 ±10% |
|---|---|---|---|---|
| W8 | 0.800 / 0.807 | 0.85 | **−5.9% / −5.1%** | 不触发 |
| W4 | 0.482 / 0.472 | 0.75 | **−35.7% / −37.1%** | 触发 |

- **W8(建议:锚不动,GO 口径成立)**:实测 dev 在 ±10% 内,条款不触发。GEMM 面 9.391(M=1)/ 9.311(M=8)ms/步 vs 假设 8.89ms:+0.42~0.50ms,由锚自带余量吸收——按段 0 GEMM 面推算 W8 C=8 TPOT ≈ 9.31+2.04+0.19 ≈ 11.5 GPU → ~12.3ms ≤ 14.0 锚(余量 ~12%);C=1 ≈ 9.39+非 GEMM ≈ ~9.9 ≤ 10.5 锚(推算口径,终仪式实测为准)。加权 η 被 kv(launch-bound,占比 4%)拖 ~0.04;**主类面(gate_up+down+qo+lm_head,96% 字节)η = 0.85–0.93,段 B 门若按「kernel η ≥ 80% roofline」读段 A kernel 质量,主类全过、加权 0.80 刚过线**。
- **W4(条款触发;两个选项留控制器裁,建议 A)**:段 0 探索级 kernel 加权 η 0.47–0.48,对 0.75 假设偏离 −36~−37%。按 η=0.48 修正口径,GEMM 面 ≈ 8.45ms,W4 C=8 TPOT 推算 ≈ 8.45+2.2+0.19 ≈ 11.2ms > 现锚 10.5 → **现 W4 锚按段 0 数据不过**。选项 A:行使一次修正,把 W4 锚改写到实测口径(η≈0.48 → C=8 ≤ ~12ms 量级,精确值由控制器按 M8 台账裁),docs 记录原推导与新锚后写死;选项 B:维持 0.75 锚 → 段 B 按门 NO-GO(W4 挪 M8),除非段 B kernel 形态实测(打包宽载、寄存器级交织等未探面)压到 0.75。段 B 开工资格门(spec §1 另取 0.80)按段 0 数据同样不过(探索级)。
- 如实边界:以上 η 为**探索级 kernel + 孤立冷口径**数字(坏数据是段 0 的目的,未修到好看);W4 未探面(宽载 unpack、scale 预广播、双 buffer)与引擎内 L2 环境都可能改变段 B 终值;W4 判定本就门控到 Task 4。

### 口径注记

同日同机(RTX 4090;物理 GPU 1,`CUDA_VISIBLE_DEVICES=1`;每腿跑前 `nvidia-smi` 验净 4MiB;前台 timeout 3500s);torch 2.13.0+cu130 / triton 3.7.1;轮转 36 套(lm_head 8 套,22.4GB 超显存的如实偏离,单套 > L2 70× 充分冷);轮转套由 randn 过**真实 quantize 函数**构建(dequant 值 ~N(0,1) 逼真分布,capture fp32 参考对拍 atol 口径才成立;字节随机性对带宽无差);QSPIKE stdout 逐字进本节(上两个 code block);证据归档 `/tmp/opencode/m7/`(spike_w8.log / spike_w4.log 全文);调试用 `--classes kv` 冒烟两轮(W8 一轮 + W4 一轮,后者暴露 unpack_w4 设备 bug)不计入账;quantize 单测重跑复绿(`4 passed`);本节树 = 694b4dc + 本 commit。M7 段 0 不打 tag、不 push(留用户)。

### 段 0 控制器裁决(2026-11-01,裁决行)

1. **W8:条款不触发,锚维持,段 A GO。** dev −5.9%/−5.1% 在 ±10% 内;C=8 推算 TPOT ~12.3 ≤ 14.0、C=1 ~9.9 ≤ 10.5(推算口径,终仪式实测为准)。Task 1 定档输入 = `bn64_bk128_w4_s4` 主形态 + down `sk2`;qo 类 autotune 开放 sk 评估(同 down 先例);down 类 TOL 单独标定(勿复用 spike atol=1.0);kv 类引擎内真实成本以 Task 2 A/B 为准(微基准为 launch-bound 下界,如实)。
2. **W4:条款触发,修正权保留不行使。** 理由:①段 0 数据为探索级,段 B 未探面(宽载 unpack、scale 预广播、双 buffer)与引擎内 L2 环境都可能改变终值;②spec §1/§2 的段 B 资格门(kernel η ≥ 0.80)是先行硬门——资格门过则锚 10.5 在 η≥0.80 口径下自动过(4.03GB@0.80 = 5.04ms GEMM → TPOT ≈ 7.9),资格门不过则 NO-GO 入档(M8 台账),锚从未被测试、无修正必要;③一次修正权留作段 B 硬化 kernel 后的生产级数据裁决(届时若资格门边缘 + 锚物理性不可达同时成立,再行使)。段 B 定档输入 = g0(组对齐 BLOCK_K=128)主形态,gather 出局,unpack 两法皆入 autotune,down sk4。

### 段 B 门裁决(2026-11-02,Task 4 = NO-GO close-out,控制器裁)

三条件读数(spec §2):①主锚 C=8:T2 A/B 轨迹 11.53–11.70 ≤ 14.0 ✓(仪式终验留 Task 5;C=1 轨迹 10.74–11.02 暂高 10.5 锚 ~3%,仪式如实裁);②精度:W8 ppl 增量 −0.088%(≤ +1% ✓;负值 = 对称噪声,不解读为量化更优);③**kernel η 资格门:FAIL——段 0 探索级加权 0.472–0.482 vs 门 0.80,缺口 ~67%;主类最高 gate_up 0.62;ALU 税(unpack + 组反量化,×0.50–0.71 of W8)为结构性,未探面合理增益 10–20% 量级,不足以跨越。判定:段 B NO-GO,W4 挪 M8 台账。**

附带否决轴(即使资格门边缘过的世界线):W4 净收益 ≈ 8.45 vs 9.31ms GEMM 面(~0.9ms/步)对已落地的 W8 微薄;精度域 RTN W4 g128 典型 +2~5% ppl,对 W8 的 −0.088% 是数量级劣化。生产判据下 W4 不值得建。

一次锚修正权:未行使,保留闭合(W8 条款未触发;W4 无需修正即 NO-GO,条款自然失效)。M8 台账 W4 复活条件:kernel 形态突破(marlin 级重构)/ 精度算法升级(GPTQ)/ 显存受限场景价值重估。

### 段 A·kernel 定档:W8 dequant-GEMM(2026-11-01,`tests/test_quant_gemm.py` + `acorninfer/ops/quant_gemm.py`,真机输出)

定档实现(段 0 裁决逐条落地):kernel = spike `_w8_kernel` 胜出形态逐字进产线(核内 dequant:int8 load → fp32 ×scale 行广播 → bf16 → tl.dot fp32 累加;BLOCK_M=16,M≤16 pad,M 为 runtime 参数零重编译);split-K 复用 decode_gemm 的 fp32 partials 常驻 workspace + 确定性 second-pass 归约(零 atomic)。**骨架复用 = `acorninfer/ops/decode_gemm.py` 按现状导入(`_bench_rot`/`_cfg_shmem_ok`/`_workspace`/`_splitk_reduce_kernel`),该文件本任务零改动**(增量幅度 = 0 行,既有 bf16 kernel 与 17 测试行为零变化,`tests/test_decode_gemm.py` 复跑 17 passed)。装配面:`quant_gemm(x, q, scale, cls, cfg=None, out=None)` + `configure()` 装载期入口(Engine 构造/graph 捕获前一次;Task 2 接线调用)+ 逐类 cfg 表 + bucket 分派(M≤16 直判;**`_bucket_for` 归桶不适用本模块,如实偏离**:decode_gemm 表按 (cls, bucket) 记 backend,本模块 cfg 按类一档 —— 段 0 实证 config M-不变,M 守门用 M≤16 直判;prefill 大 M(M>16)路径只留接口注记(`NotImplementedError` 显式报错),接线是 Task 2 范围)。生产路径:qwen3 调用点分派未接(Task 2),`--quant none` 缺省逐字节硬门不动。

装载期 autotune(网格 = 段 0 胜出点 ± 邻域小集,非全网格):数值网格 = {(64,128,4,4) 冠军, (64,128,4,3), (64,64,4,4), (128,64,4,4)}(shmem/整除预过滤照 `_cfg_shmem_ok`;warps=4 已被段 0 全网格钉死不入邻域);split 段 = down/qo top-2 × sk2(kv/gate_up/lm_head sk1 定死)。8 套轮转热测(decode_gemm autotune 同款;权重随机 int8+fp16,带宽只看字节流)。命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/opencode/m7a/loadtime_w8.py`(stdout 逐字;另跑冷缓存版 `TRITON_CACHE_DIR=/tmp/opencode/m7a/triton_cold` 同命令):

```
W8 AUTOTUNE_BEST qo cfg=(64, 128, 4, 4, 1) us=21.79 gb=770 skipped=0      (sk2 候选实测 27.70/27.70µs 全败 → sk1,同段 0)
W8 AUTOTUNE_BEST kv cfg=(64, 128, 4, 3, 1) us=17.15 gb=245 skipped=0      (s4 17.20µs,差 0.3% 噪声级;装载期实测定)
W8 AUTOTUNE_BEST gate_up cfg=(64, 128, 4, 4, 1) us=58.06 gb=867 skipped=0
W8 AUTOTUNE_BEST down cfg=(64, 128, 4, 4, 2) us=59.21 gb=850 skipped=0    (sk2 59.21 < sk1 60.29 ✓ 段 0 先例复现)
W8 AUTOTUNE_BEST lm_head cfg=(64, 128, 4, 4, 1) us=682.14 gb=913 skipped=0
W8 CONFIGURE total_s=1.59 warm_s=0.218 (warm 预算 <1s;autotune 含 JIT 编译,装载期一次) cfgs={...五档如上...}
冷缓存注记:TRITON_CACHE_DIR 全新目录(6 个编译变体全 JIT)同轮 total_s=2.70 / warm_s=0.459 —— warm <1s 预算两口径均过;autotune 总耗时为装载期一次成本,捕获/推理路径零编译。
```

capture 证据(T5 协议,调优后配置全五档;含 down sk2 split-K 双发射;`set_sync_debug_mode(2)` launch 零 host 同步):

```
W8_CAP class=down M=8 cfg=(64, 128, 4, 4, 2) capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok
W8_CAP class=gate_up M=8 cfg=(64, 128, 4, 4, 1) capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok
W8_CAP class=kv M=8 cfg=(64, 128, 4, 3, 1) capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok
W8_CAP class=lm_head M=8 cfg=(64, 128, 4, 4, 1) capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok
W8_CAP class=qo M=8 cfg=(64, 128, 4, 4, 1) capture=ok replay_bitwise=True eager_bitwise=True sync_debug=ok
```

TOL 逐类标定(协议:真实分布权重 randn→quantize_w8,五档 × M∈{1,7,8,16} × 胜出 cfg,`dequant_w8` 后 fp32 matmul 为锚;命令 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/opencode/m7a/calib_tol_w8.py`,2026-11-01,选值紧比 ≤0.38 = 绑定元素 ≥2.6× 余量,写死 `tests/test_quant_gemm.py::TOL_W8`;rtol=2e-2 全类统一,误差机制同源 = 核内 dequant bf16 舍入 × K 点积累计):

| cls | K | maxrel(\|ref\|>1) | maxabs(\|ref\|≤1) | maxrel(\|ref\|>10) | maxdiff@\|ref\| | 选值 (rtol, atol) | 紧比 | atol/maxabs |
|---|---|---|---|---|---|---|---|---|
| qo | 4096 | 0.263 | 0.468 | 0.038 | 1.024@271 | (2e-2, 1.5) | 0.322 | 3.21× |
| kv | 4096 | 0.237 | 0.315 | 0.033 | 0.917@138 | (2e-2, 1.5) | 0.232 | 4.76× |
| gate_up | 4096 | 0.317 | 0.468 | 0.044 | 1.024@271 | (2e-2, 1.5) | 0.322 | 3.21× |
| down | 12288 | 0.488 | 0.714 | 0.067 | 1.641@312 | (2e-2, **2.0**) | 0.380 | 2.80× |
| lm_head | 4096 | 0.354 | 0.496 | 0.045 | 1.308@259 | (2e-2, 1.5) | 0.368 | 3.02× |

紧比 = max\|diff\|/(atol+rtol·\|ref\|)(assert_close 判据全分布复算,余量的精确口径);maxrel(>1) 大值是 \|ref\|≈1 处 atol 覆盖区的相对误差尾,不单独选 rtol。**down K=12288 单独标定(段 0 警示证实)**:spike 级 (2e-2, 1.0) 紧比 0.711(余量仅 1.4×)→ 写死 (2e-2, 2.0);误差随 K 增长(√K 累计)五档可见(down maxabs 0.714 vs kv 0.315)。真错(索引/mask)差 O(10) 起,不可漏网。

测试账(TDD 红绿):红 = `ImportError: cannot import name 'quant_gemm'`(实现前);绿 = 26 passed(对拍 5×4=20 + splitk 确定性 1 + capture 重放逐位 2(down sk2 双发射 / qo sk1)+ dispatch 零 host 同步 1 + CPU 契约 2)。全套件:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest` → **185 passed, 6 skipped, 17 deselected**(基线 159 + 新增 26,零换金;decode_gemm 17 passed 零变化硬门成立)。对 plan 草稿偏离逐条:①文件落位 `acorninfer/ops/quant_gemm.py`(plan 写 `ops/…` 为包内简写,ops 无独立顶层目录);②`_bucket_for` 未复用(见上,归桶无消费者);③TOL 校准中发现 M=16 曾被 `_bucket_for(M) is None` 守门误拒(bucket 上限 8)——改为 M≤16 直判后红转绿,归桶偏离由此一并定案;④autotune 权重轮转用随机 int8+fp16 直构(免 spike 的 bf16→quantize 瞬态;带宽口径字节流等价,数值对拍由测试真实 quantize 链路覆盖)。kv autotune 择 (64,128,4,**3**,1) 与主形态 s4 差 0.3%(噪声级,装载期实测定,两轮复现一致)。证据归档 `/tmp/opencode/m7a/`(calib_tol_w8.log / calib_tol_w8_final.log / loadtime_w8.log / pytest_full.log)。本节树 = d83b3bc + 本 commit。不打 tag、不 push(留用户)。

### 段 A·接线账:W8A16 生产接线(2026-11-02,`tests/test_quant_wiring.py` + 接线五文件 + `scripts/m6_ab.py --quant`,真机输出)

接线面(关键现实逐条落地,调用点 = 4 前向 × 7 投影 + lm_head×4 = 23 处):`config.py::EngineConfig` 增 `quant: str = "none"`(`__post_init__` 校验 ∈ {none,w8a16,w4a16};**EngineConfig 无 tp 字段——quant×tp>1 互斥在 tp 已知的唯一构造点 `server._build_engine` 把守,构造期 ValueError**);`quantize.py` 增 `QUANT_ATTRS`(8 类末段名)+ `apply_quant(model, mode)`(walk `named_modules`,末段名命中且为 nn.Linear → int8 存储 + `_q_scale` fp16 [N] + `_m7_quant` 打标;**none = 不 walk、模块零触碰**;w4a16 = 段 B 门控,装载期显式 NotImplementedError;253 目标 0.53s 全量转完);`qwen3.py` 增 `dispatch_linear(mod, x)` 模块级帮手,23 处调用点改形(`forward`/`forward_decode`/`forward_prefill_batch`/`forward_extend` 各 4 投影 + MLP 3 + lm_head×4;qwen3_tp.py 零触碰,TP×quant 互斥无外溢面);`engine.from_config` 与 `server._build_engine`(tp==1 腿)装载后 `apply_quant` + `quant_gemm.configure(device)`(**均在 Engine 构造/graph 捕获之前**);server CLI `--quant {none,w8a16,w4a16}`(照 `--attention-backend` 旗标风格,getattr 缺省 = 旧调用方零改动);`m6_ab.py` 增 `--quant`(生产同构:同 `apply_quant`/`configure` 码)+ `ABPREFILL` 行(prefill 计时账)+ `LOADTIME` 行(装载耗时账)。

**cls 分派形态定案(两候选中选调用期 shape 查表)**:`apply_quant` walk 期不打标 cls,`dispatch_linear` 在调用点按 `(N,K)` 查 `quant_gemm._CLS_BY_SHAPE`(五档形状全唯一,host dict 一次 ~0.1µs;graph 捕获期定死,零 host 同步)。理由:apply_quant 保持纯 torch/CPU 可测(walk 单测不需要 GPU/五档形状);调用开销 253 次/decode 步 ≈ 30–50µs(<0.3% TPOT),none 腿实测(M6 带内)证实不可见。输出 `view(*x.shape[:-1], N)` 回填领先维(quant_gemm 内部 2D 化,调用点 `.view()/.float()` 语义不变;等效 m6_ab layout harness 的 make_forward 形态)。

大 M 桶(Task 2 遗留注记落地,**无 bf16 回退可选**,spec §4 单存储全路径走量化):`quant_gemm` 增 `_w8_gemm_kernel_m`(pid_m 瓦片轴 × BM∈{32,64,128},dequant 数学与 decode kernel 逐字同源;**sk1 定死**——大 M 网格 N//BN × cdiv(M,BM) 自足,split-K 的 partials/reduce 无增益,down sk2 先例只属 decode 桶)+ `bucket_for`(d≤16 走 T1 decode 路径零变化;s=17–64 / m=65–256 / l>256 三桶,桶数克制)+ `quant_gemm()` 入口按 M 分流(显式 cfg 在大 M 桶解释为 (BM,BN,BK,warps,stages),BM 首位,与 decode 桶 (BN,BK,warps,stages,SK) 字段序不同)。装载期 autotune 候选面:BM 逐桶 + BN/BK 邻域(s:BM32×4 / m:BM64×4 / l:BM64/128×4,shmem ≤96KB + 整除预过滤),代表 M = 桶上界 {64,256,1024};**warm_s=0.027–0.165s(<2s 预算 ✓;configure total 5.0s 热 / 6.97s 冷含 JIT,T1 的 1.59s 增至 3–4× 系大 M 15 轮 autotune,装载期一次,如实)**。r1 冷轮 FULL 输出(stdout 逐字):

```
W8 APPLY modules=253 s=0.53
W8 AUTOTUNE_BEST qo cfg=(64, 128, 4, 3, 1) us=21.99 gb=763 skipped=0
W8 AUTOTUNE_BEST kv cfg=(64, 128, 4, 3, 1) us=16.49 gb=255 skipped=0
W8 AUTOTUNE_BEST gate_up cfg=(64, 128, 4, 4, 1) us=63.39 gb=794 skipped=0
W8 AUTOTUNE_BEST down cfg=(64, 128, 4, 4, 2) us=64.43 gb=781 skipped=0
W8 AUTOTUNE_BEST lm_head cfg=(64, 128, 4, 3, 1) us=692.54 gb=899 skipped=0
W8 AUTOTUNE_BEST qo[s] M=64 cfg=(32, 64, 128, 4, 4) us=22.50 gb=746 skipped=0
W8 AUTOTUNE_BEST qo[m] M=256 cfg=(64, 64, 128, 4, 3) us=62.84 gb=267 skipped=0
W8 AUTOTUNE_BEST qo[l] M=1024 cfg=(128, 64, 64, 8, 3) us=223.03 gb=75 skipped=0
W8 AUTOTUNE_BEST kv[s] M=64 cfg=(32, 64, 128, 4, 4) us=19.17 gb=219 skipped=0
W8 AUTOTUNE_BEST kv[m] M=256 cfg=(64, 64, 128, 8, 3) us=33.58 gb=125 skipped=0
W8 AUTOTUNE_BEST kv[l] M=1024 cfg=(64, 64, 128, 4, 3) us=59.88 gb=70 skipped=0
W8 AUTOTUNE_BEST gate_up[s] M=64 cfg=(32, 64, 128, 4, 3) us=61.67 gb=817 skipped=0
W8 AUTOTUNE_BEST gate_up[m] M=256 cfg=(64, 64, 128, 4, 3) us=186.05 gb=271 skipped=0
W8 AUTOTUNE_BEST gate_up[l] M=1024 cfg=(128, 64, 64, 8, 3) us=695.71 gb=72 skipped=0
W8 AUTOTUNE_BEST down[s] M=64 cfg=(32, 64, 128, 4, 4) us=63.51 gb=793 skipped=0
W8 AUTOTUNE_BEST down[m] M=256 cfg=(64, 64, 128, 4, 3) us=184.44 gb=273 skipped=0
W8 AUTOTUNE_BEST down[l] M=1024 cfg=(128, 64, 64, 8, 3) us=694.22 gb=73 skipped=0
W8 AUTOTUNE_BEST lm_head[s] M=64 cfg=(32, 64, 128, 4, 4) us=1388.21 gb=449 skipped=0
W8 AUTOTUNE_BEST lm_head[m] M=256 cfg=(64, 64, 128, 8, 3) us=2896.92 gb=215 skipped=0
W8 AUTOTUNE_BEST lm_head[l] M=1024 cfg=(128, 64, 64, 8, 3) us=10648.58 gb=58 skipped=0
W8 CONFIGURE total_s=6.97 warm_s=0.165 (warm 预算 <2s;…大 M 三桶 autotune_s=5.51) cfgs={五档如上} m_cfgs={15 项如上}
```

(gb 列在大 M 桶口径失真:M=1024 GEMM 是 flop-bound 非权重流-bound,qo[l] 223µs ≈ 154 TFLOPS bf16 ≈ 93% 峰值——如实注记,大 M 桶的账看 µs/flops 不看 gb。轮转间 autotune 胜者有小幅漂移(kv 曾择 (64,64,4,4,1)/qo s3/s4 互换),均噪声级,装载期实测定。)

大 M TOL 复算(TOL 协议沿用 T1;命令 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/opencode/m7b/calib_tol_w8_m.py`,2026-11-02,stdout 归档 `/tmp/opencode/m7b/calib_tol_w8_m.log`):五档 × M∈{64,256,1024} × 桶代表 cfg,误差分布与 M≤16 桶同源(逐元素界 M 无关):maxabs(qo/kv/gate_up/lm_head ≤0.57、down ≤0.84)均在 TOL_W8 原表覆盖内,**紧比 ≤0.43(绑定元素 ≥2.3× 余量;T1 桶 ≤0.38),选值不变、零新桶容差**;`tests/test_quant_gemm.py` 新增大 M 对拍 20(五档 × M∈{17,64,256,257})+ 大 M capture 重放逐位 1 + 路由契约 1 + 桶边界 1。

**A/B 门(GPU 1,`m6_ab.py --quant {none,w8a16}`,n=3 交错 [none,w8a16]×3,生产同构接线;对照 M6 带 graph 17.91–17.96 / 19.87–19.93、eager 19.13–19.35 / 21.11–21.28)**:step_ms 主口径,`AB` 行逐字(三进程 none:ab_none_r1/r2/r3.log;三进程 w8a16:ab_w8a16_r1/r2/r3.log,`/tmp/opencode/m7b/` 归档):

| 腿 | C=1 graph | C=8 graph | C=1 eager | C=8 eager |
|---|---|---|---|---|
| none r1 | 17.92 | 19.90 | 19.17 | 21.45 |
| none r2 | 17.92 | 19.93 | 19.16 | 21.15 |
| none r3 | 17.92 | 19.89 | 19.15 | 21.12 |
| **none 中位** | **17.92** | **19.90** | **19.16** | **21.15** |
| w8a16 r1 | 10.78 | 11.54 | 16.95 | 18.13 |
| w8a16 r2 | 11.02 | 11.68 | 17.59 | 17.22 |
| w8a16 r3 | 10.74 | 11.59 | 16.65 | 16.77 |
| **w8a16 中位** | **10.78** | **11.59** | **16.95** | **17.22** |
| **graph 收益** | **7.14 ≥ 5.0 ✓** | **8.31 ≥ 4.0 ✓** | eager C=1 **−2.21 ✓** | eager C=8 **−3.93 ✓** |

**判定行:四门全过,且逐轮(r1/r2/r3 各自单独)四门均过——GO,eager 双 C 为改善非劣化。** none 腿 vs M6 带:graph C=1 17.92 带内 / C=8 19.89–19.93 带内 / eager C=1 19.15–19.17 带内 / eager C=8 21.12–21.45(r1 21.45 高出带顶 0.17,r2/r3 带内;中位 21.15 带内——r1 为首跑机器态,如实注记;dispatch 帮手理论开销 ~30–50µs/步,不可见)。锚轨迹(判定行留 Task 5 终仪式):w8a16 graph TPOT C=8 11.53–11.70(锚 ≤14.0,余量 ~20%✓);**C=1 10.74–11.02 vs 锚 ≤10.5——当前带高于锚 0.24–0.52ms,如实呈报控制器**(段 0 推算 ~9.9;差距主要在非 GEMM 面与 kv launch-bound 成本,T5 以仪式口径终判)。

**prefill 账(spec §6 R1;`ABPREFILL` 行逐字,in-process 单步 8×128 token 全准入,sync 前后夹墙钟;二轮取后轮)**:

| 腿 | C=1(M=128,m 桶) | C=8(M=1024,l 桶) |
|---|---|---|
| none(r1/r2/r3) | 32.49 / 32.44 / 32.80 | 142.12 / 142.50 / 144.09 |
| w8a16(r1/r2/r3) | 30.82 / 31.56 / 31.56 | 150.78 / 150.82 / 151.59 |
| 中位 | 32.49 → **31.56(−2.9%)✓** | 142.50 → **150.82(+5.8%)如实入账** |

C=1 prefill w8a16 更快(M=128 权重流-bound,量化字节优势兑现);C=8 prefill +5.8%(M=1024 flop-bound:dequant ALU + int8→bf16 转换纯增开销,大 M 桶 η 表同证)——**非灾难级(≫2× 阈),R1 允许回退,照实入账不粉饰;chunked prefill(预算切块)下 M 更小,向 C=1 侧移动**。TTFT 影响面:服务端 TTFT ≈ queue + 本步,单请求 128-token 场景为净改善;8 并发到齐场景 +8.3ms。

**parity(margin 门协议,M5 先例;量化 vs bf16 是不同数值域,不逐位)**:①tiny/mini 腿 = 套件内 `test_w8a16_mini_model_decode_margin`(128 倍数 mini 模型端到端 none vs w8a16 decode logits:max|Δ| 打印 + margin 门 argmax 断言,绿);②8B 单请求腿 = `/tmp/opencode/m7b/parity_8b.py`(同 seed prompt 128 token,贪心 16 步,graph 生产同构,双进程 none/w8a16):**token 流 16/16 逐位一致**;逐步 top-1 logit |Δ| ≤ 0.75(量级 |logit| 14–51 的 ~1–2%),top-2 margin 逐步非平局(两腿 min=0.25 同值),无 argmax 翻转。stdout 逐字:

```
PARITY none  tokens=[2132, 4977, 1075, 498, 3003, 3267, 291, 264, 6514, 315, 4194, 5766, 11, 17738, 11]
PARITY w8a16 tokens=[2132, 4977, 1075, 498, 3003, 3267, 291, 264, 6514, 315, 4194, 5766, 11, 17738, 11]
PARITY none  top1=[14.56, 34.5, 34.0, 33.25, 38.0, 39.0, 51.25, 45.25, 38.25, 41.25, 36.0, 41.0, 39.0, 33.75, 41.25]
PARITY w8a16 top1=[14.69, 35.0, 34.25, 34.0, 38.25, 39.75, 51.5, 45.75, 38.0, 41.0, 36.0, 41.0, 39.25, 33.75, 41.5]
```

**装载耗时(spec R5 预算 120s;`LOADTIME` 行逐字)**:none 30.8–34.3s(纯 bf16 读盘);w8a16 load 30.7–31.2 + apply 0.53(253 模块 RTN)+ configure 5.0(热)/6.97(冷,含 JIT)≈ **36.2–38.7s 全链,预算 120s 内 ✓**(triton 缓存热后 configure 稳定 5.0s;不做磁盘量化缓存,non-goal 不变)。

**none 硬门证据**:全套件 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest` → **217 passed, 6 skipped, 17 deselected**(= 基线 185 原样全绿 + 新增 32:quant_wiring 9 + quant_gemm 大 M 23;golden m0–m5 零换金,golden 文件 diff 为零);none 缺省时 `dispatch_linear` 走 `mod(x)` 原路径(CPU 逐位单测 + golden 面),A/B none 腿与 M6 带重合(上表)= 8B 生产级零漂移复核。

测试账(TDD 红绿):红 = `ImportError: cannot import name 'dispatch_linear'` + 大 M `KeyError/NotImplementedError`(实现前,24 failed);绿 = wiring 9 + gemm 49。**对 plan 草稿/关键现实偏离逐条**:①walk 测试不能用 tiny(plan 草稿"CPU/tiny 为主"):tiny hidden=64 触发 quantize_w8 零特例守门(T0 钉死契约)→ 改 128 倍数 mini 配置,none 逐位测仍用 tiny;②草稿 `mod.weight.data = q` 直接换存储实测失败(torch 禁止 requires_grad 的 .data 换非浮点——草稿注③预判命中)→ 换存储前置 `requires_grad_(False)`(int8 权重本就不可训,引擎全推理无 backward,语义无损);③cls 形态在"装载期打标 vs shape 查表"两候选中定 shape 查表(理由见上;两候选均为指令允许面,选型如实报告);④quant×tp 互斥落点 = `server._build_engine`(EngineConfig 无 tp 字段,指令已预授权"以 config.py 现状为准");⑤大 M 桶 split-K 不评估(sk1 结构性定死,网格自足;非"评估后放弃");⑥T1 注记期契约测试 `test_bucket_guard_m_gt16_prefill_note`(NotImplementedError)退役改写为大 M 契约测试;⑦prefill 账仪器 = m6_ab `ABPREFILL`(指令"bench --in-tokens 128 或等价"的等价腿);⑧warm 预算口径从 T1 的 <1s 放宽为本指令 <2s(实测 0.027–0.165s ✓)。lm_head 引擎内大 M 不出现(prefill_batch 只取各行末位,M=N≤8 → decode 桶;大 M lm_head 消费面 = Task 3 ppl window=512 / M0 oracle)。证据归档 `/tmp/opencode/m7b/`(ab_none_r1/r2/r3.log、ab_w8a16_r1/r2/r3.log、calib_tol_w8_m.log、parity 两腿 stdout、baseline_pytest)。本节树 = 6dc7c00 + 本 commit。不打 tag、不 push(留用户)。

### 段 A·精度账:自研困惑度 harness + W8 ppl(2026-11-02,`scripts/perplexity.py` + `data/wikitext-2-raw/`,真机输出)

背景:spec §3 精度锚 **W8 ppl ≤ +1% vs bf16 基线**。harness 自研(运行时零 transformers;纪律自查 grep:本文件与 import 链 `acorninfer.tokenizer`/`models.qwen3`/`quant.quantize`/`ops.quant_gemm` 零命中):分词 = 自家 `Tokenizer(f"{model_dir}/tokenizer.json")` 全文一次性 encode;模型直建 `qwen3.from_pretrained`(cache=None 无 KV 全前向,logits [B,S,vocab] fp32);量化腿生产同构(装载后 `apply_quant(model,"w8a16")` + `quant_gemm.configure(device)`,window=512 → lm_head M=4096 走大 M 桶,m_configs 表须先配置,与 m6_ab/engine 接线同码)。

**滑窗 stride 法**(window=512 stride=256 batch=8,注释钉死在脚本内):wiki.test.raw 按 "\n" 切行保序、"\n".join 拼接(空行原样保留,与原文件逐字节等价),Tokenize 全文一次 → 299,078 token。窗口 i 覆盖 [i·stride, i·stride+window);logits 行 r 预测 token s+r+1。窗口 0 计全部 window−1 个位置(行 0..window−2);后续窗口只计"新暴露"的末 stride 个位置(行 window−stride−1..window−2,与此前所有窗口目标集严格不相交 → 每 token 恰计一次;CPU 纯索引覆盖校验:299,007 目标连续无重无漏)。counted = (window−1)+(windows−1)·stride,ppl = exp(nll_sum/counted) 可复算;窗口定长,尾部 71 token 不计。fp32 logits 全量物化 [8,512,151936] ≈2.5GB/批,逐批 CE 累加(批内逐窗行求和),零跨批拼接。one-off 修正:harness 首版计数行末行切片 off-by-one(行 window−1 预测窗外 token,首批即抛 batch_size mismatch),索引校验后修正重跑,**两腿均为修正后同一版本**(首跑未产生任何入账数字)。

**数据源(R7 双源,取活者)**:首选 Smerity zip 实测活(2026-11-02),命令原样:

```bash
curl -L https://wikitext.smerity.com/wikitext-2-raw-v1.zip -o /tmp/opencode/m7/wikitext.zip && unzip
cp wikitext-2-raw/wiki.test.raw → data/wikitext-2-raw/   # 4358 行 1,290,590 字节
```

HF parquet 备选未启用;`.gitignore` 增 `data/`(数据不入 git)。环境:GPU 1(`CUDA_VISIBLE_DEVICES=1`),双腿跑前 `nvidia-smi` 各验净 4MiB;torch 2.13.0+cu130 / triton 3.7.1。

**PPL 行逐字**(stdout 全文归档 `/tmp/opencode/m7/ppl_none.log`、`ppl_w8a16.log`;各一遍):

```
DATA path=data/wikitext-2-raw/wiki.test.raw lines=4359 tokens=299078
LOADTIME quant=none load_s=31.1 apply_s=0.0 configure_s=0.0
PPL quant=none tokens=299078 nll_sum=690601.61 ppl=10.0709 windows=1167
PPLPOS quant=none counted=299007 mean_nll=2.309650 eval_s=75.7 peak_gib=18.78
DATA path=data/wikitext-2-raw/wiki.test.raw lines=4359 tokens=299078
LOADTIME quant=w8a16 load_s=31.4 apply_s=0.6 configure_s=5.0
PPL quant=w8a16 tokens=299078 nll_sum=690337.84 ppl=10.0620 windows=1167
PPLPOS quant=w8a16 counted=299007 mean_nll=2.308768 eval_s=80.1 peak_gib=15.78
```

(注:两腿同 token 同 counted,ppl 比值 = nll 比值;`lines=4359` 为 split 元素数——文件以换行结尾多出一个空尾元素,join 后与原文件逐字节等价。)

**精度账与锚对照**:

| 腿 | tokens | windows | counted | nll_sum | ppl | 相对增量 |
|---|---|---|---|---|---|---|
| none(bf16 基线) | 299,078 | 1,167 | 299,007 | 690,601.61 | **10.0709** | — |
| w8a16 | 299,078 | 1,167 | 299,007 | 690,337.84 | **10.0620** | **−0.09%** |

增量 = (ppl_w8/ppl_none − 1)×100% = (10.0620/10.0709 − 1)×100% = **−0.088%**(NLL 总额口径 −0.038% 同向)。**锚 ≤ +1%:✓ 通过,且为负增量**。初步归因:W8 per-channel 对称 RTN 逐元素误差 ≤ scale/2(+fp16 scale 表示误差),对 logits 是对称小扰动(parity 账 top-1 |Δ| ≤ 0.75 同量级);逐窗 NLL 差呈对称噪声(45.2% 窗口为正,中位 −0.20 nats,极值 −8.8/+8.7),正负相消后净差为统计涨落量级,**不解读为"量化更优"**。超标归因条款(R4)未触发,top-10 恶化窗口列表不适用;判定行(锚过/不过终裁)留控制器/Task 5。

**装载耗时(R5 账延续)**:none load 30.6–31.1s(两次进程);w8a16 load 31.4 + apply 0.6(253 模块 RTN)+ configure 5.0(triton 缓存热)≈ **37.0s 全链**,120s 预算内 ✓。eval 耗时 none 75.7 / w8a16 80.1s(+5.8%,大 M 桶 dequant ALU 开销,prefill C=8 flop-bound 结论同源);peak VRAM 18.78(none,bf16 权重+logits)/15.78(w8a16,int8 存储)GiB。

全套件复核:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest` → **217 passed, 6 skipped, 17 deselected**(本任务零生产码改动,新增仅 `scripts/perplexity.py`(非测试路径,零收集)+ docs + .gitignore,golden 零换金)。证据归档 `/tmp/opencode/m7/`(wikitext.zip、wiki.test.raw 提取物、ppl_none.log、ppl_w8a16.log、ppl_windows_none.json、ppl_windows_w8a16.json)。本节树 = 17061b6 + 本 commit。不打 tag、不 push(留用户)。

## M7 终账(2026-11-02,收官:全量回归 + in-process 终bench + 仪式三腿(轨 1,W4 腿撤销)+ 判定行落笔,真机输出)

**仪式构成(承段 B 门裁决)**:段 B NO-GO(裁决在档:W4 kernel η 资格门 FAIL,0.472–0.482 vs 门 0.80,W4 挪 M8 台账)→ 计划四腿撤销 W4 腿,**仪式 = 三腿**:vLLM bf16 8100 → AcornInfer none 8200(满配方)→ AcornInfer w8a16 8200;轨 2 免跑(TP×quant 构造期互斥,无外溢面,照 plan Task 5)。

### 终账·全量回归(零换金;跑前双卡各 4 MiB 验净)

| 套件 | 命令 | 尾行(逐字) | 判定 |
|---|---|---|---|
| fast | `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` | `217 passed, 6 skipped, 17 deselected, 1 warning in 16.19s` | = Task 2/3 后基线,零改动 ✓ |
| slow | `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q -m slow` | `17 passed, 223 deselected in 416.51s (0:06:56)` | golden 17 原样,零换金 ✓ |
| 双卡 tp 全套 | `CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest -q tests/test_tp_eager.py tests/test_tp_engine.py tests/test_tp_server.py` | `13 passed in 18.97s`;跑后双卡回落 4 MiB 双查、零计算进程 | 全绿 ✓ |

slow deselect 计数注记(如实):M6 终账基线行为 `17 passed, 161 deselected`(总收集 178);本日 223 deselected 系 M7 三个测试任务扩容后总收集 240 = 217 + 6 + 17,三套件计数自洽,golden 面本身零变化。

### 终账·in-process 终bench(GPU 1;none 零漂移 + w8a16 复核)

命令(五样本串行,stdout 逐字归档 `/tmp/opencode/m7t5/ab_*.log`):`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_ab.py --label m7t5_{none,w8a16}_s{1,2,3}`(w8a16 加 `--quant w8a16`;warmup 10 + 计 30):

| 样本 | C=1 graph | C=8 graph | C=1 eager | C=8 eager |
|---|---|---|---|---|
| none s1 | 17.92 | 19.91 | 19.19 | 21.15 |
| none s2 | 17.91 | 19.89 | 19.14 | 21.14 |
| M6 带 | 17.91–17.96 | 19.87–19.93 | 19.13–19.35 | 21.11–21.28 |
| w8a16 s1 | 10.75 | 11.49 | 17.49 | 18.34 |
| w8a16 s2 | 10.72 | 11.50 | 17.64 | 18.14 |
| w8a16 s3 | 10.73 | 11.49 | 19.25 | 19.03 |
| T2 带 | 10.74–11.02 | 11.54–11.68 | 16.65–17.59 | 16.77–18.13 |

判定(如实):none 腿 8/8 全带内 = **引擎级零漂移硬证延续**;w8a16 graph C=1 10.72–10.75 与 T2 带底 10.74 带缘重合(−0.02~+0.01 噪声级)、C=8 11.49–11.50 低于 T2 带底 0.05(−0.4%,三样本同侧一致的有利侧机时偏快——none 腿同窗贴中带,非全局机时偏快;autotune 胜者轮转漂移先例同款,如实);eager 整体略偏慢 +0.3~+1.1ms(s3 19.25/19.03 最宽,该档历史带宽本宽 + 首跑机器态离带先例同款,graph 主口径不受影响)。锚轨迹终值:**C=8 graph 11.49–11.50(锚 ≤14.0);C=1 graph 10.72–10.75(锚 ≤10.5,未达,判定行如实裁)**。

### 终账·仪式三腿(GPU 0 照 M5/M6 配方;逐腿 stdout 原样)

配方注记(照抄 + gate-basis 腿如实):命令/端口/协议照 M6 终账节原样(seed=20261027、in=128/out=32、temperature=0、max_tokens=32、warmup 1 条不计入;`ss -tlnp` 取真身 PID 定向 kill:vLLM 55424 / none 70758 / w8a16 102713,均 SIGTERM 优雅退出、端口闭、双卡 4 MiB、无计算进程;每腿起跑前双卡 4 MiB 实录在档;全程无 pkill 模糊匹配)。vLLM 就绪 ~35s;AcornInfer none 就绪 ~30s、w8a16 就绪 ~36s(Δ≈+6s = apply+configure,装载耗时账同向)。**gate-basis 腿如实入档**:派单引用门(92.4/90.3/83.4 ±1pp)的原口径 = M5/M6 in-process graph,故在派单三腿之外照 M6 配方补跑一条 in-process graph none 腿(`acorninfer.bench`,M6 终账同款命令),供同基线读数;它不是第四条验收腿,是引用门自身的口径腿。

命令原样(腿序照派单,gate-basis 腿居 none 之后):

- vLLM TP=1:`CUDA_VISIBLE_DEVICES=0 /home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90 --dtype bfloat16`
- AcornInfer HTTP 满配方 none:`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.server --model-dir /home/ernest/model-weight/Qwen3-8B --port 8200 --attention-backend triton --fused-ops --cuda-graph`
- AcornInfer HTTP 满配方 w8a16:同上加 `--quant w8a16`
- gate-basis 腿(in-process graph none,M6 终账 92.4/90.3/83.4 原口径):`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.bench --in-tokens 128 --out-tokens 32 --attention-backend triton --fused-ops --cuda-graph`
- 压测:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --base http://127.0.0.1:{8100|8200} --concurrency 1 4 8`

逐腿 stdout 逐字(server 全文归档 `/tmp/opencode/m7t5/leg{A,B,B2,C}_*.log`;`RESULT-VLLM` 前缀为 `bench_remote` 固定标签):

```
# leg A:vLLM bf16 8100
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0267s ttft_mean=0.0267s tpot_mean=16.88ms tok_s=58.17
RESULT-VLLM C=4 ttft_p50=0.0679s ttft_mean=0.0573s tpot_mean=17.87ms tok_s=207.57
RESULT-VLLM C=8 ttft_p50=0.0804s ttft_mean=0.0732s tpot_mean=17.91ms tok_s=356.12

# leg B:AcornInfer HTTP 满配方 none 8200
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0382s ttft_mean=0.0382s tpot_mean=18.32ms tok_s=52.79
RESULT-VLLM C=4 ttft_p50=0.0824s ttft_mean=0.0823s tpot_mean=20.71ms tok_s=174.23
RESULT-VLLM C=8 ttft_p50=0.1861s ttft_mean=0.1675s tpot_mean=20.95ms tok_s=276.06

# leg B2:gate-basis 腿(in-process graph none,M6 原口径)
RESULT C=1 ttft_p50=0.0346s ttft_mean=0.0346s tpot_mean=17.93ms tok_s=54.21
RESULT C=4 ttft_p50=0.0775s ttft_mean=0.0775s tpot_mean=19.55ms tok_s=187.21
RESULT C=8 ttft_p50=0.1455s ttft_mean=0.1455s tpot_mean=19.98ms tok_s=296.32

# leg C:AcornInfer HTTP 满配方 w8a16 8200
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0358s ttft_mean=0.0358s tpot_mean=10.88ms tok_s=85.74
RESULT-VLLM C=4 ttft_p50=0.1225s ttft_mean=0.1009s tpot_mean=12.11ms tok_s=266.58
RESULT-VLLM C=8 ttft_p50=0.1903s ttft_mean=0.1714s tpot_mean=12.57ms tok_s=401.84
```

**仪式表(M7 终账):**

| 腿 | C=1 tok/s | C=4 tok/s | C=8 tok/s | TTFT p50 C=1/4/8 (s) | TPOT mean C=1/4/8 (ms) |
|---|---|---|---|---|---|
| vLLM TP=1 HTTP bf16 | 58.17 | 207.57 | 356.12 | 0.0267 / 0.0679 / 0.0804 | 16.88 / 17.87 / 17.91 |
| AcornInfer HTTP 满配方 none(零漂移证明腿) | 52.79 | 174.23 | 276.06 | 0.0382 / 0.0824 / 0.1861 | 18.32 / 20.71 / 20.95 |
| AcornInfer HTTP 满配方 w8a16(**副锚腿**) | 85.74 | 266.58 | 401.84 | 0.0358 / 0.1225 / 0.1903 | 10.88 / 12.11 / 12.57 |
| AcornInfer in-process graph none(gate-basis 口径腿,M6 终账同款) | 54.21 | 187.21 | 296.32 | 0.0346 / 0.0775 / 0.1455 | 17.93 / 19.55 / 19.98 |

达标率算术(逐档,AcornInfer ÷ vLLM):

- **none HTTP**:C=1 52.79/58.17 = **90.8%**、C=4 174.23/207.57 = **83.9%**、C=8 276.06/356.12 = **77.5%**;
- **none in-process graph(gate 原口径)**:C=1 54.21/58.17 = **93.2%**、C=4 187.21/207.57 = **90.2%**、C=8 296.32/356.12 = **83.2%**;
- **w8a16(副锚,对 vLLM bf16)**:C=1 85.74/58.17 = **147.4%**、C=4 266.58/207.57 = **128.4%**、C=8 401.84/356.12 = **112.8%**(预估 ~150% 量级的实测兑现:C=1 贴近预估,C=4/C=8 递减——递减主体非 decode 面,见判读②)。

none 腿硬门读数(如实,三读法并列):

1. **字面读法**(HTTP none 达标率 vs 引用门 92.4/90.3/83.4 ±1pp):90.8/83.9/77.5 = **−1.6 / −6.4 / −5.9 pp,超带**;
2. **同基线读法 A**(HTTP none vs M6 终账同口径 HTTP 满配方腿 90.9/84.0/77.7):**−0.2 / −0.0 / −0.2 pp,带内**(tok_s 原值 −0.1% / 0.0% / −0.2%);
3. **同基线读法 B**(in-process graph none vs 引用门原口径 92.4/90.3/83.4):**+0.8 / −0.1 / −0.2 pp,三档全带内**(tok_s 原值 +1.0% / 0.0% / −0.2%)。

裁决依据(如实):字面超带是**口径错配的已知物理,非生产漂移**——引用门数字为 M5/M6 in-process graph 口径,三腿仪式的 none 腿为 HTTP 口径;HTTP/in-process 比 C=8 276.06/296.32 = ×0.932,M6 终账同口径 276.55/296.88 = ×0.932 **逐位复现**(= M6 终账判读①已入档的 SSE 全程流式开销 ×0.93)。同基线双读法 A、B 三档全带内 = **生产零漂移双证**(且 vLLM 基线侧对 M6 +0.1%/+0.0%/+0.1% 同样零漂移),「生产漂移疑云」被数据否定,**不触发 BLOCKED,终账照落**。

判读(如实):①vLLM 三档对 M6 收官 58.12/207.49/355.91 = +0.1%/+0.0%/+0.1%,基线侧零漂移;②w8a16 达标率逐档递减(147.4→128.4→112.8)的拆读:decode-only TPOT 比 = 17.91/12.57 = ×1.43(C=8)恒定强劲,递减来自 out=32 短输出协议下 tok_s span 含 TTFT 全额——我方 C=8 TTFT p50 0.1903s vs vLLM 0.0804s(prefill flop-bound +5.8% 账 + 发射合批排队轮廓),TTFT 占比越高 tok_s 越被拖低,C=1 档 TTFT 占比小故 147.4% 最接近 decode 真值;③w8a16 TTFT 轮廓与 T2 prefill 账同向(C=1 0.0358 < none 0.0382 = 量化 prefill C=1 更快的兑现;C=8 0.1903 > none 0.1861 同向劣化);④HTTP none C=8 TTFT 0.1861 复刻 M6 0.1846 发射合批排队轮廓(同款如实,主验口径方法学代价)。

### 终账·锚判定行(如实,本里程碑最重要的一行)

**主锚「W8 C=8 TPOT ≤ 14.0ms」:过。** 两读法并给(实测口径):①**TPOT 直读 = 锚的原生口径**——HTTP w8a16 腿 C=8 tpot_mean **12.57ms ≤ 14.0 ✓**(余量 10.2%),in-process graph(m6_ab 终bench)**11.49–11.50ms ≤ 14.0 ✓**(余量 ~18%);②达标率换算口径:vLLM TPOT 17.91 ÷ 达标率 112.8% = 15.88ms,表面超锚——**该换算被 TTFT 差污染,如实注记不采信**(out=32 短输出下 tok_s span 含 TTFT 全额:我方 0.1903 vs vLLM 0.0804;decode-only TPOT 比 ×1.43,若 TTFT 等额则达标率应 ~142%、换算 ~12.5ms 与直读同带)。段 0 推算 12.3ms 与仪式直读 12.57 同带,推算链闭合。

**C=1 锚「≤ 10.5ms」:未过,如实裁。** 实测:in-process graph 10.72–10.75(仪式终bench 三样本)/ HTTP 直读 10.88ms;对 10.5 = **+2.1~+2.4%(in-process)/ +3.6%(HTTP)**;三窗口互证语义一致(段 0 推算 ~9.9 → T2 轨迹 10.74–11.02 → 仪式 10.72–10.75,稳定高于锚 ~3%)。归因(数据闭环,**这是 GEMM 面假设的边界,非量化 kernel 缺陷**):①引擎内隐含 GEMM 面 9.26–9.29ms(C=1)与段 0 微基准 9.391ms 同带(隐含 η 0.81 vs 0.800)——**GEMM 面全额兑现,量化 kernel 无缺陷**(ppl −0.088%、capture 20/20+25/25 逐位、TOL 紧比 ≤0.43);②缺口主体在非投影面:M6 腿 1 实测 C=1 非投影 GPU 面 1.46ms/步(attention+norm+rope+copy+other),而段 0 锚推算假设非 GEMM ~0.5ms(9.39+0.5≈9.9 ≤ 10.5),实测口径 9.4+1.46 ≈ 10.85 ≈ 仪式实测 10.72–10.75——**锚推算对 C=1 非 GEMM 面占比低估,是未过的算术主因**;③结构性加重项:C=1 步时 10.7ms 下 kv 投影 M=1 launch/欠填充-bound(段 0 实证 η 0.28;字节占比 4% 但 72 发/步)与 lm_head M=1 大 GEMV(段 0 η 0.28–0.31 口径下界)的固定发射成本、host 面占比相对升高。10.5 锚的重开条件 = 非 GEMM 面收账(M8 台账新立项),不在 W8 kernel 域内。

**精度锚「W8 ppl ≤ +1%」:过(负增量)。** ppl none **10.0709** / w8a16 **10.0620** = **−0.088%**(NLL 总额口径 −0.038% 同向;对称噪声不解读为量化更优,Task 3 账原样)。

**段 B(W4A16):NO-GO(裁决在档)。** kernel η 资格门 FAIL(段 0 探索级加权 0.472–0.482 vs 门 0.80,缺口 ~67%;主类最高 gate_up 0.62;ALU 税 ×0.50–0.71 of W8 为结构性),W4 挪 M8;仪式 W4 腿按裁决撤销,一次锚修正权未行使、保留闭合(W8 条款未触发;W4 无需修正即 NO-GO,条款自然失效)。

### 终账·η 表(段 0 微基准 vs Task 2 引擎内账)

| 口径 | C=1(M=1) | C=8(M=8) |
|---|---|---|
| 段 0 微基准 GEMM 面(ms/步;36 套冷口径轮转) | 9.391 | 9.311 |
| 段 0 加权 η(GB/s,roofline 1008 权重字节口径) | 0.800(806.2) | 0.807(813.2) |
| Task 2 装载期 autotune 隐含 GEMM 面(五档 µs × 调用数 72/72/72/36/1) | —(装载期单 M 口径) | 10.35(η 0.726) |
| Task 2 引擎内 m6_ab graph 步时(M7 终bench 带) | 10.72–10.75 | 11.49–11.50 |
| 引擎内隐含 GEMM 面(步时 − 非投影面 1.46/2.03ms,M6 腿 1 bf16 实测口径) | 9.26–9.29 | 9.46–9.47 |
| 引擎内隐含 η(7.57GB 权重流/步) | 0.81(815–818) | 0.79(800) |
| 隐含 vs 段 0 微基准偏差 | −1.1~−1.4% | +1.6~+1.7% |

判读:引擎内隐含 GEMM 面与孤立微基准互证同带(±2% 内)——**W8 kernel 在真实引擎环境(投影间 L2 干扰/相邻发射)零增益损失,微基准 η 直接兑现为生产 TPOT**;Task 2 装载期 autotune 口径(10.35ms / 0.726)为独立 harness 的低读(轮转/占用与段 0 不同),如实并列不作锚判据。前提注记:非投影面按 bf16 口径不变假设(W8 只换投影 kernel;attention/norm/rope/copy 不动,dispatch host 开销 30–50µs/步 T2 实证不可见);kv/lm_head 引擎内大 M 不出现(prefill_batch 只取行末位,M≤8 → decode 桶)。

### 终账·ppl 三行与精度判定(引 Task 3 账)

```
PPL quant=none tokens=299078 nll_sum=690601.61 ppl=10.0709 windows=1167
PPL quant=w8a16 tokens=299078 nll_sum=690337.84 ppl=10.0620 windows=1167
PPLDELTA (10.0620/10.0709 − 1)×100% = −0.088%  → 精度锚 ≤ +1% ✓(负增量 = 对称噪声,不解读为量化更优)
```

(数据源 wikitext-2-raw wiki.test.raw 299,078 token / 滑窗 window=512 stride=256 counted=299,007;parity 账 8B 单请求 16/16 token 逐位一致、top-1 |Δ| ≤ 0.75 无 argmax 翻转,见段 A·接线账。)

### 终账·prefill/TTFT 账(引用)

T2 ABPREFILL 入账(原样):C=1 32.49→**31.56(−2.9%)✓** / C=8 142.50→**150.82(+5.8%)如实入账**(M=1024 flop-bound,dequant ALU 纯增开销;chunked prefill 预算切块下 M 更小、向 C=1 侧移动)。本终账 in-process 复核同向:C=1 31.65–32.11 vs none 32.62/32.63(量化更快兑现);C=8 153.79–154.40 vs none 141.35/144.30(罚金 +8.1%,略宽于 T2 的 +5.8%,单步墙钟口径噪声级,如实注记)。TTFT(仪式 HTTP):w8a16 C=1 0.0358s < none 0.0382s(单请求 128-token 场景净改善兑现);C=8 0.1903s > none 0.1861s(8 并发到齐 +4.2ms,同向劣化)。

### 终账·装载耗时账(引 T2/Task 3 账 + 本仪式活证)

none 30.8–34.3s(纯 bf16 读盘;本终bench 两样本 30.9/31.9);w8a16 load 30.7–31.2 + apply 0.5–0.6(253 模块 RTN)+ configure 5.0(triton 缓存热)/ 6.97(冷,含 JIT)≈ **36.2–38.7s 全链,预算 120s 内 ✓**(本终bench w8a16 s1:31.3 + 0.5 + 5.0 = 36.8)。本仪式活证:none 腿 server 就绪 ~30s vs w8a16 腿就绪 ~36s,Δ≈+6s ≈ apply+configure,与账同。磁盘量化缓存 = non-goal 不变。

### 终账·段 B NO-GO(引用,裁决在档)

段 B 门裁决(2026-11-02,Task 4 = NO-GO close-out,控制器裁)原样引用:三条件读数——①主锚 C=8:T2 A/B 轨迹 11.53–11.70 ≤ 14.0 ✓;②精度:W8 ppl 增量 −0.088% ✓;③**kernel η 资格门:FAIL——段 0 探索级加权 0.472–0.482 vs 门 0.80,缺口 ~67%;主类最高 gate_up 0.62;ALU 税为结构性 → 判定:段 B NO-GO,W4 挪 M8 台账**。附带否决轴:W4 净收益 ~0.9ms/步对已落地 W8 微薄;RTN W4 g128 精度典型 +2~5% ppl 数量级劣化。W4 复活条件:kernel 形态突破(marlin 级重构)/ 精度算法升级(GPTQ)/ 显存受限场景价值重估。

### 终账·M8 台账(承 M6 终账 M7 候选台账 + 本里程碑移入/新立)

- **W4A16 复活**(段 B NO-GO 移入):复活条件三条(形态突破/GPTQ/显存受限重估);若复活,锚按届时生产级实测重估(原 10.5 锚从未被测试,修正权已随 NO-GO 自然失效);
- **GPTQ/AWQ**(算法域另立里程碑):W4 精度路径主候选,W8 RTN→GPTQ 亦受益;
- **C=1 非 GEMM 面收账**(本终账新立,10.5 锚重开的唯一路径):C=1 非投影 GPU 面 1.46ms/步(M6 腿 1 口径)= attention/小核 launch 面(段 0 kv η 0.28 实证)+ host 提交面;与 C=8 的 2.03ms 同域,收账同时利好 C=1/C=8 双锚余量;
- **KV 量化** / **TP×quant**(构造期互斥已落地,解禁需 quant_gemm TP 语义 + 通信面评估)/ **磁盘量化缓存**(R5 账 120s 预算内,non-goal 不变);
- 承 M6 终账原样:**双存储 down_t**(+3.6GB 换 ~0.3ms,违零双存储红线)、**连续到达 bench 协议**(chunked 价值裁定的前置)、**半宽 GEMM**(TP=2 kernel 面第一候选,段 0 实测 37% 效率塌方)、**custom allreduce**(TP=2 通信栈,17.84µs 微基准在档)、**rank0 快失败**(工程健壮性)、triton decode_gemm 模块(已入库未接线,重开前提 = 融合存储或访问模式变更)。

### 终账·口径注记(照前节同款)

同日同机(RTX 4090 ×2;in-process 终bench 物理 GPU 1、仪式三腿 + gate-basis 腿物理 GPU 0、压测客户端照配方 `CUDA_VISIBLE_DEVICES=1` 零 GPU 占用;每腿跑前/收服后 nvidia-smi 双卡 4 MiB 验净 + 零计算进程双查,前台 timeout ≥1800s);全部腿 seed=20261027、in=128/out=32、temperature=0、max_tokens=32、计时前 1 条 warmup 不计入;HTTP TTFT 含客户端→服务端上行与 SSE 流式首包回传全程(`bench_remote._run_one` 口径);三腿 + gate-basis 腿 + 终bench 五样本同窗口串行,腿间以双卡 4 MiB 验净为界;关服纪律:`ss -tlnp` 取真身 PID(vLLM 55424 / none 70758 / w8a16 102713)定向 kill,无包装壳误伤、无 pkill;vLLM 0.32.0,基线环境 `/home/ernest/vllm-baseline`;7.57GB/步 = W8 权重流口径(bf16 15.13GB ÷ 2,scale 可忽略);证据归档 `/tmp/opencode/m7t5/`(三腿 server+bench stdout、gate-basis 腿、终bench 五样本、slow 尾行);本节树 = 37b528d + 本 commit(docs)。M7 不打 tag、不 push(留用户)。

## M8 段 0:n-gram 草稿器 + 贪心接受(TDD,纯 CPU,零 GPU)(2026-11-02,`tests/test_spec_ngram.py`,CPU 输出)

M8 开篇。设计定案 = `docs/superpowers/specs/2026-11-02-m08-spec-decode-design.md`(锚 A1–A4;草稿器/接受逻辑 = §2.1,verify 前向 = §2.2,KV 游标复用 = §2.3);实施计划 = `docs/superpowers/plans/2026-11-02-m08-spec-decode.md`(5 任务)。本节 = Task 0:PLD 式 n-gram 草稿器 `acorninfer/spec/ngram.py` + 贪心接受 `greedy_accept`(TDD,纯 CPU python,零 torch——草稿器跑引擎热路径,纯列表/dict)。生产其余零触碰。

### 红绿记录(TDD,纯 CPU)

- 红:`.venv/bin/pytest tests/test_spec_ngram.py` → `ModuleNotFoundError: No module named 'acorninfer.spec'`(实现前)。
- 绿:实现后 5 passed(`5 passed in 0.10s`):
  - `test_index_build_and_lookup`(键 = N-gram tuple,值 = 后继列表按出现序,重复上下文先后都记录;命中/未命中)
  - `test_propose_deterministic_and_padded`(同上下文同输出;候选 <K 补 `FILLER=0`;>K 截断到 K)
  - `test_propose_uses_prompt_plus_generated`(`extend` 增量入索引后 propose 可查到生成 token 参与的新键)
  - `test_greedy_accept_longest_prefix`(最长前缀;全中 a=K bonus=链尾;零中 a=0 仍产 bonus=首位)
  - `test_accept_is_exactly_greedy_chain`(黄金性质钉:构造 logits→argmax 链 == drafts → (K, 链尾);断链后巧合相符不计——**前缀语义**,非集合匹配)
- 全套件:228 passed, 17 deselected(本机含 CUDA,6 个 gpu 标记用例实跑;CPU-only 口径即基线 217+6 skipped+5 新增 = 222 passed, 6 skipped, 17 deselected)——零换金,零既有用例漂移。

### 对 plan 草稿的偏离(逐条记录)

1. **`greedy_accept` 签名:吃 token ids,非 logits**(plan 草稿 `greedy_accept(next_token_logits, drafts)`;任务指令与测试契约定案 `greedy_accept(next_token_ids, drafts)`)——引擎侧先 `logits.argmax(dim=-1)` 再传入(Task 3 接线);纯 CPU 可测,GPU argmax 属 Task 3。
2. **确定性定案(spec §2.1 报告口径)**:后继列表 = 纯 list,**按出现序 append、不 dedup**——出现序即唯一顺序来源,无 set 迭代、无任何随机性;propose 输出 = 后继列表前 K 位 + `FILLER` 补齐,同上下文同输出可复现。(备选 stable dedup 弃:不 dedup 更简,且首现后继已含最高频信息,dedup 反而丢弃重复证据的次序权重。)
3. 微偏离:索引内部属性公开为 `NgramProposer.index`(dict[tuple[int,...], list[int]]),测试直读钉结构(plan 草稿未命名);`propose` 键 = 已提交上下文末 n 位(prompt+已生成,verify 步首尚无新接受 token,与草稿注一致)。

## M8 段 1:pager 多槽游标(唯一保护解禁,TDD 等价钉先行)(2026-11-02,`tests/test_spec_pager.py`,CPU 输出)

Task 1(spec §2.3,D3 KV 游标复用):`cache/pager.py` 只加三个增量方法——`PagedCacheView.ensure_append_capacity_n(layer, n)`(步首容量确保,needed 公式 = write() 的 `(base+n+page-1)//page`,返回 n 个 `(page,slot)` 游标)、`PagedKVCache.scatter_verify(views, layer, k/v[B,kv,m,D], pages/slots[B,m])`(展平 advanced indexing 照 scatter_decode 的 [B]→[B*m] 法,`_lens` 提交 +m)、`PagedCacheView.set_lens(layer, new_len)`(只回退不前进,越界 ValueError)。死槽游标复用闭环:verify 写 m 位 → 接受 a → `set_lens(len+a+1)` → 下步从 len' 起复用死槽(页表项已在,零分配器回滚、free-list 语义不变),读侧按 lens 读页天然不可见死槽。

### 三方法契约(测试钉死)

- **`ensure_append_capacity_n`**:n=1 与旧 `ensure_append_capacity` 逐字节同 `(page,slot)`(空表/页内中段/页尾/恰跨页五形态 × page∈{4,16} 双页径);分配数 = write() 的 needed 公式;BlockExhausted 语义同 write(中途抛时已拿页留 table、`_lens` 未提交,释放后重试连贯);ensure 家族不提交长度。
- **`scatter_verify`**:m=1 与 `scatter_decode` 对同一输入写页逐位一致 + `_lens` 提交一致(页内尾槽/恰跨页 slot 0/页中槽三形态);m=5、B=3 时写 `[B,kv,5,D]` 于 `[B,5]` 目标,页值逐位核对 `(pages[b,j],slots[b,j]) <- k[b,:,j,:]`,`_lens` 只提交被写层 +m;容量分配不在此发生(不可能抛 BlockExhausted);层间锁步由调用方保证(注释口径同 scatter_decode)。
- **`set_lens`**:只回退不前进(`new_len ≤` 已写 len,含相等 no-op;前进越界 ValueError,空层同理);回退后 gather 读侧截到新长——死槽位不出现于任何读出张量。

### 红绿记录(TDD,纯 CPU)

- 红:`.venv/bin/pytest tests/test_spec_pager.py` → 7 failed(6 测试函数,`ensure_n_equals_old` × page∈{4,16} 参数化;实现前 AttributeError)。
- 绿:实现后 7 passed(`7 passed in 0.11s`);6 等价钉逐条落:
  - `test_ensure_n_equals_old_for_n1`(n=1 ≡ 旧 ensure 逐字节,page∈{4,16})
  - `test_ensure_n_allocates_across_pages`(n=5、page=4 base=3 起跨 2 页;分配数 = needed 公式;分配行为 ≡ write;BlockExhausted 中途抛:页留 table、lens 未提交、`assert_balanced` 过)
  - `test_scatter_verify_m1_equals_scatter_decode`(m=1 两方法写页 + `_lens` 逐位一致)
  - `test_scatter_verify_multislot_and_lens`(m=5 B=3:lens +5、页值逐位、只提交被写层)
  - `test_set_lens_rollback_guarded`(相等合法/回退合法/前进 ValueError/读侧截断)
  - `test_dead_slot_reuse_roundtrip`(写 m=5 → set_lens 回 a=2 → ensure_n(5) 从 len'=base+3 起:死槽 (p1,s1)(p1,s2) 直接复用、已分配页未用槽 (p1,s3) 零分配、跨页续接恰新拿 1 页 → 复写 → gather 只到 lens,读出 = pre ∪ k1 存活前缀 ∪ k2 复写,死槽旧值零残留)
- 既有 KV 对拍零换金:`tests/test_pager.py + tests/test_cache.py` 19 passed;fast 全套件 Task 0 复核口径(本机双卡可见,6 个 gpu 标记 TP=2 用例实跑)**235 passed, 17 deselected** = 基线 228 + 新增 7,零漂移;单卡口径(`CUDA_VISIBLE_DEVICES=1`)229 passed, 6 skipped(TP=2 需双卡的既有 skip,非本次改动)。golden/slow 零触碰。

### 草稿注①裁决:零动(非薄壳化)

旧 `ensure_append_capacity` **保持原体零动**,n=1 等价由 `test_ensure_n_equals_old_for_n1` 钉住,不以 `return self.ensure_append_capacity_n(layer, 1)[0]` 薄壳化。理由:"旧路径逐字节"硬门的最强形态是旧方法体逐字符不变(薄壳化行为等价但引入间接层,需依赖等价推辩;零动则旧 decode 路径与 `6eb1d6e` 逐字节同源,无需论证)。代价:needed 公式在 `write`/`ensure_append_capacity_n` 两处同式并存(各自带注释互引),可接受的重复。pager.py 净增 70 行(三方法 + 契约 docstring),既有方法零改动(diff 纯 +70)。

### 对 plan 草稿的偏离(逐条记录)

1. 微偏离:测试侧 `scatter_verify` 的 `[B,m]` 目标张量从 `ensure_append_capacity_n` 返回值以双重推导式构造(`[[p for p,_ in t] for t in targets]`),plan 草稿未给构造式;TDD 过程中修了三处测试自身笔误(目标张量解包、roundtrip 槽位算术漏计"已分配页未用槽"形态、gather 读侧 `[kv,len,D]` 布局误 permute)——**实现一次成型零返工**,等价钉逐位通过即契约成立的直接证据。
2. 微偏离:`scatter_verify` 的 assert 面比 plan 草稿多一条 `pages.shape[1] == slots.shape[1] == m`(防 [B,1] 与 [B,m] 混用的静默错位);k/v m 轴一致性 assert 同理。

## M8 段 2:verify attention kernel(只读派生,TOL 新标定)(2026-11-02,`tests/test_paged_verify.py`,GPU 1 输出)

Task 2(spec §2.2):`acorninfer/ops/paged_verify.py` 新文件,`paged_verify_attention(q[B,m,H,D], k/v_pages, tables, seq_lens=len+m, num_splits) -> [B,m,H,D]` —— `paged_decode_attention` 的 m 位窗口推广 + 窗口内因果。**保护原件 `ops/paged_attention.py` 零改动**(只读派生:读源码后新文件实现;combine kernel `_paged_attn_combine_kernel` 直接 import 复用,split kernel `_paged_verify_split_kernel` 新写)。

### 语义差异清单(vs 原 kernel,逐条)

1. **q 形状 / grid**:`[B,H,1,D]` → `[B,m,H,D]`(m = spec_num+1,全批统一),展平 query 行 (b, j),grid 第 0 维 `B → B*m`(row = b*M + j);q 展平 `[B*M*H, D]`(行序 (b, j, h))。
2. **每行读上限(唯一本质改动)**:decode 的标量 `L = seq_lens[b]` → 行依赖 `L_row = seq_lens[b] − (m−1−j) = len + j + 1`(j 位可见历史 len + 窗口前 j+1 位;窗口内因果;`seq_lens` 契约 = **len+m**)。decode = verify 的 j=m−1 行退化情形。
3. **不变式同源**:L_row = len+j+1 ≥ 1 → split 0 非空 → combine 首迭代 m_new 有限(decode 的 L ≥ 1 同款);空 split 存 (−inf, 0, 0),combine β=exp(−inf)=0 零贡献 —— 无分支合并照原。
4. **零改动照抄面**:页对齐分片(start/end 整页、pps 公式)、尾块 slot mask(`slot < end−t`)、k/v 偏移公式 `(((page*P+slot)*KV+kv)*D+d)`、online-softmax 两常数更新、combine kernel 整体、fp32 全程 + 末端单次 cast 数值策略、GQA 映射 kv = hq // REP。
5. **写读分离**:k/v 写入不在本 kernel(调用方 `scatter_verify` 已写页,本 kernel 只读);死槽(接受后 `set_lens` 回退段)由下步复写,读侧按 lens 截断天然不可见。
6. eager-only(v1 spec 模式不走 graph,无 capture 要求);split 启发式复用 `paged_attention.num_splits_for`(按 B*m 个 query 行估)。

### 红绿记录(TDD)

- 红:`CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest tests/test_paged_verify.py` → `ModuleNotFoundError: No module named 'acorninfer.ops.paged_verify'`(实现前)。
- 绿:实现后 **3 passed**(`3 passed in 1.35s`):
  - `test_verify_matches_dense_reference`:B∈{1,4,7} × m∈{2,5} × len∈{1,15,16,17,48} 全 30 组合(每序列 hist_b = len+b 变长批;S 按 len 分派 {1:4, 15:2, 16:1, 17:4, 48:8} 覆盖 S=1 / S>页数空 split / 多 split),fp32 紧容差(TOL_F32)+ bf16 标定容差双跑;参考 = 本文件 `paged_verify_attention_ref`(独立稠密参考:构造思路复用 `paged_decode_attention_ref` 的 gather-to-dense + fp32 softmax 展开,逐 (b,j) 行按 L_j 截断)。
  - `test_window_causal_exact`:k ≡ 0 → 可见分数恒 0 → softmax 严格均匀 → 每行输出 = 可见集 v 的精确均值(与 q 无关,独立于 gather 式参考)。三重 tripwire:①窗口后位(hist+j+1..hist+m−1 的 v ×100,行 j 泄漏任一后位 → 均值量级爆炸 ≥15 绝对 vs 容差 1e-2);②lens 外槽位(≥ hist_b+m 页内槽 = NaN + 表填充项指向全 NaN 陷阱页,越 lens 读 → NaN 传播,`isfinite` 断言);③逐行精确均匀均值(fp32,rtol 1e-4/atol 1e-2,可见集组成错即炸)。
  - `test_rollback_then_reverify_numerics`:真实 pager 路径死槽复用数值闭环(base=13 跨页界,a=2,m=5):verify 写 m=5 → `set_lens` 回 16 → `ensure_n(5)` 复用死槽(p1 槽 0,1,零分配)+ 同页续接 → `scatter_verify` 复写 → verify 读侧 ≡ 从未写过死槽的孪生 cache(**页值 + 输出逐位 `torch.equal`**)≡ 独立稠密参考(TOL_F32,不经页机制的 cat 构造)。

### TOL 标定(新标定,协议:命令/日期写死)

- 标定:2026-11-02,GPU 1,`CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest tests/test_paged_verify.py -v -s`。bf16(q/k/v bf16 输入、fp32 score/累加、输出 bf16,照原 kernel 数值口径):**30 组合中 27 个 max|diff| = 0.000000**(kernel 与参考同 fp32 全程 + 末端单次 cast → 取整后逐位同),worst = **0.001953**(B=7, m=5, len=15;次高 0.000244,B=7, m=5, len=17)= 单 bf16 ulp(@0.5–1.0 值域)。**写死 TOL_VERIFY = (rtol=4e-3, atol=4e-3)**(worst 的 ~2 ulp 余量;attention 是 L 长求和,L 增大漂移上升,扩配置按协议重标定)。逐组合表(maxrel 分母 = 该组合 |ref| 最大值):

| B | m | len | S | maxabs | maxrel |
|---|---|-----|---|--------|--------|
| 1/4/7 × m∈{2,5} × len∈{1,15,16,17,48} 的 27 个组合 | | | | 0.000000 | 0(B=4,m=5,len=1 maxrel 5.6e-8) |
| 7 | 5 | 17 | 4 | 0.000244 | 1.54e-04 |
| 7 | 5 | 15 | 2 | **0.001953** | 1.32e-03 |

### Sanity 计时(非门,记录;W8 场景 m=5 ≤ 16 与单 decode 同桶)

Qwen3-8B 口径(H=32, KV=8, D=128, P=16, bf16, hist=256, S 按启发式;warm 10 + 30 iter 均值,GPU 1,2026-11-02):

| B | verify(m=5) | 5×单位 decode | 比 | verify vs 5×decode 输出差 |
|---|---|---|---|---|
| 1 | 0.037ms(S=1) | 0.352ms(S=4) | **0.10×** | 0.000244(bf16 ulp 级) |
| 7 | 0.141ms(S=1) | 0.251ms(S=1) | **0.56×** | 0.000000(逐位) |

读法:verify 单调用产出 m=5 个位置的输出,开销远低于 5 次独立 decode(每decode 各自 split+combine 两次 launch ×5);B=7 时 decode 侧 S=1 已单桶、verify 收益比收窄但仍 1.8×。kernel 数值与 5 次单位 decode 交叉一致(ulp 级),与稠密参考对拍独立成立。

### 回归三尾行(零换金)

- fast 全量(本机双卡,6 gpu 用例实跑):**238 passed, 17 deselected** = 基线 235 + 新增 3,零漂移。
- fast 单卡(`CUDA_VISIBLE_DEVICES=1`):**232 passed, 6 skipped** = 229 + 3(既有 TP=2 skip,非本次改动)。
- slow **17 passed** / 双卡 tp 全套(`test_tp_eager+engine+server`)**13 passed in 19.57s**(跑后双卡回落 4 MiB)。
- `git diff c3261b0 -- acorninfer/ops/paged_attention.py` = 空输出(保护原件零变化证据);golden m0–m7 零触碰。

### 对 plan 草稿的偏离(逐条记录)

1. 微偏离:`test_verify_matches_dense_reference` 在指令矩阵(B×m×len 30 组合)上每序列加变长(hist_b = len + b),比草稿的定长批多覆盖变长表行宽;S 分派表 {1:4, 15:2, 16:1, 17:4, 48:8} 为草稿未定的测试自由度(覆盖 S=1/空 split/多 split 三形态)。
2. 微偏离:`test_window_causal_exact` 定案为"k≡0 → 严格均匀 → 可见集均值"构造(草稿的"位置编码式唯一值"不可零容差对拍——softmax 非退化时无法解析闭式);tripwire 的 poison×100 + NaN 陷阱页 + 逐行精确均值三重钉满足草稿意图(零泄漏窗口后位/历史外位)。
3. 微偏离:TDD 过程修三处测试自身笔误(page 张量 token-major 布局赋值、pager 写入需同设备张量、poison 索引误打历史位 `t−hb` → 改位置 `t`);kernel 实现一次成型零返工,两处 kernel 侧测试(device 断言、S= kwarg)按 decode API 同款位置参数修正。

## M8 段 3:引擎 spec 接线 + A1 无损性硬门(2026-11-02,`tests/test_spec_engine.py`,GPU 1 输出)

Task 3(spec §2.2/§2.4):`--spec-ngram/--spec-num` → `EngineConfig`/`Engine` 全链接线:`Qwen3ForCausalLM.forward_verify`(步首 prepare:`ensure_append_capacity_n` → `build_tables(lens_override=[len+m])` → 位置 `len+arange(m)` 展平 rope;层循环 `scatter_verify` + `paged_verify_attention`;全窗口 lm_head `[B,m,vocab]` fp32)→ `executor.verify_batch`(零采样,ids = [已提交末位]+drafts)→ 引擎 `_spec_step`(argmax → `greedy_accept` → 逐 token `_record` → `proposer.extend` → `set_lens` 逐层回退)。

### 接线形态(逐文件)

- `config.py`:`spec_ngram=0` / `spec_num=4` + `__post_init__` 校验(ngram 0..8;num 1..8;spec>0 须 triton;spec × decode_graph ValueError)。
- `server.py`:`--spec-ngram/--spec-num` CLI;getattr 缺省(旧调用方 args 无字段 = 0/4 逐字节旧行为);spec × tp 构造期 interdict;tp==1 路径透传。
- `engine.py`:`__init__` 同款守卫(直构面)+ `_proposers` dict(step 期惰性建 `NgramProposer(context=all_ids)`;context 失配自动重建 = 抢占重算后正确性;退休/drop 弃);`step()` spec 分支 → `_spec_step`;retire finally 同步 pop proposer。
- `executor.py::verify_batch`:ids `[B,1+K]`,K 全批统一(propose 补齐 FILLER)→ m 恒定零 padding;TP 不设分支(构造期互斥保证单卡)。
- `qwen3.py`:`forward_verify` 三级(Attention/Block/Model);engine 路径恒传 tables/lens/scatter(无自 prepare 回退,显式 ValueError);v1 verify 恒 torch rope(不接 fused_ops)。

### 红绿记录(TDD + A1 硬门抓出真缺陷)

- 红(实现前):`CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest tests/test_spec_engine.py` → 2 passed(config/server CPU 面)/ 4 failed(gpu 腿:接线未实现)。
- 接线完成后 A1 tiny **失败**:多 seq 批每批恰一 seq 早分歧(哪个 seq 随批成分漂移);spec row0 logits vs decode `max|d|=0.416314`(top-5 集合近似、量值整体压缩 ~0.94——错序特征的典型信号)。
- 定位(side-by-side 双引擎共享无状态模型,逐层捕获):layer0 全零(block in / post-LN / kernel q row0 / **kernel out row0 = 0.000000**)→ layer1 kernel q row0 `max|d|=0.878`、sa_out(post-o_proj)`0.42` → **kernel 输出与 o_proj 之间**。
- 根因:`forward_verify` 对 kernel 返回值抄了 decode 的 `out.transpose(1,2).reshape(B,m,-1)`。verify kernel 返回 **(b,j,h,d) 连续**布局(≠ decode 的 (b,h,1,d)),transpose 把 h 变逻辑外层轴,reshape 按逻辑序 h-major 复制 → `m>1 且 H>1` 时 o_proj 输入行 = 多窗口位×单头的错序混合(如 row0 = h0·j0..3)。**T2 kernel 与本缺陷无关**;T2 对拍(逐行 vs 5×decode)不含此步后处理。
- 修复:`out.reshape(B, m, -1)`(kernel 输出已 (b,j,h,d) 连续,直接合头,行内 = 头拼接与 decode 同序;代码内留"勿改回"注记)。
- 修后:verify row0 logits vs decode `max|d|=0.000000`(fp32 逐位);测试夹具修 4 处(见偏离清单)后 **6 passed in 1.28s**(`tests/test_spec_engine.py` + `tests/test_spec_ngram.py` 合跑 `11 passed in 1.30s`)。

### A1 无损性证据(逐字)

tiny(硬门,双场景:随机模型三类 prompt + copier rig 真实接受 K+bonus;verify spy 钉防全降级假绿):

```
tests/test_spec_engine.py::test_spec_greedy_bitwise_tiny PASSED
tests/test_spec_engine.py::test_spec_multitoken_outputs_order PASSED
tests/test_spec_engine.py::test_spec_dead_slot_reuse_engine PASSED
tests/test_spec_engine.py::test_spec_remaining_budget_degrades_to_decode PASSED
============================== 6 passed in 1.28s ==============================
```

8B(`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m8_spec_8b.py`,2026-11-02,GPU 1,W8 双腿 greedy,2 wiki 型 + 2 短 chat 型,stdout 逐字):

```
PROMPT wiki1: 57 tokens
PROMPT wiki2: 44 tokens
PROMPT chat1: 7 tokens
PROMPT chat2: 8 tokens
A1 wiki1: IDENTICAL (64 tokens, greedy, w8a16 both legs)
A1 wiki2: IDENTICAL (64 tokens, greedy, w8a16 both legs)
A1 chat1: IDENTICAL (64 tokens, greedy, w8a16 both legs)
A1 chat2: IDENTICAL (64 tokens, greedy, w8a16 both legs)
A1_RESULT PASS (bitwise spec on/off, w8a16, 4 prompts x 64 tokens)
```

### 回归尾行(零换金)

- fast 单卡(`CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest -q`):**238 passed, 6 skipped, 17 deselected in 15.52s** = 基线 232 + 新增 6,零漂移。
- slow:`17 passed in 444.28s`。tp 全套(双卡):**13 passed in 19.69s**。
- `m6_ab --quant w8a16` 带复核:`AB baseline C=1 mode=graph step_ms=10.94` / `C=8 step_ms=11.69`(带 10.7–11.7 ✓,quant 路径零扰动)。
- 保护面:`git diff HEAD --stat -- ops/paged_attention.py engine/scheduler.py engine/sequence.py engine/graph_runner.py tests/triton_utils.py` = 空输出;golden m0–m7 零触碰。

### 计测表(8B W8,GPU 1,2026-11-02)

| 账 | 值 | 读法 |
|---|---|---|
| decode eager 稳态步(C=4,4×128 随机 prompt,24 步均值) | 18.28 ms(p50 18.13) | 与 m6_ab eager C=8 18.13 ms 同量级 |
| spec verify 稳态步(C=4,m=5,接受 ~0) | 22.34 ms(p50 22.29) | 零接受最坏情形 +22%/步 |
| tok_s(C=4 同协议) | decode 218.8 / spec 179.0 | 零接受下回本需 mean_emitted ≥ 1.22 |
| eager W8 tok_s 基线(A2 分母,m6_ab C=8) | 8/18.20 ms ≈ **440 tok/s** | `AB baseline C=8 mode=eager step_ms=18.13 tpot_ms=18.20` |
| 草稿延迟(NgramProposer,512 位上下文,×2000) | mean **0.3 µs** / p95 0.4 / max 13.3 | 目标 <100 µs,3 个量级余量 |
| 接受账(随机 prompt 腿,26 次 verify) | mean_a=0.000 / mean_emitted=1.000 | 随机 id 无重复 → FILLER 草稿 → a=0,引擎正确每步 1 发射 |

零接受最坏情形下 spec 每步 +22%:A2(真文本接受率)是收益裁决面;随机内容账仅为下界锚(机制成本如实入账,零粉饰)。

### 边界处理(逐条)

1. 剩余生成预算 `max_tokens − len(generated) < m` → 该 seq 降级 decode 腿(末段 eos/length 截断交久经考验单 token 路径,同步与 verify 腿并存)。
2. 非贪心(temperature ≠ 0)→ 该 seq 降级 decode 腿(`_sample` 既有逐行采样;零静默 greedy)。
3. 死槽 `[len+a+1, len+m)`:`set_lens` 只回退不前进(> 现长 ValueError),下步 `ensure_append_capacity_n/scatter_verify` 复写,读侧按 lens 截断天然不可见(死槽复用引擎测试:allocate 次数钉死 + 孪生 cache 逐位)。
4. 抢占重算:proposer context == all_ids 失配 → 自动重建(索引重扫 O(len) 只发生在 admit/抢占后)。
5. eos/length 终止:逐 token `_record`,`is_finished` 即断,其后窗口位作废(不外泄)。
6. 退休/drop:proposer 随视图同步弃(step finally + drop 兜底双面)。
7. fillers:`FILLER=0` 补齐 K → m 全批统一;filler KV 照写,读侧行依赖 `end_j = len+j+1` 窗口内因果天然遮蔽。

### 对 plan 草稿的偏离(逐条记录)

1. 有意偏离(记录在案):非贪心 seq 的处理取 **step 期按 seq 降级 decode 腿**,而非 spec §2.4 的构造期禁入——采样参数是请求级,构造期不可判;step 分流零静默 greedy,正确性语义与 §2.4 意图一致。
2. 根因修复(见红绿记录):`forward_verify` 输出 reshape 错序(A1 硬门抓出),`transpose(1,2)` 移除 + 代码注记。草稿未预见此处布局差异(照抄 decode 幂等式在 m=1 时恰不触发)。
3. TDD 夹具修正 4 处:①budget 降级测试 `max_tokens=3` 恒 < m → verify 永不触发,改 7(两腿都被钉到);②copier 尾重复 4→6 位(n-gram 后继列表须 ≥ K=4 条,drafts 才全真无 FILLER);③eos-spy 改自标定(prefill 生成位会移位 n-gram 键,drafts[0] 不可静态预知 → spy 读真 drafts 强制 a=1);④多发射断言从"末 2 步"改为"任意步"(预算截断后末段必为单发)。
4. 微偏离:verify 不接 fused_ops(v1 eager-only,torch rope/残差);m=1 剩余预算不进 verify 腿(与 decode 逐字节同路径);tp==1 与单卡路径同码(server._build_engine 一处分支)。

## M8 Task 4 终账(2026-11-02,收官:语料 + 参数扫 + 仪式 + 判定行落笔,真机输出)

Task 4 = 语料切分(`make_spec_prompts.py`)+ bench 扩展(`--prompts-file` + 接受率计量)+ 施工期参数扫(N×K × 双 workload,GPU 1)+ 收官仪式(GPU 0,vLLM bf16 / W8 graph 生产参照 / W8 eager 同模式基线 / W8 spec eager 双语料;可选 vLLM ngram 腿**在场**)+ 判定行。**锚在选定配置上判:参数扫先行(两 pass),选定 N=1,K=4,仪式全程未换锚。** 本任务生产代码改动三面:`acorninfer/engine/engine.py`(spec_stats 步级计数,spec 关闭恒零值零写入)、`acorninfer/server.py`(GET /spec_stats)、`scripts/bench_remote.py`(--prompts-file/--spec-stats,缺省行为不变);新建 `scripts/make_spec_prompts.py`、`scripts/m8_param_sweep.py`、`data/spec_prompts_wiki.json`。`engine/scheduler.py`、`engine/sequence.py`、`engine/graph_runner.py`、`ops/paged_attention.py`、golden m0–m7、`tests/triton_utils.py` 零触碰(`git diff HEAD --stat` 该清单 = 空输出)。

### 终账·语料(Step 1,纯 CPU)

命令(原样):`.venv/bin/python scripts/make_spec_prompts.py`(缺省 = source `data/wikitext-2-raw/wiki.test.raw`、block 128、count 64、seed 20261102)。stdout 逐字:

```
CORPUS source=data/wikitext-2-raw/wiki.test.raw total_tokens=299078 blocks=2336 block=128 count=64 seed=20261102 eos_replaced=0 out=data/spec_prompts_wiki.json
```

口径(spec §6):全文按行保序拼接(perplexity.py 同款)→ 自家 Tokenizer 全文 encode 一次性(299,078 token,**与 M7 ppl 账同数** = 同源互证)→ 连续 128-token 块 2336 块 → `random.Random(20261102).sample` 无放回抽 64 块、索引升序(确定性,可复现)→ prompt 内 EOS 置换照 `bench._prompts` 同款(eos→(eos+1)%vocab;实测 eos_replaced=0,样本块零命中)→ JSON token-id 数组(`bench_remote --prompts-file` 直投 `/v1/completions`,不做重分词)。数量 ≥32 ✓(64)。

### 终账·bench 扩展与接受率计量口径(Step 1)

- `bench_remote.py --prompts-file data/spec_prompts_wiki.json`:各档用语料前 C 条 prompt(1⊂4⊂8 跨档可比);缺省随机语料行为逐字节不变(A4)。wiki 腿 `--out-tokens 64`(口径:64 而非 spec §6 草拟的 32,为稳住接受率统计的每请求 verify 步数,派单建议口径,如实记录;spec/eager/graph/vLLM 同腿同 out 对齐)。
- **接受率计量 = 引擎侧步级计数**(现状最顺路径,一次计数面贯穿 in-process/HTTP 双口径):`Engine._spec_step` 逐 verify 步累加 `spec_stats = {verify_steps, accepted(Σa), committed(实际入流 token), hist(committed/步分布,键 ≤ m 有界)}`;server 暴露 `GET /spec_stats`(state 锁内快照);`bench_remote --spec-stats` 每档压测后取增量打 `SPEC-STATS` 行(E_acc1 = accepted/verify_steps + 1 = E[接受+1];mean_committed;hist);参数扫脚本直读引擎计数增量。warmup 后先取基线快照 → warmup 的 verify 步(如有)不落任何档位增量。spec 关闭 = 计数面恒零值零写入,缺省路径逐字节(fast 238 证据)。
- wiki spec 腿 warmup 改用语料首条 + out=8(≥ m=7):强制服务端进程内 ≥1 个 verify 步,triton verify kernel JIT 预热,不落档位计时。

### 终账·参数扫(Step 2,GPU 1,in-process W8 生产同构接线,同口径两 pass)

命令(原样):`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m8_param_sweep.py`(模型装载一次 W8 = apply_quant + quant_gemm.configure;engine 按行重建;每行先 warmup seq(8-token × out=8 ≥ m,强制 verify 腿 JIT 预热,不计入)→ 计测 seq C=1(tok_s = bench 同款引擎记账口径 gen/(last−created),非 HTTP);N=0 行 = eager W8 参照同口径)。两 pass 全表(tok_s 单位 tok/s;E_acc1 与 hist 两 pass 逐位相同 = 接受账确定性):

**wiki workload(C=1,in=128,out=64,prompt0):**

| N | K | pass1 tok_s | pass2 tok_s | 两 pass 均值 | E_acc1 | hist(committed/步) |
|---|---|---|---|---|---|---|
| 0(eager) | – | 55.66 | 52.89 | 54.28 | – | – |
| 1 | 2 | 44.99 | 47.28 | 46.14 | 1.070 | {1:53, 2:4} |
| **1** | **4** | **50.19** | **47.21** | **48.70** | **1.073** | {1:51, 2:4} |
| 1 | 6 | 45.24 | 47.27 | 46.26 | 1.075 | {1:49, 2:4} |
| 2 | 2 | 46.07 | 44.55 | 45.31 | 1.070 | {1:53, 2:4} |
| 2 | 4 | 49.12 | 47.19 | 48.16 | 1.073 | {1:51, 2:4} |
| 2 | 6 | 48.96 | 46.65 | 47.81 | 1.075 | {1:49, 2:4} |
| 3 | 2 | 47.77 | 46.56 | 47.17 | 1.052 | {1:55, 2:3} |
| 3 | 4 | 45.77 | 46.89 | 46.33 | 1.054 | {1:53, 2:3} |
| 3 | 6 | 47.02 | 46.78 | 46.90 | 1.056 | {1:51, 2:3} |

**chat workload(C=1,in=128 随机,out=32,prompt0):**

| N | K | pass1 tok_s | pass2 tok_s | 两 pass 均值 | E_acc1 | hist |
|---|---|---|---|---|---|---|
| 0(eager) | – | 54.42 | 52.28 | 53.35 | – | – |
| 1 | 2 | 46.87 | 44.02 | 45.45 | 1.000 | {1:29} |
| 1 | 4 | 42.22 | 40.33 | 41.28 | 1.000 | {1:27} |
| 1 | 6 | 47.10 | 45.07 | 46.09 | 1.000 | {1:25} |
| 2 | 2 | 46.60 | 44.26 | 45.43 | 1.000 | {1:29} |
| 2 | 4 | 46.85 | 44.65 | 45.75 | 1.000 | {1:27} |
| 2 | 6 | 47.39 | 45.95 | 46.67 | 1.000 | {1:25} |
| 3 | 2 | 46.39 | 45.05 | 45.72 | 1.000 | {1:29} |
| 3 | 4 | 46.67 | 45.27 | 45.97 | 1.000 | {1:27} |
| 3 | 6 | 47.33 | 45.83 | 46.58 | 1.000 | {1:25} |

判读(如实):①**E_acc1 全表 ∈ [1.052, 1.075](wiki)/ = 1.000(chat),两 pass 逐位相同**——接受账由草稿器查表与 prompt 内容决定,确定性复现;②所有 9 组合 tok_s 均低于同口径 eager 行(wiki 均值 45.3–48.7 vs 54.3;chat 同向),**扫表内无一组合兑现加速**;③tok_s pass 间噪声 ±5%(pass1 N1K4 50.19 为噪声级,均值守序稳定);④N=3 一致最差(键越长命中率越低);K 增大 E_acc1 微升但 m=K+1 verify 税同步升,净效果 K=4–6 打平。

### 终账·选定缺省配置(锚判配置,仪式不许换锚——已执行)

**N=1,K=4(m=5)。** 理由:准则是「tok_s 最高,并列取 K 小者」——两 pass wiki 均值 48.70 全表最高(次高 N2K4 48.16,差 1.1% 噪声级;pass1 原始行同为 N1K4 最高);E_acc1=1.073 处 N∈{1,2} 并列组;m=5 与 T3 计测参考口径(verify 22.34ms)同窗口;K=4 = 既有 `--spec-num` 缺省,CLI 零意外。 Ceremony 全程按此配置,未换锚。

### 终账·机制上限探针(归因证据,非门)

强复制型 prompt(同句 ×16,128 token,选定配置 N=1 K=4,GPU 1,`/tmp/opencode/m8t4/ceiling_probe.py`):

```
CEILING workload=copy N=1 K=4 gen=64 verify_steps=33 E_acc1=1.879 mean_committed=1.879 hist={1: 4, 2: 29}
```

读法:纯复制输入下 E_acc1 = 1.879 ≈ 设计上限 2.0——**草稿→verify→接受→死槽复用全链机制工作正常**。wiki 1.073 的缺口是结构性而非接线 bug:本设计 `propose` 返回**同一键的前 K 个后继(备选)**而非链式续查,前缀接受语义下实际可用只有位置 0(位置 ≥1 拿备选对链尾,恰中属巧合;T3 copier rig 的全中链是构造特例)→ 实用上限 a≤1;自然 wiki 续写的首后继命中率 ~7% → E_acc1 ≈ 1.07。

### 终账·全量回归三尾行(零换金)+ A4 复核带

| 套件 | 命令 | 尾行(逐字) | 判定 |
|---|---|---|---|
| fast 单卡 | `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` | `238 passed, 6 skipped, 17 deselected, 1 warning in 15.46s` | = T3 后基线,零漂移 ✓ |
| slow | `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q -m slow` | `17 passed, 244 deselected in 444.59s (0:07:24)` | golden 17 原样,零换金 ✓ |
| 双卡 tp 全套 | `CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest -q tests/test_tp_eager.py tests/test_tp_engine.py tests/test_tp_server.py` | `13 passed in 18.95s` | 全绿 ✓,跑后双卡回落 4 MiB |

A4 复核带(m6_ab,GPU 1,本任务改动后):none graph C=1 **17.93**(带 17.91–17.96 ✓)/ C=8 **19.90**(带 19.87–19.93 ✓);w8a16 graph C=1 **10.93** / C=8 **11.68**(带 10.7–11.7 ✓)。eager 腿如实:none eager C=1 19.45 略高于历史带顶 19.35(+0.10,该档带宽 + 单样本噪声,M7 终账同款先例;graph 主口径不受影响)/ C=8 21.21 带内;w8a16 eager C=1 18.27(T5 带 17.49–19.25 带内)/ C=8 18.04(T5 带底 18.14 下缘 −0.10,噪声级)。`ABPREFILL`:none 32.92/144.53,w8a16 32.14/152.73(M7 终账同带)。

### 终账·仪式(Step 3,GPU 0;必选四腿 + R7 可选腿在场 + 口径腿 A2)

配方注记(照 M5/M6/M7 配方原样):命令/端口/GPU 0/seed=20261027 协议照旧;压测客户端 `CUDA_VISIBLE_DEVICES=1`(零 GPU 占用);每腿起服前 nvidia-smi 双卡 4 MiB 实录;**关服纪律全程 `ss -tlnp` 取真身 PID 定向 kill(无 pkill)**:vLLM plain 2613472 / vLLM ngram 2655017 / vLLM plain(口径腿)2985366 / graph 3044461 / eager 3148484 / spec 3201100,均 SIGTERM 后端口闭、双卡 4 MiB 复验(终态双查 nvidia-smi 全表零计算进程);AcornInfer 各腿就绪 ~40s,vLLM ~40–75s。注记:`ss -tlnp | grep 8200` 曾误中 nginx `0.0.0.0:18200` 行(机器常驻服务,非本仪式进程),真身以 `127.0.0.1:8200` 行 python PID 为准,未误伤。

腿构成(命令原样):

```bash
# leg A:vLLM bf16 8100(chat 缺省语料)
CUDA_VISIBLE_DEVICES=0 /home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90 --dtype bfloat16
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --base http://127.0.0.1:8100 --concurrency 1 4 8

# leg E(R7 可选腿,在场):vLLM bf16 + ngram spec(K=4,lookup 1–4;0.32.0 SpeculativeConfig 支持性先验兑现)
CUDA_VISIBLE_DEVICES=0 /home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90 --dtype bfloat16 --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 1}'
# 压测:chat 缺省语料 + wiki 语料两条(命令同款,--prompts-file data/spec_prompts_wiki.json --out-tokens 64)

# leg A2(口径腿):vLLM plain wiki 语料(与 leg E wiki 读数同基线对照;非验收腿)
# server 命令同 leg A;压测:--prompts-file data/spec_prompts_wiki.json --out-tokens 64

# leg B:AcornInfer W8 graph 8200(生产参照)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.server --model-dir /home/ernest/model-weight/Qwen3-8B --port 8200 --attention-backend triton --fused-ops --cuda-graph --quant w8a16
# 压测:chat 缺省 + wiki(--prompts-file data/spec_prompts_wiki.json --out-tokens 64)各 C=1 4 8

# leg C:AcornInfer W8 eager 8200(同模式基线)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.server --model-dir /home/ernest/model-weight/Qwen3-8B --port 8200 --attention-backend triton --fused-ops --quant w8a16
# 压测:wiki(--prompts-file --out-tokens 64)+ chat 缺省,各 C=1 4 8,均带 --spec-stats

# leg D:AcornInfer W8 spec eager 8200(选定配置 N=1 K=4)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m acorninfer.server --model-dir /home/ernest/model-weight/Qwen3-8B --port 8200 --attention-backend triton --fused-ops --quant w8a16 --spec-ngram 1 --spec-num 4
# 压测:wiki(--prompts-file --out-tokens 64)+ chat 缺省,各 C=1 4 8,均带 --spec-stats
```

逐腿 stdout 逐字(server 全文与 bench stdout 归档 `/tmp/opencode/m8t4/`;`RESULT-VLLM` 前缀为 bench_remote 固定标签):

```
# leg A:vLLM bf16 8100(chat)
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0269s ttft_mean=0.0269s tpot_mean=16.89ms tok_s=58.12
RESULT-VLLM C=4 ttft_p50=0.0659s ttft_mean=0.0566s tpot_mean=17.80ms tok_s=208.57
RESULT-VLLM C=8 ttft_p50=0.0805s ttft_mean=0.0738s tpot_mean=17.85ms tok_s=356.17

# leg E:vLLM ngram spec(chat)
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0274s ttft_mean=0.0274s tpot_mean=18.41ms tok_s=53.49
RESULT-VLLM C=4 ttft_p50=0.0704s ttft_mean=0.0592s tpot_mean=17.73ms tok_s=191.93
RESULT-VLLM C=8 ttft_p50=0.0838s ttft_mean=0.0764s tpot_mean=17.62ms tok_s=315.55

# leg E:vLLM ngram spec(wiki,out=64)
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0232s ttft_mean=0.0232s tpot_mean=15.85ms tok_s=62.64
RESULT-VLLM C=4 ttft_p50=0.0723s ttft_mean=0.0611s tpot_mean=14.50ms tok_s=225.18
RESULT-VLLM C=8 ttft_p50=0.0867s ttft_mean=0.0790s tpot_mean=14.60ms tok_s=423.23

# leg A2:vLLM plain(wiki,out=64)
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0230s ttft_mean=0.0230s tpot_mean=16.88ms tok_s=58.89
RESULT-VLLM C=4 ttft_p50=0.0681s ttft_mean=0.0575s tpot_mean=17.77ms tok_s=216.49
RESULT-VLLM C=8 ttft_p50=0.0809s ttft_mean=0.0739s tpot_mean=17.97ms tok_s=422.92

# leg B:AcornInfer W8 graph(chat)
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0449s ttft_mean=0.0449s tpot_mean=11.28ms tok_s=81.09
RESULT-VLLM C=4 ttft_p50=0.1202s ttft_mean=0.0997s tpot_mean=12.21ms tok_s=265.51
RESULT-VLLM C=8 ttft_p50=0.1941s ttft_mean=0.1748s tpot_mean=12.90ms tok_s=397.40

# leg B:AcornInfer W8 graph(wiki,out=64)
warmup done (1 request, not measured)
RESULT-VLLM C=1 ttft_p50=0.0357s ttft_mean=0.0357s tpot_mean=10.96ms tok_s=88.10
RESULT-VLLM C=4 ttft_p50=0.1175s ttft_mean=0.0974s tpot_mean=11.89ms tok_s=301.03
RESULT-VLLM C=8 ttft_p50=0.1907s ttft_mean=0.1720s tpot_mean=12.29ms tok_s=538.79

# leg C:AcornInfer W8 eager(wiki,out=64)
warmup done (1 request, not measured)
SPEC-STATS C=warmup verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}
RESULT-VLLM C=1 ttft_p50=0.0340s ttft_mean=0.0340s tpot_mean=18.08ms tok_s=54.56
SPEC-STATS C=1 verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}
RESULT-VLLM C=4 ttft_p50=0.1211s ttft_mean=0.0997s tpot_mean=18.45ms tok_s=201.82
SPEC-STATS C=4 verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}
RESULT-VLLM C=8 ttft_p50=0.1937s ttft_mean=0.1740s tpot_mean=18.39ms tok_s=382.80
SPEC-STATS C=8 verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}

# leg C:AcornInfer W8 eager(chat)
warmup done (1 request, not measured)
SPEC-STATS C=warmup verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}
RESULT-VLLM C=1 ttft_p50=0.0340s ttft_mean=0.0340s tpot_mean=17.54ms tok_s=55.39
SPEC-STATS C=1 verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}
RESULT-VLLM C=4 ttft_p50=0.1211s ttft_mean=0.0997s tpot_mean=18.54ms tok_s=188.26
SPEC-STATS C=4 verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}
RESULT-VLLM C=8 ttft_p50=0.1927s ttft_mean=0.1731s tpot_mean=19.26ms tok_s=291.08
SPEC-STATS C=8 verify_steps=0 E_acc1=n/a mean_committed=n/a hist={}

# leg D:AcornInfer W8 spec eager N=1 K=4(wiki,out=64)
warmup done (1 request, not measured)
SPEC-STATS C=warmup verify_steps=3 E_acc1=1.000 mean_committed=1.000 hist={'1': 3}
RESULT-VLLM C=1 ttft_p50=0.0343s ttft_mean=0.0343s tpot_mean=21.53ms tok_s=46.03
SPEC-STATS C=1 verify_steps=55 E_acc1=1.073 mean_committed=1.073 hist={'1': 51, '2': 4}
RESULT-VLLM C=4 ttft_p50=0.1272s ttft_mean=0.1044s tpot_mean=21.85ms tok_s=160.82
SPEC-STATS C=4 verify_steps=201 E_acc1=1.184 mean_committed=1.184 hist={'1': 165, '2': 35, '3': 1}
RESULT-VLLM C=8 ttft_p50=0.2033s ttft_mean=0.1825s tpot_mean=22.41ms tok_s=289.41
SPEC-STATS C=8 verify_steps=390 E_acc1=1.218 mean_committed=1.218 hist={'1': 308, '2': 79, '3': 3}

# leg D:AcornInfer W8 spec eager N=1 K=4(chat)
warmup done (1 request, not measured)
SPEC-STATS C=warmup verify_steps=649 E_acc1=1.194 mean_committed=1.194 hist={'1': 527, '2': 118, '3': 4}
RESULT-VLLM C=1 ttft_p50=0.0349s ttft_mean=0.0349s tpot_mean=24.58ms tok_s=40.16
SPEC-STATS C=1 verify_steps=27 E_acc1=1.000 mean_committed=1.000 hist={'1': 27}
RESULT-VLLM C=4 ttft_p50=0.1347s ttft_mean=0.1108s tpot_mean=25.54ms tok_s=136.56
SPEC-STATS C=4 verify_steps=103 E_acc1=1.049 mean_committed=1.049 hist={'1': 98, '2': 5}
RESULT-VLLM C=8 ttft_p50=0.2004s ttft_mean=0.1800s tpot_mean=25.95ms tok_s=208.30
SPEC-STATS C=8 verify_steps=165 E_acc1=1.152 mean_committed=1.152 hist={'1': 154, '2': 6, '4': 1, '5': 4}
```

(注记:leg D chat warmup 行的 649 = 服务端累计计数含此前 wiki 各档,增量口径以每档快照差分为准,chat 各档 verify_steps 27/103/165 为干净增量;全表 mean_committed 与 E_acc1 逐位相等 = 零 eos 截断损耗,committed ≡ accepted+1。)

### 终账·仪式表(GPU 0,tok_s;TTFT/TPOT 见上 stdout 原样)

| 腿 | 语料(out) | C=1 | C=4 | C=8 |
|---|---|---|---|---|
| vLLM bf16(chat) | chat(32) | 58.12 | 208.57 | 356.17 |
| vLLM bf16 plain(口径腿) | wiki(64) | 58.89 | 216.49 | 422.92 |
| vLLM ngram spec K=4 | chat(32) | 53.49 | 191.93 | 315.55 |
| vLLM ngram spec K=4 | wiki(64) | 62.64 | 225.18 | 423.23 |
| AcornInfer W8 graph(生产参照) | chat(32) | 81.09 | 265.51 | 397.40 |
| AcornInfer W8 graph(生产参照) | wiki(64) | 88.10 | 301.03 | 538.79 |
| AcornInfer W8 eager(A2 分母) | wiki(64) | **54.56** | 201.82 | 382.80 |
| AcornInfer W8 eager | chat(32) | 55.39 | 188.26 | 291.08 |
| AcornInfer W8 spec eager N1K4 | wiki(64) | **46.03** | 160.82 | 289.41 |
| AcornInfer W8 spec eager N1K4 | chat(32) | 40.16 | 136.56 | 208.30 |

达标率逐档(chat 语料,AcornInfer ÷ vLLM bf16):

- **graph(生产参照)**:C=1 81.09/58.12 = **139.5%**、C=4 265.51/208.57 = **127.3%**、C=8 397.40/356.17 = **111.6%**(M7 终账 147.4/128.4/112.8:C=1 低 7.9pp 主体为我方 C=1 TTFT 单样本波动 0.0449 vs M7 0.0358(out=32 短输出下 TTFT 全额进 tok_s span;vLLM 侧 +0.0% 零漂移;decode-only TPOT 比 17.85/11.28 = ×1.58 与 M7 ×1.42 同向强健),如实注记,非 decode 面回退);
- **eager(同模式基线)**:95.3% / 90.3% / 81.7%(graph−eager gap ≈ M6/M7 已入档的 eager 模式代价,口径自洽);
- **spec eager(选定配置)**:69.1% / 65.5% / 58.5%(A3 如实,归因见判定行)。

### 终账·锚判定行(如实,本里程碑最重要的一行)

**A1 无损性(硬门):过(证据引用 T3 + 本任务零回归)。** tiny 全量逐位(`test_spec_greedy_bitwise_tiny` 等 6 passed)+ 8B 抽样 4 prompts × 64 tokens 逐位 `A1_RESULT PASS`(M8 段 3 节原样);本任务对 spec/verify/接受链零算法改动(仅步级计数旁路),fast 238 零漂移 + `test_spec_engine.py` 全绿为延续证据。

**A2 加速(硬门):未过,如实裁 = NO-GO(特性保留,缺省关不变)。** 判定值(同模式同语料新鲜测,同窗口 GPU 0):spec(wiki)C=1 tok_s **46.03** ÷ eager W8(wiki)C=1 **54.56**(A2 分母,新鲜测)= **×0.844**,对锚 ×1.4 缺口 −39.7%(且方向为负:spec 比 eager 慢 15.6%)。配 E[接受+1]:**1.073**(仪式 HTTP 与参数扫 in-process 两口径逐位同值,55 verify 步同 hist),对 E ≥ ~1.45 的锚推算需求缺 −26%;E_acc1=1.073 下即使零开销也到不了 ×1.4(几何上限 ×1.073)。**vs 生产 graph W8 的比值(如实)**:wiki C=1 46.03/88.10 = **×0.522**(C=4 ×0.534 / C=8 ×0.537)。归因(数据闭环,三层):①**草稿器设计上限**——备选式 propose + 前缀接受语义 → 实用 a≤1 → E_acc1 上限 ~2.0(机制上限探针 1.879 兑现),锚推算的文献带 1.8–2.8 属**链式/span 式草稿器**(PLD 原法)的接受率,本设计的备选式结构拿不到;②**workload 敏感性兑现**——wiki.test.raw 自然续写的首后继命中率 ~7%(E_acc1 1.07),非复制型任务;外部对照:**vLLM 0.32.0 自带链式 ngram(lookup 1–4,K=4)同语料同协议只到 ×1.064**(62.64/58.89)——独立实现同向证实该 workload + ngram 系草稿的收益天花板远低于锚,非本工程实现质量所致(chat 语料 vLLM ngram 同样 ×0.92 负收益);③**verify 步税兑现**——spec wiki C=1 TPOT 21.53ms vs eager 18.08ms = **+19.1%**(T3 计测 18.28→22.34 = +22.3% 同带),E_acc1 增益 1.073 < 税 1.19 → 净 ×0.84,算术闭环(46.03/54.56 = 1.073/1.191 × 档位轮廓项)。**判定:spec eager v1 在选定配置上不达 A2;W8 graph(生产参照)仍是唯一正收益 decode 面;缺省 `--spec-ngram 0` 不变(A4);特性保留为显式 opt-in。**

**A3 workload 敏感性(如实):chat 全矩阵 <1,归因 = 接受率。** spec/eager:×0.725 / ×0.725 / ×0.716(C=1/4/8);spec/graph:×0.495 / ×0.514 / ×0.524。chat E_acc1:C=1 **1.000**(随机 prompt 零命中,hist 全 1)/ C=4 1.049 / C=8 1.152(均匀随机 id 的偶然 n-gram 重合,hist 出现 4/5 token 步);零命中 + 全额 verify 税(TPOT 17.54→24.58 = +40%)= 负收益算术。工作负载选择指引(spec R4 兑现):ngram 系投机只值得在**强复制型**(摘要/改写/代码续写/受限域名)显式开启;通用 chat / 自然续写不开。

**A4 缺省不变(硬门):过。** fast 238 零漂移 + slow 17 / tp 13 零换金(三尾行在档);m6_ab none/w8a16 双带复核带内;`bench_remote --prompts-file/--spec-stats` 均显式 flag、缺省路径逐字节;`/spec_stats` 端点只读;`spec_stats` 计数面 spec 关闭恒零值零写入。

### 终账·接受率矩阵(E[接受+1] = accepted/verify_steps + 1;committed/步分布)

| workload × C | verify 步数 | E[接受+1] | hist(committed/步) |
|---|---|---|---|
| wiki C=1(HTTP) | 55 | 1.073 | {1:51, 2:4} |
| wiki C=4(HTTP) | 201 | 1.184 | {1:165, 2:35, 3:1} |
| wiki C=8(HTTP) | 390 | 1.218 | {1:308, 2:79, 3:3} |
| chat C=1(HTTP) | 27 | 1.000 | {1:27} |
| chat C=4(HTTP) | 103 | 1.049 | {1:98, 2:5} |
| chat C=8(HTTP) | 165 | 1.152 | {1:154, 2:6, 4:1, 5:4} |
| wiki C=1(in-process 扫,两 pass 同) | 55 | 1.073 | {1:51, 2:4} |
| chat C=1(in-process 扫) | 27–29 | 1.000 | {1: 全} |
| copy 探针 C=1(N1K4,机制上限) | 33 | 1.879 | {1:4, 2:29} |

判读:wiki C 增大 E_acc1 上升(1.073→1.218)= 不同档位取用不同语料块(前 1/4/8 条)的内容差异,非并发效应(草稿器按 seq 独立);C=8 档 hist 出现 4/5-token 步(chat)= 随机 id 偶然重合;**全表零 eos 截断损耗**(mean_committed ≡ E_acc1)。机制税恒定面:verify 步 +19~22%(TPOT 对 eager 同语料),E_acc1 ≤ 1.22 的任何配置净收益为负——这就是本终账判定行的算术核心。

### 终账·kernel/草稿器开销账(引用 + 互证)

- T3 计测(M8 段 3 节原样引用):eager decode C=4 18.28ms/步;verify m=5 零接受 22.34ms(最坏 +22%/步);draft 延迟 mean 0.3µs(p95 0.4,目标 <100µs,3 个量级余量)——**草稿器 CPU 面与 verify 税相比是无关量级,开销账主体 = verify 前向 m=5 本身**。
- 本仪式互证:eager wiki C=1 TPOT 18.08 → spec 21.53(+19.1%),eager chat C=1 17.54 → spec 24.58(+40%,零接受纯税);与 T3 +22% 同带——8B HTTP 全链下 verify 税稳定复现,零粉饰。

### 终账·M9 台账(承 M6/M7 台账 + 本里程碑移入/新立)

- **链式/span 式草稿器(本终账新立,A2 翻案的唯一主路径)**:现备选式 propose(同键前 K 后继)的实用上限 E_acc1≈2.0(探针 1.879);改链式续查(PLD 原法:末缀匹配 → 复制后续 span)后 wiki 型 E_acc1 有望进 1.8–2.8 文献带——外部佐证 = vLLM ngram(链式)同语料 ×1.064 虽低但结构占优;若届时 E_acc1 ≥ 1.45,A2 须按新配置重开判定(本终账锚对备选式结构闭合,不自动外推);
- **自适应 k**(承 plan):低接受率档自动降 m(E_acc1 <1.1 时 m=5 纯亏,自适应可把 chat 负收益收窄,但救不了方向);
- **graph verify**(承 spec §2.4):verify 前向进 CUDA graph,收 verify 税的 launch 面(税 19% 中的 host/launch 分量;T2 kernel 面 0.037ms 已证 GPU 面非瓶颈);
- **采样域投机**(承 spec §2.4)/ **draft-model 系**(Eagle/Medusa,承 spec §5)/ **TP×spec**(构造期互斥已落地,解禁需 TP verify 通信面评估);
- 承 M7 原样:**W4 复活条件**(kernel 形态突破 / GPTQ / 显存受限重估)、**KV 量化**、**C=1 非 GEMM 面收账**、**GPTQ/AWQ**、**磁盘量化缓存**;
- 承 M6 原样:**双存储 down_t**、**连续到达 bench 协议**、**半宽 GEMM**、**custom allreduce**、**rank0 快失败**、triton decode_gemm 模块(入库未接线)。

### 终账·口径注记(照前节同款)

同日同机(RTX 4090 ×2;参数扫/回归/复核带物理 GPU 1、仪式六腿物理 GPU 0、压测客户端 `CUDA_VISIBLE_DEVICES=1` 零 GPU 占用;每腿跑前/收服后 nvidia-smi 双卡 4 MiB 验净 + 零计算进程双查,前台 timeout ≥1800s);chat 腿 seed=20261027、in=128/out=32;wiki 腿 in=128(语料块)/out=64(`--prompts-file` 各档取前 C 条);temperature=0、计时前 1 条 warmup 不计入;HTTP TTFT 含客户端→服务端上行与 SSE 流式首包回传全程(`bench_remote._run_one` 口径);tok_s = 总生成 token / span(含 TTFT 全额,out=64 腿 TTFT 占比低于 out=32);参数扫 tok_s 为 in-process 引擎记账口径(gen/(last−created),非 HTTP),与仪式 HTTP 口径不可直比(仪式 wiki C=1 spec 46.03 vs 扫表 48.70,差 = HTTP/SSE 全程流式开销,M6/M7 同款 ×0.93 带内);接受率口径 = E[accepted+1]/verify 步(E_acc1),committed ≡ accepted+1 全表成立(零截断);eager 腿 SPEC-STATS verify_steps=0 = 计量面未接 spec 路径的活证;E_acc1 跨口径复现(HTTP 55 步 = in-process 55 步,hist 逐位同)= 草稿器确定性活证;关服纪律:`ss -tlnp` 真身 PID 定向 kill 六次(nginx 18200 误匹配已注记,未误伤),终态双卡 4 MiB;vLLM 0.32.0(ngram spec 腿经 `--speculative-config` 在场,R7 兑现),基线环境 `/home/ernest/vllm-baseline`;证据归档 `/tmp/opencode/m8t4/`(六腿 server+bench stdout、参数扫两 pass、ceiling probe、回归三尾行);本节树 = 36a63ed + 本任务 commit。M8 不打 tag、不 push(留用户)。
