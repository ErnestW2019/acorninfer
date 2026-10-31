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

脚本原始输出行(逐字):graph `steps=20 wall/step=21.50ms gpu_kernel+memcpy/step=19.81ms gap/step=1.70ms`;`launches/step=1418.0 memcpy/step=3.00`。eager `steps=20 wall/step=26.61ms gpu_kernel+memcpy/step=19.97ms gap/step=6.64ms`;`launches/step=1418.0 memcpy/step=7.00`。两轮 sanity 行均逐字 `sanity: alive=7/8 generated_per_seq=25 (batch 恒定贯穿计时窗口)`(batch=7 非 8,同 M4 口径注记 2);GPU 忙时合计(20 步,Self CUDA time total)graph 396.130 ms / eager 399.331 ms。launches/memcpy 按 GPU 侧事件计,graph 模式下不反映 CPU 发射数(M4 已知属性)。

段一改动在当日账目上的实测映照(计数对照,均为 GPU 侧事件):

- eager memcpy/step 13.00 → 7.00(−6.00):与 argmax 合批的 D2H 数目变化吻合(B=7 逐序列 D2H → 1 次);
- eager kernel 事件/step 1430.0 → 1418.0(−12.0):与 argmax 逐序列 → 单次合批的 kernel 次数精简方向一致;
- graph memcpy/step = 3.00:与段一改造后的每步传输组成(2 H2D + 1 D2H)逐项一致;
- graph 与 eager 当日 kernel 事件数同为 1418.0:graph 回放的 kernel 集与 eager 发射的相同。

### 残差判定(graph gap)

**gap 实测 1.70 ms/步,段一目标 ≤1 ms 未达**(对照 T6 账 1.80 ms:−0.10 ms;占当日 graph 墙钟的 7.9%(1.70/21.50);当日 graph 相对 eager 回收发射间隙的 74.4%((6.64−1.70)/6.64))。残差构成,实测锚点与推断分别标注:

1. 实测:每步 GPU 侧 memcpy 恰 3 次(2 H2D + 1 D2H);argmax 合批后的 1 次 D2H(`logits.argmax().tolist()`)是 decode_batch 路径内唯一的设备同步点——同步返回后 GPU 队列为空,至下一次 `graph.replay()` 之间的全部图外 CPU 工作直接暴露为 gap;
2. 实测:同日 eager gap 6.64 ms → graph 1.70 ms,差值 4.94 ms 为逐 kernel CPU 发射路径被 graph 消除的部分(eager 当日账内 cuLaunchKernelEx Self CPU 13.561 ms/20 步 = 0.678 ms/步、216 次/步,为该路径的实测组成之一);
3. 图外 CPU 每步工作清单(`graph_runner.decode_batch` 代码路径可数,未单独计时):`ensure_append_capacity` 逐 view 循环;CPU 镜像填充(4+1 次 `torch.tensor` 构造 + pinned 切片赋值,`pin_tables.zero_()` + `build_tables` CPU 直填,padding 压回 3 次);`_commit_lens` 逐 view × 逐层 python 循环(B=7 × 36 层 = 252 次赋值/步);调度 python。何者主导为推断。

口径注记(均为实测事实):

1. eager wall/step 含 profiler 逐 kernel 采样开销(M4 口径注记 3 沿用);eager gap 7.38 → 6.64 ms(−0.74 ms)与 argmax 同步减少方向一致,但采样事件数同变(1430.0 → 1418.0),不作精确归因。
2. graph 模式 wall/step 21.50 ms 与 T6 账 21.64 ms 基本持平,段一两改动(同步 7→1、镜像单发 H2D)在 gap 上的当日净变化为 1.80 → 1.70 ms;bench 侧当日 graph 三档净增益仍为正(见上表)。
