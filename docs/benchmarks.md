# 性能基线

记录每个里程碑的吞吐数字,全部在 RTX 4090(物理 GPU 1)上测得。

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

M3 验收线(引擎聚合吞吐 ≥ vLLM 的 70%,同卡同权重)的对照基线。vLLM 0.32.0(pip,独立 venv `/home/ernest/vllm-baseline`,torch 2.13.0),服务命令:`CUDA_VISIBLE_DEVICES=0 vllm serve /home/ernest/model-weight/Qwen3-8B --dtype bfloat16 --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90`(物理 GPU 0,RTX 4090 24GB,与引擎测试卡同型号)。

| 并发 | TTFT p50 | TTFT mean | TPOT mean | 聚合吞吐 |
|---|---|---|---|---|
| 1 | 0.0289 s | 0.0289 s | 16.92 ms | 57.81 tok/s |
| 4 | 0.0689 s | 0.0581 s | 17.86 ms | 207.39 tok/s |
| 8 | 0.0819 s | 0.0749 s | 17.84 ms | 354.91 tok/s |

命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8`(压测端为纯 HTTP 客户端,不占 GPU;in=128/out=32,temperature=0,seed=20261027)。

协议(与引擎侧 `acorninfer.bench` 对齐):同一批随机 prompt id(`_prompts` 同 seed)以 token-id 数组直投 OpenAI `/v1/completions`(completions API 原生支持,绕开文本分词往返),temperature=0、max_tokens=32、自然 EOS 停止;TTFT/TPOT/聚合吞吐公式与 `acorninfer.bench._report` 一致;计时前有 1 条 smoke 请求暖机(不计入;该 warmup 已固化进 `scripts/bench_remote.py`——1 条 8-token prompt / min(4,out) 输出的小请求,镜像引擎侧 warmup 协议,复测无需手工预热)。两条口径注记:

1. **HTTP 往返计入两端延迟**:TTFT 含客户端→vLLM 的 HTTP 上行与 vLLM 调度开销→首 token 回传全程,即产品真实路径,70% 线按此口径(对引擎侧是偏严的口径)。
2. **completion_tokens 与引擎 generated_ids 同口径(含 EOS 终止 token)**:实测 C=8 档同 seed 下同样恰有一条请求在第 2 token 命中 EOS 提前终止(总生成 226 而非 256;vLLM 侧 `finish_reason=stop`、`completion_tokens=2`),与引擎 M2 实测逐位一致——同权重同贪心路径的交叉验证,两侧 token 计数可直接相比。

70% 验收线(基线聚合吞吐 × 0.7,逐档;TTFT/TPOT 越低越好,不设 ×0.7 线,仅作参考):

| 并发 | vLLM 基线 | 70% 线(M3 引擎聚合吞吐须 ≥) | 引擎 M2 实测(参考) |
|---|---|---|---|
| 1 | 57.81 tok/s | 40.47 tok/s | 36.99 tok/s |
| 4 | 207.39 tok/s | 145.17 tok/s | 89.06 tok/s |
| 8 | 354.91 tok/s | 248.44 tok/s | 110.68 tok/s |
