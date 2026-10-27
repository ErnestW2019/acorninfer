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
