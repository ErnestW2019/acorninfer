# 性能基线

记录每个里程碑的吞吐数字,全部在 RTX 4090(物理 GPU 1)上测得。

| 里程碑 | 配置 | 吞吐 | 备注 |
|---|---|---|---|
| M0 | Qwen3-8B bf16,greedy,无 cache 全量重算,batch=1 | 32.90 tok/s | O(n²) oracle 路径,M1 的 KV cache 是对照基线 |
