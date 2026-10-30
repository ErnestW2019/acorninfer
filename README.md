# AcornInfer

从零手搓的 LLM 推理引擎。橡子虽小,终成参天橡树——Great oaks from little acorns grow。

> 更名记录:前工作名 `suiren`(燧人)已定名 **AcornInfer**。

状态:设计文档见 `docs/superpowers/specs/`;M0、M1、M2 已完成(tag `m0-acceptance` / `m1-acceptance` / `m2-acceptance`)。M2 实测:并发 8 聚合 110.68 tok/s(单流 36.99,见 `docs/benchmarks.md`)。

M3 已收官(tag `m3-acceptance`):`acorninfer/ops/` 融合算子层落地——K1 fused_add_rmsnorm / K2 fused RoPE / K3 flash-decoding 分页 attention(SDPA 路径保留为 reference);性能门 C=1/C=4 达标、C=8 99.2% 带缺口(判定与数字见 `docs/benchmarks.md`)。

## M0 用法

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer generate --prompt "The capital of France is" --max-new-tokens 32

# 测试(慢速用例需权重与 GPU;GPU 0 被外部服务占用,统一用 GPU 1)
.venv/bin/pytest -v -m "not slow"
CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest -v -m slow
```

## M1 用法(KV cache + 采样)

```bash
# greedy(KV cache,~O(n))
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer generate --prompt "The capital of France is" --max-new-tokens 32

# 采样(temp/top-p/top-k,镜像 HF warper 顺序)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer generate --prompt "Once upon a time" --max-new-tokens 64 --temperature 0.6 --top-p 0.95 --top-k 20
```

## M2 用法(并发引擎 + OpenAI API)

```bash
# OpenAI 兼容服务(GPU 1)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.server --port 8001
curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "你好"}], "max_tokens": 64, "temperature": 0.6
}'

# 压测:吞吐 / TTFT / TPOT(直连 Engine,不经 HTTP)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 4
```
