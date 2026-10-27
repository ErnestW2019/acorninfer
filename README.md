# Suiren(燧人)

从零手搓的 LLM 推理引擎。燧人氏钻木取火——取火即取 torch。

> 工作名:当前仓库名 `suiren` 为工作名,正式名待定,改名成本低。

状态:设计文档见 `docs/superpowers/specs/`;M0、M1 已完成(tag `m0-acceptance` / `m1-acceptance`),当前准备 M2。

## M0 用法

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m suiren generate --prompt "The capital of France is" --max-new-tokens 32

# 测试(慢速用例需权重与 GPU;GPU 0 被外部服务占用,统一用 GPU 1)
.venv/bin/pytest -v -m "not slow"
CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest -v -m slow
```

## M1 用法(KV cache + 采样)

```bash
# greedy(KV cache,~O(n))
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m suiren generate --prompt "The capital of France is" --max-new-tokens 32

# 采样(temp/top-p/top-k,镜像 HF warper 顺序)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m suiren generate --prompt "Once upon a time" --max-new-tokens 64 --temperature 0.6 --top-p 0.95 --top-k 20
```
