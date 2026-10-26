# Suiren(燧人)设计文档

日期:2026-10-26
状态:已评审,待实施

## 1. 项目概述

Suiren 是一个从零手搓的 LLM 推理引擎。燧人氏钻木取火,是人类第一次从零手搓出
"torch"——本项目基于 PyTorch,从 tokenization 到 sampling 每一层亲手实现。

### 目标(按优先级排序,分阶段兑现)

1. **学习为主**:吃透推理引擎每一层原理,每个模块都懂为什么;
2. **生产可用**:在 NVIDIA RTX 4090 上真实地服务模型(continuous batching、
   paged attention、投机解码);
3. **研究试验场**:kernel 调优、投机解码、prefix cache 等思想的实验平台;
4. **为昇腾练手**:架构预留硬件抽象层,后期将 Triton kernel 平移至
   Triton-Ascend(参照 tokenspeed-kernel-npu 的先例)。

### 硬件与约束

- 开发机:2× NVIDIA RTX 4090(sm89,24 GB),128 核 CPU,1 TB 内存,openEuler;
- 主线**单卡单进程**;TP=2 仅作为 M4 拉伸目标,不进主线;
- 远程昇腾 950PR 一台,M5 阶段启用;
- 参照系:本机已安装 vLLM(性能对照)、HF transformers(数值基准);
  本地可学习代码库:vllm、sglang、tokenspeed、magnitude。

### 技术栈

- Python + PyTorch;
- 架构(调度器 / KV cache / batching / sampler)全部手写;
- 算子先用 PyTorch(reference),性能热点手写 Triton kernel;
- 硬件抽象收口于一个 `DeviceOps` 接口,M5 前不为昇腾写任何代码。

## 2. 总体架构(终态)

```
serving/    OpenAI 兼容 API(FastAPI,SSE 流式)
engine/     调度器:continuous batching、请求状态机、准入/抢占
cache/      分页 KV cache:块表、分配器、(M4)prefix 匹配
models/     模型定义:Qwen3 dense(GQA+QK-Norm+RoPE+SwiGLU)→ 后加 MoE
kernels/    算子层:registry + reference(PyTorch)+ triton 实现
backends/   设备抽象:CUDA 先行,M5 落 Triton-Ascend/NPU
bench/      压测工具:吞吐 / TTFT / TPOT,与 vLLM 同卡对比
```

核心数据流:

```
请求 → API server → scheduler 队列 → 准入(KV 容量检查)
  → 组 batch → executor(prefill 分块 / decode)→ sampler
  → detokenize → SSE 流回
```

## 3. 关键设计决策

1. **目标模型**:主线 Qwen3-8B(dense,现代架构:GQA + QK-Norm + RoPE +
   SwiGLU,权重已在本地 `~/model-weight/Qwen3-8B`);MoE
   (Qwen3-Coder-30B-A3B)为 M4 后的进阶目标。
2. **kernel registry 从第一天就有,但极简**:`kernels/reference/` 下的纯
   PyTorch 实现是数值基准;`kernels/triton/` 下的实现必须与 reference
   对齐(atol/rtol 紧容差)后才能被选择。借鉴 tokenspeed 的纪律,不搬其复杂度。
3. **单一执行路径**:prefill/decode 分离、投机、CUDA graph 都是同一路径的
   参数,不设第二路径分支(借鉴 tokenspeed 的设计哲学)。
4. **设备抽象最小化**:只收口 attention / norm / rope / gemm 的 `DeviceOps`;
   保证 M5 移植只动 `backends/` 与 `kernels/triton/`。
5. **KV cache 设计**:块表 + page size 16;分配器先做线性空闲表,
   M4 加哈希前缀匹配。

## 4. 里程碑与验收标准

| 里程碑 | 内容 | 验收(硬性) |
|---|---|---|
| **M0** 骨架 | 权重加载(safetensors)+ Qwen3 前向(纯 PyTorch,无 cache)+ greedy 生成 | 固定 prompt 下与 HF transformers 输出前 ≥50 token 一致 |
| **M1** 核心 | 朴素 KV cache + prefill/decode 分离 + 完整采样(temp/top-p/top-k) | 长文本生成与 HF 对齐;采样路径 logprob 偏差 < 容差 |
| **M2** 引擎 | 分页 KV(块表,page=16)+ continuous batching + OpenAI API + TTFT/TPOT 指标 | 并发压测下输出正确、KV 无泄漏(debug 断言分配/释放平衡) |
| **M3** 性能 | Triton kernel:分页 decode attention(flash-decoding 分 KV)、融合 RMSNorm+残差、融合 RoPE;prefill 先用 SDPA | 与 reference 数值对齐;Qwen3-8B decode 吞吐 ≥ vLLM 的 70% |
| **M4** 进阶 | prefix cache(哈希块匹配)+ n-gram 投机解码 + W8A8 量化;拉伸:TP=2 | 投机接受率/加速比可量化;量化后精度损失 < 阈值 |
| **M5** 昇腾 | 硬件抽象收口 → Triton-Ascend 移植到 950PR | NPU 上跑通小模型端到端 |

纪律:每个里程碑只有验收通过才进下一个;数值不对齐的引擎不堆功能。

## 5. 错误处理与守卫

- **请求级隔离**:单请求异常只失败该请求,不倒引擎;参数校验错误返回 4xx;
- **KV 不足**:先排队;满员后抢占最老请求(recompute 策略,M2 实现);
- **两个 debug 守卫**(防最难查的坑):
  - KV 泄漏断言:debug 模式下分配/释放必须平衡;
  - 数值对齐 gate:reference 对比测试进 CI。

## 6. 测试策略

1. **golden test**:greedy 输出与 HF 逐 token 对比(M0 建立后永久保留);
2. **数值 gate**:每个 Triton kernel 对 reference 的紧容差测试;
3. **并发 fuzz**:随机请求模式,断言响应完整性 + KV 计量;
4. **性能基线**:每阶段记录吞吐/TTFT/TPOT,与 vLLM 同卡对比。

## 7. 代码结构

```
suiren/
  pyproject.toml
  suiren/
    config.py           # EngineConfig 等配置
    tokenizers/         # HF tokenizers 封装
    models/
      qwen3.py          # 模型定义与权重映射
    kernels/
      registry.py       # register_kernel / select_kernel(极简)
      reference/        # 纯 PyTorch 数值基准
      triton/           # Triton kernel
    backends/
      cuda_backend.py   # DeviceOps 的 CUDA 实现(M5 加 npu_backend.py)
    cache/
      pager.py          # 分页分配器与块表
      prefix.py         # (M4)哈希前缀匹配
    engine/
      scheduler.py      # continuous batching 调度器
      sequence.py       # Sequence / SamplingParams / 请求状态机
      executor.py       # 批量前向执行
      sampler.py
    serving/
      server.py         # OpenAI 兼容 server
      protocol.py
    bench/
      cli.py            # 压测:吞吐 / TTFT / TPOT
  tests/                # golden / 数值 / 并发 fuzz / 性能基线
  docs/                 # 每模块的设计笔记
```

## 8. 协作方式

- 仓库:`~/github/suiren`,git 管理;
- 每个里程碑:写代码与测试 → 跑验收 → 复盘该模块的设计取舍
  (为什么 paged、为什么这样调度),学习目标在此兑现;
- 性能对照命令与数字随里程碑记录在 `docs/benchmarks.md`。

## 9. 命名

Suiren(燧人):燧人氏钻木取火,人类第一次从零手搓出火;本项目基于
PyTorch 从零手搓——取火即取 torch。
