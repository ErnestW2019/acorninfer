# M9 设计 spec — prefill/TTFT 攻坚(引擎优化为主 goal,业界标准压测为验收仪)

日期:2026-11-02 · 状态:待用户审阅 · 前置:M8 收官(`3182a86`,已推 origin)

## §0 动机与口径定案

**用户定案(两条,凌驾于既往协议)**:

1. **对比口径 = 业界标准**:双方 bf16、同模型、同卡、同负载;**量化退出对比口径**(M7 W8A16 保留为可选部署模式,不再投入);"全三档跑赢 vLLM"(147/128/112%)按此口径**降级为无效 headline**,docs 追记更正。
2. **M9 主 goal = 引擎优化**:从零手搓的目的就是精简、回归本质、性能好。测量只是尺子,不单独立里程碑。

**现状账(bf16 对 bf16,M6 收官 / M8 仪式口径)**:

| 面 | AcornInfer bf16 | vLLM 0.32 bf16 | 差距 |
|---|---|---|---|
| decode tok_s C=1/4/8 | 53.7 / 186.8 / 296.7 | 58.1 / 207.6 / 354.3 | 落后 7.6 / 10.0 / 16.3% |
| TTFT p50 C=8(chat) | ~0.19 s | ~0.08 s | **落后 ~2.4×** |
| TP=2 tok_s | 73 / 72 / 65% | — | 落后(不在本轮) |

**为什么打 TTFT**:①差距最大且从未被任何里程碑攻过(唯一没打过的墙);②prefill 现状是过渡方案(逐序列 gather-to-dense + SDPA,qwen3.py:331/379),工程债明确;③TTFT 是生产体验第一指标;④C=8 下 0.19s 对 128-token 输入结构性偏大,怀疑大头不在纯计算,spike 必有油水。

## §1 锚

- **A1 主锚(spike 后定硬数,预案值)**:官方 `vllm bench serve` 口径(同参数同负载),**TTFT p50 @C=8 ≤ vLLM ×1.3**(现状 ×2.4)。预案不科学时行使**一次修正权**(M7 段 0 先例:spike 归因表出来后定稿,docs 记录修正理由,仪式当日不许换锚)。
- **A2 零回退门(硬)**:decode 三档 tok_s(闭式 m6_ab 口径)与官方工具吞吐逐档不劣化(机时噪声带内);fast/slow 全套件零换金。
- **A3 逐位门(硬)**:缺省配置下既有行为逐字节不变;golden m0–m8 不动。
- **A4 可复现(硬)**:官方工具每条命令原文 + 原始输出逐字入 docs。

## §2 架构与施工面

**现状事实(已核)**:

- prefill:`forward_prefill_batch`(M4.1 批量形态,GEMM 级已合批);attention = 逐 b gather/SDPA(qwen3.py:331 sdpa 路径 / 379 triton 后端路径);chunked prefill(M5)budget 截断,`schedule()` 返回 (chunks, decodes)。
- TTFT 链:HTTP 收包 → tokenizer encode(同步)→ 引擎锁排队 → 调度准入 → prefill 前向 → 首 token → pump 路由 queue → SSE flush。server 契约 = 单线程引擎 + `asyncio.to_thread` pump(队列空闲兜底 120s)。

**T0 spike(M4.1 先例,先拆账后施工)**:

- 服务端:torch profiler 拆一次满载 prefill step(GEMM/attention/elementwise/host 各占毫秒)+ engine/scheduler 步级时间戳(准入等待、chunk 拆分、decode 交错);
- 链路:TTFT 全链打点(HTTP 收包→encode→锁等待→前向→queue→SSE),区分服务端计算 / 排队 / 往返;
- 产出**归因表**(行 = 面,列 = 毫秒数 + 占比),A1 硬数在表上定稿。

**施工面(优先级 = 归因表排序;只做表上有行的面)**:

- **a. paged/chunked prefill attention kernel**:tiled flash-style 分页预填,新文件(如 `ops/paged_prefill.py`)从 paged_attention 只读派生,替代 gather+逐 b SDPA;**前置条件(R4):spike 证明注意力/gather 占比显著**;
- **b. prefill GEMM/合批形态**:若归因指 GEMM 效率或合批缺失(M=128×B 的 GEMM 轮廓);
- **c. chunked budget / 调度边界调参**:纯参数零新代码(M5 遗产,若归因指 chunk 拆分过多);
- **d. 首 token 关键路径**:encode 异步化 / pump 唤醒延迟 / SSE flush 时机(server 面,常规解锁)。

**每施工点验收**:归因表对应行毫秒级改善(前后对照)+ 官方 TTFT 联动 + A2/A3 全绿。不做表上没有的面。

## §3 保护面

- **禁改**:`ops/paged_attention.py`(只读派生先例延续)、`engine/scheduler.py`(调度语义零改动;spike 若指认调度面 → 停手上会,本 spec 预算内不动)、`engine/sequence.py`、`engine/graph_runner.py`、`cache/kv.py`、`cache/pager.py`(M8 已收官回保护)、golden m0–m8、`tests/triton_utils.py`。
- **常规解锁**:`models/qwen3.py`(prefill 变体)、`engine/engine.py`、`engine/executor.py`、`server.py`(d 面)、新文件 kernel/测试。
- **零改动红线**:A3 逐位门之下,凡与 TTFT 无关的面一律不碰。

## §4 风险

- **R1**:spike 归因显示大头在调度/服务链而非 kernel → 施工面转移至 b/c/d(调度仍不动);A1 修正权行使一次。
- **R2**:chunked budget 拆块导致结构性 TTFT 高 → c 面调参修复;若需调度语义改动 → 上会裁决,不私改。
- **R3**:新 kernel 数值 → TOL 协议(标定后写死,注释记录命令/日期)+ golden 零换金;prefill 数值域 = bf16,与 SDPA 对拍。
- **R4**:torch SDPA 底层即 flash/cuDNN,自定义 kernel 未必更快 → **kernel 施工前置条件 = spike 证明 gather 拷贝/逐 b 循环占比显著**;占比小则此面 no-op 如实记录(M6 先例)。
- **R5**:官方 random 数据集两侧一致性 —— 同 tokenizer + 同 seed → 同 prompts,首跑验证一次(哈希对比)。
- **R6**:开环高速率下两 server 饱和形态不同(queue 深度、准入策略)→ 每档 num-completed 如实记录,不达标率化。

## §5 Non-goals

量化(对比口径永久退出;W8 产线保留不投入)、TP=2 重测、draft-model 系投机、graph verify、scheduler 语义改动、ShareGPT 等外部数据集、多实例编排、rank0 快失败(挪回台账)。

## §6 仪式

- 工具:`/home/ernest/vllm-baseline/bin/vllm bench serve`(官方 benchmark_serving 正体),同一命令原文分别指 vLLM server 与 AcornInfer server;
- 负载:random 数据集 seed 固定,128 入/32 出;Poisson 到达,rate ∈ {1, 2, 4, 8, 16, 32} req/s × 256 prompts;饱和点逐配置如实标注;
- 对比项:AcornInfer bf16 满配方(施工后)vs 施工前同口径(before)vs vLLM bf16 同参;eager 腿作诊断;
- 指标:官方输出原样(TTFT/ITL/E2E 的 mean/p50/p99 + 吞吐 req/s/tok_s);
- 纪律:GPU 0、关服 `ss -tlnp` 真身 kill、双卡收尾验净、timeout ≥ 1800000;
- docs:**before/after/vLLM 三列成绩单** + 归因表 + 逐施工点前后账 + A1–A4 判定行(不达照写)。
