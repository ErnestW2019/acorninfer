# M3 设计文档:Triton 算子层

日期:2026-10-30
状态:设计已经用户确认(四项决定逐条对齐),待写实施计划
上游:`2026-10-26-acorninfer-design.md` 中 M3 行

## 1. 背景与目标

M2 交付了分页 KV + 连续批处理的正确引擎,但 decode 路径是刻意的参考实现:
每层每序列 `gather`(O(seq_len) 拷贝)+ 逐序列 SDPA。实测 TPOT 从 C=1 的
26.79ms 恶化到 C=8 的 61.50ms,恶化来源就是这份拷贝与批间争用。

M3 用 Triton kernel 替换它们:

1. **分页 decode attention**(flash-decoding 分 KV)——直接按页读 KV,消灭 gather;
2. **融合 RMSNorm+残差**——省一次全量 hidden 读写;
3. **融合 RoPE**——cos/sin 不再由 torch 预计算整表。

prefill 仍留在 SDPA(spec 明确)。

环境事实(2026-10-30 实测):M2 期间占用 GPU 0 的外部 vLLM 已停止,
两张 4090 均空闲。venv 内 triton 3.7.1 / torch 2.13.0+cu130 / sm89 可用。

## 2. 已确认的四项设计决定

| # | 决定 | 内容 |
|---|------|------|
| 1 | vLLM 基线 | **GPU 0 自建**:独立 venv(不污染项目 .venv),加载同一份 `/home/ernest/model-weight/Qwen3-8B`,bf16,同款 in=128/out=32 压测协议取基线。磁盘代价 ≈5-6 GB。同权重、同型号卡、配置可控,是最干净的对照。 |
| 2 | 数值判据 | **token 逐位 + logits 容差**:golden 门沿用 M2 协议(margin 筛选 prompt + 平局换题);kernel 单测与 torch 参考比 logits `assert_close`,容差实测后写死进注释(bf16+归约序依据入档)。 |
| 3 | 接入方式 | **A1:SDPA 路径永久保留为 reference**,`attention_backend` 开关切换;golden 双后端都跑。kernel 炸了随时切回,对照永在。 |
| 4 | 开发顺序 | **B1:先小后大**——K1 RMSNorm+残差、K2 RoPE 先行,借它们建立 Triton 测试基建(编译、对拍、容差标定),再啃 K3 attention 主菜。 |

## 3. 验收门

| 门 | 判据 |
|---|---|
| 数值 | golden 门 token 逐位(双后端);kernel 单测容差有实测依据 |
| 性能 | 同权重 bf16、in=128/out=32、C=1/4/8 下,自建引擎吞吐 ≥ 自建 vLLM 基线的 **70%**;内部锚点:M2 gather 版 TPOT 61.50ms@C=8 应显著下降 |
| 回归 | 现有 fast 69 + slow 10 全绿不退化 |

**风险预案**:若 70% 线达不到(vLLM 有 CUDA graph 与多年调度打磨),
判据改为"≥ vLLM X%(Task 1 基线实测后与用户重议)+ 相对 M2 gather 版 ≥ 2×"。
此为唯一允许重新议价的门。

## 4. 架构

新建 `acorninfer/ops/` 包:三个 kernel + 各自 torch 参考实现放一起
(参考实现即单测的 oracle,也是容差标定的对照)。

接入开关两枚,**正交、可独立压测**:

```python
# EngineConfig 增:
attention_backend: str = "sdpa"   # "sdpa" | "triton"
fused_ops: bool = False           # True → K1/K2 生效
```

`forward_decode` 内按开关分派;SDPA 路径一行不动。默认值保持 M2 行为,
现有测试零改动即绿。

K3 的输入就是 M2 的既有数据结构:`cache.k_pages/v_pages`
(布局 `[num_blocks, page, kv_heads, head_dim]`,M2 埋的布局伏笔在此兑现)、
块表 `table`、有效长度 `seq_len`。**不 gather、不拷贝**;为此
`PagedCacheView` 新增 `write()`(纯 scatter)公共方法,`update()` 重构为
`write + gather`,行为逐位保持(M2 测试钉死)。

边界情况:我们的块表无空洞(连续分配),只有尾块不满 → K3 只需处理
"页内 slot ≥ 有效 token 数"一种 mask,不需要任意洞的通用掩码。

## 5. Kernel 设计

### K1 `fused_add_rmsnorm`

vLLM 同款语义:`(normed, residual) = f(x, residual, weight)`。
一个 kernel 内完成"残差累加 + RMSNorm + 双写",省一次全量 hidden 读写。
用于 `input_layernorm` / `post_attention_layernorm` 两处。

注意:Qwen3 的 per-head `q_norm` / `k_norm`(head_dim 维)**不在**融合范围,
留在 attention 前的 torch 路径(带宽占比小,M3 不动;实测后若 launch overhead
显著再议)。

### K2 `fused_rope`

输入:q/k(已 norm,`[B, H, 1, D]`)、`positions [B]`、`inv_freq [D/2]`。
kernel 内由 positions 与 inv_freq 算 cos/sin 并就地旋转 q 与 k。
替代现状:torch 预计算整表 cos/sin → 广播乘两次。

### K3 paged decode attention(flash-decoding 两阶段)——主菜

并行度账:4090 有 128 SM;B=8 × 8 KV 头 = 64 CTAs,不 split 填不满,
这正是 spec"分 KV"的依据。

- **Phase 1**:grid `(B, H, splits)`(**按 Q 头铺开**,kv 头 = `hq // rep`;
  计划期精化:初稿按 KV 头估并行度 B×8,Q 头 grid 下正确公式是 B×H,
  已含 GQA 因子——q 向量各不相同,按 KV 头算需 4 份 q 共享寄存器,得不偿失)。
  每个 program 处理一个 (序列, Q 头, KV 分片):按页遍历分片内 KV,
  online softmax(fp32 累计),产出部分和 `(acc[D], m, l)` 写 partial buffer
  (`[B, H, splits, D]` + `[B, H, splits]` ×2)。
  尾块 mask:slot ≥ 有效 token 数处,分数置 −inf。
- **Phase 2**:`(m, l)` 重标定归并各分片,GQA 按 rep=4 广播到 32 个 Q 头,
  输出 `[B, 32, 1, 128]`,与 SDPA 路径形状逐位同构。
- q_len=1,无因果掩码;数值策略:累计 fp32、输出 bf16(与 SDPA 内部
  行为同构,容差标定按此做)。
- `num_splits` 初版固定启发式:`clamp(ceil(128 / (B×8)), 1, 8)`,
  不 autotune;调参放 Task 7。

## 6. 测试基建

先于一切 kernel(B1 顺序的理由):

- `tests/triton_utils.py` + conftest GPU fixture:
  - **对拍 harness**:kernel ↔ torch 参考实现的表驱动跑批器
    (形状 × 序列长度 × 页大小组合);
  - **两级容差**:fp32 输入紧容差(验索引/mask 逻辑——浮点脏了必是 bug)+
    bf16 输入实测容差(验数值策略),标定过程与结论全部入档注释;
- kernel 单测标 `@pytest.mark.gpu`,conftest 无卡自动 skip;
- 测试一律 `CUDA_VISIBLE_DEVICES=1`(纪律不变),GPU 0 让给 vLLM 基线服务。

## 7. 任务切分(骨架,细节归实施计划)

1. **vLLM 基线**(独立 venv、GPU 0、同协议压测、基线数字入 docs)
   ——先行:70% 线的具体数字是全程开发锚点;
2. Triton 测试基建(harness + 容差标定协议);
3. K1 `fused_add_rmsnorm` + 单测 + 接入;
4. K2 `fused_rope` + 单测 + 接入;
5. K3 Phase 1(partial;先独立对拍);
6. K3 Phase 2(combine)+ `attention_backend` 接进 `forward_decode`;
7. golden M3(双后端 token-exact)+ bench 对比表
   (sdpa / triton / vLLM×70% 线)+ docs + `m3-acceptance`。

每任务照旧:实现子代理 → spec 审查 + 质量审查(Do Not Trust the Report)。
保护文件清单在派单时逐次列出(含 M2 全部 golden/engine/pager)。

## 8. 风险清单

| 风险 | 预案 |
|---|---|
| 70% 线达不到 | §3 预案:基线出来后重议 X 与 2× 复合门 |
| bf16 kernel 数值调试周期不可控 | 两级容差 + 小算子先行建流程 |
| `num_splits`/页大小初版非最优 | 功能正确优先,调参集中 Task 7 |
| vLLM 环境安装拖累(下载/驱动) | 独立 venv 隔离;失败则基线暂缓,内核开发照常,门后补 |
| sm89 编译坑(triton 3.7.1) | Task 2 用最小 kernel 先验编译链 |
