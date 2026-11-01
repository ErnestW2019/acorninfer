# M6 设计:decode 步贴顶(roofline closure)

日期:2026-11-01
前置:M5 收官(`7a20a12`,双轨终账在档);M4.1 性能遗留台账
状态:设计经五轮问答定案(方向/手段三段化/锚/打包范围/物理靶修正),用户已批

---

## 1. 背景与动机

M5 终账:轨 1(TP=1 graph)达标率 92.4/90.0/83.7%,距 M4 挂起的旧北极星"TP=1 逐档 ≥100%"缺口为 TPOT C=1 −1.35ms、C=8 −3.3ms/步。M5 段 0/段 2 与双轨仪式沉淀的物理数据指出:decode 步的主要矛盾不是"缺 kernel",而是**引擎级 GEMM 效率与步开销离物理天花板还有一段已量化的距离**:

| 口径 | 有效带宽 | 对 4090 峰值(1.008TB/s)占比 |
|---|---|---|
| 4090 HBM 峰值(24GB GDDR6X,384-bit) | 1.008 TB/s | 100% |
| 孤立微基准冷口径全宽 GEMM(M5 段 0 补充二) | 817–930 GB/s | 81–92% |
| 孤立微基准冷口径半宽 GEMM | 808–871 GB/s | 80–86% |
| 引擎 graph decode C=1(13.9GB/19.81ms) | ~702 GB/s | 70% |
| 引擎 graph decode C=8(13.9GB/21.12ms) | ~658 GB/s | 65% |
| vLLM TPOT C=1/C=8 反推 | 822 / 779 GB/s | 81% / 77% |

注:8B bf16 权重 ~13.9GB/步,decode 是权重流搬问题;M5 段 0 热口径的"1.68TB/s"是 L2(72MB)伪象,已由冷口径矩阵纠正。

**M6 主攻 = 引擎级 GEMM 效率追平孤立微基准(702→889GB/s,约 −1.5~−2.5ms/步)+ 步开销清账(残差 1.70ms gap、H2D、commit_lens,约 −1.3~−1.7ms)**;自研 triton kernel 降级为有界上限(≤7–12%)的第三段,仅当尖刀证明前两段关不上缺口时立项。

## 2. 目标与验收锚

**锚:重挂 M4 旧北极星——TP=1 graph tok_s 逐档 ≥ vLLM TP=1 同日同协议值**(基线缺口,按需增长口径:C=1 +8.2%、C=4 +11.2%、C=8 +19.4%)。

- 锚的算术:C=1 需总时序 −45ms(TPOT −1.4ms × 31 步为主);C=8 需 TPOT 净省 ~3.3ms/步。
- **不保证过,如实入档**(M5 先例);不过则北极星继续挂账,差距按可回收清单逐项归因。
- 附带义务:每项优化给出 A/B 计量账;docs 终账含"可回收清单 → 实际兑现"对照表。

## 3. 三段架构

```
段 0 尖刀(纯测量,不动生产)
  ├─ 腿 1 引擎内 GEMM 拆账(C=1/C=8 × graph/eager,按四类投影聚合)
  ├─ 腿 2 调参面扫(cublasLt workspace / bf16 归约旋钮 / CUBLAS 环境面)
  ├─ 腿 3 布局面扫(pre-transposed [K,N]、M-padding 7→8/16;孤立打分→引擎内复测)
  └─ 腿 4 步开销拆账(残差 1.70ms 主导项、memcpy 3/步、H2D 同步点、_commit_lens)
裁决门:可回收清单 Σms(C=8 口径)< 锚缺口的 70%(=2.31ms)→ 段 2 triton 立项;≥ → 休眠
段 1 落地(主攻):尖刀胜出项逐项进生产,每项 TDD + A/B 计量(<0.2ms 或劣化即回退)
段 2(条件立项)triton decode GEMM:生产默认 = cuBLAS/布局版/triton 按实测择优
```

## 4. 段 0 尖刀详细设计

新建 `scripts/m6_spike.py`(子命令分腿,风格照 `phase0_spike.py`;只产数据不动引擎)。

### 腿 1:引擎内 GEMM 拆账

- torch.profiler 包 decode loop(8B,`CUDA_VISIBLE_DEVICES=1`,C=1 与 C=8,graph 与 eager 各一遍);
- 按 kernel 名聚合四类投影(qkv/o/gate_up/down,名字 pattern 匹配)+ attention/norm/拷贝/其他;
- 输出:每类 in-situ µs/步、占比;对照孤立微基准同形状冷口径 → **差距定量与构成初判**(同名 kernel 不同时长?被插桩稀释?层间 L2 污染?时钟——`nvidia-smi dmon` 同窗记录);
- 裁决用途:决定段 1 的 GEMM 面走"调参/布局"还是需要段 2 kernel。

### 腿 2:调参面扫

Python 侧可动旋钮(如实列全,不虚构深度):`CUBLASLT_WORKSPACE_SIZE`(4MB→64/256MB)、`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`、`CUBLAS_WORKSPACE_CONFIG`、`CUDA_MODULE_LOADING`。孤立微基准 + 引擎内复测双口径;预期收益微小(R1),腿的价值是**排除**而非指望。

### 腿 3:布局面扫

- pre-transposed:权重预变换为连续 [K,N],调用侧 `torch.mm(x, w_t)`(NN 路径)vs 现行 `F.linear(x, w)`(TN 路径);
- M-padding:C=1(M=7→8/16)零填充-切片;
- 变体在孤立微基准打分(冷口径),**最优者必须引擎内复测**——孤立赢、引擎内被层间 L2 污染吞掉是 R2 主场景,复测不过即弃。

### 腿 4:步开销拆账

- 残差 1.70ms/步(graph wall 21.50 − GPU 19.81,M4.1 账)主导项逐一计时:ensure 循环 / 镜像填充 / `_commit_lens`(252 赋值/步)/ 调度 python;py-spy + profiler 双口径;
- memcpy 3/步(2 H2D + 1 D2H)内容与时延;ids/positions 的同步 H2D 点标注(non_blocking 候选,R3 约束:仅捕获外);
- 输出:每项 µs/步 + 可回收预估(压缩后残留)。

### 裁决门

段 0 产出**可回收清单**(逐项:机制、证据、预估 ms(C=8 口径)、风险)。**段 2 立项判定:Σ(腿 2/3/4 可回收)< 2.31ms(锚缺口 3.3ms × 70%)且腿 1 定因显示缺口主体在 GEMM 面 → 立项;否则休眠**(缺口主体在开销面时 triton 无益,剩余缺口如实入档)。清单同时是段 1 的施工顺序表。

## 5. 段 1 落地面

每项一个 commit,TDD + **A/B 计量协议**:改前/改后 decode loop C=1+C=8、graph+eager 各一遍(同 seed 同输入),收益 <0.2ms/步或任一口径劣化 → revert 并入档。token 流逐位一致为硬门。

1. **GEMM 调参/布局**(腿 2/3 胜出者):环境旋钮进 engine 构造(`from_pretrained`/server 两路径一致);布局层进 `weights.py` 新增函数(`pack_decode_weights(state, layout=...)` 类形态,原 `load_safetensors` 一行不动);qwen3.py **仅投影调用点**按 layout 分派,`layout=none` 逐字节旧行为。
2. **残差主导项压缩**(腿 4 排序第一者):按拆账结果施工;graph 捕获语义不动。
3. **non_blocking H2D**:ids/positions 上传异步化,**仅捕获外路径**(eager/镜像填充侧);graph 内 memcpy 语义不动(R3)。
4. **`_commit_lens` 向量化**:pager **有限解禁**(仅此函数,签名与语义不变);252 赋值/步 → 单次批量。

## 6. 段 2(条件立项):triton decode GEMM

- 目标形状:M ∈ [1,16],四类 (N,K):qkv (6144,4096)、o (4096,4096)、gate_up (24576,4096)、down (4096,12288);bf16 入出、fp32 累加;
- 手段:向量化加载(128-bit)、K 维 split(大 K 类)、N 维分 block;对标 cuBLAS 冷口径 889GB/s,**超 5% 以上才值得接入**;
- golden TOL 协议(M2/M3 先例:标定后写死,注释命令/日期);
- dispatch:装载时按 (M bucket, N, K) 实测择优三路(cuBLAS / 布局版 cuBLAS / triton),选择写进 config;**不达即收手,生产默认保持最优者**;
- 接入点:qwen3.py 投影调用点同一分派层(与段 1 布局共用)。

## 7. 保护面变更条款(本次解锁,最小侵入)

| 文件 | 条款 |
|---|---|
| `models/qwen3.py` | **最小侵入解锁**:仅四类投影 + lm_head 的调用形态(layout 分派);attention/rope/KV/norm 面一行不动;`layout=none` 时逐字节旧行为(byte-identical 硬门) |
| `cache/pager.py` | **有限解禁**:仅 `_commit_lens` 向量化,函数签名与语义不变 |
| `weights.py` | 增量:新增布局/pack 函数;`load_safetensors` 一行不动 |
| 其余 | 照旧禁改:golden m0–m5、`cache/kv.py`、`tests/triton_utils.py`、`ops/paged_attention.py`、`engine/graph_runner.py`、scheduler/sequence(本程无需再动) |

## 8. 风险表

| # | 风险 | 预案 |
|---|---|---|
| R1 | 调参面收益微小(cuBLAS 已近最优) | 腿 2 定位是排除;不恋战 |
| R2 | 布局收益孤立成立、引擎内被层间 L2 污染吞掉 | 引擎内复测为硬门,复测不过即弃 |
| R3 | non_blocking H2D 与 graph 捕获交互 | 仅捕获外路径;graph memcpy 语义不动 |
| R4 | qwen3.py/pager 解保护引发回归 | 最小侵入条款 + token 流逐位硬门 + 全套件零换金 |
| R5 | triton 不敌 cuBLAS | 择优收手,如实入档(学习目标已由过程兑现) |
| R6 | 锚不过(物理空间估满仍差口气) | M5 先例:如实入档,北极星继续挂账,差距逐项归因 |

## 9. Non-goals

- 连续到达负载 bench(M7 候选);
- TP=2 专项优化(GEMM 收益自然外溢,作副账如实记,不入锚);
- prefill 侧 GEMM/attention 优化;
- fp8/量化、投机解码、prefix caching;
- server 泵层改造(已被 T2 证伪为主要瓶颈)。

## 10. 验收协议与收口

- 段间门:段 0 裁决门(§4)、段 1 逐项 A/B 门(<0.2ms 回退)、段 2 立项门(§4 裁决门);
- 双轨仪式收口(同 M5 协议):轨 1 四腿(vLLM TP=1 / AcornInfer graph / eager / HTTP 缺省补充)主验;轨 2(TP=2)副账如实(GEMM 收益外溢量);
- docs `benchmarks.md` M6 节:尖刀账(四腿数据 + 裁决清单)、逐项 A/B 兑现表、双轨终账、达标率逐档、判定行(如实);tag/push 留用户;
- 测试底线:fast/slow 全绿零换金;token 流逐位;新 kernel(若立项)走 golden TOL。

## 11. 关联

- M5 终账:`docs/benchmarks.md` M5 节(双轨配对终账,commit `7a20a12`)
- 冷口径数据:M5 节"段 0 补充二:冷权重 GEMM 裁决口径"
- M4.1 遗留台账:残差/H2D/commit_lens 三项由本设计承接,其余(test_server flake)不在本程
