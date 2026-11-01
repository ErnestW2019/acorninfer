# M7 设计:weight-only 量化(W8A16 → W4A16 双段门控)

> 状态:设计已获批(A1–A6 全按推荐定案,2026-11-01);本文件为定案文本。
> 前置:M6 终账(`docs/benchmarks.md` M6 节,HEAD `d55f930`)。

## 0. 背景与物理依据

M6 以物理级证据关闭了 bf16 decode 的 kernel 工程空间:四类 GEMM 已在自身访问模式的读上限 99–100% 运行,cuBLAS ≈ 物理上限(~950GB/s 有效权重流),triton/布局/调参三面实证排除(赢面 +1.1~2.5% < 5% 阀)。**差距的根源是每步必须搬运的字节数**,不是搬得不快。

decode 权重流账(8B,bf16,每步):四类投影 13.89GB + lm_head 1.245GB ≈ **15.13GB/步**。这是 decode TPOT 的主导项(proj 16.45ms + lm_head 1.32ms ≈ GPU 步 19.81ms 的 90%)。

唯一翻墙手段:**少搬字节**。weight-only 量化把权重压到 int8/int4、激活保持 bf16,dequant 融入 GEMM——这正是 triton 的主场(fused dequant+GEMM 无 cuBLAS 竞品,M6"cuBLAS stands"的结论在量化域不适用),M6 入库的 `ops/decode_gemm.py` 骨架(autotune 1.16s warm、graph-capture 安全、TOL 协议)直接复用。

## 1. 目标与验收锚

主锚(graph,in-process 口径,与 M5/M6 验收口径一致):

| 锚 | 门 | 推导 |
|---|---|---|
| **W8A16 C=8 TPOT ≤ 14.0ms** | 硬门 | GEMM 面 7.56GB@η≥850GB/s = 8.89ms + 非 GEMM 2.04 + host 0.19 ≈ 11.1ms GPU → TPOT ≈ 11.9;锚留 ~18% 余量 |
| **W8A16 C=1 TPOT ≤ 10.5ms** | 硬门 | 现值 17.95 − (16.45+1.32) + 8.9 ≈ 9.1ms(GEMM 面带宽绑定,批不敏感;尖刀复核) |
| **W4A16 C=8 TPOT ≤ 10.5ms** | 段 B 硬门 | 4.03GB@η≥750GB/s = 5.37ms + 2.2 ≈ 7.6 GPU → TPOT ≈ 8.1;W4 unpack 开销压 η,余量加厚(锚推导取 η=0.75 保守值;段 B 开工资格门另取 0.80,两者口径独立) |

**锚修正条款(一次)**:段 0 微基准若实测 η 与推导假设(W8 0.85 / W4 0.75)偏离 >10%,允许一次修正锚并在 docs 记录原推导与新锚,之后写死不再动。

副锚(如实入档,不设门):对 vLLM bf16 的 tok_s 比值逐档(预估 W8 后 C=8 ~150%、C=1 ~180%,非承诺)。

精度锚:困惑度增量 vs bf16 基线——W8 ≤ +1%,W4 目标 ≤ +5%(实测定档,超标照实入档,门控裁决是否保留 W4)。

## 2. 方案架构(两段门控)

### 段 A:W8A16(全链落地)

- **量化格式**:per-channel **对称** RTN(通道 = 输出行):`scale[n] = max|w[n,:]|/127`(clamp 下限 ε=1e-8),`q = round(w/scale)` 存 int8 [N,K],scale 存 fp16 [N]。免校准数据(RTN 只看权重自身,这是选它而非 GPTQ 的首要原因)。
- **kernel**:`ops/quant_gemm.py`(新)——triton dequant-GEMM:tile 内 load int8 → 转 fp32 → ×scale[n] 广播 → `tl.dot`(fp32 累加,bf16 in/out)。复用 decode_gemm 的 autotune 网格方法、graph-capture 纪律(零 host 同步,autotune 于装载/warmup 期完成)、TOL 对拍法。M 两个 regime:decode M≤16(BLOCK_M=16)与 prefill 大 M(独立 autotune 桶)。
- **分派**:`--quant w8a16` 时投影+lm_head 走量化 kernel;`--quant none`(缺省)逐字节旧路径硬门不动。qwen3.py 调用点分派形态沿用 linear_dispatch 先例(第三次解锁)。
- 段 A 收口即段 B 门:**主锚过 + kernel η ≥ 80% roofline(权重字节口径,微基准)+ 精度 ≤ +1%** → GO;任一不过 → W4 挪 M8,如实记录死因。

### 段 B:W4A16(门控开工)

- **量化格式**:per-group **非对称** RTN,g=128:每 (输出通道, 组) 存 scale fp16 + zp fp16(`scale=(max−min)/15, zp=min`),`q = clamp(round((w−min)/scale), 0, 15)`;打包 uint8 [N, K/2](双 nibble)。K∈{4096,12288}、N∈{4096,6144,24576,151936} 全为 128 倍数,组对齐零特例。
- **kernel**:unpack(移位/掩码)→ 反量化(×scale + zp)→ tl.dot;组索引 = K-block 内偏移/g,要求 BLOCK_K 整除 g128(autotune 网格约束)。split-K 需求评估照 M6 T5 先例(down 类 N=4096 grid 不足)。
- 段 B 复用段 A 全部基建(分派/装载/ppl/仪式),新增面只有格式与 kernel 变体。

## 3. 精度评估基建(自研困惑度 harness)

- `scripts/perplexity.py`(新):wikitext-2-raw-v1 test split,本地缓存 `data/wikitext-2-raw/`(下载命令原样入 docs;首选 Smerity zip 源,备选 HF parquet,取活者并如实记录——R7);分词用自家 `Tokenizer`(运行时禁 transformers 纪律不破);滑窗 window=512 stride=256,输出 token 数、NLL 总额、ppl(可复算)。
- 流程:bf16 基线先标定(含窗口/stride 记录)→ W8 → W4,三配方同流对齐;数字照实,门在 §1。
- kernel 数值正确性独立于 ppl:**fp32 dequant 参考实现对拍**(torch 侧 reference:dequant 后 matmul),TOL 在 `tests/test_quant_gemm.py` 内新标定(命令/日期注释写死;`tests/triton_utils.py` 照旧只读复用)。
- token 流 parity:量化 vs bf16 为**不同数值域**,不做逐位断言;用 margin 门协议(M5 先例)在 tiny + 8B 抽查漂移量级并如实入档。

## 4. 权重与装载

- 装载期现场量化:`from_config`/`from_pretrained`(TP=1 路径)读 bf16 权重后 walk 模型,对四类投影 + lm_head 执行 RTN(纯 torch 数学放 `acorninfer/quant/quantize.py`,无 triton 依赖,便于单测与复用);**单存储**,bf16 原权重释放。
- 装载耗时预算 <120s(读 16GB + 量化写),超时如实入账;**不做磁盘量化缓存**(non-goal)。
- `--quant` 与 `--tp 2` **显式互斥**:同给即构造期报错(不做静默错误行为;TP×quant 留台账)。
- server:`--quant {none,w8a16,w4a16}` + `--quant-group 128`(仅 w4 生效);两条 `_build_engine` 构造路径透传,透传测试照 chunked-budget 风格。

## 5. 保护面

禁改照旧:golden m0–m5、`cache/kv.py`、`cache/pager.py`(本程无解禁需求)、`engine/scheduler.py`、`engine/sequence.py`、`engine/graph_runner.py`、`ops/paged_attention.py`、`tests/triton_utils.py`。

本程解锁/新建:
- `models/qwen3.py`:投影 + lm_head 调用点分派(第三次,linear_dispatch 形态;attention/rope/KV/norm 零触碰);
- `weights.py`:仅增量;
- `engine/config.py` + `engine/engine.py` + `server.py`:quant 配置面;
- `ops/decode_gemm.py`:骨架复用与分派面增量,**既有 bf16 kernel 与其 17 测试行为零变化**;
- 新建:`acorninfer/quant/`(quantize.py 等)、`ops/quant_gemm.py`、`scripts/perplexity.py`、`tests/test_quant_*.py`。

## 6. 风险表

| # | 风险 | 对策 |
|---|---|---|
| R1 | dequant ALU 开销侵吞带宽收益(尤其 prefill 大 M) | η 门(80%)在段 B 门与锚修正条款内;prefill 回退如实入账不粉饰 |
| R2 | W4 打包/组对齐 bug | fp32 参考 TOL 对拍 + 组对齐单测(K、N 均 128 倍数,零特例仍要钉住) |
| R3 | RTN W4 精度不足(+5% 不达) | 如实入档;门控裁决 W4 去留;GPTQ 留 M8 候选,不硬撑 |
| R4 | W8 对称 per-channel 遇离群通道 | 一次改非对称复测的机会(标定后写死);仍超 +1% 则如实 |
| R5 | 装载耗时超预算 | 预算 120s,超时如实入账;不缓存 |
| R6 | graph capture 兼容(scale 索引/guard) | M6 T5 已证形态;段 0 捕获实测复验(含 SPLIT_K 路径若启用) |
| R7 | wikitext 源失活 | 双源备选(zip / HF parquet),取活者如实记录 |

## 7. 非 goal

- GPTQ / AWQ(算法域另立里程碑);KV cache / embed / norm 量化;TP×quant(显式互斥);磁盘量化缓存;在线 checkpoint 导出/与 GPTQ 检查点互认;W4 无门控的承诺;MoE/多模态。

## 8. 仪式与收口

- **段 0(微基准先行)**:W8/W4 变体孤立微基准(逐变体 η 报告,BLOCK/流水线小网格)→ 锚修正条款复核 → 段 A 施工。
- **段 A 收口即段 B 门**(§2);段 B 完成或如实挪 M8。
- **终仪式(轨 1,GPU 0 照 M5/M6 配方)**:四腿 HTTP 对照 = vLLM bf16 / AcornInfer none / w8 / w4,各 C=1/4/8;none 腿应复现 M6 终账(零漂移硬证);达标率逐档 + TPOT 锚判定 + ppl 三行 + kernel η 表(微基准 vs 引擎内)。轨 2 免跑(TP×quant 互斥,无外溢面)。
- **判定行(如实条款)**:锚过/不过照写;段 B GO/NO-GO 与死因;精度账完整呈现;ppl 数字不达标不修饰。tag/push 留用户。
- M8 台账预埋:GPTQ/AWQ、TP×quant、KV 量化、磁盘量化缓存、双存储 down_t、连续到达 bench、半宽 GEMM、custom allreduce、rank0 快失败。
