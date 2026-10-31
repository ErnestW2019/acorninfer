# M4.1 设计:图外残差清零 + prefill 批量化

日期:2026-10-31(设计获用户批准;承接 M4 验收未达标的归因入档,`docs/benchmarks.md` M4 节)

## 0. 背景与目标

M4 当日配对验收:92.9% / 83.0% / 72.2%(C=1/4/8,graph ÷ vLLM),归因清晰:

- decode GPU 忙时 19.84ms/步(cuBLAS GEMM 占 89%)≈ vLLM 每步 19.6ms —— **kernel 侧已近平价,这是吞吐地板**;
- 图外 CPU 残差 1.80ms/步(prepare/H2D 覆写/`_commit_lens`/逐行 argmax 同步);
- prefill 串行化:调度器一步准入全部但 `Engine.step` 逐条调 `Executor.prefill`([1, L] 前向)→ C=8 TTFT 146.7ms vs vLLM 81.7ms(当日实测)。

**M4.1 目标**:① 图外残差压到 ≤1ms/步;② TTFT 合批到 ~80ms 量级;③ 预期达标率 C=1 ~94-95% / C=4 ~91% / C=8 ~84%(预估,非承诺)。**验收姿态 = 如实重测入档**,北极星"逐档 ≥100%"维持挂账,不改锚(裁决留 M5 前后)。

## 1. 决定(用户已批)

1. **两段制带中间门**:段一(残差清零)完成后跑 C=1/4/8 中间实测入档,再开工段二(prefill 批量化);中途不请示,两段数字都进 docs。
2. **prefill 批量化深度 = 合批 GEMM + 逐行 attention**:大 GEMM 按 padded [N, Lmax] 合批;KV 写入与 attention 按行走现有 `view.update` + causal SDPA。**零保护文件改动**(kv.py/pager.py 的 scatter 不动,scheduler.py 准入已就绪无需触碰)。
3. chunked prefill(M5 前后另行评估)、TP=2(M5 正题)均 non-goal。

## 2. 段一:图外残差清零

### 2.1 argmax 单次同步

- `DecodeGraphRunner.decode_batch` 返回值改为 `st.logits[:B].argmax(dim=1).tolist()` 派生(B 次 D2H 同步 → 1 次);`last_logits` 契约不变。
- `Executor.decode_batch` eager 路径:**全序列 temp==0 时**同样 `argmax(dim=1).tolist()`;混合温度批保持逐行 `_sample`(保 Sampler 语义与 RNG 交织契约)。bench/golden 全 greedy,故锚点路径同样受益。

### 2.2 H2D 合一(扁平缓冲 + pinned 镜像)

- `_BucketState` 的六个静态缓冲(ids/positions/tables/lens/pages/slots)改为**按 dtype 两块扁平 GPU 张量的切片视图**:long 一块(ids+positions+pages+slots)、int32 一块(lens+tables)——切片共享存储,graph 捕获的地址即扁平缓冲地址,重放语义不变。
- 对应** pinned CPU 镜像**;`decode_batch` 每步只写 CPU 镜像(pinned),然后**每 dtype 一次 H2D**(long 一发、int32 一发,共 2 次;现为 6-7 次小 H2D)。默认同步 `copy_`,不做 non_blocking 微调(先测清基线)。
- `build_tables(views, page_size, device, *, lens_override=None, out=None)` 加 `out=` 形参(`acorninfer/ops/paged_attention.py`,非保护文件):传入时直接把块表写进 out 指定的 CPU 缓冲(即 pinned 镜像的 tables 区),跳过内部分配与独立 H2D。tables 的 `zero_()` 与 padding 行第 0 列 = scratch 的重写全部移到 CPU 镜像上做,随单一 H2D 上卡。
- `ensure_append_capacity`(per-row,纯 CPU)与 `_commit_lens` 保持不变。

### 2.3 中间门(段一收尾)

- C=1/4/8 graph+eager 实测(GPU 1)+ `scripts/profile_decode.py` 双账(带 `--cuda-graph`),入 `docs/benchmarks.md`。
- 判读:残差 ≤1ms 记达标;不达也**不阻塞段二**,如实记录并顺延归因。

## 3. 段二:prefill 批量化

### 3.1 模型层:`forward_prefill_batch`

```
Qwen3ForCausalLM.forward_prefill_batch(input_ids[N, Lmax], views, *, real_lens) -> logits[N, vocab] (float32)
```

- padding 值任意(取 0);`real_lens[i] = L_i`。
- positions:逐行 `arange(L_i)`,padding 位填 0;RoPE 按 padded positions 批量施加(逐 token 操作,pad 上的 cos/sin 无害——pad 永不写入 KV、永不被 attend)。
- embed / RMSNorm / qkv / o_proj / MLP 全部按 [N, Lmax] 合批(收益所在:GEMM 次数 ÷N)。MLP 等对 pad 行算垃圾值无妨——**逐行 attention 保证行间隔离**,垃圾不出行。
- 每层 attention:按行切片真实 token(`[:, :, :L_i]`),走与今日 prefill 完全相同的 `view.update(layer_idx, k_i, v_i)`(写入+gather)与 causal SDPA(`past_len==0` 分支,`is_causal=True`)。attention_backend / fused_ops 均不影响 prefill 分支(现状即如此),故本路径与后端开关解耦。
- 末层 norm 后只取各行 `real_lens[i] - 1` 位置的 hidden 过 `lm_head` → `[N, vocab]`。
- 原单序列 `forward(ids, view, num_logits_to_keep=1)` **一字不动**(M0 oracle 路径与 N=1 行为不变)。

### 3.2 执行层与接线

- `Executor.prefill_batch(seqs, views) -> list[int]`:pad + 调用模型 + 逐行 `_sample`(temp==0 argmax;temp>0 逐行走 `Sampler`,语义与现状逐位一致)。
- `Engine.step`:`len(prefills) == 1` 走原逐条路径(byte-identical,零回归面);`len(prefills) > 1` 构建视图列表后一次 `prefill_batch`。视图构建、`num_pages` 记账、`_record` 时序语义不变。

### 3.3 数值契约与测试

- **单测(快速,TDD)**:tiny 模型,N=3 变长,批量 per-row logits vs 逐条 `model(ids_i, view_i, num_logits_to_keep=1)` `assert_close`——容差按 `tests/triton_utils.py` 协议:从 INIT 起跑,按实测收紧写死并注释标定命令与日期(预期近逐位,不同 M 的 GEMM 归约序可能允许低比特)。
- **引擎级**:批量引擎 vs 逐条参考引擎,token 逐位(定种子,确定性)。
- **既有 fast 测试**(多 submit 的 engine 测试)自动切换到批量路径,全绿即行为保持的证据;golden 四门的"并发 vs 顺序参考"对拍继续兜底——**prefill 合批改变 GEMM 形状,可能引发边缘 token 翻题:授权按 M0/M2/M3/M4 先例换题并记录 margin;系统性错位(大面积)≠ 换题能救 = BLOCKED 调查**。
- slow 套件计数不变(17);golden m0-m4 原样全绿是硬门。

## 4. 收尾:最终配对仪式

- 当日配对(vLLM GPU 0 + 本引擎 graph/eager GPU 1,同窗口),C=1/4/8 三腿;docs 追加 **M4.1 节**:段一中间账、段二后数字、配对表、达标率、判定行(如实)、归因更新(残差实测、TTFT 实测、GEMM 地板引用 M4 节)。
- 复盘照例交用户;`m4-acceptance` tag 仍不打(北极星未过);M5 前的裁决(北极星改锚与否)由用户带着 M4.1 数字做。

## 5. 保护条款

- **禁改**:`tests/test_golden_m0/m1/m2/m3/m4.py`、`acorninfer/cache/kv.py`、`acorninfer/cache/pager.py`、`acorninfer/engine/scheduler.py`、`acorninfer/engine/sequence.py`、`tests/triton_utils.py`。
- **限定改动**:`acorninfer/ops/paged_attention.py` 仅 `build_tables` 加 `out=` 形参(默认 None = 现行为逐位不变);`acorninfer/models/qwen3.py` 仅新增 `forward_prefill_batch`(原 forward/forward_decode 不动);engine/executor/graph_runner 按设计改。
- 运行时禁 import transformers;实测数字才可入 docs;容差只准用 TOL 协议。

## 6. 风险与预案

| 风险 | 概率 | 处置 |
|---|---|---|
| argmax 合批与逐行在 CUDA 上对并列最大值返回不同索引 | 极低(fp32 logits) | token 逐位门兜底;真翻题则回退逐行,记录 |
| pinned 分配失败 / 平台不支持 | 低 | 回退非 pinned(收益略降),docs 记录 |
| prefill 合批引发 golden 边缘翻题 | 中低 | 换题协议(已授权);系统性错位 = BLOCKED |
| 批量 GEMM 使 N=1 之外的 prefill 数值漂移伤 slow 门 | 低 | assert_close 门先拦;golden 兜底 |
| 残差压不到 ≤1ms | 中 | 不阻塞,如实记录并归因(ensure/调度 python 占比) |
