# M5 设计 spec:chunked prefill + TP=2(打包里程碑)

- 日期:2026-10-31
- 状态:设计定稿,待用户审阅
- 前置:M4.1 收官(`d7948e1`),当日配对达标率 93.4 / 90.2 / 83.5%(C=1/4/8,graph÷vLLM)
- 决策记录:①范围打包(chunked prefill + TP=2,用户定);②验收锚 = **双轨配对**(TP=2 对 vLLM TP=2 + 保留 TP=1 历史线,用户定);③**scheduler.py 解保护**(最小侵入,用户定);④混合步取 A1 共步双前向;⑤extend 用 SDPA + 显式 mask;⑥allreduce 先 NCCL,段 0 尖刀裁决。

## 0. 问题定义

M4.1 后剩余差距(当日配对,C=8):吞吐 83.5%,TTFT p50 145ms vs vLLM 81ms(1.8×)。decode TPOT 已收敛纯 decode GPU 地板(19.95 vs 19.8ms)。差距集中在:

1. **prefill/TTFT**:一次 [8×128] 批量前向 ≈ 145ms(~98 TFLOPS 有效);且 146.7 vs 81.7 的 TTFT 对比存在**方法学不对等**(引擎 in-process 同刻提交 vs vLLM HTTP 流式到达错峰),真构成未钉;
2. **GEMM 地板**:19.8ms/步的 decode GPU 时间被单卡算力封顶;TP=2 可切半,但本机 GPU0-GPU1 为 `NODE` 拓扑(同 NUMA 走 PCIe 主桥,**无 NVLink**),收益边界需实测。

M5 双拳:chunked prefill 攻 TTFT 与连续到达下的 TPOT 交互,TP=2 攻吞吐地板。

## 1. 验收锚(双轨,当日配对)

- **主轨(新锚)**:AcornInfer TP=2 vs **vLLM TP=2**(`--tensor-parallel-size 2`,同双卡,HTTP 对 HTTP),逐档 tok/s + TTFT/TPOT;北极星换锚为"TP=2 对 vLLM TP=2 逐档 ≥100%",挂账。
- **副轨(历史线)**:AcornInfer TP=1 vs vLLM TP=1,要求 ≥ M4.1 收官数字(零回归)。
- TTFT 口径:HTTP 对 HTTP(`bench_remote.py` 打 AcornInfer server 与 vLLM 各一),in-process bench 只作吞吐回归线,不再产生跨方法学 TTFT 结论。
- tag:M5 完成不自动打 tag(沿用 M4.1 姿态,验收由用户裁决)。

## 2. 段 0:可行性尖刀(先行,带裁决门)

全部实测入档 `docs/benchmarks.md` M5 节开篇:

1. **NCCL allreduce 微基准**(双卡,8KB / 57KB / 1MB / 8MB,graph 外 + **graph 内捕获**):每档延迟/带宽。**裁决门**:decode 尺寸(57KB)单次 >50µs 或 graph 内不可捕获 → 停,带数字回报用户重谈 TP 方案(custom one-shot 内核或降范围)。
2. **TP 形状 GEMM 微基准**:[7,4096]×[4096,2048](列并行半宽)等 decode/prefill 关键形状,验证半宽 GEMM 每卡效率不塌方(≥ 单卡全宽的 40%)。
3. **prefill 账目**:145ms 构成 profiler 拆账(GEMM/attention/elementwise)+ server HTTP 腿搭通(bench_remote → AcornInfer server),拿到 in-process vs HTTP 的 TTFT 差——区分"引擎慢"与"到达错峰方法学差"。
4. 段 0 结论页:TP=2 收益预期表(decode wall、prefill TTFT,实测外推)+ chunked prefill 的预期收益线。**入档后才开工段 1。**

## 3. 段 1:chunked prefill(单卡,TP 前落地)

### 3.1 数据与状态

- `Sequence` 增 `prefill_cursor: int = 0`(已写 KV 的 prefill token 数;`num_tokens` 语义不变 = len(all_ids))。sequence.py 例外 = 仅此一字段。
- scheduler.py 解保护,`schedule()` 改造:
  - 返回 `(prefill_chunks, decodes)`,`prefill_chunks: list[(seq, start, length)]`(start=cursor,length=本轮预算分配给该序列的 chunk 长度);
  - **token 预算**(默认 512,CLI 可配):FCFS 顺序切,预算耗尽即止;预算 ≥ 待 prefill 总 token 数时每序列 chunk = 全 prompt;
  - **deficit 修正**:mid-prefill running 序列本轮写 length 个 token(非 1),页预留按真实 chunk 需页算;准入簿记(need/reserve/watermark)语义不变;
  - **抢占语义不变**:mid-prefill 被抢占 = 全释放 + `prefill_cursor=0`,重准入后从头 prefill(与现有 recompute 一致)。

### 3.2 模型层 extend 前向

- 新增 `forward_extend(input_ids [N, Lc], views, *, starts) -> [N, vocab]`:q 为 chunk 多 token,对已含 start 个 token 的 KV 做 **bottom-right causal-offset** attention(SDPA + 显式 bool mask;is_causal 不适用因 past>0);
- positions = arange(start, start+Lc) 逐行;KV 经既有 `view.update` 追加(分页/页表零改动);末层取每行末位过 lm_head;
- **byte-identical 条款**:`cursor==0` 的 chunk **不走** forward_extend——继续走既有 `prefill`(N=1)/`forward_prefill_batch`(N>1)原路径,零回归面;forward_extend 只服务 cursor>0。

### 3.3 Engine 混合步(A1 共步双前向)

- `step()`:decodes 照旧走 graph/eager decode_batch;prefill_chunks 按 cursor==0 / >0 分流到原路径 / forward_extend;两类前向同步先后执行(两次发射),簿记(`_record`/num_pages/`_views`)按序保持;
- 采样:chunk 与 decode 均逐行 `_sample`,语义不变。

### 3.4 段 1 验收门

- 既有全套件零翻车(预算默认 ≥ bench prompt 长度 128 时行为与 M4.1 等价;`--chunked-budget` 调小触发真 chunk 路径的新单测);
- 新单测:变长 chunk 切分对拍(逐条全量 prefill vs 分 chunk extend,logits TOL 协议标定 + token 逐位)、预算边界、抢占后 cursor 归零、混合步簿记;
- server HTTP 腿 + bench_remote 连通,TTFT 实测对照表入档(中间账)。

## 4. 段 2:TP=2(双卡)

### 4.1 切分方案(标准 Megatron)

| 权重 | 切法 | 每卡形状(全卡 → 半卡) |
|---|---|---|
| qkv_proj | 列并行(q 16/32 头,kv 4/8 头) | [6144, 4096] → [3072, 4096](q 2048 + k 512 + v 512) |
| o_proj | 行并行(**入半**,出全求和) | [4096, 4096] → [4096, 2048],partial 和 → allreduce |
| mlp gate/up | 列并行 | [24576, 4096] → [12288, 4096](gate/up 各 6144) |
| mlp down | 行并行(**入半**,出全求和) | [4096, 12288] → [4096, 6144],partial 和 → allreduce |
| embed_tokens | 词并行(vocab 行切;查表后 allreduce) | [151936, 4096] → [75968, 4096] |
| lm_head | 词并行(vocab 列切;partial logits → all-gather) | [151936, 4096] → [75968, 4096] |
| RMSNorm/rope/qk_norm | 复制 | 不切 |

allreduce 点:每层 2 次(o 后、mlp down 后)× 36 层 = 72 次,加 embed 查表后 1 次;lm_head 走 all-gather。decode 尺寸 57KB(B=7×4096×2B),prefill(1024 tok)8MB——与 §2 尖刀档位一致。KV cache 按 kv 头切:**每卡 PagedKVCache 只存本 rank 的 4/8 头**,pager/allocator 逻辑不变(形状参数减半),KV 零通信;页表分配决策两卡一致(见 4.3)。

### 4.2 进程与通信

- `torch.distributed` NCCL,world_size=2(init 由 `--tp 2` / 环境变量触发);TP=1 = world_size 1,**TP 代码路径全部休眠,单卡行为逐字节不变**(含 CUDA graph);
- allreduce 用 `dist.allreduce`(段 0 尖刀裁决其延迟与 graph 可捕获性;不足再议 custom one-shot,本期默认不做);
- 采样:两卡 all-gather partial logits 拼 full [B, vocab] → **rank0 采样,broadcast token** → 两卡采样路径无 RNG 分叉;
- CUDA graph:TP decode 捕获含 NCCL allreduce(段 0 验证;失败走风险表 R1)。

### 4.3 控制流:SPMD 等身复制

两卡跑同一份引擎循环(scheduler/allocator/页表决策输入全同 → 决策全同),engine/model 代码不分支化;仅 sampler 侧按 4.2 广播。server/bench 进程模型:rank0 持 API,rank1 为 worker(测试用 torchrun/two-proc harness;pytest 以 `gpu_tp` marker 标注,**双卡测试跑前双卡查残留**)。

### 4.4 纪律变更

- TP 测试/运行需 `CUDA_VISIBLE_DEVICES=0,1`(双卡独占检查);单卡测试照旧 GPU 1;
- 验收仪式窗口:vLLM TP=2 与 AcornInfer TP=2 都占双卡,**串行**跑,同日完成。

## 5. 段 3:验收与收口

1. 双轨当日配对(HTTP 对 HTTP):vLLM TP=2 vs AcornInfer TP=2 × C=1/4/8(tok/s、TTFT、TPOT);vLLM TP=1 vs AcornInfer TP=1 ≥ M4.1 数字(零回归);
2. TP=1 全套件:fast 106+ / slow 17 全绿零换题(TP 代码休眠保证);
3. TP=2 冒烟:golden 风格小样(双卡,token 对拍 TP=1 vs TP=2,TOL 协议);
4. docs M5 节:段 0 尖刀账、段 1 中间账、双轨配对表、判定行(如实);北极星挂账状态记录。

## 6. 保护条款更新

| 文件 | 状态 |
|---|---|
| `engine/scheduler.py` | **M5 解保护**(chunk 预算/deficit 修正;准入、水位、抢占语义逐字保留) |
| `engine/sequence.py` | **M5 例外** = 仅增 `prefill_cursor` 字段 |
| `cache/kv.py`、`cache/pager.py` | 照旧禁改(KV 切头 = 形状参数,零逻辑改动) |
| `tests/triton_utils.py`、`tests/test_golden_m0..m4.py` | 照旧禁改(TP=1 下 golden 原样跑) |
| 容差 | 只准 TOL 协议(TP 对拍/extend 对拍标定后写死) |

## 7. non-goals

- A2 真混合批单前向(piecewise graph)——留 M6+;
- triton extend/prefill 分页内核(attention prefill 维持 SDPA 系);
- custom one-shot allreduce 内核(除非段 0 裁决 NCCL 不可用);
- pipeline 并行 / expert 并行 / FP8;
- max_position 扩展、prefix caching。

## 8. 风险表

| # | 风险 | 预案 |
|---|---|---|
| R1 | NCCL allreduce 不可入 graph 或延迟 >50µs | 段 0 裁决门,停工回报(不硬闯);备选:custom one-shot 内核立项或 TP 范围重谈 |
| R2 | PCIe 带宽不及预期,prefill TP 收益趋零 | TP 收益主要锚 decode;prefill 侧 chunked 独立收益;双轨照实报 |
| R3 | SPMD 决策漂移(两卡调度不一致) | 决策输入全同 + 单测双卡对拍页表/输出;漂移即 BLOCKED(真 bug) |
| R4 | 半宽 GEMM 效率塌方(<40%) | 段 0 微基准先钉;塌方则 TP 收益重估(吞吐账如实) |
| R5 | chunked 改动翻 golden(TP=1 路径被误染) | byte-identical 条款(cursor=0 走原路径)+ 既有全套件零翻车为段 1 硬门 |
| R6 | server HTTP 腿与 vLLM 的 bench_remote 口径差(tokenizer/长度) | 段 0 对齐同一 prompt 组与采样参数,差异入档 |
| R7 | 双卡资源竞争(resource_pool 占卡) | 双卡测试跑前双查;被占即 BLOCKED 等窗 |

## 9. 里程碑任务骨架(writing-plans 细化)

T0 段 0 尖刀 + 裁决 → T1 scheduler/sequence/extend/engine(单卡,TDD)→ T2 段 1 中间门(HTTP TTFT 对照)→ T3 TP 基建(dist init/分片加载/TP 模型/eager decode)→ T4 TP graph + extend TP + SPMD 接线 → T5 回归(TP=1 全绿 + TP=2 对拍冒烟)→ T6 双轨配对仪式 + docs 收口。
