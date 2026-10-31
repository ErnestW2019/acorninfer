# M5 chunked prefill + TP=2 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 两拳收官差距——chunked prefill 攻 TTFT 与连续到达下的调度交互,TP=2(双卡 PCIe)攻吞吐地板;双轨当日配对(TP=2 对 vLLM TP=2 + TP=1 历史线 ≥ M4.1)。

**Architecture:** 段 0 可行性尖刀(NCCL 微基准 + 半宽 GEMM + prefill 拆账,带裁决门)→ 段 1 chunked prefill(scheduler 原生 chunk 状态 + `forward_extend` SDPA offset-mask + Engine 共步双前向;cursor==0 走原路径 byte-identical)→ 段 2 TP=2(Megatron 切分,KV 按 kv 头切每卡 4/8,SPMD 等身复制,rank0 采样广播)→ 双轨仪式。

**Tech Stack:** torch.distributed NCCL / 既有 CUDA graph / SDPA;M0-M4.1 全部基建。

**Spec:** `docs/superpowers/specs/2026-10-31-m05-chunked-prefill-tp2-design.md`

---

## 全局纪律(每个任务都适用,实现者必读)

- **计划代码是草稿,测试才是契约。** 草稿错以代码现状与数值事实为准修,报告逐条记录偏离(M4/M4.1 先例:草稿 bug 由 TDD/审查逮出是常态,不是事故)。
- **保护文件(更新后)**:`tests/test_golden_m0..m4.py`、`acorninfer/cache/kv.py`、`acorninfer/cache/pager.py`、`tests/triton_utils.py` 照旧禁改;**M5 解保护例外**:`acorninfer/engine/scheduler.py`(chunk 预算/deficit 修正;准入、水位、抢占语义逐字保留)、`acorninfer/engine/sequence.py`(仅增 `prefill_cursor` 字段)。
- **环境纪律**:单卡测试照旧 `CUDA_VISIBLE_DEVICES=1` + `.venv/bin/` 前缀;**TP(双卡)测试/运行需 `CUDA_VISIBLE_DEVICES=0,1`,跑前 `nvidia-smi` 双卡都查**(共享机 resource_pool);代码写 `cuda:0`(可见性映射)。**前台长 timeout**(单卡 ≥600000;TP 双卡/8B ≥1800000),禁 `&` + 轮询。
- 运行时禁 import transformers;实测数字才可入 docs;容差只准 `tests/triton_utils.py` TOL 协议(标定后写死)。
- 当前基线:fast **106 passed, 17 deselected**;slow **17 passed**。HEAD 见 `git log`(spec 提交 `53726b1` 之后)。
- TP 代码 **world_size=1 时全部休眠**:单卡行为(含 CUDA graph、golden)逐字节不变是 R5 风险的硬门,每个 TP 任务的全套件回归都在钉它。

---

### Task 0: 段 0 可行性尖刀(裁决门)

**Files:**
- Create: `scripts/phase0_spike.py`(NCCL allreduce 微基准 + 半宽 GEMM 微基准)
- Create: `scripts/profile_prefill.py`(prefill 前向 profiler 拆账)
- Modify: `docs/benchmarks.md`(M5 节开篇:尖刀账)

- [ ] **Step 1: phase0_spike.py**(单文件两段;torchrun 双卡跑)

```python
"""段 0 尖刀:NCCL allreduce 与半宽 GEMM 微基准。用法:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 scripts/phase0_spike.py
输出逐档延迟/带宽/TFLOPS 行(stdout),数字原样进 docs。"""
# allreduce:尺寸档 [8KB, 57KB, 1MB, 8MB];bf16;warmup 10 + 计 50 次,
# torch.cuda.Event 计时;每档两跑:graph 外 eager 循环 + **graph 内捕获**
# (torch.cuda.CUDAGraph 捕获 100 次 allreduce 的图,replay 计均摊延迟)。
# 图内不可捕获 → 打印 "GRAPH_CAPTURE_FAIL" 并继续(这是裁决输入,不是异常)。
# GEMM:decode 半宽 [7,4096]x[4096,2048]、[7,4096]x[4096,3072];
#       prefill 半宽 [1024,4096]x[4096,2048];对照全宽 [7,4096]x[4096,4096];
# 各计 TFLOPS。判定线:半宽 ≥ 全宽 TFLOPS 的 40%。
```

- [ ] **Step 2: profile_prefill.py**(单卡 GPU 1,torch.profiler 包 `forward_prefill_batch`)

8B 模型、[8×128] 批量 prefill × 10 步,输出:总时长、GEMM/attention/elementwise/其他自用时间占比、有效 TFLOPS。对照 vLLM 81ms 与 145ms 的差距定位(引擎慢 or 方法学差)。跑前 `nvidia-smi` 查 GPU 1。

- [ ] **Step 3: HTTP 方法学对齐腿**

本引擎 server(GPU 1,`acorninfer.server`,端口 8200)+ `scripts/bench_remote.py --port 8200 --concurrency 1 4 8`:拿到 HTTP 口径 TTFT/tok/s,与 in-process 同日数字并列 → 量化"到达错峰"贡献。跑后 PID kill(禁 pkill -f)+ nvidia-smi 确认回落。

- [ ] **Step 4: docs 追加 M5 节开篇**(尖刀账:allreduce 表(graph 内/外)、GEMM TFLOPS 表、prefill 拆账、HTTP vs in-process 对照;只写实测)

- [ ] **Step 5: 裁决门(控制器执行,实现者停在报告)**

实现者报告后**停**,控制器按 spec §2 裁决:57KB 图内 allreduce >50µs 或 GRAPH_CAPTURE_FAIL → 回报用户重谈 TP 方案;半宽 GEMM <40% → R4 预案(收益重估);否则 GO,续 Task 1。

- [ ] **Step 6: Commit**

```bash
git add scripts/phase0_spike.py scripts/profile_prefill.py docs/benchmarks.md
git commit -m "feat(scripts): phase-0 spike — NCCL/half-width GEMM/prefill accounts (M5 Task 0)"
```

---

### Task 1: chunked prefill(单卡,TDD)

**Files:**
- Modify: `acorninfer/engine/sequence.py`(+`prefill_cursor` 字段,仅此一处)
- Modify: `acorninfer/engine/scheduler.py`(schedule() → (chunks, decodes);预算/deficit/抢占 cursor 归零)
- Modify: `acorninfer/models/qwen3.py`(+`forward_extend` 三层;原路径零改动)
- Modify: `acorninfer/engine/executor.py`(+`extend`)
- Modify: `acorninfer/engine/engine.py`(step 消费 chunks)
- Test: `tests/test_scheduler.py`(如无则建)、`tests/test_forward_decode.py`、`tests/test_engine.py`

- [ ] **Step 1: scheduler 失败测试**(预算切分/deficit/抢占,新建 tests/test_scheduler.py 照既有测试风格)

```python
def test_schedule_chunk_budget_splits_fcfs():
    """budget=16、三条 prompt 10/10/10:第一条拿 6(16-10 已被先到的分完?——
    以实现为准的精确切分断言);断言:FCFS 顺序、预算不超发、chunk=(seq,start,len)
    三元组、cursor 推进语义(start=seq.prefill_cursor)。"""

def test_schedule_deficit_counts_chunk_pages():
    """mid-prefill 序列(已写 4/10 token,chunk 本轮写 6)本轮需页按 6 算而非 1:
    构造 free 块卡在边界,断言不 OOM(旧公式会漏 Reserve → BlockExhausted)。"""

def test_preempt_mid_prefill_resets_cursor():
    """抢占 mid-prefill → _release + prefill_cursor 归零;重准入后 chunk 从 0 起。"""
```

(草稿注:三个测试都是红——当前 schedule() 返回 (prefills, decodes) 且无预算概念;`Scheduler.__init__` 增 `chunked_budget: int | None = None` 形参,None = 无预算(等价旧行为,返回的 chunks 全为全量 prompt)。)

- [ ] **Step 2: 跑红** → `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_scheduler.py -v`(FAIL:schedule 无该签名)

- [ ] **Step 3: scheduler 实现**(草稿;准入/水位/抢占语义逐字保留,只动 chunk 相关)

```python
    def schedule(self) -> tuple[list[tuple[Sequence, int, int]], list[Sequence]]:
        """返回 (本轮 prefill chunks [(seq, start, len)], 本轮 decode);副作用同旧。

        预算分配:FCFS 先到先得,覆盖 running 中 mid-prefill(cursor < len(prompt_ids))
        与 waiting 新准入;chunk 后 deficit 按真实 chunk 需页算(每 chunk 需
        pages(start+len) - pages(start) —— mid-prefill 首步 = 尾页已含在准入簿记,
        实际追加 = pages(累计写) - num_pages,注意与 num_pages 同步时机)。
        decode 判据:cursor == len(prompt_ids)(prefill 完)且已生成首 token 前
        仍属 decode 首步(与旧 prefills/decodes 分工一致:本轮无 chunk 的 running)。"""
```

(草稿关键点,实现时按现状推导:①预算先分 mid-prefill(running 顺序 = FCFS)再分新准入;②deficit = Σ(mid-prefill chunk 追加页) + Σ(decode 序列至多 1 页) —— 原公式 `num_tokens > num_pages*page` 只对 prefill 完成态成立,mid-prefill 换成按 chunk 算;③抢占 `_release` 处 `victim.prefill_cursor = 0`;④无预算时行为 = 旧:每 waiting 序列一条全量 chunk,deficit 公式退化回原式。)

- [ ] **Step 4: forward_extend 失败测试**(tests/test_forward_decode.py 追加)

```python
@pytest.mark.gpu
def test_forward_extend_chunked_matches_full_prefill():
    """分块 == 一次到位:prompt 12 tokens,先 forward_prefill_batch 前 5,
    再 forward_extend 后 7(starts=[5]);对照整条 forward_prefill_batch:
    末位 logits TOL 协议标定一致;两腿 KV 逐层 _lens/table 逐位一致
    (同 view.update 追加语义);再验 N=2 变长 chunk 行间零串扰。"""
```

- [ ] **Step 5: forward_extend 实现**(草稿;qwen3.py 三层各加一法,原路径零改动;rope 第 4 份内联拷贝 —— docstring 沿用 Task 4 风格加"改公式需 N 处同步"警告)

```python
    # Qwen3ForCausalLM.forward_extend(self, input_ids [N, Lc], views, *, starts, chunk_lens)
    #   pos: 每行 arange(starts[i], starts[i]+chunk_lens[i]);展平一次 rope
    #   hidden: embed → 层循环 forward_extend → norm → 每行 (starts[i]+chunk_lens[i]-1) 位过 lm_head → .float()
    # Qwen3Attention.forward_extend(..., starts, chunk_lens):
    #   qkv GEMM [N, Lc] → q_norm/k_norm → rope → 逐行 b:
    #     k_full, v_full = view.update(layer, k[b:b+1,:,:len_b], v[b:b+1,:,:len_b])  # 追加 chunk
    #     mask = torch.ones(len_b, past+len_b).tril(diagonal=past)[None,None]  # bottom-right causal
    #     outs[b,:,:len_b] = SDPA(q[b:b+1,:,:len_b], _repeat_kv(k_full), _repeat_kv(v_full), attn_mask=mask)
    #   (past = starts[b];SDPA bool mask True=参与;is_causal 不用——past>0)
```

- [ ] **Step 6: executor.extend + engine.step 消费 chunks**(草稿)

```python
    # Executor.extend(self, seqs, views, chunks) -> list[int]:
    #   chunks=[(seq,start,len)];ids [N, Lcmax] 右 padding → model.forward_extend
    #   → 逐行 _sample(与 prefill_batch 同语义)
    # Engine.step: chunks, decodes = self.scheduler.schedule()
    #   fresh = [(s,c) for ... if c[1]==0];cont = [...cursor>0]
    #   fresh 全量腿:chunk len == prompt 全长时走既有 prefill/prefill_batch(原路径,byte-identical)
    #   **cursor==0 但 len < prompt_len(预算切半)也走 forward_prefill_batch(实长=len)** ——
    #   它就是"更短的 prefill",数值路径与今日同源
    #   cont 腿:executor.extend
    #   簿记:每个 chunk 前向后 num_pages=len(view.table)、_record 按 chunks 顺序
```

Engine 级测试(tests/test_engine.py 追加):`test_engine_chunked_matches_unchunked`(budget=8 强制切块 vs budget=None,同 seed,token 流逐位 + assert_balanced)、`test_engine_mixed_step_chunk_and_decode`(3 条长 prompt + 2 条已在 decode 的短批,断言两类同步推进、输出序正确)。

- [ ] **Step 7: 全套件 + 等价性硬门**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q`
Expected: 既有 106 全绿零改动(Engine 默认 budget=None 时行为与 M4.1 等价)+ 新增 ~8-10 个(以实跑计数为准,报告说明)。

- [ ] **Step 8: Commit**

```bash
git add acorninfer/engine/scheduler.py acorninfer/engine/sequence.py acorninfer/models/qwen3.py acorninfer/engine/executor.py acorninfer/engine/engine.py tests/
git commit -m "feat(engine): chunked prefill — scheduler-native chunks + SDPA extend (M5 Task 1)"
```

---

### Task 2: 段 1 中间门(HTTP TTFT 对照)

**Files:**
- Modify: `docs/benchmarks.md`(M5 节段 1 中间账)

- [ ] **Step 1**: server(GPU 1, 端口 8200, `--chunked-budget` 默认与 512 两跑)+ `bench_remote --concurrency 1 4 8` 对照 Task 0 的 HTTP 基线;in-process bench graph/eager 双跑(吞吐回归线,须 ≥ M4.1 收官数字 ± 漂移带)。跑前双查 nvidia-smi;收服 PID kill。
- [ ] **Step 2**: docs 段 1 中间账(RESULT 逐字、TTFT 对照表、chunked 收益判定,如实)。
- [ ] **Step 3**: Commit `docs(benchmarks): M5 phase-1 interim — HTTP TTFT alignment (M5 Task 2)`。

---

### Task 3: TP=2 基建(eager decode 先行)

**Files:**
- Create: `acorninfer/distributed.py`(dist init/rank 工具/broadcast-gather 封装)
- Modify: `acorninfer/weights.py`(+分片加载,原函数不动)
- Create: `acorninfer/models/parallel.py`(Column/RowParallelLinear + qkv/gate-up 结构化切分)
- Create: `acorninfer/models/qwen3_tp.py`(TP 模型:attention/mlp/block/ForCausalLM;rope 第 5 份拷贝警告入档)
- Test: `tests/test_tp_eager.py`(双卡 harness)

- [ ] **Step 1: parallel.py + 切分工具**(草稿)

```python
class ColumnParallelLinear(nn.Module):
    """weight [out/world, in];qkv/gate-up 用结构化切分(按头/按 gate-up 边界),
    不做裸 dim 对半 —— qkv 裸切会切在 q 头中间(6144 = q4096+k1024+v1024)。"""
def shard_qkv(w, n_q, n_kv, head_dim, rank, world) -> Tensor:   # [3072, 4096]
def shard_gate_up(w, rank, world) -> Tensor                     # [12288, 4096]
class RowParallelLinear(nn.Module):
    """weight [out, in/world];输入已半,输出 partial;forward(skip_reduce=) 控制 allreduce"""
```

- [ ] **Step 2: qwen3_tp.py**(TP 全家:Qwen3TPAttention(kv 头 4/8,rope/norm 复制)、Qwen3TPMLP、Qwen3TPBlock、Qwen3TPForCausalLM;embed 词并行查表+allreduce,lm_head 词并行 partial → all-gather;与 qwen3.py 零共享状态,原文件零改动)

- [ ] **Step 3: weights 分片加载**

```python
def load_safetensors_sharded(model_dir, shard_fn, rank, world_size,
                             device="cpu", dtype=torch.bfloat16):
    """load_safetensors 全量载 CPU → shard_fn(name, t, rank, world) 切 → free 全量。
    (CPU 全量 ~16GB/进程,双进程 ~32GB host RAM——机器足够,不优化)"""
```

- [ ] **Step 4: 双卡 harness 测试**(tests/test_tp_eager.py;multiprocessing spawn ×2 子进程,file:// init,结果落 JSON,父进程断言)

```python
@pytest.mark.gpu_tp
def test_tp2_eager_decode_matches_tp1():
    """tiny 模型:TP=2(双进程)eager decode 生成 token 流 == 单进程 TP=1 逐位;
    末位 logits all-gather 后与 TP=1 的 max|diff| 按协议标定写死。"""
```

- [ ] **Step 5: 跑红 → 实现 → 跑绿**(TP 双卡命令:`CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest tests/test_tp_eager.py -v`;跑前双卡 nvidia-smi)
- [ ] **Step 6: 单卡全套件回归**(TP 路径休眠零污染):`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` 全绿零改动
- [ ] **Step 7: Commit** `feat(model): TP=2 foundation — parallel linears, sharded load, eager decode (M5 Task 3)`

---

### Task 4: TP graph decode + extend TP + SPMD 接线

**Files:**
- Modify: `acorninfer/engine/graph_runner.py`(仅当捕获路径需要 TP 感知——目标零改动,allreduce 在 model 前向内)
- Create: `acorninfer/engine/tp_engine.py`(SPMD 壳:submit/drop/step 广播命令;两卡同跑 Engine)
- Modify: `acorninfer/server.py`(rank0 持 API;TP 模式下 engine 由 tp_engine 包)
- Modify: `acorninfer/bench.py`、`acorninfer/server.py` CLI(`--tp 2`、`--chunked-budget`)
- Test: `tests/test_tp_engine.py`

- [ ] **Step 1**: graph+TP 捕获验证(T0 尖刀已证可行;decode graph 捕获 TP 前向,双卡对拍 replay 输出 vs eager)
- [ ] **Step 2**: SPMD 壳(草稿:每公开入口 `dist.broadcast` 命令/参数 → 两卡对称执行;采样 = all-gather full logits → rank0 `_sample` → broadcast token;决策漂移 = 断言双卡页表一致,漂移即 raise)
- [ ] **Step 3**: 测试:`test_tp2_graph_matches_eager`(graph replay vs eager 双卡)、`test_tp2_engine_spmd_tokens`(两卡 token 流一致 + 与 TP=1 对拍)、`test_tp2_prefill_extend`(chunked + TP 组合冒烟)
- [ ] **Step 4**: 单卡全套件回归(TP 休眠)全绿;Commit `feat(engine): TP graph + SPMD wiring (M5 Task 4)`

---

### Task 5: 回归 + TP=2 冒烟

- [ ] **Step 1**: 单卡全量:fast 全绿(≥106 + 新增)、slow `-m slow` 17 全绿零换金(golden 原样 = TP 休眠 + chunked 默认等价的双重证明)
- [ ] **Step 2**: 8B TP=2 冒烟(双卡):Qwen3-8B TP=1 vs TP=2 各生成 64 token,token 流对拍(TOL 协议标定 logits;greedy token 允许尾部 ulp 分歧?**不允**:对拍不过 = BLOCKED 上报,不静默换题)
- [ ] **Step 3**: Commit(如有修正)`test: M5 regression hardening (M5 Task 5)`

---

### Task 6: 双轨配对仪式 + docs 收口

- [ ] **Step 1: 仪式**(同日窗口,串行):① vLLM TP=1 基线(GPU 0)+ AcornInfer TP=1 graph(GPU 1)——历史线;② vLLM TP=2(双卡)+ AcornInfer TP=2(双卡)——主轨;TTFT 全部 HTTP 对 HTTP(server vs bench_remote)。收服 PID kill + 双卡回落验证。
- [ ] **Step 2: docs M5 收口**:双轨配对表、达标率(两轨各算)、TTFT/TPOT 判读(chunked 兑现?)、段 0/1/2 账目汇总、判定行(如实)+ 北极星挂账(TP=2 对 vLLM TP=2 逐档 ≥100%)+ tag 不打。
- [ ] **Step 3**: Commit `docs(benchmarks): M5 dual-track paired final (M5 Task 6)`。

---

## 计划自审记录

1. **Spec 覆盖**:§1 双轨 → Task 6;§2 段 0 尖刀/裁决门 → Task 0;§3 chunked(cursor/预算/deficit/抢占/extend/混合步/byte-identical)→ Task 1-2;§4 TP(切分表/NCCL/SPMD/采样/纪律)→ Task 3-4;§5 回归 → Task 5;§6 保护条款 → 全局纪律;§7 non-goals 未越界(A2 真混合批、triton extend、custom allreduce 均未入任务);§8 风险 → R1/R4 = Task 0 裁决门,R3 = Task 4 Step 2 断言,R5 = 各任务单卡回归硬门,R6 = Task 0 Step 3,R7 = 全局纪律。
2. **API 对照(读源码核实)**:`Sequence` 现字段面(field 列表,prefill_cursor 加默认值零破坏);`load_safetensors` 平 dict 返回 → 分片函数独立新增;server = `create_app(engine)` + FastAPI,engine 由外部构造 → TP 壳在构造层注入,server 逻辑零改动目标;`graph_runner._capture` 在 `torch.cuda.graph(st.graph)` 内调 model 前向 → allreduce 随前向入图,runner 理论零改动(若有隐含单卡假设,Task 4 处理并记录);qkv 全宽 [6144,4096] = q 4096 + k 1024 + v 1024,裸对半会切在 q 头中间 → 结构化切分是硬要求(parallel.py 草稿已注)。
3. **类型一致性**:schedule() 返回 `list[(Sequence,int,int)]` 全任务一致;`forward_extend(input_ids, views, *, starts, chunk_lens)` 模型层定义 = executor 调用一致;`load_safetensors_sharded(model_dir, shard_fn, rank, world_size, device, dtype)` = qwen3_tp 构造一致。
4. **计数账**:fast 106 基线;T1 +8~10(scheduler 3 + extend 2 + engine 2 + 杂项)、T3 +1~2、T4 +2~3、T5 视情况——以实跑为准报告,slow 17 不变(零换金是硬门)。
5. **薄弱点**:草稿 scheduler 预算/deficit 推导标注"以现状推导"(M4.1 先例:这里最可能藏 off-by-one,红绿测试先行);TP harness(multiprocessing spawn + file init)在 CI 式环境的首跑风险;TP=2 8B 对拍若 greedy 分歧 = BLOCKED 不换题。
