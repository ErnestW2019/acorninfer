# M9 prefill/TTFT 攻坚 实施计划(引擎优化为主 goal,官方压测为验收仪)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 官方 `vllm bench serve` 口径(bf16 对 bf16,开环 Poisson)把 TTFT @C=8 从"落后 vLLM ×2.4"打到 ≤×1.3(A1 预案,spike 后定稿);decode 三档与全套件零回退(A2);逐位门(A3)。

**Architecture:** spike 归因表定靶(只施工表上有行的面)→ 条件任务(激活 = 归因占比 ≥10%,段 0 裁决)→ 官方工具 before/after/vLLM 三列仪式。

**Tech Stack:** torch profiler / 全链打点 / triton paged prefill kernel(只读派生,备选)/ SDPA enable_gqa / chunked 调参。

**Spec:** `docs/superpowers/specs/2026-11-02-m09-prefill-ttft-design.md`(锚/保护面/风险以 spec 为准)

---

## 全局纪律(每个任务都适用,实现者必读)

- **计划代码是草稿,测试才是契约。** 草稿错以代码现状与实测为准修,报告逐条记录偏离。
- **保护文件**(spec §3):`ops/paged_attention.py`(**只读派生,一行不改**)、`engine/scheduler.py`(**语义零改动**;归因若指调度面 → 停手上会)、`engine/sequence.py`、`engine/graph_runner.py`、`cache/kv.py`、`cache/pager.py`(M8 收官回保护)、golden m0–m8、`tests/triton_utils.py`。常规解锁:`models/qwen3.py`(prefill/extend 变体)、`engine/engine.py`、`engine/executor.py`、`server.py`(d 面)、`acorninfer/config.py`、新文件 kernel/测试、`scripts/`。
- **逐位门**:`paged_attention.py` diff 零字节是硬门;数值对拍走 TOL 协议(新标定,命令/日期注释);`enable_gqa` 类无 kernel 改造必须逐位同 SDPA(同函数同输入,非 TOL)。
- **环境纪律**:单卡施工 `CUDA_VISIBLE_DEVICES=1` + `.venv/bin/`(跑前 nvidia-smi 查残留);官方 bench 双边 server 在 **GPU 0**(仪式配方,关服 `ss -tlnp` 真身 PID 定向 kill,收尾双卡 ~4MiB 验净);timeout ≥ 1800000;运行时禁 import transformers(golden 面不变);golden/slow 零换金。
- **基线**:M8 收官 `3182a86`(spec 后 HEAD `f489267`);fast 全量 238 / 单卡 232+6 skipped,slow 17 / tp 13 零换金;bf16 账:vLLM 58.12/208.57/356.17,我方满配方 53.7/186.8/296.7(M6 收官口径);TTFT C=8 ~0.19 vs ~0.08s。
- **官方工具**(已验):`/home/ernest/vllm-baseline/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:<port> --model /home/ernest/model-weight/Qwen3-8B --tokenizer /home/ernest/model-weight/Qwen3-8B --dataset-name random --random-input-len 128 --random-output-len 32 --random-range-ratio 1 --num-prompts 256 --request-rate <λ> --seed 42 --percentile-metrics ttft,itl,e2e`;λ ∈ {1,2,4,8,16,32};双边同参,`RESULT-*` 原样入 docs。
- **激活条款(预注册,段 0 裁决)**:条件任务 T1–T4 仅当归因表对应面**占比 ≥10% 且预期收益 ≥5ms**(C=8 单步口径)时激活;未激活的任务 no-op 入档(M6 先例,注明归因行数值)。

---

### Task 0: spike 归因 + before 成绩单(纯测量,零产线改动)

**Files:**
- Extend: `scripts/profile_prefill.py`(M4.1 骨架在档,扩链路打点)、`scripts/m6_ab.py`(只读复用)
- Create: `docs/benchmarks.md` M9 节开篇(口径定案追记 + 归因表 + before 表)

- [ ] **Step 1: 服务端拆账**:torch profiler 拆满载 prefill step(N=8×L=128,W8 不涉及,bf16):GEMM 面 / attention 面(含 `view.update` gather + `_repeat_kv` 物化拷贝)/ elementwise / host 间隙各占毫秒;`scripts/profile_prefill.py` 扩展出**归因表**(行 = 面,列 = ms + 占比)。
- [ ] **Step 2: 探针腿(临时,不进产线)**:(a) SDPA `enable_gqa=True` 替代 `_repeat_kv`(独立脚本 A/B,逐位对拍验证);(b) gather-free 直读页变体可行性估算(只测,不写 kernel);(c) chunked 拆块行为审计:C=8 满载时每请求实际 chunk 数与步数(临时打点,提交前移除,diff 干净)。
- [ ] **Step 3: 全链打点**:TTFT 链 = encode 耗时(独立测 tokenizer batch=8)/ HTTP→首字节 / queue 等待 / prefill 前向 / SSE flush;区分计算 vs 排队 vs 往返。`vllm bench serve` before 全 rate 扫:双边 server(GPU 0,配方在档)各 λ 一轮,`RESULT-*` 逐字入 docs(**before 成绩单**)。
- [ ] **Step 4: 段 0 材料提交**:归因表 + A1 硬数定稿提案(预案 ×1.3 行使/放弃修正权)+ 条件任务激活建议清单 → **控制器段 0 裁决**;commit `docs(m9): T0 spike attribution + before scorecard`。

**红线**:本任务产线零改动(git diff 仅 scripts/ + docs/);官方工具首跑验证 R5(同 seed 两侧 prompts 哈希一致)。

---

### Task 1: [条件] prefill/extend 注意力改造(激活 = 归因占比 ≥10%)

**Files(草稿,以 spike 为准):**
- Modify: `models/qwen3.py`(`forward_prefill_batch` / `forward_extend` 的 attention 段;投影/rope 不动)
- Create(若走 kernel): `ops/paged_prefill.py`(从 paged_verify 结构只读派生)
- Test: `tests/test_prefill_attention.py`(新)

- [ ] **Step 1: 两段式施工(先便宜后贵,每段官方 TTFT 复测)**:
  - **1a. gather-free 化**:`view.update`(O(L) 全量 gather)+ `_repeat_kv`(×4 物化)→ 直读页/`enable_gqa`;若 1a 达锚,kernel 段 no-op 如实入档(M6 先例);
  - **1b. triton paged prefill kernel**(仅当 1a 不够):q [N,L,H,D] 行 (b,j),读上限 `end_j = starts[b]+j+1`(past=0 纯 prefill 与 past>0 extend 同式,bottom-right 因果);KV 由调用方 `write()` scatter 后 kernel 直读页;pad 行契约照旧(`outs` zeros + `[:L]` 切片写);
- [ ] **Step 2: TDD**:逐 b SDPA 稠密参考对拍(TOL 标定:随机输入 × N∈{1,4,8} × L∈{1,15,16,17,128} × past∈{0,16,48});因果精确性(可区分输入钉 mask 语义);1a 段**逐位钉**(同 SDPA 函数,非 TOL);fast 全套件零换金。
- [ ] **Step 3: A2 复核**(m6_ab 三档带内)+ 官方工具 TTFT 中间复测(C=8 单档);docs 施工账 + commit `feat(ops): prefill attention rework (M9 Task 1)`。

---

### Task 2: [条件] prefill GEMM/合批形态(激活条款同上)

**Files(草稿)**:`models/qwen3.py` / `engine/engine.py`(prefill 前向形态)/ `tests/`。

- [ ] **Step 1**:按归因行施工(候选:M 维度轮廓差 [N,Lmax] 右 padding 的浪费率;prefill 与 decode 步的发射交错;多请求 Lmax 不齐的 padding 浪费);**Step 2**:前后归因行对照 + 官方 TTFT 联动 + A2/A3;**Step 3**:docs + commit `feat(engine): prefill batching rework (M9 Task 2)`。
- 注:若归因指向"prefill 与 decode 不并发"等调度形态 → **停手上会**(scheduler 禁改边界),不得私改。

---

### Task 3: [条件] chunked budget / 调度边界调参(激活条款同上;零新代码)

**Files**:`acorninfer/config.py` 缺省值(如需)+ `server.py` 旗标缺省 + docs。

- [ ] **Step 1**:budget ∈ {None, 128, 256, 512} × C=8 TTFT / decode TPOT 双指标扫(官方工具 C=8 单档 + m6_ab);**Step 2**:选值定档(准则:TTFT 改善 ≥10% 且 decode TPOT 劣化 ≤2%),逐 byte 核对缺省行为变化仅限显式旗标路径;**Step 3**:docs 参数表 + commit `feat(config): chunked budget retune (M9 Task 3)`。

---

### Task 4: [条件] 首 token 关键路径(激活条款同上;server 面常规解锁)

**Files(草稿)**:`server.py`(encode 异步化 / pump 唤醒 / flush 时机)、`tests/test_server_spec.py` 追加。

- [ ] **Step 1**:按归因行施工(候选:encode 移出锁外/线程化、pump 空转唤醒延迟、首 token flush 时机);**Step 2**:TDD(异步测试照既有 server 测试风格)+ 全链打点前后对照 + 官方 TTFT 联动;**Step 3**:A4 缺省门(全套件零换金)+ docs + commit `feat(server): first-token path (M9 Task 4)`。

---

### Task 5: 仪式 + docs 终账

**Files**:`docs/benchmarks.md`(M9 终账);双边 server 配方照 M8(M5/M6)在档。

- [ ] **Step 1: after 成绩单**:官方 `vllm bench serve` 全 rate 扫(命令原样)× {AcornInfer 满配方(施工后), vLLM bf16 同参, AcornInfer eager(诊断)};饱和点逐配置标注。
- [ ] **Step 2: 判定**:A1(TTFT @C=8 p50 比值,spike 定稿硬数)、A2(decode 三档 m6_ab 带内 + 官方吞吐逐档)、A3(全套件三尾行:fast 全量 / slow 17 / tp 13 零换金 + `paged_attention.py` 零 diff)、A4(命令原文);**每施工点归因行前后对照**(毫秒级兑现)。
- [ ] **Step 3: docs 终账**:before/after/vLLM 三列表 + 归因表前后 + **§0 定案落实**:147.4/128.4/112.8% headline 追记更正(量化 vs 非量化口径,不作 engine 胜负);M10 台账(decode C=8 残余缺口、TP=2、rank0 快失败、spec chain-mode、KV 量化)+ 判定行;commit `docs(benchmarks): M9 final — prefill/TTFT accounts`。

---

## 计划自审记录

1. **Spec 覆盖**:§1 A1→T0 定稿+T5 判定、A2→T1/2/3 Step 复核+T5、A3→全局纪律+T1 逐位钉、A4→T5;§2 四施工面→T1–T4(激活条款预注册);§3 保护面→全局纪律;§4 R1/R2→T2 注+停手上会条款、R3→T1 TDD、R4→T1 两段式、R5→T0 红线、R6→T5 标注;§6→T5。
2. **占位扫描**:条件任务均为预注册激活程序(归因占比阈值),非 TBD;A1 硬数走"预案+一次修正权"条款。
3. **计数账**:fast 238 基线;T1 ~4、T4 ~2(条件激活才有);slow 17 / tp 13 零换金硬门。
4. **薄弱点**:①归因可能指向多处平摊(无单一 ≥10% 面)→ 段 0 可批多任务组合,但每个仍须有归因行;②T1b kernel 若 1a 已达锚则 no-op,防过度施工;③官方工具对非 vLLM 后端的容错(首跑即验,R5);④encode/lock 等链路面的临时打点必须提交前移除(红线明文)。
