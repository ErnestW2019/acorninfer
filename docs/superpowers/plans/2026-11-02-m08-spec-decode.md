# M8 投机解码 实施计划(n-gram 草稿 + 贪心验证,eager v1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PLD 式 n-gram 投机解码全链:wiki 续写 C=1 tok_s ×≥1.4 vs 同模式 eager W8(A2 硬门),无损性 spec 贪心 ≡ 非投机贪心逐位(A1 硬门),chat 型如实(A3),缺省关逐字节(A4)。

**Architecture:** 草稿器(纯 CPU 查表)→ `forward_verify`(m=K+1 统一,pager 死槽游标复用)→ 引擎级接受逻辑 → 语料/参数扫/仪式。

**Tech Stack:** 纯 python 草稿器 / triton verify kernel(从 paged_attention 只读派生)/ pager 三方法增量 / 自研语料。

**Spec:** `docs/superpowers/specs/2026-11-02-m08-spec-decode-design.md`(锚/保护面/风险以 spec 为准)

---

## 全局纪律(每个任务都适用,实现者必读)

- **计划代码是草稿,测试才是契约。** 草稿错以代码现状与实测为准修,报告逐条记录偏离。
- **保护文件**:golden m0–m7、`ops/paged_attention.py`(**只读派生,一行不改**)、`engine/scheduler.py`、`engine/sequence.py`、`engine/graph_runner.py`(eager v1 零触碰)、`tests/triton_utils.py` 照旧禁改。**解禁**:`cache/pager.py`(仅三增量方法,见 T1;既有方法行为逐字节不变)、`models/qwen3.py`(新增 forward_verify 变体 + 调用点,既有 4 前向不动)、`engine/engine.py` + `executor.py` + `server.py` + `acorninfer/config.py`(常规解锁面)。
- **byte-identical 硬门**:`--spec-ngram 0`(缺省)时 token 流与全套件与 `a3e1cf5` 逐字节一致;TOL 协议只用于 verify kernel 对拍(新标定,命令/日期注释);A1 无损性是**逐位**断言(非 TOL)。
- **环境纪律**:单卡 `CUDA_VISIBLE_DEVICES=1` + `.venv/bin/`;跑前 nvidia-smi;timeout ≥ 1800000;运行时禁 import transformers;golden/slow 零换金。
- **基线**:HEAD `a3e1cf5`;fast **217 passed, 6 skipped, 17 deselected** / slow 17 / tp 13;W8 eager C=1 step ~17.0ms(T2/T5 带 16.65–19.25,仪式取新鲜基线);W8 graph C=1 10.72–11.02(生产参照);kernel 对拍参考 = `paged_decode_attention_ref`(稠密,只读复用)。
- verify 形状口径:m = spec_num+1;query 绝对位置 = len..len+m−1(len = 该 seq 前写长度);`seq_lens`(kernel 读侧)= len+m(含本步全部窗口位);接受 a → lens 回 = len+a+1(死槽 len+a+1..len+m−1 下步复用)。

---

### Task 0: 草稿器 + 接受逻辑(TDD,纯 CPU,零 GPU)

**Files:**
- Create: `acorninfer/spec/__init__.py`、`acorninfer/spec/ngram.py`
- Test: `tests/test_spec_ngram.py`(新;纯 CPU)

- [ ] **Step 1: 失败测试**

```python
def test_index_build_and_lookup():
    """prompt 建索引:键 = N-gram tuple,值 = 后继(按出现序);查表命中/未命中。"""
def test_propose_deterministic_and_padded():
    """同上下文同输出;候选 <K 用 filler=0 补齐(spec §2.1);K 候选截断。"""
def test_propose_uses_prompt_plus_generated():
    """索引随生成增量更新:生成 token 进入上下文后可被查到(增量 append O(1))。"""
def test_greedy_accept_longest_prefix():
    """greedy_accept(argmax_ids[m], drafts[K]) -> (a, bonus):最长前缀;全中给 bonus;零中 a=0 仍产 bonus。"""
def test_accept_is_exactly_greedy_chain():
    """黄金性质钉:构造 logits 使贪心链逐 token == drafts → (a=K, bonus=链尾);
    任何 draft 与 argmax 不符即断,其后位置即使巧合相符也不计(前缀语义)。"""
```

- [ ] **Step 2: 跑红 → Step 3: 实现(草稿)**

```python
"""PLD 式 n-gram 草稿器 + 贪心接受(spec §2.1)。纯 CPU,确定性。"""
FILLER = 0  # 固定填充 token(spec §2.1 定案)

class NgramProposer:
    def __init__(self, prompt_ids: list[int], n: int, k: int): ...
    def extend(self, new_ids: list[int]) -> None:          # 生成 token 增量入索引
    def propose(self) -> list[int]:                        # 键 = 末 n 位;不足 K 补 FILLER

def greedy_accept(next_token_logits, drafts) -> tuple[int, int]:
    # next_token_logits [m, vocab] float(argmax 已取);返回 (接受数 a, bonus token)
```

(草稿注:接口签名以测试为准;propose 的键取"末 n 位已提交上下文"——verify 步首尚未接受任何新 token,故 = prompt+已生成尾部。)

- [ ] **Step 4: docs 起笔(M8 节开篇:设计引用 + 本任务记录)+ commit `feat(spec): n-gram drafter + greedy acceptance (M8 Task 0)`**

---

### Task 1: pager 多槽游标(唯一保护解禁,TDD 等价钉先行)

**Files:**
- Modify: `cache/pager.py`(仅三增量方法;既有方法零改动)
- Test: 既有 KV 对拍套件 + `tests/test_spec_pager.py`(新)

- [ ] **Step 1: 失败测试(等价钉)**

```python
def test_ensure_n_equals_old_for_n1():        # ensure_append_capacity_n(layer,1) 与旧 ensure 逐字节同 (page,slot)
def test_ensure_n_allocates_across_pages():   # n 跨 1..page_size 边界:分配语义 = write() 的 needed 公式
def test_scatter_verify_m1_equals_scatter_decode():  # m=1 时两方法写页与 _lens 提交逐位一致
def test_scatter_verify_multislot_and_lens(): # m=5:写 [B,kv,5,D] 于 [B,5] 目标;lens += 5;层间锁步
def test_set_lens_rollback_guarded():         # set_lens(new_len ≤ 已写 len);越界 ValueError;engine 侧 36 层循环由锁步调用方保证
def test_dead_slot_reuse_roundtrip():         # 写 m=5 → set_lens 回 a=2 → 下步从 len' 复写:页值被覆盖、读侧(lens)永不泄死槽
```

- [ ] **Step 2: 跑红 → Step 3: 实现(三增量方法,草稿)**

```python
def ensure_append_capacity_n(self, layer_idx: int, n: int) -> list[tuple[int, int]]:
    # needed = (base + n + page - 1) // page;分配循环同 write();返回 n 个 (page,slot)
def scatter_verify(self, views, layer_idx, k, v, pages, slots) -> None:
    # k/v [B, kv, m, D];pages/slots [B, m] long;单次展平 advanced indexing(照 scatter_decode
    # 的 [B*m] 展平法);_lens 提交 + m;assert 语义照 scatter_decode(容量已在步首 ensure)
def set_lens(self, layer_idx: int, new_len: int) -> None:   # PagedCacheView 方法
    # new_len ≤ self._lens[layer_idx] 否则 ValueError(只回退不前进;前进只属 scatter/write)
```

(草稿注:①旧 `ensure_append_capacity` 改为 `return self.ensure_append_capacity_n(layer_idx, 1)[0]` 的薄壳**或**保持原体零动——二选一以"旧路径逐字节"为准,报告说明;②`scatter_verify` 内层锁步断言照 scatter_decode 注释口径。)

- [ ] **Step 4: 既有 KV 对拍全绿(零换金)+ docs + commit `feat(cache): multi-slot verify cursor (M8 Task 1)`**

---

### Task 2: verify attention kernel(ops/paged_verify.py,新文件只读派生)

**Files:**
- Create: `ops/paged_verify.py`(split+combine 结构从 `paged_attention._paged_attn_split_kernel` 派生;**原件零触碰**)
- Test: `tests/test_paged_verify.py`(新)

- [ ] **Step 1: 失败测试(数值对拍,fp32/稠密参考)**

```python
TOL_VERIFY = (rtol=…, atol=…)  # 标定后写死(命令/日期注释;参考 = 同输入 gather-to-dense + sdpa/fp32 实现)
def test_verify_matches_dense_reference():   # B∈{1,4,7} × m∈{2,5} × len 跨页组合(含 len%16==0 边界)
def test_window_causal_exact():              # 逐 query 位可见集 = 历史 + 窗口前缀(构造可区分输入钉 mask 语义)
def test_rollback_then_reverify_numerics():  # 回退后再 verify:死槽复写后读侧结果与"从未写过死槽"一致
```

- [ ] **Step 2: 跑红 → Step 3: kernel 实现(草稿要点)**

- q 展平 [B*m, H, D](行 → (b, j));每行读上限 `end = seq_len[b] − (m − 1 − j)`(= len+j+1;seq_lens 传 len+m);split kernel 逐行因果,页遍历/偏移公式照 `paged_decode_attention` 现状(L 的口径差异是唯一本质改动);
- combine 照原结构;输出 [B*m, H, D] → view [B, m, H, D];
- eager-only(无 capture 要求);k/v 写入不在 kernel 内(调用方 scatter_verify 已写,读侧只读页);
- sanity 计时:m=5 B=1 verify vs 5 次单位 decode 调用(同卡同口径,记录,非门)。

- [ ] **Step 4: TOL 标定表 + docs + commit `feat(ops): paged verify attention kernel (M8 Task 2)`**

---

### Task 3: forward_verify + 引擎接线 + 无损性硬门

**Files:**
- Modify: `models/qwen3.py`(forward_verify 变体:Model 级 + ForCausalLM 级 + Qwen3Attention.forward_verify + Qwen3MLP 无新路径)、`engine/executor.py`(verify_batch)、`engine/engine.py`(spec 步分支)、`acorninfer/config.py`(spec 字段)、`server.py`(CLI + 互斥)
- Test: `tests/test_spec_engine.py`(新)

- [ ] **Step 1: 失败测试**

```python
def test_config_spec_validation():        # spec_ngram ∈ {0}∪[1,8];spec_num ∈ [1,8];spec>0 须 triton 后端;
                                          # spec × decode_graph 构造期 ValueError(tp 互斥在 _build_engine 照 quant 先例)
def test_spec_greedy_bitwise_tiny():      # A1 核心:tiny 模型同 prompts,spec on/off 全量生成逐位一致(多条 prompts)
def test_spec_multitoken_outputs_order(): # 接受多 token 时 TokenOutput 顺序 = 位置序;eos/max_tokens 截断语义
def test_spec_dead_slot_reuse_engine():   # 连续 verify 步:KV 对拍 + 页表断言(复用不泄漏;assert_balanced 过)
def test_server_passthrough_spec():       # --spec-ngram/--spec-num 透传 + interdicts(照 quant 先例)
```

- [ ] **Step 2: 跑红 → Step 3: 接线(草稿要点)**

- `forward_verify(input_ids [B,m], views)`:prepare(`ensure_append_capacity_n(·,m)` + `build_tables(lens_override=[len+m])`)+ 位置 `len+arange(m)`(镜像 forward_decode 的位置推导)+ 层循环 `layer.forward_verify(...)`(attention=paged_verify + `scatter_verify` 写入;投影分派复用既有 linear/quant 路径)+ lm_head **全 m 位** `[B, m, vocab].float()`;
- `executor.verify_batch(seqs, views, drafts)`:ids [B,m] = [last] + drafts;返回 logits;
- `engine.step` spec 分支(decodes 全批):proposers[seq_id].propose() → verify_batch → 逐 seq `greedy_accept` → 接受 token 逐个过 eos/max_tokens 截断(现有单 token 方法组合)→ `generated_ids.extend` → `set_lens`×36 层回 len+a+1 → 多 TokenOutput 发射 → `num_pages` 同步 → 退休检查;proposers 生命周期 = admit 建 / retire 弃(engine 侧 dict,scheduler 不动);
- 边界:剩余容量 < m 的 seq → 该步降级 m=1 纯 decode 路径(引擎级,批量统一性让位于正确性;测试钉住);
- server CLI `--spec-ngram 0` / `--spec-num 4` 透传;`spec × tp` 互斥在 `_build_engine`(quant 先例)。

- [ ] **Step 4: 无损性硬门 + A/B**

- tiny 全量 A1(测试内)+ **8B 抽样 A1**:GPU 1,同 prompts(spec on/off,均 greedy W8)≥4 条(含 wiki 型 + 短 chat 型),token 流逐位 diff 脚本;
- **A4 缺省门**:全套件 fast(基线 217+新增)零换金 + `m6_ab.py --quant w8a16` 带(10.7–11.7)复核;
- 计测:verify step_ms(m=5)vs 纯 decode eager(gpu 1,`CUDA_VISIBLE_DEVICES=1`),draft 耗时(<100µs 目标);eager W8 tok_s 基线出数(= A2 分母,仪式前先钉)。
- [ ] **Step 5: docs 段 A 账 + commit `feat(engine): speculative decode wiring (M8 Task 3)`**

---

### Task 4: 语料 + 参数扫 + 仪式 + 终账

**Files:**
- Create: `scripts/make_spec_prompts.py`(wiki.test.raw → 128-token 块,Tokenizer 分词,seed 确定性,JSON 出 `data/spec_prompts_wiki.json`)
- Modify: `scripts/bench_remote.py`(--prompts-file 支持)、`docs/benchmarks.md`(M8 终账)

- [ ] **Step 1: 语料 + bench 扩展**(命令原样入 docs;A4:缺省语料行为不变)

- [ ] **Step 2: 施工期参数扫**(GPU 1,in-process 或 HTTP 同口径):N∈{1,2,3} × K∈{2,4,6} × wiki workload C=1,各 ×(接受率、tok_s);**选定缺省配置 docs 记录(锚判在选定配置上,仪式当天不许换锚)**;chat workload 同扫如实。

- [ ] **Step 3: 仪式**(GPU 0,M5/M6 配方,关服 ss -tlnp 真身 kill 纪律):

- 腿:vLLM bf16 8100 / AcornInfer W8 graph 8200(生产参照)/ W8 eager(同模式基线)/ **W8 spec eager(wiki + chat 双 workload)**;可选 vLLM ngram spec 腿(0.32 支持性先验证,不行如实缺席);
- 各腿 `bench_remote.py` C=1/4/8;达标率逐档;**A2 锚判定 = spec(wiki)/ eager-W8(wiki)tok_s 比**(同模式同语料);接受率矩阵 + tokens/step 分布;A1/A4 硬门证据引用;
- 全量回归三尾行(fast ≥217+新增 / slow 17 / tp 13 零换金)。

- [ ] **Step 4: docs M8 终账**:锚判定行(A1–A4 逐条,不达照写)+ 接受率/参数扫矩阵 + workload 敏感性归因 + kernel/草稿器开销账 + M9 台账(自适应 k、graph verify、采样域、draft-model 系、TP×spec、W4 复活条件、KV 量化、连续到达、半宽 GEMM、custom allreduce、rank0 快失败);commit `docs(benchmarks): M8 final — speculative decoding accounts (M8 Task 4)`。

---

## 计划自审记录

1. **Spec 覆盖**:§1 锚 A1→T3 Step4/A2→T4 Step3/A3→T4 矩阵/A4→T3 Step4+T4 Step1;§2.1→T0;§2.2→T2/T3;§2.3→T1;§2.4 互斥→T3 Step1/3;§3 保护面→全局纪律;§4 R1→T4 Step2、R2→T2、R3→T1/T3、R4→T4 矩阵、R5→T3 Step4、R6→T3 Step1、R7→T4 Step3;§5 非 goal 未越界;§6 仪式→T4。
2. **占位扫描**:全任务含草稿或预注册协议;TOL 数值留给标定(协议明文);缺省参数待参数扫(预注册程序,非 TBD)。
3. **类型一致性**:`NgramProposer(prompt_ids, n, k)`/`extend/propose`/`greedy_accept(logits, drafts)->(a,bonus)` 全任务一致;`ensure_append_capacity_n(layer,n)`/`scatter_verify(views,layer,k,v,pages,slots)`/`set_lens(layer,new_len)` 三签名贯穿 T1/T2/T3;m=spec_num+1 统一口径(全局纪律)。
4. **计数账**:fast 217 基线;T0 ~5、T1 ~6、T2 ~3、T3 ~5、T4 0(仪式);slow 17 / tp 13 零换金硬门。
5. **薄弱点**:①`ensure_append_capacity` 薄壳化 vs 零动二选一(T1 报告);②forward_verify 的位置推导须与 forward_decode 逐位同源(T3 测试经 A1 无损性间接钉);③大 m 时 lm_head logits 显存 [8,6,151936]≈29MB 可接受(K≤8 → ≤50MB);④spec × quant 兼容(m≤16 桶内 ✓,测试在 T3 tiny+8B 双跑覆盖)。
