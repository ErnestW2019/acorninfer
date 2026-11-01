# M6 decode 步贴顶 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 decode 步从引擎级 702GB/s 推向孤立微基准 889GB/s + 步开销清账,重挂旧北极星:TP=1 graph tok_s 逐档 ≥ vLLM(现 92.4/90.0/83.7%)。

**Architecture:** 段 0 尖刀(引擎内 GEMM 拆账/调参面/布局面/步开销四腿,可回收清单 + 70% 裁决门)→ 段 1 落地(每项 TDD + A/B 计量,<0.2ms 回退)→ 段 2 triton(条件立项)→ 双轨仪式收口。

**Tech Stack:** torch.profiler/py-spy/cublasLt 环境面/权重布局预变换/可选 triton kernel;M0–M5 全部基建。

**Spec:** `docs/superpowers/specs/2026-11-01-m06-decode-roofline-design.md`

---

## 全局纪律(每个任务都适用,实现者必读)

- **计划代码是草稿,测试才是契约。** 草稿错以代码现状与实测为准修,报告逐条记录偏离;数据依赖的施工面(段 0 榜单)以尖刀实测排序为准,偏离即记录。
- **保护文件(本次更新)**:golden m0–m5、`cache/kv.py`、`tests/triton_utils.py`、`ops/paged_attention.py`、`engine/graph_runner.py`、`engine/scheduler.py`、`engine/sequence.py` 照旧禁改。**M6 解保护例外**:`models/qwen3.py`(仅投影调用点分派,attention/rope/KV/norm 面一行不动)、`cache/pager.py`(仅 `_commit_lens` 相关批量提交,签名与语义不变)、`weights.py`(仅新增布局函数)。
- **byte-identical 硬门**:所有新路径默认关闭(`layout="none"`、旋钮缺省)时,token 流与全套件行为与 M5 收官逐字节一致;布局/分派路径的数值等价用 **TOL 协议 + margin 门 argmax**(M5 T3 先例;bf16 换核不保逐位,平局翻转向已证结构性)。
- **环境纪律**:单卡 `CUDA_VISIBLE_DEVICES=1` + `.venv/bin/`;双卡测试(仅 T6 仪式)双查 nvidia-smi;前台 timeout ≥ 1800000;运行时禁 import transformers;容差只准 `tests/triton_utils.py` TOL 协议(标定后写死,注释命令/日期)。
- 基线:HEAD `7a20a1a` 后为 `5c43f0e`(spec);fast **138 passed, 6 skipped, 17 deselected** / slow **17 passed** / 双卡 tp 全套 **16 passed**;锚基线数字:TPOT 17.95/21.12ms(C=1/C=8),tok_s 53.66/186.81/296.71,达标率 92.4/90.0/83.7%。

---

### Task 0: 段 0 尖刀(四腿 + 可回收清单 + 裁决门)

**Files:**
- Create: `scripts/m6_spike.py`(子命令:`insitu` / `knobs` / `layout` / `overhead`)
- Create: `scripts/m6_ab.py`(A/B 计时协议,段 1 各项复用)
- Modify: `docs/benchmarks.md`(M6 节开篇:尖刀账)

- [ ] **Step 1: `m6_ab.py`** —— A/B 计时协议(段 1 每项的度量衡,先行建好)

```python
"""M6 A/B 计时:decode loop 8B,C=1/C=8 × graph/eager 各 N 步。
用法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/m6_ab.py --label <名字>
输出行:AB <label> C=<c> mode=<graph|eager> step_ms=<v> tpot_ms=<v>(warmup 10 + 计 30)
同 seed 同输入;对照基线:graph 21.50/…(C=8)wall-step,TPOT 17.95/21.12。
结构照 scripts/profile_decode.py 的引擎构造方式(先读它)。"""
```

- [ ] **Step 2: `m6_spike.py insitu`(腿 1:引擎内 GEMM 拆账)**

torch.profiler 包 decode loop(C=1/C=8 × graph/eager),按 kernel 名聚合四类投影(qkv/o/gate_up/down 的名字 pattern)+ attention/norm/copy/其他;输出 `INSITU C=<c> mode=<m> class=<k> us_per_step=<v>` + 引擎 GEMM 合计 vs 孤立微基准冷口径(15.64ms,引用 M5 段 0)差距;`nvidia-smi dmon -s puc` 同窗记录时钟。**judge 字段**:差距归因候选(同名 kernel 时长不同/插桩稀释/层间 L2 污染/降频)各给证据行。

- [ ] **Step 3: `m6_spike.py knobs`(腿 2:调参面扫)**

矩阵:{CUBLASLT_WORKSPACE_SIZE ∈ 缺省/64MB/256MB} × {allow_bf16_reduced_precision_reduction ∈ True/False} × {CUBLAS_WORKSPACE_CONFIG 缺省/:4096:8};跑法:env 设定后子进程跑 `m6_ab.py --label <组合>`(env 必须在 CUDA 初始化前生效);孤立微基准(复用 `phase0_spike.py --gemm-cold` 口径)+ 引擎内双口径。输出 `KNOB <组合> <口径> <值>`;预期收益微小——腿的价值是排除。

- [ ] **Step 4: `m6_spike.py layout`(腿 3:布局面扫)**

孤立打分:变体 {TN 现行(F.linear)、NN 预变换(mm(x, w_t),w_t=.t().contiguous() 装载期)、pad8(零填充 8 行切回)、pad16},四类形状,冷口径 36 套轮转(照 `gemm-cold` 协议);输出 `LAYOUT <variant> <class> us=<v> gb=<v>`。**最优变体进引擎内复测**:`m6_ab.py --layout <v>`(引擎侧布局接线 = Task 1 的 `pack_linear_weight` + `linear_dispatch`,本腿允许先行以最小接线实现——仅接线,不 commit 生产默认);孤立赢、引擎内输 = R2 实录。

- [ ] **Step 5: `m6_spike.py overhead`(腿 4:步开销拆账)**

graph 步 wall−GPU = 1.70ms 拆解:profiler CPU 侧事件 + py-spy 采样,逐项计时:`ensure_append_capacity` 循环(graph_runner.py:146)、镜像填充 4 次 `torch.tensor`(graph_runner.py:150-157)、padding 常量场压回(:158-168)、`_commit_lens`(:172,252 赋值)、H2D 两次(:170-171)、D2H/同步点(argmax/item)、scheduler.schedule python、`num_pages`/簿记;输出 `OHEAD <item> us_per_step=<v> evidence=<行号/采样>` + 可回收预估。

- [ ] **Step 6: RECOVER 清单 + docs 段 0 节 + 裁决门**

汇总 `RECOVER <item> ms_c8=<v> confidence=<H/M/L> risk=<R#>`(C=8 口径,Σ 与锚缺口 3.3ms 对比);docs `benchmarks.md` M6 节开篇(四腿数据 + 清单,**判定行留控制器**);commit `feat(scripts): m6 spike — four-leg accounts + recover list (M6 Task 0)`。

- [ ] **Step 7: 裁决门(控制器执行,实现者停在报告)**

按 spec §4:Σ(腿 2/3/4)< 2.31ms 且缺口主体在 GEMM 面 → 段 2 triton 立项(Task 5 激活);否则休眠。同时裁定段 1 施工顺序(榜单序)与 Task 1 的实际形态(布局接线 vs 仅旋钮)。

---

### Task 1: GEMM 布局/调参落地(形态由 Task 0 裁决定;下文按"布局胜出"起草)

**Files:**
- Create: `acorninfer/models/gemm_dispatch.py`
- Modify: `acorninfer/weights.py`(新增 pack 函数)、`acorninfer/models/qwen3.py`(仅投影调用点)、`acorninfer/engine/engine.py`(EngineConfig + from_pretrained 接线)、`acorninfer/server.py`(CLI `--gemm-layout`,两条 _build_engine 路径透传)
- Test: `tests/test_gemm_dispatch.py`(新)

- [ ] **Step 1: 失败测试**

```python
# tests/test_gemm_dispatch.py
def test_dispatch_none_is_bitwise_flinear():
    """layout none:linear_dispatch(m, x) 与 F.linear(x, m.weight) 逐位一致。"""

def test_dispatch_t_matches_flinear_within_tol():
    """layout t:mm(x, w_t) vs F.linear(x, w):TOL 协议标定(换核不保逐位),
    标定后写死,注释命令/日期。"""

def test_pack_layout_none_is_noop_and_t_transposes():
    """pack_linear_weight(none) 不动 weight;pack(t) 后 weight 连续 [K,N],
    _m6_layout=="t";再 pack(none) 不回切(单向,报告注明)。"""

def test_apply_gemm_layout_walks_projection_only():
    """apply_gemm_layout(model, "t") 只给四类投影 + lm_head 打标;
    embed/norm 不动(枚举比对 module id 集合)。"""
```

- [ ] **Step 2: 跑红** → `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_gemm_dispatch.py -v`(FAIL:模块不存在)

- [ ] **Step 3: gemm_dispatch.py 实现**

```python
"""投影 GEMM 分派(M6 段 1 布局层)。none = 原路径逐字节;t = 装载期预变换。"""
import torch

PROJ_ATTRS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")

def pack_linear_weight(module, layout: str) -> None:
    if layout == "t":
        module.weight.data = module.weight.data.t().contiguous()  # [K, N]
        module._m6_layout = "t"

def apply_gemm_layout(model, layout: str) -> None:
    if layout == "none":
        return
    for attr in PROJ_ATTRS:
        for name, mod in model.named_modules():
            if name.endswith(attr):
                pack_linear_weight(mod, layout)

def linear_dispatch(module, x: torch.Tensor) -> torch.Tensor:
    if getattr(module, "_m6_layout", "none") == "t":
        return torch.mm(x, module.weight.data)          # weight 现为 [K, N]
    return torch.nn.functional.linear(x, module.weight)  # 原路径
```

(草稿注:`PROJ_ATTRS` 的属性名以 qwen3.py 现状为准校正——先读 attention/mlp 的模块命名;gate_up 若是融合单模块则列表照实。)

- [ ] **Step 4: qwen3.py 调用点分派(解锁条款内)**

投影调用点改 `linear_dispatch(self.<proj>, x)`;attention/rope/KV/norm 面零改动;`linear_dispatch` import 自 gemm_dispatch。lm_head 各 forward 调用点(418/495/539/586 附近)同改。

- [ ] **Step 5: 配置接线**

`EngineConfig` 增 `gemm_layout: str = "none"`(校验 ∈ {none,t});`from_config`/`from_pretrained` 装载后 `apply_gemm_layout(model, cfg.gemm_layout)`;server CLI `--gemm-layout`(缺省 none)两条 `_build_engine` 路径透传。透传测试照 `test_build_engine_chunked_budget_passthrough` 风格。

- [ ] **Step 6: 等价硬门 + A/B 门**

全套件(fast 全绿零改动);`m6_ab.py --label baseline` vs 布局开启腿:**收益 ≥0.2ms/步且无口径劣化**才保留,否则 revert(数据入 docs);token 流 parity 用 margin 门协议(非逐位,换核平局已证)。commit `feat(model): gemm layout dispatch — M6 spike winner (M6 Task 1)`。

---

### Task 2: 步开销·镜像批量提交(graph_runner 路径)

**Files:**
- Modify: `acorninfer/engine/graph_runner.py`(:139-172 每步路径)
- Test: 既有 graph parity 套件 + `tests/` 新增镜像等价钉(如无可复用既有对拍)

- [ ] **Step 1: 失败测试(等价钉)**

```python
def test_mirror_fill_flat_equals_segmented():
    """扁平构建 pin_long[0:4bk](含常量 gap 段)与现行 4 段填充 + 常量场压回,
    产出的镜像逐位一致(构造同一 seqs/views 输入对照)。"""

def test_commit_lens_batch_matches_per_view_loop():
    """_commit_lens 批量(dict.fromkeys update)后各 view._lens 与逐赋值版全等。"""
```

- [ ] **Step 2: 跑红 → Step 3: 实现(草稿)**

```python
# 镜像填充:一次扁平构建(4 段真值 + 中间常量段)替 4 次 torch.tensor + 4 次切片赋值
flat = [0] * (4 * bk)
for i, s in enumerate(seqs):            flat[i] = s.generated_ids[-1]
for i, p in enumerate(pre):             flat[bk + i] = p
for i, (p, _) in enumerate(targets):    flat[2 * bk + i] = p
for i, (_, s) in enumerate(targets):    flat[3 * bk + i] = s
# 常量段:ids/positions 的 padding 段维持既有语义(陈旧无害,照旧不重置);
# pages/slots padding 段 = scratch/0(:166-167 同职责,合并进本表)
for i in range(B, bk):
    flat[2 * bk + i] = self.scratch
    flat[3 * bk + i] = 0
st.pin_long[0:4 * bk] = torch.tensor(flat, dtype=torch.long)
# _commit_lens:36 次逐赋值/view → 一次 dict 重建
layers = range(self.cache.num_layers)
for v, n in zip(views, new_len):
    v._lens.update(dict.fromkeys(layers, n))
```

(草稿注:①现实现的常量场细节以 :105-114 init 与 :158-168 每步职责为准,扁平表必须逐位复刻含 `pin_tables` 路径不动;②`pin_tables.zero_()` + `build_tables` 不在本任务面;③`_commit_lens` 现体以源码为准,此处为等价重写。)

- [ ] **Step 4: 等价 + A/B 门**:graph parity 全绿;`m6_ab.py` C=8 graph 收益 ≥0.2ms/步,否则 revert 入档。commit `perf(engine): batched mirror fill + commit_lens (M6 Task 2)`。

---

### Task 3: H2D/D2H 同步点清账(形态由腿 4 数据定,允许"如实 no-op"收尾)

**Files:**
- Modify: 视腿 4 榜单而定(候选:executor 采样同步点 / graph_runner 镜像 copy)
- Test: 相应 parity 套件

- [ ] **Step 1**:按腿 4 的 memcpy/同步点实测榜施工;候选形态:镜像 `copy_` 已是 pinned 异步则改异步化收尾;D2H 同步(argmax/item)若上榜,评估与下一 enqueue 的重叠安全面(不改 stream 语义,graph 外才动)。
- [ ] **Step 2**:A/B 门同 Task 1(≥0.2ms 或 revert);**若数据显示该面无可回收 ≥0.2ms,任务如实以 no-op 收尾**(数据入 docs,不算失败)。
- [ ] **Step 3**:全套件回归;commit(如无代码变更则为 docs 附注,`docs(benchmarks): M6 overhead leg account (M6 Task 3)`)。

---

### Task 4: `_commit_lens` 向量化(eager 路径 + pager 有限解禁)

**Files:**
- Modify: `acorninfer/cache/pager.py`(:112-113 scatter_decode 的逐 view 提交——仅此)
- Test: 既有 KV 对拍套件(等价性钉:scatter_decode 批量提交后各 view._lens 与逐赋值全等)

- [ ] **Step 1: 失败测试(等价钉)**——构造 B views × 单层 scatter_decode 调用,断言 `_lens` 与逐赋值参考全等 + `assert_balanced` 语义不变。
- [ ] **Step 2: 实现(草稿)**——eager 侧每层调用 B 次赋值是调用结构固有(每层一次 scatter_decode);可回收面 = 腿 4 若显示此循环上榜,方案 = `scatter_decode` 增 keyword `commit_lens=True`(签名向后兼容),graph 旁路(eager decode 步)由调用方一次性批量提交。**以腿 4 榜单定去留:不上榜则 pager 零触碰,如实记录。**
- [ ] **Step 3: 全套件(KV 对拍零换金)+ A/B 门 + commit** `perf(cache): batched lens commit — M6 overhead leg (M6 Task 4)`(或 no-op docs 附注)。

---

### Task 5: (条件立项)triton decode GEMM —— **仅当 Task 0 裁决门激活**

**Files:**
- Create: `acorninfer/ops/decode_gemm.py`(triton kernel:四类 (N,K),M∈[1,16],fp32 累加,128-bit 加载,K-split 大 K 类)
- Modify: `acorninfer/models/gemm_dispatch.py`(第三路分派)、`acorninfer/weights.py`(如需 triton 布局 pack)
- Test: `tests/test_decode_gemm.py`(golden TOL 协议,标定后写死)

- [ ] **Step 1: 失败测试**——`test_triton_gemm_matches_cublas_tol`(四类形状 × M∈{1,7,8,16},对照 cuBLAS,TOL 标定);`test_dispatch_picks_best_per_shape`(三路择优表装载期实测)。
- [ ] **Step 2: kernel 实现 + 对拍标定** → 冷口径 GB/s 对照表(triton vs cuBLAS vs 布局版);**超 cuBLAS 最优 5% 以上才接入 dispatch**,否则收手入档(R5)。
- [ ] **Step 3: 全套件 + A/B 门(≥0.2ms)+ commit** `feat(ops): triton decode gemm — M6 stage-2 (M6 Task 5)`(未激活则本任务跳过,报告注明)。

---

### Task 6: 回归 + 双轨仪式收口 + docs 终账

- [ ] **Step 1: 全量回归**:fast 全绿(≥138,零换金)、slow 17 全绿;`m6_ab.py` 终版数字(C=1/C=8 × graph/eager)对照基线(17.95/21.12 TPOT)。
- [ ] **Step 2: 双轨仪式**(同 M5 协议,串行验净):轨 1 四腿(vLLM TP=1 8100 / AcornInfer graph 8200 满配方 / eager / HTTP 缺省补充)主验 + 达标率逐档;轨 2(TP=2)副账如实(GEMM 收益外溢量);收服 PID kill + nvidia-smi 验净。
- [ ] **Step 3: docs M6 终账**:可回收清单 → 兑现对照表、逐项 A/B 账、达标率逐档、判定行(如实;锚不过则北极星继续挂账 + 差距归因)、段 2 立项与否实录;tag/push 留用户。commit `docs(benchmarks): M6 final — roofline closure accounts (M6 Task 6)`。

---

## 计划自审记录

1. **Spec 覆盖**:§4 四腿 → Task 0;§5 四项 → Task 1-4;§6 → Task 5(条件);§7 保护条款 → 全局纪律;§8 风险(R1 腿 2 排除定位/R2 腿 3 引擎内复测硬门/R3 Task 3 捕获外约束/R4 解锁条款+byte-identical/R5 Task 5 择优收手/R6 Task 6 如实判定);§10 验收 → Task 6。§9 non-goals 未越界。
2. **占位扫描**:Task 2/3/4/5 均含完整草稿或"数据定形态 + 如实 no-op 收尾"的显式协议(不是 TBD;A/B 门与证据要求写死)。
3. **类型一致性**:`pack_linear_weight(module, layout)`/`linear_dispatch(module, x)`/`apply_gemm_layout(model, layout)` 三签名全任务一致;`m6_ab.py --label <名>` 输出行格式全任务复用;A/B 门阈值 0.2ms 统一。
4. **计数账**:fast 138 基线;T1 新增 ~4、T2 ~2、T4 ~1、T5 ~2(条件);slow 17 不变(零换金硬门)。
5. **薄弱点**:Task 1 的 `PROJ_ATTRS` 命名以源码为准(草稿已注);Task 2 扁平镜像必须逐位复刻常量场语义(既有 parity 钉子守门);段 2/Task 3 的 no-op 收尾是合法结局,要求证据入 docs。
