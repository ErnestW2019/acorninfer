# M7 weight-only 量化 实施计划(W8A16 → W4A16 门控)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 decode 权重流从 15.13GB/步压到 W8 7.56GB / W4 ~4.03GB,主锚 W8 C=8 TPOT ≤14.0ms、C=1 ≤10.5ms(段 B 门控后 W4 C=8 ≤10.5ms),精度锚 W8 ppl ≤+1%。

**Architecture:** 段 0 RTN 数学 + 微基准(η 表 → 锚修正一次条款)→ 段 A W8 kernel 定档 → 生产接线 → ppl harness → 段 B 门控(W4)→ 仪式四腿收口。

**Tech Stack:** 纯 torch RTN 数学 / triton dequant-GEMM(复用 `ops/decode_gemm.py` 全套骨架)/ 自研困惑度 harness(自家 Tokenizer)。

**Spec:** `docs/superpowers/specs/2026-11-01-m07-weight-quant-design.md`(锚/门/保护面/风险以 spec 为准)

---

## 全局纪律(每个任务都适用,实现者必读)

- **计划代码是草稿,测试才是契约。** 草稿错以代码现状与实测为准修,报告逐条记录偏离;kernel 形态、autotune 网格以微基准数据为准。
- **保护文件**:golden m0–m5、`cache/kv.py`、`cache/pager.py`、`engine/scheduler.py`、`engine/sequence.py`、`engine/graph_runner.py`、`ops/paged_attention.py`、`tests/triton_utils.py`(TOL 只用不改)照旧禁改。**解锁**:`models/qwen3.py`(投影+lm_head 调用点分派,第三次;attention/rope/KV/norm 一行不动)、`weights.py`(仅增量)、`acorninfer/config.py` + `engine/engine.py` + `server.py`(quant 配置面)、`ops/decode_gemm.py`(骨架复用/增量,**既有 bf16 kernel 与其 17 测试行为零变化**)。
- **byte-identical 硬门**:`--quant none`(缺省)时 token 流与全套件与 M6 收官逐字节一致;量化路径数值等价用 **fp32 dequant 参考实现对拍 + 新标定 TOL**(命令/日期注释写死);量化 vs bf16 是不同数值域,parity 用 margin 门协议抽查漂移量级(M5 先例),不做逐位断言。
- **环境纪律**:单卡 `CUDA_VISIBLE_DEVICES=1` + `.venv/bin/`;跑前 nvidia-smi(应 ~4MiB);timeout ≥ 1800000;运行时禁 import transformers(含 perplexity.py);容差只准 TOL 协议标定后写死。
- **基线**:M6 终账(HEAD `d55f930`):TPOT 17.95/21.12(C=1/C=8);m6_ab step 带 graph 17.91–17.96 / 19.87–19.93,eager 19.13–19.35 / 21.11–21.28;fast **155 passed, 6 skipped, 17 deselected** / slow **17** / tp **13**(如实口径);η 推导假设 W8 0.85 / W4 0.75(锚修正条款:>10% 偏离允许一次修正,docs 记录)。
- 权重流账(复核口径):q (4096,4096) 33.55MB、k/v (1024,4096) 各 8.39MB、o (4096,4096) 33.55MB、gate/up (12288,4096) 各 100.66MB、down (4096,12288) 100.66MB → 每层 385.86MB ×36 = 13.89GB + lm_head (151936,4096) 1245MB = **15.13GB/步**;W8 ≈ 7.56GB(scale 可忽略),W4 ≈ 3.78GB + g128 元数据 ~6.3% ≈ 4.03GB。

---

### Task 0: RTN 数学 + 段 0 微基准(锚修正复核 + 段 A GO 数据)

**Files:**
- Create: `acorninfer/quant/__init__.py`、`acorninfer/quant/quantize.py`(纯 torch,免校准 RTN)
- Create: `scripts/m7_spike.py`(变体孤立微基准;冷口径复用 `decode_gemm._bench_rot` 思路)
- Test: `tests/test_quantize.py`(新;纯 CPU 可跑)

- [ ] **Step 1: quantize.py 失败测试**(数值性质钉:`dequant(q)≈w` 逐元素误差 ≤ scale 量化步长理论界;round-trip;打包/解包 round-trip;shape/边界:K=4096、12288 与 g128 整除;EPS clamp)

```python
# tests/test_quantize.py(节选;实现者补全)
def test_w8_roundtrip_within_theoretical_bound():
    """w8:|dequant(q,s) - w| <= scale/2 + eps(逐元素,CPU 随机张量)。"""
def test_w4_roundtrip_within_theoretical_bound():
    """w4 g128:|dequant - w| <= scale/2 + eps(组内)。"""
def test_pack_w4_roundtrip():
    """pack(unpack(p)) == p;低 nibble = 偶数列(约定钉死,注释写明)。"""
def test_shape_guards():
    """K/N 非 128 倍数 → ValueError(spec §2 零特例口径的守门)。"""
```

- [ ] **Step 2: 跑红 → Step 3: quantize.py 实现(草稿)**

```python
"""RTN weight-only 量化数学(纯 torch;spec §2/§4)。"""
import torch
EPS = 1e-8

def quantize_w8(w):            # w [N,K] bf16 → int8 [N,K] + scale fp16 [N]
    w32 = w.float()
    scale = w32.abs().amax(dim=1).clamp_min(EPS) / 127.0
    q = torch.round(w32 / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale.half()

def quantize_w4(w, group=128): # → uint8 [N,K](0..15)+ scale/zp fp16 [N, K//g]
    N, K = w.shape; assert K % group == 0
    wg = w.float().view(N, K // group, group)
    wmax, wmin = wg.amax(dim=2), wg.amin(dim=2)
    scale = ((wmax - wmin) / 15.0).clamp_min(EPS)
    zp = wmin
    q = torch.round((wg - zp[..., None]) / scale[..., None]).clamp_(0, 15)
    return q.to(torch.uint8).view(N, K), scale.half(), zp.half()

def pack_w4(q):                # [N,K] → [N,K//2]:byte = q_even | q_odd<<4
    q = q.view(q.shape[0], q.shape[1] // 2, 2)
    return (q[..., 0] | (q[..., 1] << 4)).contiguous()

def dequant_w8(q, scale): ...  # fp32 参考实现(TOL 对拍锚)
def dequant_w4(q, scale, zp): ...
```

- [ ] **Step 4: `m7_spike.py` 微基准(段 0)**

变体面(triton 探索级 kernel,直接写在 spike 脚本;生产 kernel Task 1 定档):
- W8:`{BLOCK_N 64/128/256} × {BLOCK_K 64/128} × {warps 4/8} × {stages 2/3/4}`(dequant 在核内:int8 load → fp32 ×scale → bf16 → tl.dot);scale 广播方式(行向量 per-channel)固定。
- W4:同网格 + unpack 两法(位运算交织 vs `tl.interleave`,取实测优)+ BLOCK_K≤g128 约束两形态(BLOCK_K=g128 vs gather 索引 (k0+i)//g)。
- 类形状五档:q/o (4096,4096)、kv (1024,4096)、gate/up (12288,4096)、down (4096,12288)、lm_head (151936,4096);M ∈ {1, 8}(decode 口径;M≤16 全 pad 16)。
- 冷口径:36 套轮转(权重字节口径;照 `_ROT_SETS` 法);输出行 `QSPIKE <fmt> <variant> <class> M=<m> us=<v> gb=<v>`(gb = 量化权重字节/us)。
- SPLIT_K 需求:down/lm_head 的 N-不足类照 `_autotune_config` 先例评估,fp32 partial + second-pass 确定性归零 atomic 路线复用。
- **capture 实测**:胜出配置的 kernel 进 `torch.cuda.CUDAGraph` 捕获 + 重放逐位钉(照 M6 T5 协议)。
- 汇总:η 表(vs roofline 1.008TB/s 权重字节口径)+ 锚修正判定数据(实测 η vs 0.85/0.75)。

- [ ] **Step 5: docs 段 0 节 + commit `feat(quant): RTN math + m7 spike microbench (M7 Task 0)`**

- [ ] **Step 6: 控制器门(实现者停在报告)**:η 若偏离推导 >10% → 锚一次修正(docs 记录);段 A 施工 GO 确认;W4 kernel 胜出形态记录供段 B。

---

### Task 1: W8 kernel 定档(ops/quant_gemm.py)

**Files:**
- Create: `ops/quant_gemm.py`(定档 kernel + autotune + dispatch)
- Modify: `ops/decode_gemm.py`(仅骨架复用面:`_bench_rot`/`_bucket_for`/workspace 模式导入或按现状共享;bf16 kernel 零变化)
- Test: `tests/test_quant_gemm.py`(新)

- [ ] **Step 1: 失败测试**(fp32 参考 = `dequant_w8` 后 fp32 matmul)

```python
TOL_W8 = (rtol=…, atol=…)  # 随机 + 真实分布权重标定,命令/日期注释写死
def test_w8_gemm_matches_fp32_reference():    # 五类形状 × M∈{1,7,8,16}
def test_w8_splitk_deterministic():            # 若启用:同输入两次重放逐位同
def test_w8_capture_replay_bitwise():          # graph 捕获含 kernel,重放逐位(T5 协议)
def test_dispatch_no_host_sync():              # set_sync_debug_mode 下 launch 干净
```

- [ ] **Step 2: 跑红 → Step 3: 定档实现**(Task 0 胜出变体进产线;autotune 装载期,warm <1s 预算;`quant_gemm(x, q, scale, cls, cfg)` + bucket 分派)

- [ ] **Step 4: 全套件(decode_gemm 17 测试零变化)+ commit `feat(ops): W8 dequant-GEMM kernels (M7 Task 1)`**

---

### Task 2: W8A16 生产接线(段 A 落地)

**Files:**
- Modify: `acorninfer/quant/quantize.py`(增 `apply_quant(model, mode)` walk)、`models/qwen3.py`(调用点分派)、`acorninfer/config.py`(EngineConfig)、`engine/engine.py`、`server.py`
- Test: `tests/test_quant_wiring.py`(新;CPU/tiny 为主)

- [ ] **Step 1: 失败测试**

```python
def test_config_validates_quant():            # EngineConfig.quant ∈ {none,w8a16,w4a16};非法 ValueError
def test_quant_tp2_mutually_exclusive():      # quant × tp>1 构造期报错(spec §4)
def test_apply_quant_walk_targets():          # 8 模块类打标 _m7_quant;embed/norm/KV 零触碰(枚举断言)
def test_none_path_bitwise():                 # quant none:linear_dispatch 全部走 F.linear 旧路径(tiny 全等)
def test_server_passthrough_quant():          # _build_engine 透传(照 chunked-budget 风格)
```

- [ ] **Step 2: 跑红 → Step 3: 接线(草稿形态)**

```python
# quantize.py 增量:walk 后权重换 int8 存储 + scale 挂 attr(参数仍在 target device)
QUANT_ATTRS = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head")
def apply_quant(model, mode, group=128):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in QUANT_ATTRS:
            q, s = quantize_w8(mod.weight.data)
            mod.weight.data = q            # int8 Parameter(data 换存储;T1b 先例)
            mod._q_scale = s; mod._m7_quant = "w8a16"
# qwen3.py 调用点:q = dispatch_linear(self.q_proj, hidden) 形态(4 前向 × 7 投影 + lm_head×4);
# dispatch_linear:none → mod(x)(F.linear 原路径逐字节);w8a16 → quant_gemm(x, mod.weight.data, mod._q_scale, cls=…)
# cls 由 shape 查表(5 档);EngineConfig.quant 校验 + tp 互斥;engine.from_config / server._build_engine 装载后 apply_quant
```

(草稿注:①k/v N=1024 也是 128 倍数 ✓;②module 名末段匹配用 `name.split(".")[-1]` 防跨层误伤;③`_m7_quant` attr 命名以本任务实现为准,全任务一致即可。)

- [ ] **Step 4: 等价硬门 + A/B 门**

- 全套件(fast 全绿零换金;none 缺省逐字节);
- **A/B**:`m6_ab.py --quant w8a16`(harness 增此旗标,生产同构)vs M6 带:门 = **C=8 graph 收益 ≥4.0ms 且 C=1 graph 收益 ≥5.0ms 且 eager 双 C 不劣化**;任一不过 → 停手报告(不 revert 生产接线之外的东西;接线本身问题则修到过);
- margin 门 parity 抽查(tiny + 8B 单请求,漂移量级入 docs)。
- [ ] **Step 5: commit `feat(model): W8A16 production wiring (M7 Task 2)`**

---

### Task 3: 困惑度 harness + W8 精度账

**Files:**
- Create: `scripts/perplexity.py`
- Create: `data/wikitext-2-raw/`(下载;不入 git,.gitignore 增量)
- Modify: `docs/benchmarks.md`(精度账节)

- [ ] **Step 1: wikitext-2-raw 本地缓存**(命令原样入 docs;首选 `https://wikitext.smerity.com/wikitext-2-raw-v1.zip`,备选 HF parquet;R7 取活者)

- [ ] **Step 2: `scripts/perplexity.py`**(自家 Tokenizer;`--quant {none,w8a16}` `--window 512 --stride 256`;直接 `qwen3.from_pretrained` 建模型走无 KV 全前向,CE=sum NLL/token 数;输出 token 数、NLL 总额、ppl 可复算;`CUDA_VISIBLE_DEVICES=1`)

- [ ] **Step 3: bf16 基线标定 → W8 计量** → 精度锚判定(≤+1%;超标照实,一次非对称改形机会按 spec R4 由控制器裁)

- [ ] **Step 4: docs 精度账 + commit `feat(scripts): perplexity harness + W8 accuracy account (M7 Task 3)`**

---

### Task 4: (门控)段 B W4A16 —— **仅当控制器按 spec §2 三条件裁决 GO**

**Files:**
- Modify: `ops/quant_gemm.py`(W4 kernel 定档)、`acorninfer/quant/quantize.py`(walk 增 w4 分支 + pack)、`models/qwen3.py`/`server.py`(w4a16 透传)、`scripts/m6_ab.py`(--quant w4a16)
- Test: `tests/test_quant_gemm.py` 增 W4 面

- [ ] **Step 1: 失败测试**(W4 fp32 参考 TOL;unpack 约定钉(低 nibble=偶数列);组 gather 索引钉;capture 重放;split-K 若启用)
- [ ] **Step 2: kernel 定档(Task 0 胜出形态)+ autotune**(BLOCK_K/g 约束按 Task 0 数据)
- [ ] **Step 3: 接线(apply_quant w4 分支:pack 存储 + scale/zp attrs;dispatch w4a16 路)**;全套件零换金
- [ ] **Step 4: A/B 门(C=8 graph 收益对照 W8 再 ≥1.5ms 或按锚修正后口径;eager 不劣化)+ ppl(W4 vs bf16 ≤+5% 目标,实测定档)**
- [ ] **Step 5: docs 段 B 账 + commit `feat(ops): W4A16 group dequant kernel + wiring (M7 Task 4)`**(NO-GO 则 `docs(m7): stage-B gate NO-GO — <死因> (M7 Task 4)`)

---

### Task 5: 回归 + 仪式四腿 + docs 终账

- [ ] **Step 1: 全量回归**:fast(≥155+新增全绿零换金)/ slow 17 / tp 13;`m6_ab.py --quant none` 复现 M6 带(零漂移硬证)+ w8/w4 终版数字。
- [ ] **Step 2: 仪式四腿(GPU 0 照 M5/M6 配方)**:vLLM bf16 8100 / AcornInfer none 8200 满配方 / w8a16 / w4a16(如 GO),各 `bench_remote.py` C=1/4/8;none 腿应复现 M6 终账达标率(92.4/90.3/83.4 带内);关服 `ss -tlnp` 真身 PID 定向 kill,双卡回落 ~4MiB 验净。轨 2 免跑(TP×quant 互斥)。
- [ ] **Step 3: docs M7 终账**:锚判定行(过/不过如实)+ 锚修正记录(如触发)+ ppl 三行 + η 表(微基准 vs 引擎内)+ 段 B GO/NO-GO + M8 台账更新;仪式命令原样。commit `docs(benchmarks): M7 final — quant anchors & accuracy accounts (M7 Task 5)`。

---

## 计划自审记录

1. **Spec 覆盖**:§1 锚 → T0 修正条款 + T5 判定;§2 两段门控 → T0-T4;§3 ppl → T3;§4 装载/互斥 → T2;§5 保护面 → 全局纪律;§6 R1-R7(R1→η 门/T0、R2→T4 Step1、R3→T4 Step4、R4→T3 Step3、R5→T3(装载耗时在 T2 报告)、R6→T0/T1、R7→T3 Step1);§7 non-goals 未越界;§8 仪式 → T5。
2. **占位扫描**:全部任务含完整草稿或预注册门;T4 条件任务有 NO-GO 收尾协议;TOL 数值留给标定(协议明文要求命令/日期,非 TBD)。
3. **类型一致性**:`quantize_w8/quantize_w4/pack_w4/dequant_*` 签名 T0-T4 一致;`apply_quant(model, mode, group)` 贯穿;A/B 输出行沿用 m6_ab `AB <label> …`;五类形状清单(含 kv (1024,4096))全任务一致——**注意 M6 的"gate_up (24576,4096)"是聚合口径,生产是 gate/up 各 (12288,4096),微基准与 autotune cls 键按生产五档走**。
4. **计数账**:fast 155 基线;T0 ~4、T1 ~4、T2 ~5、T4 ~4(条件);slow 17 不变(零换金)。
5. **薄弱点**:`mod.weight.data = q`(int8 Parameter data 换存储)与 nn.Linear 元数据交互以 T2 实测为准(草稿注③);W4 nibble 交织的 triton 实现形态(位运算 vs `tl.interleave`)以 T0 实测为准;lm_head M=decode batch ≤8 在 M≤16 桶内,无需特殊处理。
