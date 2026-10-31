# M4.1 图外残差清零 + prefill 批量化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M4 验收归因出的两块非 GPU 地板差距清掉——图外 CPU 残差 1.8ms/步(argmax 逐行同步 + 6-7 次小 H2D)与 prefill 串行化(TTFT 147ms vs vLLM 82ms)——预期达标率 C=1 ~94-95% / C=4 ~91% / C=8 ~84%,当日配对仪式如实入档。

**Architecture:** 段一:`DecodeGraphRunner` 静态缓冲改为按 dtype 两块扁平 GPU 张量的切片视图 + pinned CPU 镜像(每步 2 次 H2D),argmax 合批单次同步(runner 与 eager greedy 路径同享);`build_tables` 加 `out=` 直填 CPU 镜像。段二:`Qwen3ForCausalLM.forward_prefill_batch`(padded [N, Lmax] 合批 GEMM,逐行 `view.update` + causal SDPA 保证行间隔离与语义同构)+ `Executor.prefill_batch` + `Engine.step` 按 `len(prefills)` 分流(N=1 走原路径 byte-identical)。scheduler/kv/pager/sequence **零触碰**。

**Tech Stack:** torch 2.13 pinned memory / CUDA graph(已落地)/ SDPA;既有 M0-M4 全部基建。

**Spec:** `docs/superpowers/specs/2026-10-31-m41-residual-and-batched-prefill-design.md`

---

## 全局纪律(每个任务都适用,实现者必读)

- **计划代码是草稿,测试才是契约。** 本计划所有代码块都是意图与契约的草稿(M4 期间计划草稿 5 处 bug 被测试/审查逮出)。对拍不过时,以数值事实与推导为准修代码,并在报告里记录偏离。
- **保护文件(禁改)**:`tests/test_golden_m0/m1/m2/m3/m4.py`、`acorninfer/cache/kv.py`、`acorninfer/cache/pager.py`、`acorninfer/engine/scheduler.py`、`acorninfer/engine/sequence.py`、`tests/triton_utils.py`。**换题例外(仅 Task 6,预先授权)**:若 golden m4 因 prefill 合批翻题,允许按 M0 先例对 `tests/test_golden_m4.py` 的 PROMPTS 换题并记录 margin(单独 commit,控制器复签);m0-m3 无此例外。
- **环境纪律**:GPU 0 留给验收仪式的 vLLM;开发/测试一律 `CUDA_VISIBLE_DEVICES=1` 前缀 + `.venv/bin/` 前缀,代码写 `cuda:0`;跑前 `nvidia-smi` 查 GPU 1 残留(共享机)。**前台长 timeout**(≥600000;8B 相关 ≥1800000),禁 `&` + 轮询。
- 运行时禁 import transformers;实测数字才可入 docs;容差只准用 `tests/triton_utils.py` 的 TOL 协议(标定后写死)。
- 当前基线:fast **99 passed, 13 deselected**;slow **17 passed**(golden m0-m4)。HEAD 见 `git log`(spec 提交 `81825b0` 之后)。

---

### Task 1: argmax 合批(单次 D2H 同步)

**Files:**
- Modify: `acorninfer/engine/graph_runner.py`(decode_batch 返回值)
- Modify: `acorninfer/engine/executor.py`(decode_batch eager 全 greedy 分流)
- Test: `tests/test_engine.py`(追加 2 测试)

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_engine.py`,先读该文件 import/marker 风格)

```python
@pytest.mark.gpu
def test_executor_decode_batch_greedy_matches_per_row_argmax():
    """eager decode(graph off)全 greedy:合批 argmax(dim=1) 与逐行 argmax
    token 逐位一致(防实现滑步);这正是 bench eager 锚点走的路径。"""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    model.attention_backend = "triton"
    model.fused_ops = True
    model.decode_graph = False                 # Executor 无 runner → eager 路径
    engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
    gen = torch.Generator(device="cuda").manual_seed(3)
    sids, seqs = [], []
    for L in [6, 10]:
        ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
        sids.append(engine.submit(ids[0].tolist(),
                                  SamplingParams(temperature=0.0), max_tokens=4))
    # 直接钉 Executor:同 logits 下合批返回 == 逐行 argmax
    seq_objs = [engine.seqs[sid] for sid in sids]
    views = [PagedCacheView(engine.cache, seq.block_table) for seq in seq_objs]
    # 两序列尚无 KV(prefill 前)——改为走 engine 两步后再钉,或用 stub:
    # 简化:stub seqs + 真实 views 不可行;改走 engine 全流程对拍:
    toks = {sid: [] for sid in sids}
    while engine.has_pending():
        for o in engine.step():
            toks[o.seq_id].append(o.token_id)
    for sid in sids:
        engine.drop(sid)
    engine.assert_balanced()
    model.to("cpu")
    # 同 seed 重建逐行参考引擎(逐行 argmax = _sample 现行为),token 逐位
    torch.manual_seed(0)
    model2 = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    model2.attention_backend = "triton"
    model2.fused_ops = True
    model2.decode_graph = False
    engine2 = Engine(model2, None, num_blocks=64, page_size=4, max_running=8)
    gen2 = torch.Generator(device="cuda").manual_seed(3)
    sids2 = []
    for L in [6, 10]:
        ids = torch.randint(0, 256, (1, L), generator=gen2, device="cuda")
        sids2.append(engine2.submit(ids[0].tolist(),
                                    SamplingParams(temperature=0.0), max_tokens=4))
    toks2 = {sid: [] for sid in sids2}
    while engine2.has_pending():
        for o in engine2.step():
            toks2[o.seq_id].append(o.token_id)
    for sid in sids2:
        engine2.drop(sid)
    engine2.assert_balanced()
    model2.to("cpu")
    assert [toks[s] for s in sids] == [toks2[s] for s in sids2]
```

**注意**:此测试在实现前也应通过(两腿都是旧实现)——它不是红绿门,是**防回归对拍门**(Task 1 Step 3 改完后仍须绿)。真正的红绿钉子是第二个测试:

```python
@pytest.mark.gpu
def test_executor_decode_batch_mixed_temperature_falls_back_per_row():
    """混合温度批(graph off):temp>0 行必须走逐行 _sample/Sampler(RNG 交织
    契约),全 greedy 才允许合批 argmax。钉住分流条件。"""
    from types import SimpleNamespace
    import acorninfer.engine.executor as executor_mod
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    ex = executor_mod.Executor(model)
    greedy = SimpleNamespace(sampling=SamplingParams(temperature=0.0),
                             generated_ids=[7])
    sampled = SimpleNamespace(sampling=SamplingParams(temperature=0.5, top_p=0.9),
                              generated_ids=[11])
    # 同一 eager 前向喂两种批:全 greedy vs 混合——混合批必须仍然工作且
    # greedy 行结果与全 greedy 批中该行一致(同输入行)
    from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
    alloc = BlockAllocator(32)
    cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    view = PagedCacheView(cache, [])
    model(torch.full((1, 4), 5, device="cuda", dtype=torch.long), view)  # 预热 KV
    torch.manual_seed(123)
    mixed = ex.decode_batch([greedy, sampled], [view, view])
    torch.manual_seed(123)
    all_greedy = ex.decode_batch(
        [greedy, SimpleNamespace(sampling=SamplingParams(temperature=0.0),
                                 generated_ids=[11])], [view, view])
    assert mixed[0] == all_greedy[0]   # greedy 行与批构成无关(确定性)
    assert isinstance(mixed[1], int)
```

(草稿:SimpleNamespace stub 的字段以 `Executor.decode_batch`/`_sample` 实际读取为准 —— `generated_ids[-1]` 与 `sampling.temperature`;若实现还读别的字段,补 stub。decode 对同一 view 连跑两次会推进 `_lens`,两次调用之间 KV 状态不一致不影响本断言(各行输出只由该行 logits 决定,两次调用各自 argmax/Sampler)。若实测发现状态耦合导致断言不可靠,改为两套独立 cache/views 再跑,并在注释记录。)

- [ ] **Step 2: 跑测试**(新增测试当前应绿——它们钉"改后行为不变";真正验证靠 Step 4 改后全部既有测试仍绿)

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_engine.py -v -k "greedy_matches or mixed_temperature"`
Expected: 2 passed

- [ ] **Step 3: 实现**

`acorninfer/engine/graph_runner.py` `decode_batch` 尾部:

```python
        self.last_logits = st.logits
        # 合批 argmax:B 次 D2H 同步 → 1 次(图外残差清理,Task 1)
        return [int(t) for t in st.logits[:B].argmax(dim=1).tolist()]
```

`acorninfer/engine/executor.py` `decode_batch` eager 尾部改为:

```python
        logits = self.model.forward_decode(x, views)  # [B, vocab]
        if all(s.sampling.temperature == 0 for s in seqs):
            # 全 greedy:与 runner 同款合批 argmax(单次 D2H)
            return [int(t) for t in logits.argmax(dim=1).tolist()]
        return [self._sample(s, logits[i]) for i, s in enumerate(seqs)]
```

- [ ] **Step 4: 全套件**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q`
Expected: **101 passed, 13 deselected**(99 + 2;所有既有测试零改动通过 = 行为保持证据)

- [ ] **Step 5: Commit**

```bash
git add acorninfer/engine/graph_runner.py acorninfer/engine/executor.py tests/test_engine.py
git commit -m "perf(engine): fused batched argmax — one D2H sync per decode (M4.1 Task 1)"
```

---

### Task 2: 扁平缓冲 + pinned 镜像(H2D 6-7 次 → 2 次)

**Files:**
- Modify: `acorninfer/engine/graph_runner.py`(`_build_state` + `decode_batch`)
- Modify: `acorninfer/ops/paged_attention.py`(仅 `build_tables` 加 `out=`)
- Test: `tests/test_graph_runner.py`、paged-attention 既有测试文件(先 `ls tests/` 定位,K3 相关)

- [ ] **Step 1: build_tables 的 out= 测试**(追加到既有 paged-attention 测试文件)

```python
def test_build_tables_out_fills_cpu_buffer_without_device_transfer():
    """out= 契约:直接填调用方的 CPU 缓冲(≤ [B, W]),零 H2D;返回的
    tables 就是 out 的前 B 行视图,lens 为 CPU 张量。默认路径行为不变。"""
    class _V:
        def __init__(self, table):
            self.table = table
    views = [_V([3, 7, 9]), _V([1])]
    out = torch.full((4, 8), 99, dtype=torch.int32)   # 故意比需要大
    t, lens = build_tables(views, 4, "cpu", lens_override=[11, 2], out=out)
    assert t.data_ptr() == out.data_ptr()             # 零拷贝视图
    assert t.shape[0] == 2
    assert out[0, :3].tolist() == [3, 7, 9] and out[0, 3:].tolist() == [0] * 5
    assert out[1, :1].tolist() == [1] and out[1, 1:].tolist() == [0] * 7
    assert out[2:].tolist() == [[99] * 8, [99] * 8]   # 未触及行保持原值(调用方自己管 padding)
    assert lens.tolist() == [11, 2]
    # 默认路径(不传 out)不受影响
    t2, lens2 = build_tables(views, 4, "cpu", lens_override=[11, 2])
    assert t2.tolist() == out[:2].tolist()
```

(草稿:stub 视图只需 `table` 属性——按 `build_tables` 现状核实其读取面;`out[: len(views)]` 的 zero 语义 = 填充值 0,与默认路径 `torch.zeros` 一致;out 未触及行调用方自管——runner 每步全量 zero 自己的镜像区。)

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest <paged-attention 测试文件> -k out -v`
Expected: FAIL(TypeError: unexpected keyword argument 'out')

- [ ] **Step 3: build_tables 加 out=**(`acorninfer/ops/paged_attention.py`,docstring 追加 out 契约)

```python
def build_tables(views, page_size: int, device, *,
                 lens_override=None, out=None) -> tuple[torch.Tensor, torch.Tensor]:
    # ……原 docstring 保留,追加:
    # out(M4.1):给定时把块表直接填进调用方的 CPU 缓冲(out[:B],先清零),
    # 跳过内部分配与 H2D(pinned 镜像路径);返回 (out 前 B 行视图, CPU lens)。
    if lens_override is not None:
        lens = [int(L) for L in lens_override]
    else:
        lens = [int(v.seq_len) for v in views]
    max_pages = max(1, max(-(-L // page_size) for L in lens))
    if out is not None:
        assert out.shape[0] >= len(views) and out.shape[1] >= max_pages, \
            "out 缓冲必须 ≥ [B, max_pages]"
        t = out[: len(views)]
        t.zero_()
    else:
        t = torch.zeros((len(views), max_pages), dtype=torch.int32)
    for i, v in enumerate(views):
        n = -(-lens[i] // page_size)
        t[i, :n] = torch.tensor(v.table[:n], dtype=torch.int32)
    if out is not None:
        return t, torch.tensor(lens, dtype=torch.int32)
    return t.to(device), torch.tensor(lens, dtype=torch.int32, device=device)
```

(以现状为准:现返回语句若为 `t.to(device), torch.tensor(lens, ...)` 照此;out 路径的 lens 留 CPU —— runner 不用它。)

- [ ] **Step 4: `_build_state` 扁平化 + pinned 镜像**(草稿;字段名/切片布局以现状为准,`_BucketState` 相应加字段)

```python
    def _build_state(self, bucket: int) -> _BucketState:
        dev = self.device
        W = self.allocator.total
        # 扁平 GPU 缓冲:静态字段 = 切片视图(共享存储 = graph 捕获地址)
        flat_long = torch.zeros(4 * bucket, dtype=torch.long, device=dev)
        flat_i32 = torch.zeros(bucket * (1 + W), dtype=torch.int32, device=dev)
        st = _BucketState(
            bucket=bucket,
            flat_long=flat_long,
            flat_i32=flat_i32,
            ids=flat_long[0:bucket].view(bucket, 1),
            positions=flat_long[bucket:2 * bucket],
            pages=flat_long[2 * bucket:3 * bucket],
            slots=flat_long[3 * bucket:4 * bucket],
            lens=flat_i32[0:bucket],
            tables=flat_i32[bucket:].view(bucket, W),
            views=[PagedCacheView(self.cache, [self.scratch]) for _ in range(bucket)],
            graph=torch.cuda.CUDAGraph(),
            logits=None,
        )
        # pinned CPU 镜像(同构布局);平台不支持 pinned 时回退普通内存
        st.pin_long = self._pinned(4 * bucket, torch.long)
        st.pin_i32 = self._pinned(bucket * (1 + W), torch.int32)
        st.pin_long[2 * bucket:3 * bucket].fill_(self.scratch)  # padding 常量场
        st.pin_i32[0:bucket].fill_(1)                           # padding lens 恒 1
        st.pin_tables = st.pin_i32[bucket:].view(bucket, W)
        st.pin_tables[:, 0].fill_(self.scratch)                 # padding 行 tables 第 0 列
        return st

    @staticmethod
    def _pinned(numel: int, dtype) -> torch.Tensor:
        try:
            return torch.empty(numel, dtype=dtype, pin_memory=True)
        except RuntimeError:   # 平台/驱动不支持 pinned:回退(收益略降,正确性不变)
            return torch.empty(numel, dtype=dtype)
```

`_BucketState` dataclass 加字段:`flat_long`、`flat_i32`、`pin_long`、`pin_i32`、`pin_tables`(各带一行注释;pinned 镜像可标注"非 dataclass 字段亦可,实例属性即可"——以实现整洁为准,但必须与草稿字段名一致以便 decode_batch 引用)。

- [ ] **Step 5: `decode_batch` 改走镜像单发**(草稿;保持 Task 1 的合批 argmax)

```python
    def decode_batch(self, seqs, views):
        """与 Executor.decode_batch 同参;seqs 仅取 generated_ids[-1]。"""
        B = len(views)
        assert 1 <= B <= BUCKETS[-1]
        assert len(seqs) == B, "seqs/views 必须一一对应"
        st = self.states[min(b for b in BUCKETS if b >= B)]
        bk = st.bucket
        targets = [v.ensure_append_capacity(0) for v in views]
        pre = [int(v.seq_len) for v in views]
        new_len = [p + 1 for p in pre]
        # CPU 镜像填充(pinned;零 H2D)——padding 场为 init 常量,[:B] 之外不动
        st.pin_long[0:B] = torch.tensor([s.generated_ids[-1] for s in seqs],
                                        dtype=torch.long)
        st.pin_long[bk:bk + B] = torch.tensor(pre, dtype=torch.long)
        st.pin_long[2 * bk:2 * bk + B] = torch.tensor([p for p, _ in targets],
                                                      dtype=torch.long)
        st.pin_long[3 * bk:3 * bk + B] = torch.tensor([s for _, s in targets],
                                                      dtype=torch.long)
        st.pin_i32[0:B] = torch.tensor(new_len, dtype=torch.int32)
        st.pin_tables.zero_()                                # 真实行 + padding 全清
        build_tables(views, self.page_size, self.device,
                     lens_override=new_len, out=st.pin_tables)   # 直填 CPU 镜像
        st.pin_tables[B:, 0] = self.scratch                  # padding 行常量场(zero 后重写)
        # 单发 H2D(每 dtype 一次;pinned → 扁平 GPU 缓冲,切片视图地址 = 捕获地址)
        st.flat_long.copy_(st.pin_long)
        st.flat_i32.copy_(st.pin_i32)
        self._commit_lens(views, new_len)
        st.graph.replay()
        # 图输出缓冲原址覆写:仅本步有效;parity 测试的对拍观察点
        self.last_logits = st.logits
        return [int(t) for t in st.logits[:B].argmax(dim=1).tolist()]
```

(要点:① GPU 侧 padding 字段不再逐步 fill_/zero_ —— 单发 H2D 全量覆盖,pinned 侧已保证 padding 常量;② tables 的 zero_/padding 重写全在 CPU 镜像;③ 旧 `st.tables[B:, 0] = self.scratch` 等行删除。)

- [ ] **Step 6: 跑绿**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_graph_runner.py -v`(既有对拍门 = 逐位契约,必须原样全绿)
Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q`
Expected: **102 passed, 13 deselected**(101 + build_tables out= 1)

- [ ] **Step 7: Commit**

```bash
git add acorninfer/engine/graph_runner.py acorninfer/ops/paged_attention.py tests/  # 测试文件按实际
git commit -m "perf(engine): flat static buffers + pinned staging — 2 H2D per decode (M4.1 Task 2)"
```

---

### Task 3: 中间门(段一收尾实测)

**Files:**
- Modify: `docs/benchmarks.md`(M4.1 节开篇)

- [ ] **Step 1: bench 双跑**(graph + eager 各一轮,C=1 4 8)

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 --attention-backend triton --fused-ops --cuda-graph
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 --attention-backend triton --fused-ops
```
(跑前 `nvidia-smi` 查残留;前台 timeout ≥ 1800000。)

- [ ] **Step 2: profiler 双账**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py --steps 20 --cuda-graph
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py --steps 20
```
Expected:graph 账目 wall/step 较 21.64ms 进一步下降;残差(gap)≤1ms 记达标,**不达也照实记录**并给出归因(残差构成 = ensure 循环/调度 python/镜像填充);memcpy/step 应从 14 显著下降(≈2 H2D + argmax D2H + 图内既有)。launches/step 的 GPU 侧计数语义不变(图内 kernel 仍逐条入账)。

- [ ] **Step 3: docs 追加 M4.1 节开篇**(只写实测;数字与 stdout 逐字一致)

`## M4.1`(日期 2026-10-31):段一改动摘要(argmax 合批/扁平缓冲/2×H2D)、两轮 RESULT 行逐字、profiler 双账表、残差判定与归因。命令照 M4 节格式。

- [ ] **Step 4: Commit**

```bash
git add docs/benchmarks.md
git commit -m "docs(benchmarks): M4.1 phase-1 interim account (M4.1 Task 3)"
```

---

### Task 4: `forward_prefill_batch`(模型层批量 prefill)

**Files:**
- Modify: `acorninfer/models/qwen3.py`(`Qwen3Attention`/`Qwen3Block`/`Qwen3ForCausalLM` 各加一个批量 prefill 方法;原 forward/forward_decode 一字不动)
- Test: `tests/test_forward_decode.py`(追加;该文件已是模型层对拍的家)

- [ ] **Step 1: 写失败测试**

```python
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.gpu
def test_forward_prefill_batch_matches_sequential(fused):
    """批量 prefill vs 逐条 forward:per-row 末位 logits 紧容差一致(TOL 协议
    标定写死);KV 写入逐层逐行等长(行间零串扰);fused 开关不影响 prefill
    分支(现状即如此),参数化钉住这一解耦。"""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    model.attention_backend = "triton"   # prefill 与后端解耦;取一个非默认值
    model.fused_ops = fused
    lens = [5, 9, 3]
    gen = torch.Generator(device="cuda").manual_seed(11)
    prompts = [torch.randint(0, 256, (1, L), generator=gen, device="cuda")
               for L in lens]
    # 批量腿
    alloc = BlockAllocator(32)
    cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    Lmax = max(lens)
    ids = torch.zeros(3, Lmax, dtype=torch.long, device="cuda")
    views = []
    for i, p in enumerate(prompts):
        ids[i, : lens[i]] = p
        views.append(PagedCacheView(cache, []))
    batch_logits = model.forward_prefill_batch(ids, views, real_lens=lens)
    # 逐条参考腿(独立 cache/views,页表互不污染)
    alloc2 = BlockAllocator(32)
    cache2 = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc2,
                          device="cuda", dtype=torch.bfloat16)
    ref_views = []
    ref = []
    for i, p in enumerate(prompts):
        v = PagedCacheView(cache2, [])
        ref.append(model(p, v, num_logits_to_keep=1)[0, -1])
        ref_views.append(v)
    ref = torch.stack(ref)
    # 容差:先以 TOL_BF16 跑通,再按实测 max|diff| 收紧写死,注释记录标定命令与日期
    torch.testing.assert_close(batch_logits, ref, **TOL_BF16)
    for i in range(3):
        assert views[i]._lens == ref_views[i]._lens   # 逐层 KV 长度一致
        assert views[i].table == ref_views[i].table   # 页表一致(无跨行写)
```

(标定步骤:跑一次打印 `max_abs_diff(batch_logits, ref)`,若 ≤1e-3 量级按协议收紧到实测 2-3 倍并写注释;禁止拍脑袋容差。)

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_forward_decode.py -k prefill_batch -v`
Expected: FAIL(AttributeError: ... forward_prefill_batch)

- [ ] **Step 3: 实现**(三个方法;原 forward/forward_decode 零改动)

`Qwen3Attention`:

```python
    def forward_prefill_batch(self, hidden, cos, sin, views, layer_idx, *, real_lens):
        """批量 prefill:qkv/o 大 GEMM 按 [N, Lmax] 合批;KV 写入与 attention
        逐行 —— 与单序列 forward 的 view.update + causal SDPA 同一语义。
        real_lens 截断保证 pad 永不入 KV、行间零串扰;cos/sin: [N, Lmax, head_dim]。"""
        N, Lmax, _ = hidden.shape
        q = self.q_proj(hidden).view(N, Lmax, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # 批量 RoPE(逐 token 操作):cos/sin 广播 [N,1,Lmax,D];pad 位截断丢弃
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        rep = self.num_heads // self.num_kv_heads
        outs = torch.zeros(N, self.num_heads, Lmax, self.head_dim,
                           dtype=hidden.dtype, device=hidden.device)
        for b, view in enumerate(views):
            L = real_lens[b]
            k_full, v_full = view.update(layer_idx, k[b:b + 1, :, :L], v[b:b + 1, :, :L])
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
            # 镜像单序列 prefill 分支:is_causal=(s>1);L==1 单 query 全可见等价
            outs[b, :, :L] = F.scaled_dot_product_attention(
                q[b:b + 1, :, :L], k_full, v_full, is_causal=(L > 1))[0]
        out = outs.transpose(1, 2).reshape(N, Lmax, -1)
        return self.o_proj(out)
```

`Qwen3Block`:

```python
    def forward_prefill_batch(self, hidden, cos, sin, views, layer_idx, *, real_lens):
        hidden = hidden + self.self_attn.forward_prefill_batch(
            self.input_layernorm(hidden), cos, sin, views, layer_idx, real_lens=real_lens)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden
```

`Qwen3ForCausalLM`:

```python
    def forward_prefill_batch(self, input_ids, views, *, real_lens):
        """批量 prefill(M4.1):input_ids [N, Lmax](右 padding,值任意)+
        real_lens[i] -> 各行末真实位 logits [N, vocab] (float32)。

        大 GEMM 合批 embed/RMSNorm/qkv/o/MLP(pad 行垃圾值只在其行内传播,
        逐行 attention 天然隔离);KV 写入与 attention 逐行 = 单序列 forward
        同一语义;末层只对各行 real_lens[i]-1 位置过 lm_head。N=1 时与
        forward 等价(合批 GEMM 形状差允许低比特漂移,TOL 协议标定)。
        """
        if len(views) != input_ids.shape[0] or len(real_lens) != input_ids.shape[0]:
            raise ValueError("views/real_lens 必须与 batch 维一一对应")
        N, Lmax = input_ids.shape
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        # 逐行位置(pad 位 0);展平为一次 rope 调用
        pos = torch.zeros(N * Lmax, dtype=torch.long, device=device)
        for i, L in enumerate(real_lens):
            pos[i * Lmax: i * Lmax + L] = torch.arange(L, device=device)
        cos, sin = self.rope(pos, dtype, device)      # [N*Lmax, head_dim]
        cos = cos.view(N, Lmax, -1)
        sin = sin.view(N, Lmax, -1)
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_prefill_batch(hidden, cos, sin, views, i,
                                                 real_lens=real_lens)
        hidden = self.norm(hidden)
        last = torch.tensor([L - 1 for L in real_lens], device=device)
        hidden = hidden[torch.arange(N, device=device), last]
        return self.lm_head(hidden).float()
```

- [ ] **Step 4: 跑绿 + 标定**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_forward_decode.py -v`
Expected: 全绿(parametrize 两腿);按 docstring 记录标定值。

- [ ] **Step 5: 全套件 + Commit**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` → **103 passed, 13 deselected**

```bash
git add acorninfer/models/qwen3.py tests/test_forward_decode.py
git commit -m "feat(model): forward_prefill_batch — batched GEMM, row-wise attention (M4.1 Task 4)"
```

---

### Task 5: `Executor.prefill_batch` + `Engine.step` 接线

**Files:**
- Modify: `acorninfer/engine/executor.py`(prefill_batch)
- Modify: `acorninfer/engine/engine.py`(step 分流)
- Test: `tests/test_engine.py`(追加 2 测试)

- [ ] **Step 1: 写失败测试**

```python
@pytest.mark.gpu
def test_engine_prefill_batch_matches_staggered_bitwise():
    """一次性提交 4 序列(一步批量 prefill)vs 逐条提交逐条排空(N=1 原路径):
    生成 token 逐位一致;钉死批量路径确曾走过(spy),且逐条腿零批量调用。"""
    def run(stagger, pin_calls):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
        model.attention_backend = "sdpa"   # prefill 批量化与后端开关解耦
        model.fused_ops = False
        model.decode_graph = False
        engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
        gen = torch.Generator(device="cuda").manual_seed(13)
        prompts = []
        for L in [6, 10, 14, 5]:
            ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
            prompts.append(ids[0].tolist())
        calls = {"n": 0}
        real_pb = engine.executor.prefill_batch
        def spy_pb(seqs, views):
            calls["n"] += 1
            return real_pb(seqs, views)
        if pin_calls:
            engine.executor.prefill_batch = spy_pb
        sids, toks = [], {}
        if stagger:
            for p in prompts:
                sid = engine.submit(p, SamplingParams(temperature=0.0), max_tokens=6)
                sids.append(sid)
                toks[sid] = []
                while engine.has_pending():
                    for o in engine.step():
                        toks[o.seq_id].append(o.token_id)
        else:
            for p in prompts:
                sids.append(engine.submit(p, SamplingParams(temperature=0.0),
                                          max_tokens=6))
            toks = {sid: [] for sid in sids}
            while engine.has_pending():
                for o in engine.step():
                    toks[o.seq_id].append(o.token_id)
        if pin_calls:
            engine.assert_balanced()
            model.to("cpu")
            del engine.executor.prefill_batch   # 卸实例 shadow(恢复类方法)
            return [toks[sid] for sid in sids], calls["n"]
        for sid in sids:
            engine.drop(sid)
        engine.assert_balanced()
        model.to("cpu")
        return [toks[sid] for sid in sids], calls["n"]

    batched, pb_calls = run(stagger=False, pin_calls=True)
    staggered, _ = run(stagger=True, pin_calls=False)
    assert pb_calls >= 1                # 钉:批量腿确曾走 prefill_batch
    assert batched == staggered         # token 逐位


@pytest.mark.gpu
def test_executor_prefill_batch_mixed_temperature_per_row_sampling():
    """批量 prefill 采样语义:temp>0 行走逐行 Sampler(与单序列一致),
    greedy 行 argmax;混合批两行各自正确(定种子确定性)。"""
    from types import SimpleNamespace
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    ex = Executor(model)
    from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
    alloc = BlockAllocator(64)
    cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(17)
    prompts = [torch.randint(0, 256, (1, L), generator=gen, device="cuda")
               for L in [5, 9]]
    seqs = [SimpleNamespace(all_ids=p[0].tolist(),
                            sampling=SamplingParams(temperature=0.0))
            for p in prompts]
    seqs[1] = SimpleNamespace(all_ids=prompts[1][0].tolist(),
                              sampling=SamplingParams(temperature=0.5, top_p=0.9))
    views = [PagedCacheView(cache, []) for _ in range(2)]
    torch.manual_seed(456)
    batched = ex.prefill_batch(seqs, views)
    # 逐条参考(独立页表;同 seed → 采样行同值)
    alloc2 = BlockAllocator(64)
    cache2 = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc2,
                          device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(456)
    single = [ex._sample(seqs[i], model(prompts[i], PagedCacheView(cache2, []),
                                        num_logits_to_keep=1)[0, -1])
              for i in range(2)]
    assert batched == single
    assert views[0]._lens == views[1]._lens or True  # 变长不比长度,比 KV 等长性:
    assert all(views[i]._lens == {} or True for i in range(2))  # 占位——以实现为准
```

(第二测试草稿的 KV 断言占位**必须按实际修**:prefill 后 `views[i]._lens[li] == lens[i]` 逐层——参照 Task 4 测试的 `views[i]._lens == ref_views[i]._lens` 写法,这里逐条参考在 cache2 上,可直接对拍两个 views dict。SimpleNamespace stub 字段按 `_sample`/`prefill_batch` 实际读取面补齐。)

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_engine.py -k "prefill_batch" -v`
Expected: FAIL(Executor 无 prefill_batch 属性 → spy 构造即 AttributeError;第二测试同)

- [ ] **Step 3: 实现**

`acorninfer/engine/executor.py`(prefill 之后加):

```python
    def prefill_batch(self, seqs: list[Sequence], views) -> list[int]:
        """多序列同批 prefill(M4.1):右 padding 合批前向,逐行采样
        (_sample 语义不变:temp==0 argmax;temp>0 逐行 Sampler)。"""
        lens = [len(s.all_ids) for s in seqs]
        Lmax = max(lens)
        ids = torch.zeros(len(seqs), Lmax, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids[i, : lens[i]] = torch.tensor(s.all_ids, dtype=torch.long,
                                             device=self.device)
        logits = self.model.forward_prefill_batch(ids, views, real_lens=lens)
        return [self._sample(s, logits[i]) for i, s in enumerate(seqs)]
```

`acorninfer/engine/engine.py` `step()` 的 prefill 段改为(原逐条路径在 N=1 分支原样保留):

```python
        try:
            built = []
            for seq in prefills:
                # 块表共享同一 list 对象:调度器 _release 清空它即同时清空视图
                view = PagedCacheView(self.cache, seq.block_table)
                self._views[seq.seq_id] = view
                built.append((seq, view))
            if len(built) == 1:
                # N=1 原路径:byte-identical(零回归面)
                seq, view = built[0]
                token = self.executor.prefill(seq, view)
                seq.num_pages = len(view.table)
                self._record(seq, token, outputs)
            elif built:
                # M4.1:同批多序列一次批量 prefill(调度器本就一步全准入)
                tokens = self.executor.prefill_batch(
                    [s for s, _ in built], [v for _, v in built])
                for (seq, view), token in zip(built, tokens):
                    seq.num_pages = len(view.table)
                    self._record(seq, token, outputs)
            if decodes:
                ...(decode 段原样不动)
```

- [ ] **Step 4: 跑绿 + 全套件**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_engine.py -v`(全绿)
Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q`
Expected: **105 passed, 13 deselected**(103 + 2;既有测试零改动 —— 所有"提交多条再步进"的既有测试此刻已自动走批量路径,全绿即行为保持证据)

- [ ] **Step 5: Commit**

```bash
git add acorninfer/engine/executor.py acorninfer/engine/engine.py tests/test_engine.py
git commit -m "feat(engine): batched prefill wiring — N>1 one forward per step (M4.1 Task 5)"
```

---

### Task 6: slow/golden 回归(换题预案)

**Files:**
- Modify: `docs/benchmarks.md`(若换题,记录 margin;否则零改动)
- 可能 Modify(仅换题预案,控制器复签):`tests/test_golden_m4.py`(PROMPTS 单条替换)

- [ ] **Step 1: 全 slow 套件**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -m slow -q`(timeout ≥ 3600000;跑前 nvidia-smi)
Expected: **17 passed**(golden m0-m4 原样全绿)。golden m4 的 C≥2 门此刻已在批量 prefill 路径上(多 submit → 一步批量 prefill)。

- [ ] **Step 2: 若 golden m4 翻题(换题预案,预先授权)**

单条 prompt 翻题:按 M0 先例换入新 prompt(同 in_tokens 生成方式)、跑三引擎筛选 margin、docstring 记录换题与日期,单独 commit `test(golden): swap prompt <id> (M4.1 prefill-batch flip, margin <v>)`。**系统性错位(多条/成模式)= 不是换题能救 = BLOCKED**,原样报告。

- [ ] **Step 3: Commit**(如有 docs/换题改动;无改动则本任务零 commit)

---

### Task 7: 最终配对仪式 + docs M4.1 收口

**Files:**
- Modify: `docs/benchmarks.md`(M4.1 节收口)

- [ ] **Step 1: 配对仪式**(M4 Task 7 同款配方,读 `scripts/bench_remote.py` docstring 与 docs M4 节记录的命令)

preflight `nvidia-smi` 双卡空闲 → GPU 0 起 vLLM(`/home/ernest/vllm-baseline/bin/vllm serve /home/ernest/model-weight/Qwen3-8B --port 8100 --max-model-len 512 --gpu-memory-utilization 0.90 --dtype bfloat16`,nohup + /health 轮询)→ `scripts/bench_remote.py --concurrency 1 4 8` → GPU 1 本引擎 graph(`--cuda-graph`)与 eager 双跑 → 关服 + 清理验证(双卡回 4 MiB)。全窗口同一工作会话(当日配对)。timeout ≥ 3600000。

- [ ] **Step 2: docs M4.1 收口**(只写实测)

配对表(vLLM / graph / eager × C=1/4/8,RESULT 行逐字)、达标率(graph ÷ 当日 vLLM)、与 M4 验收的对照列(92.9/83.0/72.2 → 今值)、判定行(如实)、归因更新(TTFT 实测 vs 82ms 线、残差实测、GEMM 地板引用)。

- [ ] **Step 3: Commit**

```bash
git add docs/benchmarks.md
git commit -m "docs(benchmarks): M4.1 paired A/B final — batched prefill + residual cuts (M4.1 Task 7)"
```

---

## 计划自审记录(写作时已对照现状逐项核实)

1. **Spec 覆盖**:§2.1 argmax → Task 1;§2.2 扁平缓冲/pinned/out= → Task 2;§2.3 中间门 → Task 3;§3.1 模型层 → Task 4;§3.2 接线 → Task 5;§3.3 数值门/换题 → Task 4/5/6;§4 仪式 → Task 7;§5 保护条款 → 全局纪律(Task 6 换题例外为 spec §3.3 预授权的落实);§6 风险 → 各任务草稿注 + Task 3/6 判读。
2. **API 对照(读源码核实)**:`RotaryEmbedding.forward(positions 1D, dtype, device) -> cos/sin [q_len, head_dim]`;`_apply_rope` 广播 `[1,1,s,D]`(批量版直接用 `[:, None, :, :]` 内联,不再走 `_apply_rope`);`Qwen3Attention.forward` prefill 分支 = `view.update`(写+gather)→ `_repeat_kv` → `SDPA(is_causal=(s>1))`,past_len==0;q_norm → k_norm → rope 顺序(line 97-100);`PagedCacheView.update(layer_idx, k[1,kv,s,d], v)` 返回全量 (k_full, v_full);`Scheduler.schedule()` 已一步准入多条(`prefills: list`)→ scheduler 零触碰;`Engine.step` prefill 循环在 line 126-132;`Executor.prefill` 单序列 `[1, all_ids]`;`build_tables` 现返回 `t.to(device), torch.tensor(lens, ...)`(out= 路径留 CPU);`_BucketState` 现字段 ids/positions/tables/lens/pages/slots/views/graph/logits。
3. **类型一致性**:`forward_prefill_batch(input_ids, views, *, real_lens)` Task 4 定义 = Task 5 调用一致;`prefill_batch(seqs, views)` Task 5 定义 = Engine 调用一致;`build_tables(..., out=)` Task 2 定义 = runner 调用一致;`_BucketState` 新字段名(pin_long/pin_i32/pin_tables/flat_long/flat_i32)Task 2 内一致;argmax 合批返回形式 Task 1 两处一致。
4. **计数账**:fast 基线 99 → T1 +2 = 101 → T2 +1 = 102 → T4 +1 = 103 → T5 +2 = 105;slow 17 不变。若实现者按草稿注增加 parametrize/变体,以实际计数 + 全绿为准,报告中说明。
5. **已知薄弱点(草稿注已标)**:Task 1 两测试为"改后不变"对拍门而非红绿门(红相由全套件既有绿 + 改后仍绿保证);Task 5 第二测试 KV 断言有占位注(实现者按实际修);Task 1 stub 字段面按实现核实。测试才是契约,草稿错按 M4 先例修并在报告记录。
