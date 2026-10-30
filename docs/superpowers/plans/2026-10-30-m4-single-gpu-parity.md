# M4 单卡追平 vLLM(CUDA graph)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 1×RTX 4090 上,引擎 decode 路径经 CUDA graph 静态化 + 必要的 kernel 调参,聚合吞吐(C=1/4/8,同协议)逐档 ≥ 100% 官方 vLLM 0.32.0 当日配对实测。

**Architecture:** 唯一新模块 `acorninfer/engine/graph_runner.py`(`DecodeGraphRunner`:批大小桶 {1,2,4,8} 在 Engine 构造期对全桶预捕获;静态缓冲图外原位覆写、图内重放;CPU 逻辑全部留图外)。`Qwen3ForCausalLM.forward_decode` 加 additive `prepared=None` 形参(graph 路径图外传 bundle,避免捕获期 CPU→H2D 被冻结;eager 行为逐位不变)。三级 reference 链:SDPA → triton eager → triton graph,每级可对拍;golden 逐位门全程护航。

**Tech Stack:** torch 2.13.0+cu130(`torch.cuda.CUDAGraph` / `torch.profiler`)、Triton 3.7.1、本项目 M0–M3 全部基建(pager / ops 三 kernel / Engine / bench)。

---

## 全局纪律(每个任务都适用,实现者必读)

- **计划代码是草稿,测试才是契约。** 本计划所有代码块都是给出意图与契约的草稿——Task 3 的草稿里可能藏错(M0–M3 共揪出 6 处计划 bug)。对拍不过时,以数值事实与推导为准修代码,并在报告里记录偏离。
- **保护文件**:`tests/test_golden_m0/m1/m2/m3.py`、`acorninfer/cache/kv.py`、`acorninfer/cache/pager.py`、`acorninfer/engine/scheduler.py`、`acorninfer/engine/sequence.py` 禁改。`acorninfer/models/qwen3.py` 仅 Task 2 的 `prepared` 形参一条改动,torch 路径与 eager triton 路径 byte-identical。`acorninfer/ops/*` 仅 Task 6 条件性微调。`tests/triton_utils.py` 容差值不动。
- **GPU 分工**:开发测试全部 `CUDA_VISIBLE_DEVICES=1` 前缀(代码写 `cuda:0`);GPU 0 仅 Task 7 验收日跑 vLLM 服务。Python/pytest 用 `.venv/bin/` 前缀。
- **数值门**:tiny 级逐位 / token 逐位双门(见各任务);bf16 容差只准用 `tests/triton_utils.py` 既有 TOL,不许新造;实测数字才可写入 docs。
- **长命令前台长 timeout**(子代理不会因后台命令完成而自动恢复);每条 bench 前 `nvidia-smi` 确认 GPU 1 无残留。
- 仓库 `/home/ernest/github/acorninfer`,branch `main`,计划基线 HEAD = spec 提交(spec + 本计划一起提交后的 HEAD)。fast 套件基线 **94 passed, 13 deselected**;slow 套件基线 **13 passed**。

## 文件结构(职责边界)

- `acorninfer/engine/graph_runner.py`(新):DecodeGraphRunner —— 桶状态、捕获、每步 prepare+覆写+重放。唯一懂 CUDA graph 的地方。
- `acorninfer/models/qwen3.py`(改):`forward_decode` 增 `prepared=None` 形参。唯一与模型有关的改动。
- `acorninfer/config.py`(改):`EngineConfig.decode_graph` 字段 + 校验。
- `acorninfer/engine/engine.py`、`acorninfer/engine/executor.py`(改):runner 构造与分流。
- `acorninfer/bench.py`(改):`--cuda-graph` 开关。
- `scripts/profile_decode.py`(新):T1 profiler 账目。
- `tests/test_graph_runner.py`(新)、`tests/test_golden_m4.py`(新)、既有测试文件追加。

---

### Task 1: profiler 账目 —— 把"推断"换成实测

**Files:**
- Create: `scripts/profile_decode.py`
- Modify: `docs/benchmarks.md`(M4 节追加实测账目;此节若不存在则新建"## M4"小节)

- [ ] **Step 1: 写 profiler 脚本**

```python
"""decode 步 profiler 账目(M4 T1):docs 的"推断"升级为实测。

跑法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py
输出:每步墙钟、CUDA kernel 累计、发射间隙(gap = 墙钟 − kernel)、
每步 kernel 启动次数与 H2D/D2H 次数、top-15 kernel 表。
8 序列(in=128)prefill 完成后测 N 个 decode 步(默认 20),temp=0。
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from acorninfer.bench import DEFAULT_MODEL, _prompts
from acorninfer.engine.engine import Engine
from acorninfer.models.qwen3 import from_pretrained
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = from_pretrained(DEFAULT_MODEL, args.device, torch.bfloat16)
    model.attention_backend = "triton"
    model.fused_ops = True
    engine = Engine(model, None, num_blocks=1024, page_size=16, max_running=8)
    try:
        vocab = model.cfg.vocab_size
        prompts = _prompts(8, 128, vocab, 20261027)
        ids = [engine.submit(p, SamplingParams(temperature=0.0), max_tokens=args.steps + 2)
               for p in prompts]
        engine.step()  # prefill 步
        while any(len(engine.seqs[i].generated_ids) == 0 for i in ids):
            engine.step()
        # 全员进入 decode;先空跑 3 步出稳定性,再计时 + profile
        for _ in range(3):
            engine.step()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                engine.step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        evs = prof.key_averages()
        cuda_us = sum(getattr(e, "self_device_time_total", 0) for e in evs)
        launches = sum(e.count for e in evs if getattr(e, "self_device_time_total", 0) > 0)
        memcpy = sum(e.count for e in evs if "Memcpy" in e.key or "memcpy" in e.key)
        n = args.steps
        print(f"steps={n} wall/step={wall / n * 1000:.2f}ms "
              f"cuda_kernel/step={cuda_us / n / 1000:.2f}ms "
              f"gap/step={(wall - cuda_us / 1e6) / n * 1000:.2f}ms")
        print(f"launches/step≈{launches / n:.0f} memcpy/step≈{memcpy / n:.1f}")
        print(evs.table(sort_by="self_cuda_time_total", row_limit=15))
    finally:
        model.to("cpu")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
```

(草稿:`_prompts`/`SamplingParams` 签名以 `acorninfer/bench.py`、`acorninfer/sampler.py` 现状为准;`self_device_time_total` 属性名按 torch 2.13 实际 API 校正。)

- [ ] **Step 2: 跑账目**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_decode.py --steps 20`
Expected: 打印每步墙钟(预期 ~40ms 量级)、kernel 累计、gap、top-15 kernel 表。gap 占比即 CUDA graph 的理论收益上限。

- [ ] **Step 3: docs 追加 M4 节(只写实测)**

`docs/benchmarks.md` 新增 `## M4` 小节:T1 账目数字(每步墙钟 / kernel / gap / launches / memcpy / top kernel 摘要)、脚本名与命令、日期。明确写"以下归因自此为实测"。

- [ ] **Step 4: Commit**

```bash
git add scripts/profile_decode.py docs/benchmarks.md
git commit -m "feat(profile): decode-step profiler accounting (M4 Task 1)"
```

---

### Task 2: `forward_decode` 增 `prepared` 形参(additive,eager 逐位不变)

**Files:**
- Modify: `acorninfer/models/qwen3.py`(`forward_decode` 签名与 prepare 块守卫)
- Modify: `tests/test_forward_decode.py`(追加对拍测试)

- [ ] **Step 1: 写失败测试(追加到 `tests/test_forward_decode.py`)**

```python
@pytest.mark.gpu
def test_forward_decode_prepared_bundle_bitwise(monkeypatch):
    """prepared 四件套路径 ≡ 自 prepare 路径(逐位);且传入 prepared 时
    forward_decode 内部零 ensure 调用 —— 否则 graph 捕获会冻结 CPU 源地址。"""
    import acorninfer.cache.pager as pager_mod
    from acorninfer.ops.paged_attention import build_tables

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    model.attention_backend = "triton"
    model.fused_ops = True
    alloc = BlockAllocator(32)
    cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    views, tokens = [], []
    for i, L in enumerate([5, 9]):
        ids = torch.randint(0, 256, (1, L), device="cuda")
        view = PagedCacheView(cache, [])
        model(ids, view)  # prefill:写入 L 个 token 的 KV
        views.append(view)
        tokens.append(int(ids[0, -1]))

    # 关键时序:pre-write 状态与 scatter 目标必须在参考步**之前**取 ——
    # decode 一步会提交 _lens(L → L+1),之后同状态无法再复现
    pre = [int(v.seq_len) for v in views]
    targets = [v.ensure_append_capacity(0) for v in views]
    x = torch.tensor([[t] for t in tokens], device="cuda")
    ref = model.forward_decode(x, views)  # eager 自 prepare

    # runner 同款图外 prepare → prepared(四件套,build_tables 返回 (tables, lens))
    calls = {"n": 0}
    real_ensure = pager_mod.PagedCacheView.ensure_append_capacity
    def spy_ensure(self, layer_idx):
        calls["n"] += 1
        return real_ensure(self, layer_idx)
    monkeypatch.setattr(pager_mod.PagedCacheView, "ensure_append_capacity", spy_ensure)
    tables, lens = build_tables(views, 4, "cuda", lens_override=[p + 1 for p in pre])
    prepared = (
        torch.tensor(pre, device="cuda"),
        tables,
        lens,
        (torch.tensor([p for p, _ in targets], device="cuda"),
         torch.tensor([s for _, s in targets], device="cuda")),
    )
    out = model.forward_decode(x, views, prepared=prepared)
    assert calls["n"] == 0  # 测试自己的 ensure 在 spy 装配前;prepared 路径内部必须零调用
    assert torch.equal(out, ref)
```

(逐位相等的根据:两条调用以完全相同的输入张量走同一 eager 核(prepared call 的 scatter 会把 `_lens` 再提交到 L+2,但断言在其前,无影响)。prepared 四件套顺序 = `(positions, tables, lens, (pages, slots))`,与 Task 3 `_bundle` 一致。)

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_forward_decode.py -k prepared -v`
Expected: FAIL(`forward_decode() got an unexpected keyword argument 'prepared'`)

- [ ] **Step 3: 实现(最小改动)**

`forward_decode` 签名改 `def forward_decode(self, input_ids: torch.Tensor, views, *, prepared=None) -> torch.Tensor:`,docstring 追加一段 prepared 契约(四件套顺序、图外构建理由、与自 prepare 逐位一致)。主体改为纯缩进搬移(零重排):原有 positions 行、fused_ops if/else 块、`tables = lens = scatter = None` 行、triton prepare 块**原样移入 else 分支**(仅加一层缩进):

```python
        if prepared is not None:
            # M4:graph 捕获路径。prepare 的 CPU torch.tensor 构建 + H2D 若在
            # 图内执行,重放会从捕获期冻结的源地址拷贝陈旧数据 —— 故由调用方
            # 图外完成并传入。契约:四件套 (positions, tables, lens, (pages, slots)),
            # 语义与下方 else 自 prepare 逐位一致(Task 2 单测钉死)。
            positions, tables, lens, scatter = prepared
        else:
            positions = torch.tensor([v.seq_len for v in views], device=input_ids.device)
            if self.fused_ops:
                # ……原有 K2 注释与两行赋值,原样缩进……
                cos = sin = None
                inv_freq = self.rope.inv_freq
            else:
                # ……原有非融合 rope 五行,原样缩进……
            tables = lens = scatter = None
            if self.attention_backend == "triton":
                # ……原有 M3 Task 9 prepare 全体(局部导入/ensure/scatter/build_tables),原样缩进……
```

(原 prepare 注释全部保留,随块搬入 else。fuse 分支与 prepare 无数据依赖,不交换顺序。torch 路径不走此函数,天然无感;eager triton 路径 `prepared=None` 走 else = 原执行序不变。)

- [ ] **Step 4: 跑绿 + 全套件**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_forward_decode.py -v` → 全绿(含既有 6 项);
Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` → **95 passed, 13 deselected**(94 基线 + 本任务新增 1;eager 路径 byte-identical 的证据:全部既有测试零改动通过)。

- [ ] **Step 5: Commit**

```bash
git add acorninfer/models/qwen3.py tests/test_forward_decode.py
git commit -m "feat(model): prepared-bundle param in forward_decode for graph capture (M4 Task 2)"
```

---

### Task 3: DecodeGraphRunner(桶预捕获 + 静态缓冲重放)

**Files:**
- Create: `acorninfer/engine/graph_runner.py`
- Create: `tests/test_graph_runner.py`

- [ ] **Step 1: 写失败测试(`tests/test_graph_runner.py`)**

```python
"""DecodeGraphRunner 对拍:graph 路径 vs eager 路径。

门(双环境独立 —— 同 seed 造同权重的两套 model/cache/allocator,页表互不污染):
1. 3 序列(桶 4)× 6 步:每步 argmax token 逐位;logits assert_close(TOL_BF16;
   前 3 步两路 B 不同(3 vs 桶 4 的 GEMM 形状差)允许低比特漂移,token 不许翻);
2. 第 4 步起丢 1 序列(模拟 EOS 退休)→ 桶 4→2 切换,余下序列仍逐位;
3. 每 runner 步后 views 的 _lens 与 eager 步后逐层相等(_commit_lens 契约);
4. 收尾:显式 free 全部测试页后,runner 侧 alloc 恰 1 块 live(scratch 常驻),
   eager 侧 assert_balanced 零 live。
"""
import pytest
import torch
from types import SimpleNamespace

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import Qwen3Config
from acorninfer.engine.graph_runner import DecodeGraphRunner
from acorninfer.models.qwen3 import Qwen3ForCausalLM
from triton_utils import TOL_BF16

LENS = [5, 9, 13]


def _env(seed=0):
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    model.attention_backend = "triton"
    model.fused_ops = True
    alloc = BlockAllocator(64)
    cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    return model, alloc, cache


def _prefill(model, cache, L, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (1, L), generator=g, device="cuda")
    view = PagedCacheView(cache, [])
    model(ids, view)
    return view, int(ids[0, -1])


@pytest.mark.gpu
def test_runner_matches_eager_and_bucket_switch():
    model, alloc, cache = _env()
    runner = DecodeGraphRunner(model, cache, alloc)
    views, tokens = map(list, zip(*[_prefill(model, cache, L, 100 + i)
                                    for i, L in enumerate(LENS)]))

    # eager 参考环境(同权重同 prompt,独立页表)
    model2, alloc2, cache2 = _env()
    views2, _ = map(list, zip(*[_prefill(model2, cache2, L, 100 + i)
                                for i, L in enumerate(LENS)]))

    retired = []  # 退休视图的页由测试自行 free(引擎外无 scheduler 代劳)
    for step in range(6):
        if step == 3:  # 模拟 EOS 退休:批 3→2,桶 4→2
            retired.append((alloc, views.pop()))
            retired.append((alloc2, views2.pop()))
        x = torch.tensor([[t] for t in tokens], device="cuda")
        eager_out = model2.forward_decode(x, views2)
        stubs = [SimpleNamespace(generated_ids=[t]) for t in tokens]
        graph_tokens = runner.decode_batch(stubs, views)
        torch.testing.assert_close(runner.last_logits[: len(views)], eager_out, **TOL_BF16)
        assert graph_tokens == [int(r.argmax()) for r in eager_out]
        for v, v2 in zip(views, views2):
            assert v._lens == v2._lens  # _commit_lens 契约:全层逐层等长
        tokens = graph_tokens

    for a, v in retired + [(alloc, v) for v in views] + [(alloc2, v) for v in views2]:
        for b in v.table:
            a.free(b)
    assert alloc.num_live == 1  # 仅 runner 常驻 scratch;其余零 live
    alloc2.assert_balanced()
```

(草稿:① 退出步的序列必须与退休后一一对应 —— `views.pop()` 同时作用于两环境,保序;② `runner.last_logits` 是冻结输出 [桶, vocab],切片 `[:n]` 后与 eager [n, vocab] 同形;③ token 逐位断言覆盖跨桶(桶 4 步与桶 2 步)与长程(6 步)两个维度 —— warmup/capture 的副作用若污染后续真步,在这里炸。)

(收尾记账注:scratch 块由 runner 常驻持有 —— 这是 Engine 契约"排空后恰 1 块 live"的根据,Task 4 的 `Engine.assert_balanced` 按此实现。)

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_graph_runner.py -v`
Expected: FAIL(ModuleNotFoundError: acorninfer.engine.graph_runner)

- [ ] **Step 3: 实现 `acorninfer/engine/graph_runner.py`**

```python
"""Decode CUDA graph runner(M4):批大小分桶预捕获 + 静态缓冲重放。

静态化契约(与 eager 逐位的根据):
- 图体 = model.forward_decode(ids, views, prepared=bundle)。bundle 四件
  (positions/tables/lens/scatter)全部来自静态缓冲,每步重放前原位覆写;
- 图内 python(局部导入、asserts、num_splits_for、scatter_decode 的
  _lens 提交)只在捕获期执行一次,重放不执行 —— _lens 提交由 runner 在
  prepare 阶段代行(_commit_lens),捕获期那次与代行同值,无副作用;
- padding 行:views 尾部补 dummy view;scatter 索引指向 scratch 块(init
  期从 allocator 常驻分配,永不读);输出由调用方丢弃;
- 捕获在 runner 构造期完成(全 dummy views):请求路径零捕获抖动;捕获前
  3 次 side-stream warmup 与正式步计算同值(确定性 + 同 (page,slot) 写同
  k/v 的幂等性)。
预期用途:attention_backend="triton"、全 greedy(temp=0)、B ≤ 8;
不满足时由 Executor 分流回 eager。

scratch 契约:init 期从 allocator 常驻分配 1 块,永不释放 —— KV 排空后的
收支平衡判据因此为"恰 1 块 live"(Engine.assert_balanced 按此实现,Task 4);
本 runner 单测按同一契约以 num_live == 1 收尾。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from acorninfer.cache.pager import PagedCacheView
from acorninfer.ops.paged_attention import build_tables

BUCKETS = (1, 2, 4, 8)
WARMUP_STEPS = 3


@dataclass
class _BucketState:
    bucket: int
    ids: torch.Tensor        # [B, 1] long
    positions: torch.Tensor  # [B] long
    tables: torch.Tensor     # [B, W] int32;W = allocator.total(单序列页数上界)
    lens: torch.Tensor       # [B] int32
    pages: torch.Tensor      # [B] long
    slots: torch.Tensor      # [B] long
    views: list              # [B] dummy PagedCacheView(捕获期 views 实参)
    graph: torch.cuda.CUDAGraph
    logits: torch.Tensor | None  # [B, vocab] float32;捕获后由图冻结


class DecodeGraphRunner:
    def __init__(self, model, cache, allocator):
        assert getattr(model, "attention_backend", "sdpa") == "triton", \
            "DecodeGraphRunner 仅支持 triton 后端"
        self.model = model
        self.cache = cache
        self.allocator = allocator
        self.num_layers = model.cfg.num_hidden_layers
        self.page_size = cache.page_size
        self.device = next(model.parameters()).device
        self.scratch = allocator.allocate()  # 常驻;padding 行 KV 落点
        self.states = {b: self._build_state(b) for b in BUCKETS}
        for st in self.states.values():
            self._capture(st)

    # ---- 构造期:状态与捕获 ----

    def _build_state(self, bucket: int) -> _BucketState:
        dev = self.device
        st = _BucketState(
            bucket=bucket,
            ids=torch.zeros(bucket, 1, dtype=torch.long, device=dev),
            positions=torch.zeros(bucket, dtype=torch.long, device=dev),
            tables=torch.zeros(bucket, self.allocator.total, dtype=torch.int32, device=dev),
            lens=torch.ones(bucket, dtype=torch.int32, device=dev),
            pages=torch.full((bucket,), self.scratch, dtype=torch.long, device=dev),
            slots=torch.zeros(bucket, dtype=torch.long, device=dev),
            views=[PagedCacheView(self.cache, [self.scratch]) for _ in range(bucket)],
            graph=torch.cuda.CUDAGraph(),
            logits=None,
        )
        return st

    def _capture(self, st: _BucketState) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(WARMUP_STEPS):
                self.model.forward_decode(st.ids, st.views, prepared=self._bundle(st))
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(st.graph):
            st.logits = self.model.forward_decode(st.ids, st.views, prepared=self._bundle(st))

    def _bundle(self, st: _BucketState):
        return (st.positions, st.tables, st.lens, (st.pages, st.slots))

    # ---- 每步路径 ----

    def decode_batch(self, seqs, views):
        """与 Executor.decode_batch 同参;seqs 仅取 generated_ids[-1]。"""
        B = len(views)
        assert 1 <= B <= BUCKETS[-1]
        st = self.states[min(b for b in BUCKETS if b >= B)]
        targets = [v.ensure_append_capacity(0) for v in views]
        pre = [int(v.seq_len) for v in views]
        new_len = [p + 1 for p in pre]
        st.pages[:B].copy_(torch.tensor([p for p, _ in targets], dtype=torch.long))
        st.slots[:B].copy_(torch.tensor([s for _, s in targets], dtype=torch.long))
        st.positions[:B].copy_(torch.tensor(pre, dtype=torch.long))
        st.lens[:B].copy_(torch.tensor(new_len, dtype=torch.int32))
        t, _ = build_tables(views, self.page_size, self.device, lens_override=new_len)
        st.tables.zero_()
        st.tables[:B, : t.shape[1]].copy_(t)
        st.tables[B:, 0] = self.scratch  # padding 行的常量场(zero_ 后重写)
        st.ids[:B].copy_(torch.tensor([[s.generated_ids[-1]] for s in seqs], dtype=torch.long))
        self._commit_lens(views, new_len)
        st.graph.replay()
        self.last_logits = st.logits
        return [int(row.argmax()) for row in st.logits[:B]]

    def _commit_lens(self, views, new_len) -> None:
        # pager 私有状态的图外代行(pager 本里程碑禁改):与 scatter_decode 的
        # per-layer 提交同值同义;重放体内无 python,提交只能发生在图外。
        for v, n in zip(views, new_len):
            for li in range(self.num_layers):
                v._lens[li] = n
```

(草稿风险点,实现时以测试为裁判:① `st.tables[:, 0] = self.scratch` 的 padding 常量在 `zero_()` 后必须重写 —— 草稿已放在 copy_ 之后;② capture 期 dummy views 的 `_lens` 提交写入 dummy dict,无害;③ warmup 与 capture 在默认 stream 上下文之外的副作用(_lens、页写)是否影响后续真步 —— 同值幂等,但要在测试里被 6 步长程对拍覆盖;④ `torch.cuda.graph` 捕获期不允许 host 侧 sync —— prepared 路径无 `.item()`/`.cpu()`,若炸先查 build_tables 之外残留的隐式 sync。)

- [ ] **Step 4: 跑绿**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_graph_runner.py -v`
Expected: 全绿(多步逐位 + 桶切换 + _lens 等价)。若 logits 容差炸在低比特,先核 padding 行 lens/tables 场,再核两条路径 B 不同(3 vs 桶 4)的 GEMM 形状差 —— token 逐位必须过,logits 允许 TOL_BF16。

- [ ] **Step 5: 全套件 + Commit**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` → 全绿,fast 计数按任务累加(Task 2 后基线 95,本任务 +1 → 96), 13 deselected

```bash
git add acorninfer/engine/graph_runner.py tests/test_graph_runner.py
git commit -m "feat(engine): DecodeGraphRunner — bucketed capture/replay for decode (M4 Task 3)"
```

---

### Task 4: 接线 —— EngineConfig / Engine / Executor / bench

**Files:**
- Modify: `acorninfer/config.py`(EngineConfig 字段 + 校验)
- Modify: `acorninfer/engine/engine.py`(runner 构造)
- Modify: `acorninfer/engine/executor.py`(分流)
- Modify: `acorninfer/bench.py`(`--cuda-graph`)
- Modify: `tests/test_engine.py`(追加 graph on/off 等价 + EOS 缩批测试)

- [ ] **Step 1: 写失败测试(追加到 `tests/test_engine.py`)**

```python
@pytest.mark.gpu
def test_engine_decode_graph_tokens_bitwise():
    """graph 引擎 vs eager 引擎:同 submits 同步序,生成 token 逐位一致;
    runner 启用钉死(构造期捕获,属性翻转不重建 runner 则必然不一致);
    排空后 KV 平衡(graph 常驻 scratch 例外,见 Engine.assert_balanced)。"""
    outs = []
    for graph in (False, True):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
        model.attention_backend = "triton"
        model.fused_ops = True
        model.decode_graph = graph          # 必须在 Engine 构造前置位
        engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
        assert (engine.graph_runner is not None) == graph
        gen = torch.Generator().manual_seed(7)
        sids = []
        for i, L in enumerate([6, 10, 14, 18]):
            ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
            sids.append(engine.submit(ids[0].tolist(),
                                      SamplingParams(temperature=0.0), max_tokens=6))
        toks = {sid: [] for sid in sids}
        while engine.has_pending():
            for o in engine.step():
                toks[o.seq_id].append(o.token_id)
        for sid in sids:
            engine.drop(sid)
        engine.assert_balanced()
        outs.append([toks[sid] for sid in sids])
        model.to("cpu")
    assert outs[0] == outs[1]


@pytest.mark.gpu
def test_engine_decode_graph_eos_shrinks_batch():
    """decode 中途 EOS 退休使批缩小、桶切换(4→2)。

    tiny 词表(256)吐不出 QWEN3_EOS_IDS,max_tokens=1 又在 prefill 就终
    (缩不了 decode 批)—— 用两阶段陷阱:先跑无陷阱 eager 拿全量 token 流,
    取各陷阱序列的第 2 个生成 token(= 首个 decode 产出)作 eos_ids 覆写,
    两引擎同构重跑;批在第 1 个 decode 步后 4→2,余者仍逐位、跑满 max_tokens。"""

    def run(graph, traps):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
        model.attention_backend = "triton"
        model.fused_ops = True
        model.decode_graph = graph
        engine = Engine(model, None, num_blocks=64, page_size=4, max_running=8)
        gen = torch.Generator().manual_seed(7)
        prompts = []
        for i, L in enumerate([6, 10, 14, 18]):
            ids = torch.randint(0, 256, (1, L), generator=gen, device="cuda")
            prompts.append(ids[0].tolist())
        sids = []
        for i, p in enumerate(prompts):
            kw = {"eos_ids": frozenset({traps[i]})} if i in traps else {}
            sids.append(engine.submit(p, SamplingParams(temperature=0.0),
                                      max_tokens=8, **kw))
        toks = {sid: [] for sid in sids}
        while engine.has_pending():
            for o in engine.step():
                toks[o.seq_id].append(o.token_id)
        for sid in sids:
            engine.drop(sid)
        engine.assert_balanced()
        model.to("cpu")
        return [toks[sid] for sid in sids]

    full = run(False, {})                              # 阶段 1:无陷阱全量流
    traps = {0: full[0][1], 1: full[1][1]}             # 生成流第 2 个 token = 首个 decode 产出
    eager = run(False, traps)
    graph = run(True, traps)
    assert eager == graph                              # 缩批后 token 仍逐位
    assert len(eager[0]) == 2 and len(eager[1]) == 2   # 陷阱序列恰在第 1 个 decode 步终止
    assert all(len(s) == 8 for s in eager[2:])         # 余者跑满 → 桶 4→2 切换确有发生
```

(草稿:陷阱 token 撞上贪心自重复的概率非零(prefill token == 首个 decode token 时陷阱提前到 prefill 期触发)—— 种子 7 下是确定性的,若真撞上,把陷阱改为取 `full[i][2]`(第 3 个生成 token)并在注释里记录,断言长度同步改 3。)

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_engine.py -k graph -v`
Expected: FAIL(`model.decode_graph` 属性无消费方 —— 第一组(graph=False)过、第二组(graph=True)与 eager 输出不一致或属性无效果导致断言路径缺失;以实际失败形态为准,若"无效果但恰好逐位一致"则属假绿,须加 runner 启用断言:`engine.graph_runner is not None`)

- [ ] **Step 3: 实现**

`acorninfer/config.py` EngineConfig(`fused_ops` 之后):

```python
    decode_graph: bool = False       # True → decode 走 CUDA graph(M4;须 triton 后端)
```

`__post_init__` 追加:

```python
        if self.decode_graph and self.attention_backend != "triton":
            raise ValueError("decode_graph requires attention_backend='triton'")
```

`acorninfer/engine/engine.py` `__init__`(executor 构造行改为):

```python
        self.graph_runner = None
        if getattr(model, "decode_graph", False):
            from acorninfer.engine.graph_runner import DecodeGraphRunner
            self.graph_runner = DecodeGraphRunner(model, self.cache, self.allocator)
        self.executor = Executor(model, self.graph_runner)
```

同文件 `assert_balanced` 改为(Task 3 的 scratch 常驻契约;显式 raise 而非
assert 语句,理由同 `BlockAllocator.assert_balanced` —— 不能在 `python -O`
下静默失效):

```python
    def assert_balanced(self) -> None:
        if self.graph_runner is None:
            self.allocator.assert_balanced()
        else:
            # graph runner 常驻 1 块 scratch(padding 行 KV 落点,永不读):
            # 排空后恰它一块 live,其余任何 live 块 = KV 泄漏
            if self.allocator.num_live != 1:
                raise AssertionError(
                    f"KV leak: {self.allocator.num_live - 1} block(s) never freed")
```

`from_config`(`model.fused_ops = cfg.fused_ops` 之后加):

```python
        model.decode_graph = cfg.decode_graph
```

`acorninfer/engine/executor.py`:

```python
class Executor:
    def __init__(self, model: Qwen3ForCausalLM, graph_runner=None):
        self.model = model
        self.graph_runner = graph_runner
        self.device = next(model.parameters()).device
```

`decode_batch` 头部加分流:

```python
    def decode_batch(self, seqs: list[Sequence], views) -> list[int]:
        if (self.graph_runner is not None
                and len(views) <= BUCKETS[-1]
                and all(s.sampling.temperature == 0 for s in seqs)):
            return self.graph_runner.decode_batch(seqs, views)
        x = torch.tensor(...)  # 既有 eager 体不动
```

(`from acorninfer.engine.graph_runner import BUCKETS` 置顶。)

`acorninfer/bench.py`:argparse 加 `parser.add_argument("--cuda-graph", action="store_true")`;属性赋值块整体移到 `Engine(...)` 构造**之前**并更新注释:

```python
    # M3/M4 后端开关:decode_graph 必须在 Engine 构造前置位(graph runner 在
    # Engine.__init__ 捕获各桶);backend/fused 为运行时属性,一并提前便于一处管理
    model.attention_backend = args.attention_backend
    model.fused_ops = args.fused_ops
    model.decode_graph = args.cuda_graph
    engine = Engine(...)
```

(engine-ready 日志追加 `cuda_graph={model.decode_graph}`。)

- [ ] **Step 4: 跑绿 + 全套件**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q` → 全绿(95 → Task 3 后 96 → 本任务 +2~3), 13 deselected;
Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_config.py -q` → 绿(validation 用例:若既有 config 测试文件缺 graph 校验用例,补一条 `EngineConfig(model_dir="x", attention_backend="sdpa", decode_graph=True)` raises ValueError)。

- [ ] **Step 5: Commit**

```bash
git add acorninfer/config.py acorninfer/engine/engine.py acorninfer/engine/executor.py acorninfer/bench.py tests/test_engine.py tests/test_config.py
git commit -m "feat(engine): decode_graph switch — runner wiring + bench flag (M4 Task 4)"
```

---

### Task 5: golden M4(三引擎对拍 + fuzz)+ golden M3 回归

**Files:**
- Create: `tests/test_golden_m4.py`(`@pytest.mark.slow`,镜像 `test_golden_m3.py` 协议;golden m3 只读)

- [ ] **Step 1: 通读 `tests/test_golden_m3.py`(只读),镜像其结构写 `tests/test_golden_m4.py`**

docstring 契约:

```python
"""M4 验收门:graph 后端引擎的并发输出 == 顺序参考(token 逐位)。

协议镜像 test_golden_m3.py:同 margin 筛选 prompt,三引擎(同一模型对象,
属性切换后各建 Engine):
  - sdpa:回归门(与 m3 同);
  - triton eager:回归门(与 m3 同);
  - triton graph:M4 新门 —— decode_graph=True 须在 Engine 构造前置位;
  - fuzz no-leak:graph 开,跑完 assert_balanced。
失误协议:graph 门若翻题按 M0/M2/M3 先例换题并记录新 margin;系统性错位才是 bug。
"""
```

四个测试体:`test_golden_m4_sdpa_matches_sequential`、`test_golden_m4_triton_eager_matches_sequential`(镜像 m3,回归)、`test_golden_m4_graph_matches_sequential`(同 m3 triton 门,但 `model.decode_graph = True` **置于 Engine 构造前**;并保留 m3 的防空转钉:属性前置自检 + runner 启用断言 `engine.graph_runner is not None` + 可选 spy)、`test_golden_m4_fuzz_no_leak`(graph 开,20 条混合)。

- [ ] **Step 2: 跑 golden M4 + 全 slow**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_golden_m4.py -v -m slow
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -m slow -q
```

Expected: 新 4 项 passed;slow 总 ≥ 17(原 13 + 新 4)。**golden m0/m1/m2/m3 任一挂 = BLOCKED。**

- [ ] **Step 3: Commit**

```bash
git add tests/test_golden_m4.py
git commit -m "test(golden): M4 graph-engine gate + fuzz no-leak (M4 Task 5)"
```

---

### Task 6: 条件 kernel 调参(条件不满足则记录跳过理由)

**Files:**
- Modify(仅条件触发): `acorninfer/ops/paged_attention.py`、`acorninfer/ops/rmsnorm.py`、`acorninfer/ops/rope.py` 的 `num_warps`/grid 参数
- Modify: `docs/benchmarks.md`(调参 A/B 记录或跳过理由)

- [ ] **Step 1: 判定触发条件(依据 Task 1 账目 + Task 5 后的一次 C=8 中间实测)**

触发需同时满足:① T1 账目显示 CUDA kernel 时间占 TPOT ≥ 20%;② graph 落地后 C=8 距 100% 线差距 ≤ 10%。否则在本计划执行报告中记录"跳过:kernel 占比 X% / 差距 Y%"并直接进 Task 7。

- [ ] **Step 2(若触发): 数值保持类 A/B**

允许:K3 `num_warps`(现默认 4,试 2/8)、K1/K2 `num_warps`、K3 grid 三维次序。禁默认改:exp→exp2、num_splits(改归约序 ⇒ golden 可能翻题;若确需,走 M0/M2/M3 换题协议并在 golden_m4 docstring 记录)。每个候选:改 → `pytest tests/test_graph_runner.py tests/test_forward_decode.py -q` → golden m4 `-m slow` → bench C=8 单档 A/B(`--attention-backend triton --fused-ops --cuda-graph --concurrency 8`)→ 只保留严格更快者,docs 记录前后数字(同日同环境)。

- [ ] **Step 3: Commit(若触发)**

```bash
git add acorninfer/ops/ docs/benchmarks.md
git commit -m "perf(ops): graph-era kernel tuning A/B (M4 Task 6)"
```

---

### Task 7: 当日配对 A/B 验收 + docs + verdict

**Files:**
- Modify: `docs/benchmarks.md`(M4 验收表)、`README.md`(状态行)

- [ ] **Step 1: 当日配对 A/B(同时间窗口,先 vLLM 后引擎;两条命令前各自 `nvidia-smi` 查残留)**

```bash
# ① vLLM 基线(GPU 0):起服务(等 /health 200)
CUDA_VISIBLE_DEVICES=0 nohup /home/ernest/vllm-baseline/bin/vllm serve \
  /home/ernest/model-weight/Qwen3-8B --dtype bfloat16 --port 8100 \
  --max-model-len 512 --gpu-memory-utilization 0.90 \
  > /tmp/opencode/vllm-m4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8
pkill -f "vllm serve"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# ② 引擎(GPU 1):graph 全开
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 \
  --attention-backend triton --fused-ops --cuda-graph
# ③ 回归锚点(同日):eager triton
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 \
  --attention-backend triton --fused-ops
```

- [ ] **Step 2: 判定 + docs**

`docs/benchmarks.md` M4 节:配对 A/B 表(C=1/4/8 × vLLM/graph/eager,吞吐+TTFT+TPOT,全实测)、逐档 ≥100% 判定、graph vs eager 增益、TPOT 对齐情况。**逐档 ≥100% → 达标;任何一档 <100% → 不放宽,如实写,报告标 BLOCKED(spec §5 预案:≥95% 且归因带 T1 实测 → 交用户裁决)。** README M4 状态行同步。

- [ ] **Step 3: Commit(不打 tag —— m4-acceptance 由控制器在最终整体审查后打)**

```bash
git add docs/benchmarks.md README.md
git commit -m "docs(m4): paired A/B acceptance run + verdict"
```

---

## 计划自审记录(写作时已对照现状逐项核实)

1. **Spec 覆盖**:§1 四决定 → Task 7(≥100% + 配对 A/B)/全任务(主轴);§2 架构 → Task 2(prepared)+ Task 3(runner)+ Task 4(接线/开关);§3 数值契约 → Task 2/3 逐位门 + Task 5 golden 双门 + Task 6 调参纪律;§4 六任务 → 本计划 7 任务(T6 条件任务为 §4 Task 5 的展开);§5 中间门 → Task 6 判定步骤 + Task 7 BLOCKED 预案;§6 保护 → 全局纪律。
2. **API 对照(读源码核实,非凭记忆)**:`Engine.submit(prompt_ids, sampling, *, max_tokens, eos_ids=None)`、`step() -> list[TokenOutput]`、`has_pending()/drop()/assert_balanced()`;`PagedCacheView.ensure_append_capacity(layer_idx) -> (page, slot)`、`_lens: dict[int,int]`、scatter_decode 提交语义 = `get(layer,0)+1`;`PagedKVCache(num_layers, num_kv_heads, head_dim, page_size, allocator, device, dtype)`;`build_tables(views, page_size, device, *, lens_override) -> (tables, lens)`;`Qwen3Config.tiny()` = 2 层/hidden 64/kv 2/head_dim 16/vocab 256;`bench._prompts(count, in_tokens, vocab, seed)`、`triton_utils.TOL_BF16` 均存在;`Qwen3ForCausalLM.forward_decode` 现签名 `(input_ids, views)`、prepare 块位于层循环前(M3 Task 9)。
3. **类型一致性**:`prepared = (positions, tables, lens, (pages, slots))` 在 Task 2/3/4 一致;`DecodeGraphRunner(model, cache, allocator)` 构造签名 Task 3/4 一致;`BUCKETS` Task 3/4 一致;`decode_batch(seqs, views)` 与 `Executor.decode_batch` 同参一致。
4. **本轮设计修订(对照现状揪出的 3 处草稿 bug,已改正文)**:① Task 2 对拍的 pre/targets 必须取在参考步**之前**(decode 会提交 `_lens`,事后无法复现同状态);② runner 常驻 scratch 块使排空后的 `assert_balanced` 永假 —— Engine 侧改"恰 1 块 live"契约,Task 3/4 测试同步;③ tiny 词表(256)吐不出 `QWEN3_EOS_IDS`、`max_tokens=1` 又在 prefill 即终 —— EOS 缩批改两阶段陷阱法(`submit(eos_ids=)` 覆写,批量 4→2 触发真桶切换)。
