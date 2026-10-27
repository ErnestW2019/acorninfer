# M2 分页引擎实施计划(Paged KV + Continuous Batching + OpenAI API + TTFT/TPOT)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M1 的单序列引擎升级为可并发的推理引擎:分页 KV cache(page=16)、continuous batching、OpenAI 兼容 API、TTFT/TPOT 基准,通过"并发压测输出正确 + KV 无泄漏"验收。

**Architecture:** 分页存储(`[num_blocks, page, kv_heads, head_dim]` 每层)+ 块表;`PagedCacheView` 镜像 M1 `KVCache` 的 `update()/seq_len` 接口使单序列 forward 零改动复用;批量 decode 走新 `forward_decode`(GEMM 批量化 + 逐序列 SDPA);调度器 FCFS 准入 + 水位 + 抢占最老(recompute);Engine 同步 `step()` 循环,FastAPI 用后台 worker 桥接;bench 直连 Engine 测 TTFT/TPOT。

**Tech Stack:** Python 3.12 + PyTorch 2.13(bf16)、FastAPI + uvicorn + pydantic v2(已装)、tokenizers DecodeStream(增量 detok)、pytest。

**环境约定(每个任务都适用):**
- 仓库:`/home/ernest/github/acorninfer`(master;2026-10-27 已由 `suiren` 改名,包名 `acorninfer`,HEAD `9b44c0c`,tag `m0-acceptance`/`m1-acceptance` 在);
- 一切命令用 `.venv/bin/` 前缀;GPU 测试统一 `CUDA_VISIBLE_DEVICES=1`(GPU 0 被外部 vLLM 服务占用,代码写 `cuda:0`);
- 模型权重:`/home/ernest/model-weight/Qwen3-8B`(只读);
- TDD 纪律:red → implement → green → commit;改动之外零越界;golden 测试挂了 = BLOCKED 不许改断言;
- M1 的关键数值结论沿用:`logprobs` 截断后语义、EOS 先 append 再 break、bf16 跨路径噪声底(容差类断言必须写明校准依据)。

**非目标(明确推迟):** 分块 prefill 调度(准入按整段 prompt 需求检查,不足则排队)、prefix cache(M4)、beam/n>1 采样、per-request seed、CUDA graph、鉴权、jinja chat template(硬编码 ChatML)、Triton kernel(M3;M2 的逐序列 gather+SDPA 就是 M3 flash-decoding 的数值 reference)。

---

## 文件地图(Task 落地后的终态)

```
acorninfer/
  config.py            # + EngineConfig、QWEN3_EOS_IDS(Task 1/3)
  sampler.py           # + generator/NaN 校验(Task 1)
  models/qwen3.py      # + num_logits_to_keep(Task 2)、forward_decode 三件套(Task 6)
  cache/pager.py       # 新:BlockAllocator + PagedKVCache + PagedCacheView(Task 3/4)
  engine/
    __init__.py        # 新(Task 7)
    sequence.py        # 新:Sequence/RequestStatus(Task 7)
    scheduler.py       # 新:FCFS + 水位 + 抢占(Task 7)
    executor.py        # 新:prefill/decode_batch(Task 8)
    engine.py          # 新:Engine step 循环(Task 8)
  serving/
    __init__.py        # 新(Task 9)
    protocol.py        # 新:pydantic 模型 + ChatML(Task 9)
    server.py          # 新:FastAPI + SSE + worker(Task 9)
  bench/
    __init__.py        # 新(Task 10)
    cli.py             # 新:吞吐/TTFT/TPOT(Task 10)
  generate.py          # import EOS 改 config(Task 1);generate_cached prefill 用 logits_to_keep(Task 2)
tests/
  conftest.py          # 新:tiny_model fixture + QWEN3_8B(Task 1)
  test_pager.py        # 新(Task 3/4)
  test_paged_forward.py# 新(Task 5)
  test_forward_decode.py# 新(Task 6)
  test_scheduler.py    # 新(Task 7)
  test_engine.py       # 新(Task 8)
  test_server.py       # 新(Task 9)
  test_golden_m2.py    # 新:验收门(Task 10)
docs/benchmarks.md     # M2 行(Task 10)
README.md              # M2 用法(Task 10)
pyproject.toml         # addopts(Task 1)
```

接口契约总览(先立后用,后续任务不得偏离):

```python
# cache/pager.py
class BlockExhausted(Exception): ...
class BlockAllocator:
    def __init__(self, num_blocks: int): ...
    num_free: int; num_live: int; total: int
    def allocate(self) -> int: ...          # 无块可分 → BlockExhausted
    def free(self, block: int) -> None: ... # 重复 free → ValueError
    def assert_balanced(self) -> None: ...  # 有 live 块 → AssertionError

class PagedKVCache:
    def __init__(self, num_layers, num_kv_heads, head_dim, page_size,
                 allocator: BlockAllocator, device, dtype): ...
    # k_pages/v_pages: list per layer, 每层 [num_blocks, page, kv_heads, head_dim]

class PagedCacheView:                        # 每序列一个;镜像 KVCache 接口
    def __init__(self, cache: PagedKVCache, table: list[int] | None = None): ...
    table: list[int]                         # 与 Sequence 共享同一 list 对象
    seq_len: int
    def update(self, layer_idx, k_new, v_new) -> tuple[Tensor, Tensor]:
        # 写入新 token 的 k/v(按需扩块)→ 返回全量 [1, kv, seq_len, D]

# engine/sequence.py
class RequestStatus(Enum): WAITING; RUNNING; FINISHED
@dataclass
class Sequence:
    seq_id: int; prompt_ids: list[int]; sampling: SamplingParams
    max_tokens: int; eos_ids: frozenset[int]
    generated_ids: list[int]; block_table: list[int]   # block_table 与 view 共享
    num_pages: int                                     # engine 与 view.table 长度同步
    status: RequestStatus; stop_reason: str | None
    created_at: float; first_token_at: float | None; last_token_at: float | None
    num_tokens: int        # prompt + generated
    all_ids: list[int]     # prompt + generated(抢占 recompute 的 prefill 输入)

# engine/scheduler.py
class Scheduler:
    def __init__(self, allocator, page_size, max_running=8, watermark_blocks=0): ...
    waiting: deque[Sequence]; running: list[Sequence]
    def add(self, seq) -> None
    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        # 返回 (本轮要 prefill 的, 本轮要 decode 的);内部完成抢占与准入

# engine/executor.py
class Executor:
    def __init__(self, model): ...
    def prefill(self, seq, view) -> int            # 返回首个新 token
    def decode_batch(self, seqs, views) -> list[int]

# engine/engine.py
class Engine:
    def __init__(self, model, tokenizer, *, num_blocks=1024, page_size=16,
                 max_running=8, watermark_blocks=8): ...
    seqs: dict[int, Sequence]
    @classmethod
    def from_config(cls, cfg: EngineConfig, tokenizer) -> "Engine"
    def submit(self, prompt_ids, sampling, *, max_tokens, eos_ids=None) -> int
    def step(self) -> list[TokenOutput]            # 单线程契约:同一时刻只有一个调用方
    def has_pending(self) -> bool
    def assert_balanced(self) -> None
```

---

### Task 1: 速赢清理(conftest / addopts / EOS 入 config / Sampler 加固)

M1 终审的 quick wins,全部是 M2 各任务的直接依赖。

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_forward_cache.py`、`tests/test_generate_cached.py`、`tests/test_qwen3_model.py`(删本地 tiny_model,改用 fixture)
- Modify: `pyproject.toml`(addopts)
- Modify: `acorninfer/config.py`(QWEN3_EOS_IDS)、`acorninfer/generate.py`(import)
- Modify: `acorninfer/sampler.py`(generator + NaN 校验)
- Test: `tests/test_sampler.py`(追加 2 个测试)

- [ ] **Step 1: 写 conftest**

`tests/conftest.py`:

```python
"""Shared test helpers."""

import pytest
import torch

from acorninfer.config import Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"


@pytest.fixture
def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config.tiny()).to(dtype=torch.bfloat16)
```

三个测试文件:删除各自的 `def tiny_model(...)`,所有调用 `tiny_model()` 的测试函数签名改为接收 fixture 参数 `def test_xxx(tiny_model):`(已核实:三个文件中所有调用点都是无参调用,`test_qwen3_model.py` 的 device 参数从未被使用过)。

- [ ] **Step 2: pyproject 加 addopts(默认跳过 slow)**

`[tool.pytest.ini_options]` 段改为:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs model weights / GPU"]
addopts = "-m 'not slow'"
```

说明:CLI 显式 `-m slow` 会覆盖 addopts(后写的 -m 生效),所以现有所有 slow 命令不受影响;裸 `pytest` 从此只跑快集。

- [ ] **Step 3: EOS 常量入 config**

`acorninfer/config.py` import 区之后加:

```python
# Qwen3 生成终止符:<|endoftext|>(基座)、<|im_end|>(chat)。
# 引擎默认值;请求级 eos 覆盖在 Engine.submit 传入。
QWEN3_EOS_IDS: frozenset = frozenset({151643, 151645})
```

`acorninfer/generate.py`:删除 `EOS_IDS = frozenset(...)`,import 区加 `from acorninfer.config import QWEN3_EOS_IDS`,原位置保留兼容别名:

```python
EOS_IDS = QWEN3_EOS_IDS  # 兼容别名;新代码请用 acorninfer.config.QWEN3_EOS_IDS
```

- [ ] **Step 4: Sampler 加固(generator + NaN 校验)**

`acorninfer/sampler.py` 的 `__init__` 与 `process`/`sample` 改为:

```python
    def __init__(self, params: SamplingParams, generator: torch.Generator | None = None):
        if params.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < params.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if params.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables)")
        self.params = params
        self.generator = generator

    def process(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature -> top-k -> top-p; returns processed [vocab] logits."""
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError("logits contain NaN/inf; model forward is broken")
        logits = logits / self.params.temperature
```

`sample` 里 multinomial 加 generator:

```python
        token = int(torch.multinomial(logprobs.exp(), num_samples=1, generator=self.generator))
```

- [ ] **Step 5: 追加 sampler 测试**

`tests/test_sampler.py` 末尾追加:

```python
def test_nan_logits_rejected():
    s = Sampler(SamplingParams())
    with pytest.raises(ValueError, match="NaN/inf"):
        s.process(torch.tensor([1.0, float("nan")]))


def test_generator_makes_sampling_reproducible():
    g1, g2 = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    s1 = Sampler(SamplingParams(), generator=g1)
    s2 = Sampler(SamplingParams(), generator=g2)
    torch.manual_seed(0)
    logits = torch.randn(1000)
    seq1 = [s1.sample(logits.clone())[0] for _ in range(8)]
    seq2 = [s2.sample(logits.clone())[0] for _ in range(8)]
    assert seq1 == seq2
```

- [ ] **Step 6: 跑绿 + 回归**

```bash
.venv/bin/pytest tests/test_sampler.py tests/test_forward_cache.py tests/test_generate_cached.py tests/test_qwen3_model.py -v
.venv/bin/pytest -q          # 现在 = not-slow 全集
```

Expected: 全绿(裸 pytest 收集数应等于原 not-slow 计数 27 + 新增 2;以实际收集为准,全绿即可)。

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: M2 前速赢 — conftest/addopts/EOS 入 config/Sampler generator+NaN 校验"
```

---

### Task 2: `num_logits_to_keep`(prefill 只算末位 logits,TTFT 直接收益)

prefill 的 lm_head 在所有 prompt 位置上算 logits,但生成只需要最后一个位置。Qwen3-8B 上这是 4096×4096×151936 的白算矩阵乘;`hidden[:, -k:]` 切片后再过 lm_head 即可。

**Files:**
- Modify: `acorninfer/models/qwen3.py`(forward 签名)
- Modify: `acorninfer/generate.py`(generate_cached 的 prefill 调用)
- Test: `tests/test_qwen3_model.py`(追加 1 个测试)

- [ ] **Step 1: 写失败测试**

`tests/test_qwen3_model.py` 末尾追加(fixture 来自 Task 1):

```python
def test_num_logits_to_keep_matches_full(tiny_model):
    ids = torch.randint(0, 256, (1, 8))
    with torch.inference_mode():
        full = tiny_model(ids)
        kept = tiny_model(ids, num_logits_to_keep=3)
        kept1 = tiny_model(ids, num_logits_to_keep=1)
    assert kept.shape == (1, 3, 256)
    assert kept1.shape == (1, 1, 256)
    torch.testing.assert_close(kept, full[:, -3:])
    torch.testing.assert_close(kept1, full[:, -1:])
```

- [ ] **Step 2: 跑红**

```bash
.venv/bin/pytest tests/test_qwen3_model.py::test_num_logits_to_keep_matches_full -v
```

Expected: FAIL(`TypeError: forward() got an unexpected keyword argument 'num_logits_to_keep'`)。

- [ ] **Step 3: 实现**

`acorninfer/models/qwen3.py` 的 `Qwen3ForCausalLM.forward` 整体替换为:

```python
    def forward(
        self,
        input_ids: torch.Tensor,
        cache=None,
        num_logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """input_ids: [batch, seq] -> logits: [batch, seq, vocab] (float32).

        cache=None: 全量前向(M0 oracle 路径)。传入 KVCache 时为增量路径:
        位置由 cache.seq_len 推导,prefill(cache 空)与 decode(cache 非空)共用。
        num_logits_to_keep: 只对最后 k 个位置算 lm_head(prefill 生成只需末位;
        None = 全位置,M0 路径不变)。
        """
        s = input_ids.shape[1]
        past_len = 0 if cache is None else cache.seq_len
        positions = torch.arange(past_len, past_len + s, device=input_ids.device)
        dtype = self.embed_tokens.weight.dtype
        cos, sin = self.rope(positions, dtype, input_ids.device)
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, i)
        hidden = self.norm(hidden)
        if num_logits_to_keep is not None:
            hidden = hidden[:, -num_logits_to_keep:, :]
        return self.lm_head(hidden).float()
```

`acorninfer/generate.py` 的 prefill 行改为:

```python
    # prefill:整段 prompt 一次前向,取最后位置的分布(lm_head 只算末位)
    last = model(
        torch.tensor([out], device=device, dtype=torch.long),
        cache,
        num_logits_to_keep=1,
    )[0, -1]
```

- [ ] **Step 4: 跑绿 + 全套件**

```bash
.venv/bin/pytest -q
CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest -m slow -q
```

Expected: 全绿。关键正确性证据:`test_cached_greedy_matches_oracle_*`(cached 与 oracle 逐 id 相等)证明 prefill 末位 logits 数值不变——这正是切片只省"用不到的位置"的证明。

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "perf: num_logits_to_keep — prefill 只算末位 logits"
```

---

### Task 3: EngineConfig + BlockAllocator

物理块分配器:空闲线性表(LIFO 栈序,刚释放的块优先复用,对 cache 友好)。`assert_balanced` 是 spec §5 的 KV 泄漏 debug 守卫的分配器侧。

**Files:**
- Create: `acorninfer/cache/__init__.py`(空文件)、`acorninfer/cache/pager.py`(本任务只写分配器部分)
- Modify: `acorninfer/config.py`(EngineConfig)
- Test: `tests/test_pager.py`

- [ ] **Step 1: EngineConfig**

`acorninfer/config.py` 顶部 import 加 `import torch`,文件末尾追加:

```python
@dataclass(frozen=True)
class EngineConfig:
    """Engine-level configuration (M2)."""

    model_dir: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    num_blocks: int = 1024      # 物理块数;Qwen3-8B 每块 16 token ≈ 2.3MB(全 36 层)
    page_size: int = 16
    max_running: int = 8        # decode batch 上限
    watermark_blocks: int = 8   # 水位:free 低于此值不再准入,降低抢占概率
```

- [ ] **Step 2: 写失败测试** `tests/test_pager.py`:

```python
import pytest

from acorninfer.cache.pager import BlockAllocator, BlockExhausted


def test_allocate_returns_unique_blocks_then_exhausts():
    a = BlockAllocator(2)
    b1, b2 = a.allocate(), a.allocate()
    assert {b1, b2} == {0, 1}
    assert a.num_free == 0 and a.num_live == 2 and a.total == 2
    with pytest.raises(BlockExhausted):
        a.allocate()


def test_free_returns_block_to_pool_lifo():
    a = BlockAllocator(2)
    x, y = a.allocate(), a.allocate()
    a.free(x)
    a.free(y)
    assert a.allocate() == y  # LIFO:后释放先复用
    assert a.allocate() == x


def test_double_free_raises():
    a = BlockAllocator(1)
    b = a.allocate()
    a.free(b)
    with pytest.raises(ValueError, match="double free"):
        a.free(b)


def test_assert_balanced_detects_leak():
    a = BlockAllocator(1)
    b = a.allocate()
    with pytest.raises(AssertionError, match="leak"):
        a.assert_balanced()
    a.free(b)
    a.assert_balanced()  # 不抛
```

- [ ] **Step 3: 跑红**

```bash
.venv/bin/pytest tests/test_pager.py -v
```

Expected: FAIL(`ModuleNotFoundError: acorninfer.cache`)。

- [ ] **Step 4: 实现** `acorninfer/cache/__init__.py` 空文件;`acorninfer/cache/pager.py`:

```python
"""Paged KV cache: block allocator (this file grows PagedKVCache in Task 4).

Spec §3.5: 块表 + page size 16;分配器先做线性空闲表,M4 加哈希前缀匹配。
Spec §5:   KV 泄漏 debug 守卫 —— assert_balanced 在引擎排空后必须通过。
"""

from __future__ import annotations

import torch


class BlockExhausted(Exception):
    """物理块耗尽;调度器以此触发排队/抢占。"""


class BlockAllocator:
    """Linear free-list allocator for physical KV blocks (LIFO reuse)."""

    def __init__(self, num_blocks: int):
        self._free: list[int] = list(range(num_blocks))[::-1]  # LIFO
        self._live: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_live(self) -> int:
        return len(self._live)

    @property
    def total(self) -> int:
        return len(self._free) + len(self._live)

    def allocate(self) -> int:
        if not self._free:
            raise BlockExhausted("no free KV blocks")
        b = self._free.pop()
        self._live.add(b)
        return b

    def free(self, block: int) -> None:
        if block not in self._live:
            raise ValueError(f"double free: block {block} is not live")
        self._live.discard(block)
        self._free.append(block)

    def assert_balanced(self) -> None:
        """Debug 守卫:引擎排空后必须零 live 块,否则 = KV 泄漏。"""
        assert not self._live, f"KV leak: blocks {sorted(self._live)} never freed"
```

- [ ] **Step 5: 跑绿 + Commit**

```bash
.venv/bin/pytest tests/test_pager.py -v   # 4 passed
git add -A && git commit -m "feat: BlockAllocator 线性空闲表 + KV 泄漏守卫"
```

---

### Task 4: PagedKVCache + PagedCacheView(页式存储与块表)

物理布局:每层 `k_pages/v_pages: [num_blocks, page_size, kv_heads, head_dim]`。
`PagedCacheView` 是**每序列**的视图:持有块表(与 `Sequence.block_table` 共享同一个 list 对象)、`update()` 写入+gather、`seq_len`。它与 M1 `KVCache` 接口一致,这是 Task 5 零改动复用 forward 的关键。

**Files:**
- Modify: `acorninfer/cache/pager.py`(追加)
- Test: `tests/test_pager.py`(追加)

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_pager.py`)

```python
import torch

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache

PAGE = 4  # 小页方便测试跨页


def make_cache(num_blocks=4, kv_heads=2, head_dim=8, layers=2):
    alloc = BlockAllocator(num_blocks)
    return PagedKVCache(layers, kv_heads, head_dim, PAGE, alloc, "cpu", torch.float32)


def test_view_update_grows_and_gathers_like_concat():
    cache = make_cache()
    view = PagedCacheView(cache)
    k1 = torch.randn(1, 2, 3, 8)
    v1 = torch.randn(1, 2, 3, 8)
    k_full, v_full = view.update(0, k1, v1)
    assert view.seq_len == 3 and len(view.table) == 1
    torch.testing.assert_close(k_full, k1)
    # 跨页:3 + 3 = 6 > PAGE=4,需要第 2 块
    k2 = torch.randn(1, 2, 3, 8)
    v2 = torch.randn(1, 2, 3, 8)
    k_full, v_full = view.update(0, k2, v2)
    assert view.seq_len == 6 and len(view.table) == 2
    assert torch.equal(k_full[:, :, :3], k1)
    assert torch.equal(k_full[:, :, 3:], k2)
    assert torch.equal(v_full[:, :, :3], v1)
    assert torch.equal(v_full[:, :, 3:], v2)


def test_pages_independent_across_layers_and_views():
    cache = make_cache()
    va, vb = PagedCacheView(cache), PagedCacheView(cache)
    ka = torch.randn(1, 2, 2, 8)
    va.update(0, ka, torch.randn(1, 2, 2, 8))
    kb = torch.randn(1, 2, 5, 8)
    vb.update(1, kb, torch.randn(1, 2, 5, 8))  # 另一层,互不干扰
    assert va.seq_len == 2 and vb.seq_len == 5
    k_back, _ = va.update(0, torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))
    assert torch.equal(k_back[:, :, :2], ka)  # 另一 view 的写入不影响本 view


def test_gather_matches_naive_concat_exactly():
    """分页存储 + 块表 gather 必须与朴素拼接逐位一致(M2 等价性的根)。"""
    cache = make_cache()
    view = PagedCacheView(cache)
    naive_k = []
    for s in (5, 1, 4):  # 故意跨 3 块:5+1+4=10 > 2 页
        k = torch.randn(1, 2, s, 8)
        naive_k.append(k[0])
        view.update(0, k, torch.randn(1, 2, s, 8))
    k_full, _ = view.update(0, torch.randn(1, 2, 0, 8), torch.randn(1, 2, 0, 8))
    assert torch.equal(k_full[0], torch.cat(naive_k, dim=0))


def test_view_allocation_exhausts_pool():
    alloc = BlockAllocator(2)
    cache = PagedKVCache(2, 2, 8, PAGE, alloc, "cpu", torch.float32)
    view = PagedCacheView(cache)
    with pytest.raises(BlockExhausted):
        view.update(0, torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8))
    # 中断的 update 已把拿到的块记进 table;释放后守卫必须通过
    for b in view.table:
        alloc.free(b)
    alloc.assert_balanced()
```

- [ ] **Step 2: 跑红**

```bash
.venv/bin/pytest tests/test_pager.py -v
```

Expected: 新增测试 FAIL(`ImportError: cannot import name 'PagedKVCache'`)。

- [ ] **Step 3: 实现**(追加到 `acorninfer/cache/pager.py`)

```python
class PagedKVCache:
    """物理页存储:每层 k_pages/v_pages [num_blocks, page_size, kv_heads, head_dim]。

    序列通过块表(list[block_id])引用自己的页;token 顺序 = 块表序 × 页内偏移。
    M3 的 Triton paged attention 将直接按页读取;本实现的 gather-to-contiguous
    就是它的数值 reference。
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        allocator: BlockAllocator,
        device,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.allocator = allocator
        self.device = device
        self.dtype = dtype
        shape = (allocator.total, page_size, num_kv_heads, head_dim)
        self.k_pages = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v_pages = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]


class PagedCacheView:
    """每序列视图:块表 + update()/seq_len,与 M1 KVCache 接口镜像。

    table 与 Sequence.block_table 共享同一 list 对象(engine 负责传递),
    因此调度器/引擎释放块时清空 seq.block_table 即同时清空视图。
    """

    def __init__(self, cache: PagedKVCache, table: list[int] | None = None):
        self.cache = cache
        self.table: list[int] = table if table is not None else []
        self._len = 0

    @property
    def seq_len(self) -> int:
        return self._len

    def update(
        self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写入新 token 的 k/v 并返回全量 [1, kv_heads, seq_len, head_dim]。

        k_new/v_new: [1, kv_heads, s, head_dim];块不足时按需 allocate。
        """
        c = self.cache
        s = k_new.shape[2]
        page = c.page_size
        needed = (self._len + s + page - 1) // page
        while len(self.table) < needed:
            self.table.append(c.allocator.allocate())
        pos = torch.arange(self._len, self._len + s)
        # dtype=torch.long 必须显式:s=0 时空 list 会默认造出 float 张量,当索引会炸
        rows = torch.tensor(
            [self.table[p // page] for p in pos.tolist()], device=c.device, dtype=torch.long
        )
        slots = (pos % page).to(c.device)
        c.k_pages[layer_idx][rows, slots] = k_new[0].to(c.dtype)
        c.v_pages[layer_idx][rows, slots] = v_new[0].to(c.dtype)
        self._len += s
        return self._gather(layer_idx, c.k_pages), self._gather(layer_idx, c.v_pages)

    def _gather(self, layer_idx: int, pages: list[torch.Tensor]) -> torch.Tensor:
        c = self.cache
        t = pages[layer_idx][torch.tensor(self.table, device=c.device)]  # [n, page, kv, D]
        t = t.reshape(-1, c.num_kv_heads, c.head_dim)[: self._len]       # [len, kv, D]
        return t.permute(1, 0, 2)[None]                                  # [1, kv, len, D]
```

- [ ] **Step 4: 跑绿**

```bash
.venv/bin/pytest tests/test_pager.py -v   # 8 passed
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: PagedKVCache 页式存储 + PagedCacheView 块表视图(等价朴素拼接)"
```

---

### Task 5: 单序列分页 forward 等价(M1 接口设计的兑现时刻)

`PagedCacheView` 的 `update()/seq_len` 与 `KVCache` 同签名 → **`Qwen3ForCausalLM.forward` 一行都不用改**。本任务用 4 个等价性测试证明它(这正是 M1 计划里"接口兼容 M2 分页"承诺的兑现,也是 M1 终审指定的 TDD 入口)。

**Files:**
- Test: `tests/test_paged_forward.py`(新建;源码零改动)

- [ ] **Step 1: 写等价性测试**

```python
"""单序列分页路径 == 朴素 cache 路径 == 无 cache 路径(逐位)。

gather 只做拷贝不改数值,SDPA 输入逐位相同 → 输出必须 torch.equal。
任何一处 fail 都意味着分页语义(块表序、页内偏移、部分页截断)有 bug。
"""

import pytest
import torch

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM


@pytest.fixture
def env(tiny_model):
    alloc = BlockAllocator(16)
    cache = PagedKVCache(
        tiny_model.cfg.num_hidden_layers,
        tiny_model.cfg.num_key_value_heads,
        tiny_model.cfg.head_dim,
        4,  # 小页,强制频繁跨页
        alloc,
        "cpu",
        torch.bfloat16,
    )
    return tiny_model, cache, alloc


def _fresh_view(cache):
    return PagedCacheView(cache)


def _free_view(view, alloc):
    for b in view.table:
        alloc.free(b)


def test_paged_prefill_equals_no_cache(env):
    model, cache, alloc = env
    ids = torch.randint(0, 256, (1, 9))
    with torch.inference_mode():
        plain = model(ids)
        view = _fresh_view(cache)
        paged = model(ids, view)
    assert torch.equal(paged, plain)
    _free_view(view, alloc)


def test_paged_decode_step_equals_full_forward(env):
    model, cache, alloc = env
    ids = torch.randint(0, 256, (1, 6))
    with torch.inference_mode():
        full = model(ids)
        view = _fresh_view(cache)
        _ = model(ids[:, :5], view)
        stepped = model(ids[:, 5:], view)
    assert torch.equal(stepped[:, -1], full[:, -1])
    _free_view(view, alloc)


def test_paged_multi_step_equals_full_forward(env):
    model, cache, alloc = env
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 6))
    with torch.inference_mode():
        full = model(ids)
        view = _fresh_view(cache)
        last = model(ids[:, :3], view)
        for t in range(3, 6):
            last = model(ids[:, t : t + 1], view)
    assert torch.equal(last[:, -1], full[:, -1])
    _free_view(view, alloc)


def test_paged_chunked_prefill_equals_full_forward(env):
    """分支 3(非空 cache + 多 token)走分页 gather。"""
    model, cache, alloc = env
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (1, 11))  # 11 token,页=4 → 至少跨 2 次页边界
    with torch.inference_mode():
        full = model(ids)
        view = _fresh_view(cache)
        _ = model(ids[:, :3], view)
        mid = model(ids[:, 3:7], view)   # 4-token 块,显式掩码分支
        end = model(ids[:, 7:], view)
    assert torch.equal(mid, full[:, 3:7])
    assert torch.equal(end, full[:, 7:])
    _free_view(view, alloc)
```

- [ ] **Step 2: 跑绿(这一步应当直接绿——如果红了,说明 Task 4 的分页语义有 bug,修 Task 4 而不是这里)**

```bash
.venv/bin/pytest tests/test_paged_forward.py -v   # 4 passed
.venv/bin/pytest -q                                # 全套无回归
```

这一步**应当直接绿**:分页路径没有改任何模型代码,只是换了存储。如果红了,说明 Task 4 的分页语义有 bug(块表序/页内偏移/部分页截断),修 Task 4 而不是这里。

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: 单序列分页 forward 与朴素/无 cache 逐位等价(forward 零改动)"
```

---

### Task 6: `forward_decode`(批量 decode:GEMM 批量化 + 逐序列 SDPA)

continuous batching 的吞吐来源:decode 是权重带宽瓶颈(每步读全量 16GB),B 条序列合并成一次前向 = 权重读一次服务 B 条。GEMM 全部批量化;attention 因为各序列 KV 长度不同,逐序列 SDPA(q_len=1 无需掩码,M1 分支 2 的推广)。写回用 `view.update`(gather 结果正好就是本步 attention 需要的全量 K/V)。

**Files:**
- Modify: `acorninfer/models/qwen3.py`(Qwen3Attention/Block/ForCausalLM 各加 `forward_decode`)
- Test: `tests/test_forward_decode.py`

- [ ] **Step 1: 写失败测试**

```python
"""批量 decode 正确性:
- 单序列 forward_decode == 现有 forward 路径(逐位;同形状同内核);
- 批量 forward_decode 的 token 序列 == 逐序列单独 decode(token 序列是真正的门;
  logits 允许批处理内核的 ulp 级差异)。
"""

import torch

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM

import pytest


@pytest.fixture
def env(tiny_model):
    alloc = BlockAllocator(32)
    cache = PagedKVCache(
        tiny_model.cfg.num_hidden_layers,
        tiny_model.cfg.num_key_value_heads,
        tiny_model.cfg.head_dim,
        4,
        alloc,
        "cpu",
        torch.bfloat16,
    )
    return tiny_model, cache, alloc


def test_decode_single_matches_forward_path(env):
    model, cache, alloc = env
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 5))
    with torch.inference_mode():
        ref_view = PagedCacheView(cache)
        ref = model(ids[:, :4], ref_view)
        ref = model(ids[:, 4:], ref_view)          # 旧路径 decode 一步
        v = PagedCacheView(cache)
        _ = model(ids[:, :4], v)                    # 同一 prefill
        new = model.forward_decode(ids[:, 4:], [v])  # [1, vocab]
    assert torch.equal(new[0], ref[0, -1])
    for b in v.table:
        alloc.free(b)
    for b in ref_view.table:
        alloc.free(b)


def test_decode_batch_matches_individual_token_sequences(env):
    model, cache, alloc = env
    torch.manual_seed(1)
    lens = [3, 5, 7]
    prompts = [torch.randint(0, 256, (1, n)) for n in lens]
    views_indiv = [PagedCacheView(cache) for _ in range(3)]
    views_batch = [PagedCacheView(cache) for _ in range(3)]
    with torch.inference_mode():
        for p, vi, vb in zip(prompts, views_indiv, views_batch):
            model(p[:, :-1], vi)
            model(p[:, :-1], vb)
        n_steps = 4
        for _ in range(n_steps):
            batch_ids = torch.tensor([[vi.seq_len and int(torch.tensor(0)) ] for vi in views_indiv])
            break  # 占位,见下方真实实现
    for b in views_indiv + views_batch:
        for blk in b.table:
            alloc.free(blk)
```

上面第二个测试的循环体写成真实版本(整体替换 `test_decode_batch_matches_individual_token_sequences`):

```python
def test_decode_batch_matches_individual_token_sequences(env):
    model, cache, alloc = env
    torch.manual_seed(1)
    lens = [3, 5, 7]
    prompts = [torch.randint(0, 256, (1, n)) for n in lens]
    views_indiv = [PagedCacheView(cache) for _ in range(3)]
    views_batch = [PagedCacheView(cache) for _ in range(3)]
    indiv_seqs = [[] for _ in range(3)]
    batch_seqs = [[] for _ in range(3)]
    with torch.inference_mode():
        for p, vi, vb in zip(prompts, views_indiv, views_batch):
            model(p[:, :-1], vi)   # prefill(留 1 个 token 做 decode 输入)
            model(p[:, :-1], vb)
        for _ in range(4):
            # 逐序列:自己上一步的 token 喂自己
            for i, (p, vi) in enumerate(zip(prompts, views_indiv)):
                prev = p[0, -1] if not indiv_seqs[i] else indiv_seqs[i][-1]
                logits = model(torch.tensor([[int(prev)]]), vi)
                indiv_seqs[i].append(int(logits[0, -1].argmax()))
            # 批量:forward_decode 一次吃 B 条
            prevs = [p[0, -1] if not batch_seqs[i] else batch_seqs[i][-1]
                     for i, p in enumerate(prompts)]
            x = torch.tensor([[int(t)] for t in prevs])
            logits = model.forward_decode(x, views_batch)
            for i in range(3):
                batch_seqs[i].append(int(logits[i].argmax()))
    assert indiv_seqs == batch_seqs
    for v in views_indiv + views_batch:
        for blk in v.table:
            alloc.free(blk)
```

- [ ] **Step 2: 跑红**

```bash
.venv/bin/pytest tests/test_forward_decode.py -v
```

Expected: FAIL(`AttributeError: 'Qwen3ForCausalLM' object has no attribute 'forward_decode'`)。

- [ ] **Step 3: 实现**(`acorninfer/models/qwen3.py` 三处追加)

`Qwen3Attention` 追加方法:

```python
    def forward_decode(self, hidden, cos, sin, views, layer_idx):
        """批量 decode: hidden [B, 1, hidden], cos/sin [B, 1, 1, D]。

        GEMM 批量化(continuous batching 的收益所在);attention 逐序列
        (KV 长度不同,q_len=1 无需掩码 —— M1 分支 2 的推广)。
        """
        B, s, _ = hidden.shape
        q = self.q_proj(hidden).view(B, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE:每个序列自己的位置;cos/sin 已是 [B,1,1,D],不走 _apply_rope(形状约定不同)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        rep = self.num_heads // self.num_kv_heads
        outs = []
        for b, view in enumerate(views):
            k_full, v_full = view.update(layer_idx, k[b : b + 1], v[b : b + 1])
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
            outs.append(F.scaled_dot_product_attention(q[b : b + 1], k_full, v_full))
        out = torch.cat(outs, dim=0)                 # [B, H, 1, D]
        out = out.transpose(1, 2).reshape(B, s, -1)
        return self.o_proj(out)
```

`Qwen3Block` 追加方法:

```python
    def forward_decode(self, hidden, cos, sin, views, layer_idx):
        hidden = hidden + self.self_attn.forward_decode(
            self.input_layernorm(hidden), cos, sin, views, layer_idx
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden
```

`Qwen3ForCausalLM` 追加方法:

```python
    def forward_decode(self, input_ids: torch.Tensor, views) -> torch.Tensor:
        """批量 decode:input_ids [B, 1] + 每序列的 PagedCacheView -> [B, vocab]。

        每序列位置 = 各自 cache.seq_len(互不相同);新 token 的 K/V 在
        attention 内写入各自分页块。
        """
        B = input_ids.shape[0]
        positions = torch.tensor([v.seq_len for v in views], device=input_ids.device)
        dtype = self.embed_tokens.weight.dtype
        cos, sin = self.rope(positions, dtype, input_ids.device)  # [B, head_dim]
        cos = cos[:, None, None, :]  # [B, 1, 1, D]
        sin = sin[:, None, None, :]
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_decode(hidden, cos, sin, views, i)
        hidden = self.norm(hidden)
        return self.lm_head(hidden).float()[:, 0, :]  # [B, vocab]
```

- [ ] **Step 4: 跑绿**

```bash
.venv/bin/pytest tests/test_forward_decode.py -v   # 2 passed
.venv/bin/pytest -q                                 # 无回归
```

说明:`torch.equal` 在 CPU tiny 模型上是稳定预期(gather 同值、同形状内核)。若未来 GPU 上出现 ulp 级不等,处理方式与 M1 同:改 assert_close(容差写明 bf16 依据),token 序列断言不许放松。

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: forward_decode — 批量 GEMM + 逐序列分页 SDPA"
```

---

### Task 7: Sequence + Scheduler(状态机、FCFS 准入、抢占 recompute)

调度器是纯逻辑(不碰张量):块需求都是整数算术。`Sequence.block_table` 与视图共享 list(Engine 负责接线);`num_pages` 由 Engine 在每次前向后同步。

**Files:**
- Create: `acorninfer/engine/__init__.py`(空)、`acorninfer/engine/sequence.py`、`acorninfer/engine/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: 实现 Sequence**(先写这个,调度器测试需要它;它的行为由调度器/引擎测试覆盖)

`acorninfer/engine/sequence.py`:

```python
"""Sequence: 引擎内的请求状态机。"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from acorninfer.sampler import SamplingParams


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt_ids: list[int]
    sampling: SamplingParams
    max_tokens: int
    eos_ids: frozenset[int]
    generated_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)  # 与 PagedCacheView 共享
    num_pages: int = 0                                    # engine 每次前向后同步
    status: RequestStatus = RequestStatus.WAITING
    stop_reason: str | None = None
    created_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    last_token_at: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.generated_ids)

    @property
    def all_ids(self) -> list[int]:
        """prompt + generated;抢占 recompute 的 prefill 输入。"""
        return self.prompt_ids + self.generated_ids

    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED
```

- [ ] **Step 2: 写调度器失败测试** `tests/test_scheduler.py`:

```python
"""调度器纯逻辑测试:准入顺序、水位、max_running、抢占最老(recompute)。

页面数学(易错,先算清楚):
- PAGE=4;mkseq(i, 5) → num_tokens=5 → 需 ceil(5/4)=2 块;
- 抢占场景 mkseq(i, 3, gen=1) → num_tokens=4 == 1 页满 → 下一 token 必须新块。
"""

from acorninfer.cache.pager import BlockAllocator
from acorninfer.engine.scheduler import Scheduler
from acorninfer.engine.sequence import RequestStatus, Sequence
from acorninfer.sampler import SamplingParams

PAGE = 4


def mkseq(sid, prompt_len, gen=0):
    s = Sequence(
        seq_id=sid,
        prompt_ids=[sid * 100 + i for i in range(prompt_len)],
        sampling=SamplingParams(),
        max_tokens=64,
        eos_ids=frozenset({999}),
    )
    s.generated_ids = list(range(1000, 1000 + gen))
    return s


def test_fcfs_admission_order():
    alloc = BlockAllocator(8)
    sch = Scheduler(alloc, PAGE)
    for i in range(3):
        sch.add(mkseq(i, 5))
    prefills, decodes = sch.schedule()
    assert [s.seq_id for s in prefills] == [0, 1, 2]
    assert decodes == []


def test_max_running_cap():
    alloc = BlockAllocator(64)
    sch = Scheduler(alloc, PAGE, max_running=2)
    for i in range(3):
        sch.add(mkseq(i, 5))
    prefills, _ = sch.schedule()
    assert len(prefills) == 2
    assert sch.waiting[0].seq_id == 2


def test_admission_stops_at_watermark():
    alloc = BlockAllocator(8)
    sch = Scheduler(alloc, PAGE, max_running=4, watermark_blocks=3)
    for i in range(3):
        sch.add(mkseq(i, 5))  # 每条需 2 块
    prefills, _ = sch.schedule()
    # 准入数学:s0 后 free 8≥2 ✓;s1 后 8-4=4,预算 4-3=1 < 2 → s2 不准入
    assert [s.seq_id for s in prefills] == [0, 1]
    for s in sch.running:  # 模拟 prefill 已各占 2 块
        s.num_pages = 2
        s.block_table = [alloc.allocate(), alloc.allocate()]
    # free=4,水位 3 → 预算 1 < need 2 → 仍不准入
    prefills2, _ = sch.schedule()
    assert prefills2 == []
    sch.watermark = 0  # 水位放开 → 立即可入
    prefills3, _ = sch.schedule()
    assert [s.seq_id for s in prefills3] == [2]


def test_preempt_oldest_running_recompute():
    alloc = BlockAllocator(3)
    sch = Scheduler(alloc, PAGE)
    s0, s1 = mkseq(0, 3, gen=1), mkseq(1, 3, gen=1)  # num_tokens=4 == 1 页满
    for s in (s0, s1):
        sch.add(s)
    sch.schedule()  # 双双准入(此时尚未真实占块)
    for s in (s0, s1):
        s.status = RequestStatus.RUNNING
        s.num_pages = 1
        s.block_table = [alloc.allocate()]
    assert alloc.num_free == 1
    # 两条都页满:deficit=2 > free=1 → 抢占最老 s0(释放 1 块 → free=2 ≥ 1)
    prefills, decodes = sch.schedule()
    assert s0.status is RequestStatus.WAITING
    assert s0.seq_id == sch.waiting[0].seq_id  # 回到队首(FCFS 重试优先)
    assert s0.num_pages == 0 and s0.block_table == []
    assert [s.seq_id for s in decodes] == [1]
    assert prefills == []  # 刚被抢占者本轮不再准入(防抖;下轮重试)
    assert alloc.num_free == 2 and alloc.num_live == 1
```

- [ ] **Step 3: 跑红**

```bash
.venv/bin/pytest tests/test_scheduler.py -v
```

Expected: FAIL(`ModuleNotFoundError: acorninfer.engine`)。

- [ ] **Step 4: 实现** `acorninfer/engine/scheduler.py`:

```python
"""FCFS 调度器:准入 + 水位 + 抢占最老(recompute)。spec §5。

策略:KV 不足先排队;running 增长可能耗尽块时,抢占最老请求
(释放其全部块,回 waiting 队首,重被准入时对 all_ids 重新 prefill)。
"""

from __future__ import annotations

from collections import deque

from acorninfer.cache.pager import BlockAllocator
from acorninfer.engine.sequence import RequestStatus, Sequence


class Scheduler:
    def __init__(
        self,
        allocator: BlockAllocator,
        page_size: int,
        max_running: int = 8,
        watermark_blocks: int = 0,
    ):
        self.allocator = allocator
        self.page_size = page_size
        self.max_running = max_running
        self.watermark = watermark_blocks
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def _release(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.allocator.free(b)
        seq.block_table.clear()
        seq.num_pages = 0

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """返回 (本轮 prefill, 本轮 decode);副作用:抢占与准入。"""
        # 抢占:保证每个 running 序列本轮 decode 至多再要 1 块
        preempted_ids: set[int] = set()
        while True:
            deficit = sum(
                1 for s in self.running if s.num_tokens == s.num_pages * self.page_size
            )
            if self.allocator.num_free >= deficit:
                break
            victim = self.running.pop(0)  # 最老
            victim.status = RequestStatus.WAITING
            self._release(victim)
            self.waiting.appendleft(victim)
            preempted_ids.add(victim.seq_id)

        # 准入:FCFS,受水位与 max_running 约束;刚被抢占者本轮不准入(防抖:
        # 否则同一步"抢占→重准入"空转,下轮重试)
        prefills: list[Sequence] = []
        while self.waiting and len(self.running) < self.max_running:
            cand = self.waiting[0]
            if cand.seq_id in preempted_ids:
                break
            need = -(-cand.num_tokens // self.page_size) - cand.num_pages
            if self.allocator.num_free - self.watermark < need:
                break
            self.waiting.popleft()
            cand.status = RequestStatus.RUNNING
            self.running.append(cand)
            prefills.append(cand)

        prefill_ids = {s.seq_id for s in prefills}
        decodes = [s for s in self.running if s.seq_id not in prefill_ids]
        return prefills, decodes
```

- [ ] **Step 5: 跑绿**

```bash
.venv/bin/pytest tests/test_scheduler.py -v   # 3 passed
.venv/bin/pytest -q                            # 无回归
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: Sequence 状态机 + FCFS 调度器(水位 + 抢占最老 recompute)"
```

---

### Task 8: Executor + Engine(step 循环与请求生命周期)

Engine 是唯一知道"视图与序列怎么接线"的地方;`step()` 单线程契约(服务器/基准各自独占调用)。抢占 recompute 的 prefill 输入 = `all_ids`(prompt+已生成),greedy 下逐位复现。

**Files:**
- Create: `acorninfer/engine/executor.py`、`acorninfer/engine/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: 实现 Executor**

`acorninfer/engine/executor.py`:

```python
"""批量前向执行:prefill 单序列 / decode 批量化,采样按序列参数。"""

from __future__ import annotations

import torch

from acorninfer.engine.sequence import Sequence
from acorninfer.models.qwen3 import Qwen3ForCausalLM
from acorninfer.sampler import Sampler


class Executor:
    def __init__(self, model: Qwen3ForCausalLM):
        self.model = model
        self.device = next(model.parameters()).device

    def prefill(self, seq: Sequence, view) -> int:
        """整段(all_ids)一次前向,选出新 token。抢占重放同一路径。"""
        ids = torch.tensor([seq.all_ids], device=self.device, dtype=torch.long)
        logits = self.model(ids, view, num_logits_to_keep=1)[0, -1]
        return self._sample(seq, logits)

    def decode_batch(self, seqs: list[Sequence], views) -> list[int]:
        x = torch.tensor(
            [[s.generated_ids[-1]] for s in seqs], device=self.device, dtype=torch.long
        )
        logits = self.model.forward_decode(x, views)  # [B, vocab]
        return [self._sample(s, logits[i]) for i, s in enumerate(seqs)]

    @staticmethod
    def _sample(seq: Sequence, logits: torch.Tensor) -> int:
        if seq.sampling.temperature == 0:  # OpenAI 语义:temp=0 即 greedy
            return int(logits.argmax())
        return Sampler(seq.sampling).sample(logits)[0]
```

- [ ] **Step 2: 写 Engine 失败测试** `tests/test_engine.py`:

```python
"""Engine 端到端(tiny 模型,CPU):并发等价、连续提交、抢占回收、泄漏守卫。"""

import pytest
import torch

from acorninfer.engine.engine import Engine
from acorninfer.generate import generate_cached
from acorninfer.sampler import SamplingParams


@pytest.fixture
def engine(tiny_model):
    eng = Engine(
        tiny_model,
        tokenizer=None,
        num_blocks=64,
        page_size=4,
        max_running=4,
        watermark_blocks=0,
    )
    yield eng
    # 未完成的序列(如有)排空后再验平衡,避免中间态泄漏误报
    while eng.has_pending():
        eng.step()
    eng.assert_balanced()


def _ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)[0].tolist()


def test_two_concurrent_greedy_match_individual(engine):
    p1, p2 = _ids(5, 0), _ids(7, 1)
    with torch.inference_mode():
        ref1, _ = generate_cached(engine.model, p1, max_new_tokens=8)
        ref2, _ = generate_cached(engine.model, p2, max_new_tokens=8)
    s1 = engine.submit(p1, SamplingParams(temperature=0.0), max_tokens=8)
    s2 = engine.submit(p2, SamplingParams(temperature=0.0), max_tokens=8)
    while engine.has_pending():
        engine.step()
    assert engine.seqs[s1].all_ids == ref1
    assert engine.seqs[s2].all_ids == ref2
    assert {engine.seqs[s].stop_reason for s in (s1, s2)} == {"length"}


def test_midflight_submission_continuous_batching(engine):
    p1, p2 = _ids(5, 0), _ids(7, 1)
    with torch.inference_mode():
        ref1, _ = generate_cached(engine.model, p1, max_new_tokens=6)
        ref2, _ = generate_cached(engine.model, p2, max_new_tokens=6)
    s1 = engine.submit(p1, SamplingParams(temperature=0.0), max_tokens=6)
    engine.step()  # s1 prefill
    s2 = engine.submit(p2, SamplingParams(temperature=0.0), max_tokens=6)  # 中途插入
    while engine.has_pending():
        engine.step()
    assert engine.seqs[s1].all_ids == ref1
    assert engine.seqs[s2].all_ids == ref2


def test_metrics_timestamps_and_first_token(engine):
    s = engine.submit(_ids(5, 0), SamplingParams(temperature=0.0), max_tokens=3)
    outs = engine.step()
    assert len(outs) == 1 and outs[0].seq_id == s
    seq = engine.seqs[s]
    assert seq.first_token_at is not None and seq.last_token_at is not None
    assert seq.first_token_at >= seq.created_at


def test_stop_reason_eos(engine, tiny_model):
    # 让 eos 等于该序列第一步必选的 greedy token → 一步即停,reason=stop
    with torch.inference_mode():
        probe = engine.submit(_ids(5, 0), SamplingParams(temperature=0.0), max_tokens=1)
        engine.step()
        t0 = engine.seqs[probe].generated_ids[0]
    s = engine.submit(_ids(5, 0), SamplingParams(temperature=0.0), max_tokens=8, eos_ids=frozenset({t0}))
    outs = engine.step()
    assert outs[-1].finished and engine.seqs[s].stop_reason == "stop"


def test_preemption_under_block_pressure(tiny_model):
    """3 块小池(page=4 → 单序列上限 12 token):两条互相挤兑 → 必然抢占;
    输出仍与无压力运行逐位一致(recompute 的 greedy 决定论)。
    参数推算:prompt 3 + max_tokens 8 = 11 token ≤ 12,必须可完成,防死循环。"""
    eng = Engine(tiny_model, tokenizer=None, num_blocks=3, page_size=4,
                 max_running=4, watermark_blocks=0)
    p1, p2 = _ids(3, 0), _ids(3, 1)
    with torch.inference_mode():
        ref1, _ = generate_cached(tiny_model, p1, max_new_tokens=8)
        ref2, _ = generate_cached(tiny_model, p2, max_new_tokens=8)
    eng.submit(p1, SamplingParams(temperature=0.0), max_tokens=8)
    eng.submit(p2, SamplingParams(temperature=0.0), max_tokens=8)
    preempted = False
    while eng.has_pending():
        before = {s.seq_id: s.num_pages for s in eng.scheduler.running}
        eng.step()
        for s in eng.scheduler.waiting:
            if s.num_pages == 0 and s.seq_id in before and before[s.seq_id] > 0:
                preempted = True
    assert preempted, "场景未触发抢占,压力设置失效"
    assert all(s.is_finished for s in eng.seqs.values())
    assert [s.all_ids for s in eng.seqs.values()] == [ref1, ref2] or \
           [s.all_ids for s in eng.seqs.values()] == [ref2, ref1]
    eng.assert_balanced()
```

- [ ] **Step 3: 跑红**

```bash
.venv/bin/pytest tests/test_engine.py -v
```

Expected: FAIL(`ModuleNotFoundError: acorninfer.engine.engine`)。

- [ ] **Step 4: 实现 Engine**

`acorninfer/engine/engine.py`:

```python
"""Engine:请求生命周期 + step 循环。单线程契约(同一时刻一个 step 调用方)。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import QWEN3_EOS_IDS, EngineConfig
from acorninfer.engine.executor import Executor
from acorninfer.engine.scheduler import Scheduler
from acorninfer.engine.sequence import RequestStatus, Sequence
from acorninfer.models.qwen3 import Qwen3ForCausalLM, from_pretrained
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer


@dataclass
class TokenOutput:
    seq_id: int
    token_id: int
    finished: bool
    stop_reason: str | None = None


class Engine:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        tokenizer: Tokenizer | None,
        *,
        num_blocks: int = 1024,
        page_size: int = 16,
        max_running: int = 8,
        watermark_blocks: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        cfg = model.cfg
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.allocator = BlockAllocator(num_blocks)
        self.cache = PagedKVCache(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            page_size,
            self.allocator,
            device,
            dtype,
        )
        self.scheduler = Scheduler(self.allocator, page_size, max_running, watermark_blocks)
        self.executor = Executor(model)
        self.page_size = page_size
        self.seqs: dict[int, Sequence] = {}
        self._views: dict[int, PagedCacheView] = {}
        self._next_id = 0

    @classmethod
    def from_config(cls, cfg: EngineConfig, tokenizer: Tokenizer) -> "Engine":
        model = from_pretrained(cfg.model_dir, cfg.device, cfg.dtype)
        return cls(
            model,
            tokenizer,
            num_blocks=cfg.num_blocks,
            page_size=cfg.page_size,
            max_running=cfg.max_running,
            watermark_blocks=cfg.watermark_blocks,
        )

    def submit(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        *,
        max_tokens: int,
        eos_ids: frozenset[int] | None = None,
    ) -> int:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if -(-len(prompt_ids) // self.page_size) > self.allocator.total:
            raise ValueError("prompt longer than total KV capacity; reject early")
        seq = Sequence(
            seq_id=self._next_id,
            prompt_ids=list(prompt_ids),
            sampling=sampling,
            max_tokens=max_tokens,
            eos_ids=frozenset(eos_ids) if eos_ids is not None else QWEN3_EOS_IDS,
        )
        self._next_id += 1
        self.seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    @torch.inference_mode()
    def step(self) -> list[TokenOutput]:
        prefills, decodes = self.scheduler.schedule()
        outputs: list[TokenOutput] = []
        for seq in prefills:
            view = PagedCacheView(self.cache, seq.block_table)  # 共享块表
            self._views[seq.seq_id] = view
            token = self.executor.prefill(seq, view)
            seq.num_pages = len(view.table)
            self._record(seq, token, outputs)
        if decodes:
            views = [self._views[s.seq_id] for s in decodes]
            tokens = self.executor.decode_batch(decodes, views)
            for seq, token in zip(decodes, tokens):
                seq.num_pages = len(self._views[seq.seq_id].table)
                self._record(seq, token, outputs)
        # 回收:释放已完成序列的块
        for seq in [s for s in self.scheduler.running if s.is_finished]:
            self.scheduler.running.remove(seq)
            seq.status = RequestStatus.FINISHED
            self._views.pop(seq.seq_id)  # 块表与 view 共享,free 由块表驱动
            for b in seq.block_table:
                self.allocator.free(b)
            seq.block_table.clear()
            seq.num_pages = 0
        return outputs

    def _record(self, seq: Sequence, token_id: int, outputs: list[TokenOutput]) -> None:
        seq.generated_ids.append(token_id)
        now = time.perf_counter()
        if seq.first_token_at is None:
            seq.first_token_at = now
        seq.last_token_at = now
        finished, reason = False, None
        if token_id in seq.eos_ids:
            finished, reason = True, "stop"
        elif len(seq.generated_ids) >= seq.max_tokens:
            finished, reason = True, "length"
        if finished:
            seq.stop_reason = reason
            seq.status = RequestStatus.FINISHED
        outputs.append(TokenOutput(seq.seq_id, token_id, finished, seq.stop_reason))

    def has_pending(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def assert_balanced(self) -> None:
        self.allocator.assert_balanced()
```

注意:抢占发生时 `Scheduler._release` 已把 `seq.block_table` 清空并 free——此时 `self._views[seq_id]` 还挂着旧 view(其 `table` 就是同一个已清空的 list),无害;重准入时 `PagedCacheView(self.cache, seq.block_table)` 会重新绑定同一 list。但旧 view 对象需要从 `self._views` 移除/覆盖:`prefill` 循环里的 `self._views[seq.seq_id] = view` 覆盖即可,无需删除。实现时按此写,不要额外清理逻辑。

- [ ] **Step 5: 跑绿**

```bash
.venv/bin/pytest tests/test_engine.py -v   # 5 passed
.venv/bin/pytest -q                         # 无回归
```

诊断纪律:`test_two_concurrent_greedy_match_individual` 若不等,先看首个分歧 step 两条路径各自的 argmax 与 top-2 margin(参照 M0 平局纪律);`test_preemption_under_block_pressure` 若未触发抢占,调小 `num_blocks` 或加长 `max_tokens`,不许放松 `preempted` 断言。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: Engine step 循环 — 连续提交/抢占 recompute/泄漏守卫/指标时间戳"
```

---

### Task 9: OpenAI 兼容 API(serving/)

支持 `/v1/completions` 与 `/v1/chat/completions`,流式(SSE)+ 非流式;`temperature=0` 即 greedy(OpenAI 语义);请求级校验失败返回 4xx,引擎不受影响。chat 模板硬编码 Qwen3 ChatML(不用 jinja,注明 naive)。

**Files:**
- Create: `acorninfer/serving/__init__.py`(空)、`acorninfer/serving/protocol.py`、`acorninfer/serving/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: 实现 protocol**

`acorninfer/serving/protocol.py`:

```python
"""OpenAI 兼容协议模型(我们支持的子集)。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = "*"
    prompt: str
    max_tokens: int = Field(default=16, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(CompletionRequest):
    messages: list[ChatMessage]


def render_chat(messages: list[ChatMessage]) -> str:
    """Qwen3 ChatML 硬编码(naive:不做模板注入防护,roles 不校验)。"""
    parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)
```

- [ ] **Step 2: 实现 server**

`acorninfer/serving/server.py`:

```python
"""OpenAI 兼容 server:FastAPI + SSE;engine 在后台 worker 里逐步推进。"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from tokenizers import DecodeStream

from acorninfer.engine.engine import Engine
from acorninfer.sampler import SamplingParams
from acorninfer.serving.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    render_chat,
)


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="AcornInfer")
    app.state.engine = engine
    app.state.queues: dict[int, asyncio.Queue] = {}
    app.state.worker: asyncio.Task | None = None

    async def _worker() -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not engine.has_pending():
                await asyncio.sleep(0.01)
                continue
            try:
                outputs = await asyncio.to_thread(engine.step)
            except Exception as e:  # 引擎级故障:通知所有等待者后退出
                for q in app.state.queues.values():
                    loop.call_soon_threadsafe(q.put_nowait, e)
                return
            for out in outputs:
                q = app.state.queues.get(out.seq_id)
                if q is not None:
                    loop.call_soon_threadsafe(q.put_nowait, out)

    def _ensure_worker() -> None:
        if app.state.worker is None or app.state.worker.done():
            app.state.worker = asyncio.create_task(_worker())

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        if app.state.worker is not None:
            app.state.worker.cancel()

    async def _wait_finished(seq_id: int):
        """产出 (output, None) 直到 finished;引擎故障时产出 (None, exc)。"""
        q = app.state.queues[seq_id]
        while True:
            item = await q.get()
            if isinstance(item, Exception):
                raise HTTPException(status_code=500, detail=f"engine failure: {item}")
            yield item
            if item.finished:
                return

    def _submit(prompt_ids: list[int], req: CompletionRequest) -> int:
        return app.state.engine.submit(
            prompt_ids,
            SamplingParams(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k),
            max_tokens=req.max_tokens,
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        _ensure_worker()
        try:
            prompt_ids = app.state.engine.tokenizer.encode(req.prompt)
            seq_id = _submit(prompt_ids, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        app.state.queues[seq_id] = asyncio.Queue()
        try:
            if req.stream:
                return StreamingResponse(
                    _stream_sse(app, seq_id, req.model, "text_completion", "text"),
                    media_type="text/event-stream",
                )
            detok = DecodeStream(skip_special_tokens=True)
            chunks: list[str] = []
            reason = None
            async for out in _wait_finished(seq_id):
                chunks.append(detok.step(out.token_id) or "")
                if out.finished:
                    reason = out.stop_reason
            return {
                "id": f"cmpl-{seq_id}",
                "object": "text_completion",
                "model": req.model,
                "choices": [{"text": "".join(chunks), "finish_reason": reason, "index": 0}],
            }
        finally:
            app.state.queues.pop(seq_id, None)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        _ensure_worker()
        try:
            prompt_ids = app.state.engine.tokenizer.encode(render_chat(req.messages))
            seq_id = _submit(prompt_ids, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        app.state.queues[seq_id] = asyncio.Queue()
        try:
            if req.stream:
                return StreamingResponse(
                    _stream_sse(app, seq_id, req.model, "chat.completion.chunk", "content"),
                    media_type="text/event-stream",
                )
            detok = DecodeStream(skip_special_tokens=True)
            chunks: list[str] = []
            reason = None
            async for out in _wait_finished(seq_id):
                chunks.append(detok.step(out.token_id) or "")
                if out.finished:
                    reason = out.stop_reason
            return {
                "id": f"chatcmpl-{seq_id}",
                "object": "chat.completion",
                "model": req.model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "".join(chunks)},
                        "finish_reason": reason,
                        "index": 0,
                    }
                ],
            }
        finally:
            app.state.queues.pop(seq_id, None)

    async def _stream_sse(app: FastAPI, seq_id: int, model: str, obj: str, field: str):
        detok = DecodeStream(skip_special_tokens=True)
        try:
            async for out in _wait_finished(seq_id):
                delta = detok.step(out.token_id) or ""
                chunk = {
                    "id": f"{obj.split('.')[0]}-{seq_id}",
                    "object": obj,
                    "model": model,
                    "choices": [
                        {field: delta, "finish_reason": out.stop_reason if out.finished else None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return app


def main() -> None:
    import argparse

    from acorninfer.config import EngineConfig
    from acorninfer.tokenizer import Tokenizer

    parser = argparse.ArgumentParser(prog="acorninfer-server")
    parser.add_argument("--model-dir", default="/home/ernest/model-weight/Qwen3-8B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-blocks", type=int, default=1024)
    args = parser.parse_args()

    cfg = EngineConfig(model_dir=args.model_dir, num_blocks=args.num_blocks)
    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    engine = Engine.from_config(cfg, tok)
    import uvicorn

    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 写测试** `tests/test_server.py`(tiny 模型 CPU,不占 GPU):

```python
"""server 合同测试(tiny 引擎):非流式/流式/chat/校验 4xx/请求隔离。"""

import json

import pytest
import torch
from fastapi.testclient import TestClient

from acorninfer.cache.pager import BlockAllocator, PagedKVCache
from acorninfer.engine.engine import Engine
from acorninfer.serving.server import create_app


@pytest.fixture(scope="module")
def client():
    torch.manual_seed(0)
    model = ...  # 与 conftest tiny_model 相同构造,但 module 级只建一次:
    from acorninfer.config import Qwen3Config
    from acorninfer.models.qwen3 import Qwen3ForCausalLM
    from acorninfer.tokenizer import Tokenizer

    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to(dtype=torch.bfloat16)
    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    engine = Engine(model, tok, num_blocks=64, page_size=4, max_running=4, watermark_blocks=0)
    return TestClient(create_app(engine))
```

注意:上面 fixture 里的 `QWEN3_8B` 从 `tests.conftest` 拿不到(模块级常量可以 import:`from tests.conftest import QWEN3_8B` 在 rootdir 下可行但脆)——直接在文件顶部写死 `QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"`。tiny 模型只有 256 词表,真实 BPE tokenizer 编出的 id 会越界(embedding 越界崩)!**所以 server 测试的 tokenizer 必须用一个截断包装**:在 `client` fixture 里包一层:

```python
    class TinyTok:
        def encode(self, text):
            ids = tok.encode(text)
            return [i % 256 for i in ids] or [1]
        def decode(self, ids):
            return tok.decode([i % 256 for i in ids])

    engine = Engine(model, TinyTok(), ...)
```

(greedy 确定性不受取模影响;decode 只是测试可读性。)

测试正文:

```python
def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_completion_non_stream(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert 0 < len(body["choices"][0]["text"])  # tiny 词表取模后可能多为空格,仅验证非空列表路径


def test_completion_stream_sse(client):
    with client.stream("POST", "/v1/completions", json={"prompt": "hi", "max_tokens": 3, "stream": True}) as r:
        assert r.status_code == 200
        events = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(e[len("data: "):]) for e in events[:-1]]
    assert all("choices" in c for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] is not None


def test_chat_completions(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["role"] == "assistant"


def test_validation_400_empty_prompt(client):
    r = client.post("/v1/completions", json={"prompt": "", "max_tokens": 4})
    assert r.status_code == 400


def test_validation_400_bad_params(client):
    r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0})
    assert r.status_code == 422  # pydantic ge=1
    r = client.post("/v1/completions", json={"prompt": "x", "temperature": -1})
    assert r.status_code == 422


def test_request_isolation_after_bad_request(client):
    assert client.post("/v1/completions", json={"prompt": "", "max_tokens": 4}).status_code == 400
    r = client.post("/v1/completions", json={"prompt": "still alive", "max_tokens": 2})
    assert r.status_code == 200
```

- [ ] **Step 4: 跑绿**

```bash
.venv/bin/pytest tests/test_server.py -v   # 7 passed
.venv/bin/pytest -q                         # 无回归
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: OpenAI 兼容 API — completions/chat,SSE 流式,4xx 请求隔离"
```

---

### Task 10: bench CLI + M2 验收门 + 文档 + tag

**Files:**
- Create: `acorninfer/bench/__init__.py`(空)、`acorninfer/bench/cli.py`
- Create: `tests/test_golden_m2.py`
- Modify: `docs/benchmarks.md`、`README.md`

- [ ] **Step 1: 实现 bench CLI**

`acorninfer/bench/cli.py`:

```python
"""吞吐 / TTFT / TPOT 压测(直连 Engine,不经 HTTP,测的是引擎本体)。"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from acorninfer.config import EngineConfig
from acorninfer.engine.engine import Engine
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"
SEED_TEXT = "The quick brown fox jumps over the lazy dog. " * 64


def main() -> None:
    parser = argparse.ArgumentParser(prog="acorninfer-bench")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-blocks", type=int, default=1024)
    args = parser.parse_args()

    tok = Tokenizer(f"{args.model_dir}/tokenizer.json")
    base = tok.encode(SEED_TEXT)
    while len(base) < args.input_tokens:
        base = base + base
    prompts = [base[i : i + args.input_tokens] for i in range(args.num_requests)]

    cfg = EngineConfig(model_dir=args.model_dir, num_blocks=args.num_blocks)
    engine = Engine.from_config(cfg, tok)

    submitted, completed = 0, 0
    inflight: dict[int, int] = {}

    def submit_next() -> None:
        nonlocal submitted
        if submitted < args.num_requests:
            sid = engine.submit(
                prompts[submitted],
                SamplingParams(temperature=args.temperature),
                max_tokens=args.output_tokens,
            )
            inflight[sid] = submitted
            submitted += 1

    t0 = time.perf_counter()
    for _ in range(min(args.concurrency, args.num_requests)):
        submit_next()
    while inflight:
        for out in engine.step():
            if out.finished:
                inflight.pop(out.seq_id)
                completed += 1
                submit_next()
    wall = time.perf_counter() - t0

    gen_tokens = sum(len(s.generated_ids) for s in engine.seqs.values())
    ttfts = [(s.first_token_at - s.created_at) for s in engine.seqs.values()]
    tpots = [
        (s.last_token_at - s.first_token_at) / (len(s.generated_ids) - 1)
        for s in engine.seqs.values()
        if len(s.generated_ids) > 1
    ]
    e2es = [(s.last_token_at - s.created_at) for s in engine.seqs.values()]

    def ms(xs):
        return f"avg {statistics.mean(xs)*1000:.0f}ms p50 {statistics.median(xs)*1000:.0f}ms"

    print(f"requests={completed} concurrency={args.concurrency} "
          f"in={args.input_tokens} out={args.output_tokens}")
    print(f"wall={wall:.2f}s throughput={gen_tokens / wall:.2f} tok/s (output only)")
    print(f"TTFT {ms(ttfts)}")
    print(f"TPOT {ms(tpots)}")
    print(f"E2E  {ms(e2es)}")
    engine.assert_balanced()
    engine.model.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 bench 拿实测数字(写进 benchmarks.md,不许编造)**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench.cli --num-requests 16 --concurrency 1 --input-tokens 128 --output-tokens 32
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench.cli --num-requests 16 --concurrency 4 --input-tokens 128 --output-tokens 32
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench.cli --num-requests 16 --concurrency 8 --input-tokens 128 --output-tokens 32
```

记录三条的 throughput / TTFT / TPOT(concurrency=1 是单流基线,对照 M1 的 36.32 tok/s)。

- [ ] **Step 3: 写 M2 验收门** `tests/test_golden_m2.py`:

```python
"""M2 验收(spec §4):并发压测输出正确 + KV 无泄漏。

- 并发 greedy 与逐请求参考(generate_cached,M1 路径)逐 id 一致;
  若分歧:打印首个分歧步两侧 argmax 与 top-2 margin —— 真值平局(margin<0.2)
  按 M0 先例换 prompt 并记录;系统性错位 = BLOCKED。
- 混合负载 fuzz 后 allocator 必须零 live 块(spec §5 泄漏守卫)。

单卡顺序协议:本文件 module 级只加载一次引擎(16GB);CUDA_VISIBLE_DEVICES=1。
"""

import pytest
import torch

from acorninfer.config import EngineConfig
from acorninfer.engine.engine import Engine
from acorninfer.sampler import SamplingParams
from acorninfer.generate import generate_cached
from acorninfer.tokenizer import Tokenizer

QWEN3_8B = "/home/ernest/model-weight/Qwen3-8B"
PROMPTS = [
    "The capital of France is",
    "The first three prime numbers are",
    "Water boils at",
    "The largest planet in the solar system is",
]


@pytest.fixture(scope="module")
def engine():
    tok = Tokenizer(f"{QWEN3_8B}/tokenizer.json")
    eng = Engine.from_config(
        EngineConfig(model_dir=QWEN3_8B, num_blocks=512, max_running=4, watermark_blocks=2),
        tok,
    )
    yield eng
    eng.assert_balanced()
    eng.model.to("cpu")
    torch.cuda.empty_cache()


@pytest.mark.slow
def test_concurrent_greedy_matches_sequential(engine):
    tok = engine.tokenizer
    refs = {}
    for p in PROMPTS:
        ids = tok.encode(p)
        out, _ = generate_cached(engine.model, ids, max_new_tokens=32)
        refs[p] = out
    sids = [
        engine.submit(tok.encode(p), SamplingParams(temperature=0.0), max_tokens=32)
        for p in PROMPTS
    ]
    while engine.has_pending():
        engine.step()
    for sid, p in zip(sids, PROMPTS):
        got = engine.seqs[sid].all_ids
        if got != refs[p]:
            i = next(k for k, (a, b) in enumerate(zip(got, refs[p])) if a != b)
            raise AssertionError(
                f"divergence at step {i} for {p!r}; concurrent={got[i:i+3]} ref={refs[p][i:i+3]}"
            )
        assert got == refs[p]


@pytest.mark.slow
def test_no_leak_under_mixed_fuzz(engine):
    tok = engine.tokenizer
    import random

    rng = random.Random(20261027)
    pool = PROMPTS * 5
    for i, p in enumerate(pool):
        temp = 0.0 if i % 2 == 0 else 0.6
        engine.submit(
            tok.encode(p),
            SamplingParams(temperature=temp, top_p=0.95, top_k=20),
            max_tokens=rng.randint(4, 24),
        )
    finished = 0
    while engine.has_pending():
        for out in engine.step():
            if out.finished:
                finished += 1
    assert finished == len(pool)
    for s in engine.seqs.values():
        assert s.stop_reason in ("stop", "length")
        assert len(s.generated_ids) >= 1
    engine.assert_balanced()
```

- [ ] **Step 4: 跑验收 + 全套件**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest tests/test_golden_m2.py -v -m slow
CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest -q -m slow        # 全部 slow
.venv/bin/pytest -q                                        # 全部 not-slow
```

Expected: 全绿(首次跑 M2 门预计 4-8 分钟,含 16GB 加载;总收集数以实际为准,全绿即可)。

- [ ] **Step 5: 更新 docs/benchmarks.md**

表格追加(数字用 Step 2 实测):

```markdown
| M2 | Qwen3-8B bf16,greedy,in=128/out=32,分页 KV + 连续批处理 | 并发1: <t1> tok/s;并发4: <t4> tok/s;并发8: <t8> tok/s | TTFT/TPOT 见 bench 输出;M1 单流 36.32 tok/s 为对照 |
```

测量协议段追加:

```markdown
M2 协议(`python -m acorninfer.bench.cli`):in=128/out=32/greedy,burst 准入,并发 1/4/8 三档;TTFT=提交→首 token,TPOT=(末 token-首 token)/(n-1)。
```

- [ ] **Step 6: 更新 README(M2 用法段)**

```markdown
## M2 用法(并发引擎 + OpenAI API)

```bash
# OpenAI 兼容服务(GPU 1)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.serving.server --port 8001
curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "你好"}], "max_tokens": 64, "temperature": 0.6
}'

# 压测:吞吐 / TTFT / TPOT
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench.cli --concurrency 4
```
```

- [ ] **Step 7: 全套件最终回归 + Commit + tag**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/pytest -q -m slow
.venv/bin/pytest -q
git add -A && git commit -m "feat: bench CLI + M2 验收门 + 文档 — 分页引擎完成"
git tag m2-acceptance
```

---

## 自审记录(controller 写完计划后过一遍)

1. **Spec 覆盖**:分页 KV(块表 page=16)→ Task 3/4/5;continuous batching → Task 6/7/8;OpenAI API → Task 9;TTFT/TPOT → Task 10 bench;"KV 不足先排队、满员抢占最老 recompute"(§5)→ Task 7 调度器;泄漏 debug 守卫(§5)→ Task 3 `assert_balanced` + Task 8/10 集成;并发压测输出正确(§4 验收)→ Task 10 `test_concurrent_greedy_matches_sequential`。✓ 无缺口。
2. **推演修正(已直接改进正文,落地者无需再查)**:
   - Task 4:`rows` 索引构造补 `dtype=torch.long`(空 list 默认 float,作索引会炸);删掉占位版测试,只留真实断言版;
   - Task 5:fixture 清理为纯 return,测试签名统一收 `env`;
   - Task 7(最重):原水位测试的准入数学算错(8 块水位 3 时 s2 实际能准入),重算后改 max_running=4/watermark=3 分两段断言;原抢占测试 `range(10)` 使 num_tokens=13(注释却说 4),重写为 `gen=1`;发现"抢占后同一步被重新准入"的空转问题 → 调度器加 `preempted_ids` 防抖,测试同步断言 `prefills == []`;
   - Task 8:原抢占压力参数(3+14=17 token > 3 块容量)会**死循环**,修正为 3+8=11 ≤ 12 并在 docstring 写明推算;fixture 去掉与 Engine 内部重复的 alloc/cache 构造;teardown 排空后再验平衡(防中间态误报);`Engine.step` 加 `@torch.inference_mode()`;
   - Task 9:SamplingParams import 提到模块顶层。
3. **类型一致性**:`PagedCacheView(cache, table)` 构造签名在 Task 4/5/8 一致 ✓;`Engine.submit(prompt_ids, sampling, *, max_tokens, eos_ids=None)` 与 server/bench/tests 调用一致 ✓;`forward(input_ids, cache, num_logits_to_keep)` 与 executor 位置参数顺序一致 ✓;`Sequence.num_pages` 由 engine 同步、scheduler 只读 ✓;`block_table` 共享 list 的三方(Sequence/View/Scheduler._release)语义在 Task 4/7/8 各自注明 ✓。
4. **已知风险,落地时注意**:
   - 批量 decode 与逐序列 decode 的 token 序列一致性依赖 argmax 对 ulp 级 logits 差不敏感(CPU tiny 模型成立;真实模型若在 M2 门分歧,按门内 docstring 的 margin 纪律处理);
   - FastAPI `on_event("shutdown")` 是 deprecated API,当前版本可用,不引入 lifespan 重构(YAGNI);
   - server 测试的 TinyTok 取模包装是为 256 词表 tiny 模型做的适配,真实模型路径(Task 10 门)不用它;
   - bench 的 TTFT/TPOT 依赖 `time.perf_counter` 差值,受 CPU 调度抖动影响,数字看量级不看个位。
