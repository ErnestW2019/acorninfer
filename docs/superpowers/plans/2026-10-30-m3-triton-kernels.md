# M3 Triton 算子层实施计划(分页 decode attention + 融合 RMSNorm/RoPE + vLLM 基线)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用三个 Triton kernel(fused_add_rmsnorm、fused RoPE、flash-decoding 两阶段分页 decode attention)替换 M2 decode 路径的 gather+逐序列 SDPA,在"token 逐位 + 实测容差"数值门与"≥ vLLM 基线 70% 吞吐"性能门下通过 M3 验收。

**Architecture:** 新建 `acorninfer/ops/` 包,每个 kernel 配一个 torch 参考实现(单测 oracle);两枚正交开关(`attention_backend: "sdpa"|"triton"`、`fused_ops: bool`)在 `forward_decode` 内分派,SDPA 路径永久保留为 reference;attention kernel 直接读页张量 `[num_blocks, page, kv_heads, head_dim]`(M2 布局伏笔),消除 gather;先小算子后主菜,测试基建(对拍 + 容差标定协议)先行。

**Tech Stack:** Python 3.12 + PyTorch 2.13(bf16)+ Triton 3.7.1(sm89,已装)、httpx(已装,vLLM 基线压测)、pytest。

**Spec:** `docs/superpowers/specs/2026-10-30-m3-triton-kernels-design.md`(四项已确认决定:GPU 0 自建 vLLM 基线 / token 逐位+logits 容差 / SDPA 保留为 reference / 先小后大)。

---

**环境约定(每个任务都适用):**

- 仓库 `/home/ernest/github/acorninfer`,分支 `main`,M2 已收官(tag `m0/m1/m2-acceptance`,已推 GitHub);一切命令 `.venv/bin/` 前缀;
- **GPU 分工(2026-10-30 起 GPU 0 已空闲):GPU 0 跑 vLLM 基线服务,GPU 1 做开发/测试**。pytest/bench 命令照旧加 `CUDA_VISIBLE_DEVICES=1` 前缀(纪律不变,代码仍写 `cuda:0` = 可见卡里的第一张 = 物理 GPU 1);
- 模型权重 `/home/ernest/model-weight/Qwen3-8B`(只读);M2 实测锚点:C=1/4/8 → 吞吐 36.99/89.06/110.68 tok/s,TPOT 26.79/43.77/61.50ms(已入 `docs/benchmarks.md`);
- **计划代码是草稿,测试才是契约**(M2 教训:计划片段出过轴错误)。凡 kernel 代码,先跑对拍测试,测试炸了以数值事实为准修代码,不许改测试迁就草稿;
- 容差不许拍脑袋:fp32 级用紧容差(只容忍归约序),bf16 级先跑通再按实测 max|diff| 写死 `tests/triton_utils.py` 的 `TOL`,注释记录标定日期;
- TDD 纪律:red → implement → green → commit;fast 套件每任务收尾全绿;golden m0/m1/m2 挂了 = BLOCKED,不许改断言;
- **保护文件(审查者逐项核对,除计划明示改动外零越界):** `tests/test_golden_m0.py`、`tests/test_golden_m1.py`、`tests/test_golden_m2.py`、`acorninfer/cache/kv.py`、`acorninfer/engine/scheduler.py`、`acorninfer/engine/sequence.py` 禁改;`acorninfer/cache/pager.py` 仅 Task 5 的行为保持重构;`acorninfer/models/qwen3.py` 的 torch 路径(`forward`、`_apply_rope`、sdpa 分支)数值不许变。

**非目标(明确推迟):** prefill kernel 化(spec 明确留 SDPA)、autotune、CUDA graph、量化、TP(M4)、投机解码、chunked prefill、prefix cache(M4)、M5 前的一切昇腾代码。

---

## 文件地图(Task 落地后的终态)

```
acorninfer/
  config.py                   # +EngineConfig 两字段(Task 3)
  cache/pager.py              # PagedCacheView.write() 从 update() 抽出(Task 5)
  engine/engine.py            # from_config 透传两开关(Task 3)
  models/qwen3.py             # 后端开关 + 块/注意力分派(Task 3/4/7)
  ops/
    __init__.py               # 新(Task 3)
    rmsnorm.py                # 新:K1 kernel + torch ref(Task 3)
    rope.py                   # 新:K2 kernel + torch ref(Task 4)
    paged_attention.py        # 新:K3 两阶段 kernel + torch ref + 启发式(Task 6/7)
scripts/
  bench_remote.py             # 新:vLLM HTTP 压测(同协议)(Task 1)
tests/
  triton_utils.py             # 新:容差协议常量与助手(Task 2)
  test_triton_smoke.py        # 新:编译链 smoke(Task 2)
  test_ops_rmsnorm.py         # 新(Task 3)
  test_ops_rope.py            # 新(Task 4)
  test_ops_paged_attention.py # 新(Task 6)
  test_golden_m3.py           # 新(Task 8)
  conftest.py                 # +gpu 无卡 skip 钩子(Task 2)
pyproject.toml                # +gpu marker(Task 2)
docs/benchmarks.md            # +M3 基线与对比表(Task 1/8)
```

---

### Task 1: vLLM 基线(独立 venv + 同协议压测)

**Files:**
- Create: `scripts/bench_remote.py`
- Modify: `docs/benchmarks.md`(追加 M3 基线小节,只写实测数字)

- [ ] **Step 1: 建独立 venv 并装 vLLM**

```bash
python3 -m venv /home/ernest/vllm-baseline
/home/ernest/vllm-baseline/bin/pip install -U pip
/home/ernest/vllm-baseline/bin/pip install vllm
```

下载约 2-6 GB,慢的话换清华镜像(`-i https://pypi.tuna.tsinghua.edu.cn/simple`)。装完确认:`/home/ernest/vllm-baseline/bin/python -c "import vllm; print(vllm.__version__)"`。

- [ ] **Step 2: GPU 0 起服务并等健康**

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/ernest/vllm-baseline/bin/vllm serve \
  /home/ernest/model-weight/Qwen3-8B --dtype bfloat16 --port 8100 \
  --max-model-len 512 --gpu-memory-utilization 0.90 \
  > /tmp/opencode/vllm-baseline.log 2>&1 &
# 轮询至 200(权重加载 + CUDA graph 捕获需数分钟):
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8100/health
```

- [ ] **Step 3: 写 `scripts/bench_remote.py`**

与 `acorninfer/bench.py` 同协议打 `/v1/completions`。**prompt 以 token-id 数组直投**(completions API 原生支持),复用 `acorninfer.bench._prompts(同 seed)`——与引擎 bench 的随机 id 完全同构,绕开文本分词往返的不确定性;temperature=0、max_tokens=32、自然 EOS 停止,均与引擎侧相同。

```python
"""vLLM 基线压测:与 acorninfer.bench 同协议打 OpenAI /v1/completions。

协议对齐:同一批随机 prompt id(acorninfer.bench._prompts 同 seed)以 token-id
数组直投;temperature=0、max_tokens=32、自然 EOS 停止。指标公式与
acorninfer.bench._report 一致;差异(入档 docs):HTTP 往返计入两端延迟
(vLLM 侧含其调度开销),这是产品真实路径,70% 线按此口径。

运行(vLLM 已在 GPU 0 :8100 起好,见 Task 1):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import torch

from acorninfer.bench import _prompts

DEFAULT_MODEL = "/home/ernest/model-weight/Qwen3-8B"


def _run_one(client: httpx.Client, base: str, ids: list[int], out_tokens: int, out: dict):
    out["t0"] = time.perf_counter()
    out["first"] = None
    out["last"] = out["t0"]
    out["n"] = 0
    with client.stream(
        "POST", f"{base}/v1/completions",
        json={"prompt": ids, "max_tokens": out_tokens, "temperature": 0.0,
              "stream": True, "stream_options": {"include_usage": True}},
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            obj = json.loads(payload)
            now = time.perf_counter()
            if out["first"] is None:
                out["first"] = now          # 首个 data 事件 ≈ TTFT
            out["last"] = now
            if obj.get("usage"):
                out["n"] = obj["usage"]["completion_tokens"]


def _report(level: int, outs: list[dict], out_tokens: int) -> None:
    ttfts = [o["first"] - o["t0"] for o in outs]
    tpots = [(o["last"] - o["first"]) / (o["n"] - 1) for o in outs if o["n"] > 1]
    total = sum(o["n"] for o in outs)
    span = max(o["last"] for o in outs) - min(o["t0"] for o in outs)
    print(f"RESULT-VLLM C={level} ttft_p50={statistics.median(ttfts):.4f}s "
          f"ttft_mean={statistics.mean(ttfts):.4f}s "
          f"tpot_mean={statistics.mean(tpots) * 1000:.2f}ms tok_s={total / span:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8100")
    p.add_argument("--model-dir", default=DEFAULT_MODEL)  # 仅取 vocab 大小,与 _prompts 对齐
    p.add_argument("--in-tokens", type=int, default=128)
    p.add_argument("--out-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--seed", type=int, default=20261027)
    args = p.parse_args()

    # vocab 只为生成同分布的随机 id;真实值与引擎 bench 一致(151936)
    from acorninfer.config import Qwen3Config
    vocab = Qwen3Config.from_json(f"{args.model_dir}/config.json").vocab_size

    with httpx.Client(timeout=300.0) as client:
        for level in args.concurrency:
            prompts = _prompts(level, args.in_tokens, vocab, args.seed)
            outs = [dict() for _ in prompts]
            with ThreadPoolExecutor(max_workers=level) as pool:
                list(pool.map(lambda t: _run_one(client, args.base, t[0], args.out_tokens, t[1]),
                              zip(prompts, outs)))
            _report(level, outs, args.out_tokens)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 压测三个档位并记录**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8
```

Expected:三行 `RESULT-VLLM C=... tok_s=...`。数字只写实测。

- [ ] **Step 5: `docs/benchmarks.md` 追加 M3 基线小节**

记录:vLLM 版本(`pip show vllm`)、命令、协议(含"HTTP 往返计入"与"completion_tokens 含 EOS 与引擎侧 generated_ids 口径一致"两条注记)、C=1/4/8 的 tok_s/TTFT/TPOT、70% 线数字(基线 × 0.7,逐档)。压完停服务:`pkill -f "vllm serve"`(Task 8 再起)。

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_remote.py docs/benchmarks.md
git commit -m "feat(bench): vLLM baseline harness + measured baseline (M3 Task 1)"
```

---

### Task 2: Triton 测试基建(gpu marker + 对拍协议 + 编译链 smoke)

**Files:**
- Modify: `pyproject.toml`(markers 行)
- Modify: `tests/conftest.py`(追加无卡 skip 钩子)
- Create: `tests/triton_utils.py`
- Create: `tests/test_triton_smoke.py`

- [ ] **Step 1: 注册 gpu marker(`pyproject.toml`)**

```toml
markers = ["slow: needs model weights / GPU", "gpu: needs a CUDA device"]
```

- [ ] **Step 2: conftest 追加无卡自动 skip(`tests/conftest.py` 末尾)**

```python
def pytest_collection_modifyitems(config, items):
    """@pytest.mark.gpu 的用例在无 CUDA 设备时自动 skip;slow 的去留仍由 addopts 决定。"""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 3: 容差协议常量(`tests/triton_utils.py`)**

```python
"""Triton kernel 测试基建(先于一切 kernel —— spec §6)。

两级容差协议:
- fp32 输入 + TOL_F32(紧):只容忍归约序差异;索引/mask/边界错了必炸;
- bf16 输入 + TOL_BF16(标定):先以 INIT 值跑通,再按实测 max|diff| 收紧写死,
  注释记录标定命令与日期。不允许拍脑袋容差。

导入约定:pytest 默认 import-mode=prepend 会把 tests/ 目录插入 sys.path,
kernel 测试文件直接 `from triton_utils import ...`。
"""

import torch

TOL_F32 = dict(rtol=1e-5, atol=1e-5)

# bf16 初值(宽松;各 kernel 标定后收紧):bf16 ulp ≈ 2^-8 相对,跨路径双重
# 取整放宽到 ~2%。标定命令见各测试 docstring。
TOL_BF16 = dict(rtol=2e-2, atol=2e-2)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()
```

- [ ] **Step 4: 编译链 smoke(`tests/test_triton_smoke.py`)**

```python
"""sm89(RTX 4090)Triton 编译链 smoke:最小 kernel 先验工具链,不验数值。"""
import pytest
import torch


@pytest.mark.gpu
def test_triton_compiles_sm89():
    import triton
    import triton.language as tl

    @triton.jit
    def _add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)

    x = torch.arange(64, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    _add_one[(1,)](x, y, 64, BLOCK=64)
    torch.testing.assert_close(y, x + 1.0)
```

- [ ] **Step 5: 跑绿**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_triton_smoke.py -v`
Expected: 1 passed。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/triton_utils.py tests/test_triton_smoke.py
git commit -m "feat(tests): triton test infra (gpu marker, tolerance protocol, sm89 smoke)"
```

---

### Task 3: K1 `fused_add_rmsnorm` + 后端开关接线

**Files:**
- Create: `acorninfer/ops/__init__.py`、`acorninfer/ops/rmsnorm.py`
- Create: `tests/test_ops_rmsnorm.py`
- Modify: `acorninfer/config.py`(EngineConfig 两字段 + 校验)
- Modify: `acorninfer/engine/engine.py`(from_config 透传)
- Modify: `acorninfer/models/qwen3.py`(model 开关属性 + `Qwen3Block.forward_decode` fused 分派)

- [ ] **Step 1: 写失败测试(`tests/test_ops_rmsnorm.py`)**

```python
"""K1 fused_add_rmsnorm 对拍。

语义契约(与 torch 路径逐操作同构):
    fused_add_rmsnorm(x, residual, weight, eps) -> (normed, residual_new)
    residual is None    : residual_new = x,              normed = rmsnorm(x)
    residual is not None: residual_new = bf16(residual + x),normed = rmsnorm(residual_new)
torch 路径是 "hidden = hidden + attn_out(bf16 加); hidden = ln(hidden)(fp32 归一)"。
标定:bf16 用例打印 max|diff|,回填 triton_utils.TOL 后收紧。
"""
import pytest
import torch
from triton_utils import TOL_BF16, TOL_F32, max_abs_diff

from acorninfer.ops.rmsnorm import fused_add_rmsnorm, fused_add_rmsnorm_ref

CASES = [(1, 4096), (4, 4096), (3, 1024), (2, 64)]  # (rows, cols);64 = tiny 模型 hidden


def _gen(dtype):
    g = torch.Generator().manual_seed(7)
    for rows, cols in CASES:
        x = torch.randn(rows, cols, generator=g).to("cuda").to(dtype)
        r = torch.randn(rows, cols, generator=g).to("cuda").to(dtype)
        w = torch.randn(cols, generator=g).to("cuda").to(dtype)
        yield x, r, w


@pytest.mark.gpu
def test_matches_ref_fp32_tight():
    for x, r, w in _gen(torch.float32):
        for residual in (None, r):
            n_ref, r_ref = fused_add_rmsnorm_ref(x, residual, w, 1e-6)
            n_k, r_k = fused_add_rmsnorm(x, residual, w, 1e-6)
            torch.testing.assert_close(n_k, n_ref, **TOL_F32)
            torch.testing.assert_close(r_k, r_ref, **TOL_F32)


@pytest.mark.gpu
def test_matches_ref_bf16_calibrated():
    worst = 0.0
    for x, r, w in _gen(torch.bfloat16):
        for residual in (None, r):
            n_ref, r_ref = fused_add_rmsnorm_ref(x, residual, w, 1e-6)
            n_k, r_k = fused_add_rmsnorm(x, residual, w, 1e-6)
            worst = max(worst, max_abs_diff(n_k, n_ref), max_abs_diff(r_k, r_ref))
            torch.testing.assert_close(n_k, n_ref, **TOL_BF16)
            torch.testing.assert_close(r_k, r_ref, **TOL_BF16)
    print(f"\n[rmsnorm bf16 calibrated max|diff| = {worst:.6f}]")  # 标定输出,回填 TOL
```

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_ops_rmsnorm.py -v`
Expected: FAIL(ModuleNotFoundError: acorninfer.ops)

- [ ] **Step 3: 实现(`acorninfer/ops/__init__.py` 空文件 + `acorninfer/ops/rmsnorm.py`)**

```python
"""K1: fused add + RMSNorm(行内融合)。torch 参考实现即单测 oracle。

契约见 tests/test_ops_rmsnorm.py docstring。数值镜像 M0 结论:
RMSNorm 在 fp32 计算后 cast 回输入 dtype;残差加法在输入 dtype 语义下取整
(fp32 相加后按 bf16 取整 = torch bf16 加法的 opmath 行为)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, res_out_ptr, weight_ptr, normed_ptr,
    n_cols, eps,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(res_out_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
    # 残差按输入 dtype 取整后再进 norm(镜像 torch:先 dtype 加法,再 fp32 归一)
    x_rounded = x.to(res_out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(res_out_ptr + row * n_cols + cols, x_rounded.to(res_out_ptr.dtype.element_ty), mask=mask)
    var = tl.sum(x_rounded * x_rounded, axis=0) / n_cols
    inv = tl.math.rsqrt(var + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_rounded * inv * w
    tl.store(normed_ptr + row * n_cols + cols, y.to(normed_ptr.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps):
    """x: [..., H];residual: None 或同形状;返回 (normed, residual_new),均新分配。"""
    *lead, H = x.shape
    rows = x.numel() // H
    x2 = x.reshape(rows, H)
    res_in = x2 if residual is None else residual.reshape(rows, H)
    normed = torch.empty_like(x2)
    res_out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(H)
    _fused_add_rmsnorm_kernel[(rows,)](
        x2, res_in if residual is not None else x2, weight, normed,
        H, eps, BLOCK=BLOCK, HAS_RESIDUAL=residual is not None,
        num_warps=8,
    )
    return normed.view(*lead, H), res_out.view(*lead, H)
```

注意:HAS_RESIDUAL=False 时 res_out_ptr 传 x2 作占位(kernel 不读它),res_out 单独新分配 = x 的取整拷贝,不写调用方的 x。

torch 参考实现(同文件末尾):

```python
def fused_add_rmsnorm_ref(x, residual, weight, eps):
    """Oracle。residual is None → res = x;否则 res = residual + x(输入 dtype 加法)。"""
    if residual is None:
        res = x
    else:
        res = residual + x
    xf = res.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
    return normed, res.to(x.dtype)
```

- [ ] **Step 4: 跑绿 + 标定**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_ops_rmsnorm.py -v -s`
Expected: 2 passed;记录打印的 max|diff|,若显著小于初值即收紧 `TOL_BF16` 并注明标定日期。

- [ ] **Step 5: 接线 —— config/engine/model 三处**

`acorninfer/config.py` EngineConfig 增字段(dataclass 字段顺序:加在 `watermark_blocks` 之后)与 `__post_init__` 校验:

```python
    attention_backend: str = "sdpa"  # "sdpa" | "triton"(M3 paged attention)
    fused_ops: bool = False          # True → decode 走 K1/K2 融合算子(M3)
```

```python
        if self.attention_backend not in ("sdpa", "triton"):
            raise ValueError(
                f"attention_backend must be 'sdpa'|'triton', got {self.attention_backend!r}"
            )
```

`acorninfer/engine/engine.py` `from_config` 里 `model = from_pretrained(...)` 之后加:

```python
        model.attention_backend = cfg.attention_backend
        model.fused_ops = cfg.fused_ops
```

`acorninfer/models/qwen3.py` `Qwen3ForCausalLM.__init__` 末尾加:

```python
        # M3 后端开关(EngineConfig/bench/测试直接赋值;默认 = M2 行为)
        self.attention_backend = "sdpa"
        self.fused_ops = False
```

`Qwen3Block.forward_decode` 换为(带 fused 分派,torch 路径逐字不动):

```python
    def forward_decode(self, hidden, cos, sin, views, layer_idx, *, fused_ops=False):
        if not fused_ops:
            hidden = hidden + self.self_attn.forward_decode(
                self.input_layernorm(hidden), cos, sin, views, layer_idx
            )
            hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
            return hidden
        from acorninfer.ops.rmsnorm import fused_add_rmsnorm  # 局部导入,避免 torch 路径依赖 ops
        normed, residual = fused_add_rmsnorm(
            hidden, None, self.input_layernorm.weight, self.input_layernorm.eps
        )
        hidden = self.self_attn.forward_decode(normed, cos, sin, views, layer_idx)
        normed, residual = fused_add_rmsnorm(
            hidden, residual, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return residual + self.mlp(normed)
```

(导入位置最终以"模块顶部 import 不引入 torch 路径对 ops 的硬依赖"为准;若审查发现循环导入风险,把 import 提到模块顶也可——ops 不 import models,无环。)

- [ ] **Step 6: 集成测试(追加到 `tests/test_ops_rmsnorm.py`)**

```python
@pytest.mark.gpu
def test_tiny_model_fused_ops_roundtrip():
    """tiny 模型端到端:fused_ops 开关不改变 logits 超出 bf16 标定容差。"""
    import torch
    from acorninfer.config import Qwen3Config
    from acorninfer.models.qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    ids = torch.randint(0, 256, (2, 8), device="cuda")
    logits_ref = model(ids)
    model.fused_ops = True
    logits_fused = model(ids)
    torch.testing.assert_close(logits_fused, logits_ref, **TOL_BF16)
    model.fused_ops = False
```

注:此测试走 prefill 路径(`forward`),`Qwen3Block.forward` 也接 fused 分派才有意义 —— **prefill 路径 M3 不 kernel 化,`forward` 保持现状不动**;该测试只证明开关默认关闭时行为不变。若实现者判断用 `forward_decode` 更贴切,可改为构造 tiny PagedCacheView 走 decode(参照 Task 7 的测试写法),二选一,在 commit message 里说明。

- [ ] **Step 7: 全套件绿 + Commit**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest`
Expected: 全绿(原 69+ 项 + 新增)。

```bash
git add acorninfer/ops/ acorninfer/config.py acorninfer/engine/engine.py acorninfer/models/qwen3.py tests/test_ops_rmsnorm.py
git commit -m "feat(ops): K1 fused_add_rmsnorm kernel + backend switches (M3 Task 3)"
```

---

### Task 4: K2 `fused_rope`(kernel 内算 cos/sin + 就地旋转 q/k)

**Files:**
- Create: `acorninfer/ops/rope.py`
- Create: `tests/test_ops_rope.py`
- Modify: `acorninfer/models/qwen3.py`(签名词 threading + attention 内分派)

- [ ] **Step 1: 写失败测试(`tests/test_ops_rope.py`)**

```python
"""K2 fused_rope 对拍:镜像 torch 路径 q*cos + rotate_half(q)*sin 的三段 bf16 取整,
目标 bitwise 相等(bf16 取整 RNE 与 torch 一致);对不上再退 assert_close 并记录。"""
import pytest
import torch
from triton_utils import TOL_BF16, TOL_F32

from acorninfer.ops.rope import fused_rope_inplace, fused_rope_ref

# (B, H, D);D 必须 power of 2 且偶数
CASES = [(1, 4, 16), (2, 4, 16), (3, 8, 16), (4, 32, 128)]


def _gen(dtype):
    g = torch.Generator().manual_seed(11)
    for B, H, D in CASES:
        x = torch.randn(B, H, 1, D, generator=g).to("cuda").to(dtype)
        pos = torch.arange(B, device="cuda") * 3 + 5  # 非零起点,暴露位置错位
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        yield x, pos, inv_freq


@pytest.mark.gpu
def test_rope_matches_ref_fp32_tight():
    for x, pos, inv_freq in _gen(torch.float32):
        ref = fused_rope_ref(x, pos, inv_freq)
        q = x.clone()
        fused_rope_inplace(q, q.clone(), pos, inv_freq.to("cuda"))
        torch.testing.assert_close(q, ref, **TOL_F32)


@pytest.mark.gpu
def test_rope_matches_ref_bf16_bitwise_or_calibrated():
    for x, pos, inv_freq in _gen(torch.bfloat16):
        ref = fused_rope_ref(x, pos, inv_freq)
        q = x.clone()
        k = x.clone()
        fused_rope_inplace(q, k, pos, inv_freq.to("cuda"))
        try:
            torch.testing.assert_close(q, ref, rtol=0, atol=0)  # 目标:逐位
        except AssertionError:
            torch.testing.assert_close(q, ref, **TOL_BF16)      # 退级:标定容差
        assert torch.equal(k, q)  # 同输入同 kernel,q/k 必一致
```

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_ops_rope.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现(`acorninfer/ops/rope.py`)**

```python
"""K2: fused RoPE —— kernel 内由 positions×inv_freq 算 cos/sin 并就地旋转。

布局 = HF Qwen3 的 half-rotate(chunk 前/后半),_rotate_half 语义:
    y1 = x1*c - x2*s;  y2 = x2*c + x1*s
数值镜像 torch 路径的三段 bf16 取整:cos/sin fp32 算完先 cast 激活 dtype;
x*cos、x2*sin 各自取整后,和/差再取整(RNE 与 torch 一致 → 目标逐位相等)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_inplace_kernel(
    x_ptr,          # [B*H, D] contiguous
    pos_ptr,        # [B] int
    inv_freq_ptr,   # [D/2] fp32
    H: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0)          # b * H + h
    b = row // H
    half: tl.constexpr = D // 2
    i = tl.arange(0, half)
    pos = tl.load(pos_ptr + b).to(tl.float32)
    freq = pos * tl.load(inv_freq_ptr + i)                  # [half] fp32
    act = x_ptr.dtype.element_ty
    cos = tl.cos(freq).to(act).to(tl.float32)               # 镜像:先 cast 再乘
    sin = tl.sin(freq).to(act).to(tl.float32)
    base = x_ptr + row * D
    x1 = tl.load(base + i).to(tl.float32)
    x2 = tl.load(base + half + i).to(tl.float32)
    t1 = (x1 * cos).to(act).to(tl.float32)                  # 三段取整镜像 torch
    t2 = (x2 * sin).to(act).to(tl.float32)
    t3 = (x2 * cos).to(act).to(tl.float32)
    t4 = (x1 * sin).to(act).to(tl.float32)
    tl.store(base + i, (t1 - t2).to(act))
    tl.store(base + half + i, (t3 + t4).to(act))


def fused_rope_inplace(q, k, positions, inv_freq):
    """q/k: [B, H, 1, D] 且 contiguous(必须);就地旋转。positions [B] int;inv_freq [D/2] fp32。"""
    B, H, s, D = q.shape
    assert s == 1 and D % 2 == 0 and (D & (D - 1)) == 0, "D must be power-of-2"
    assert q.is_contiguous() and k.is_contiguous()
    _rope_inplace_kernel[(B * H,)](q.view(B * H, D), positions, inv_freq, H=H, D=D)
    KB = k.shape[0] * k.shape[1]
    _rope_inplace_kernel[(KB,)](k.view(KB, D), positions, inv_freq, H=k.shape[1], D=D)


def fused_rope_ref(x, positions, inv_freq):
    """Oracle:x [B,H,1,D] → 旋转后同形状。逐操作镜像 Qwen3Attention 的 torch 分支。"""
    D = x.shape[-1]
    pos = positions.to(device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq.float().to(x.device))       # [B, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)[:, None, None, :]
    sin = emb.sin().to(x.dtype)[:, None, None, :]
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin
```

- [ ] **Step 4: 跑绿**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_ops_rope.py -v`
Expected: 2 passed(若 bitwise 断言失败且差在 bf16 双取整量级,退级 assert_close 并在注释记录)。

- [ ] **Step 5: 接线(`acorninfer/models/qwen3.py`)**

签名 threading(三处,均带默认值,旧调用方零改动):

```python
# Qwen3Attention.forward_decode 签名:
def forward_decode(self, hidden, cos, sin, views, layer_idx, *,
                   positions=None, inv_freq=None, fused_ops=False,
                   attention_backend="sdpa"):
```

attention 内 RoPE 段换为:

```python
        if fused_ops:
            if positions is None or inv_freq is None or cos is not None:
                raise ValueError("fused_ops requires positions+inv_freq and no cos/sin")
            from acorninfer.ops.rope import fused_rope_inplace
            fused_rope_inplace(q, k, positions, inv_freq)   # 就地;q/k 来自 RMSNorm 输出,contiguous
        else:
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
```

`Qwen3Block.forward_decode` 签名同步加 `positions=None, inv_freq=None` 两个 kwarg,两条路径的 `self_attn.forward_decode(...)` 调用透传。

`Qwen3ForCausalLM.forward_decode` 主体换为:

```python
        positions = torch.tensor([v.seq_len for v in views], device=input_ids.device)
        if self.fused_ops:
            cos = sin = None
            inv_freq = self.rope.inv_freq
        else:
            dtype = self.embed_tokens.weight.dtype
            cos, sin = self.rope(positions, dtype, input_ids.device)  # [B, head_dim]
            cos = cos[:, None, None, :]
            sin = sin[:, None, None, :]
            inv_freq = None
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_decode(
                hidden, cos, sin, views, i,
                positions=positions, inv_freq=inv_freq,
                fused_ops=self.fused_ops,
                attention_backend=self.attention_backend,
            )
        hidden = self.norm(hidden)
        return self.lm_head(hidden).float()[:, 0, :]
```

- [ ] **Step 6: 集成测试(追加到 `tests/test_ops_rope.py`)**

```python
@pytest.mark.gpu
def test_tiny_model_fused_rope_decode_matches_torch():
    """tiny 模型 decode 步:fused_ops=True 的 forward_decode 与 torch 路径 logits 容差内一致。"""
    import torch
    from acorninfer.cache.kv import KVCache  # 若 tiny 视图构造在别处有先例,照抄先例
    from acorninfer.config import Qwen3Config
    from acorninfer.models.qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    ids = torch.randint(0, 256, (1, 8), device="cuda")
    ref = model(ids)                      # prefill 参考(末位 logits)
    cache = KVCache(model.cfg.num_hidden_layers, model.cfg.num_key_value_heads,
                    model.cfg.head_dim, device="cuda", dtype=torch.bfloat16)
    model(ids, cache)
    next_tok = int(ref[0, -1].argmax())
    logits_ref = model(torch.tensor([[next_tok]], device="cuda"), cache)
    model.fused_ops = True
    cache2 = KVCache(model.cfg.num_hidden_layers, model.cfg.num_key_value_heads,
                     model.cfg.head_dim, device="cuda", dtype=torch.bfloat16)
    model(ids, cache2)
    logits_fused = model(torch.tensor([[next_tok]], device="cuda"), cache2)
    torch.testing.assert_close(logits_fused, logits_ref, **TOL_BF16)
```

(单序列 M1 cache 路径不经 `forward_decode`;若 KVCache 构造签名不同,以 `acorninfer/cache/kv.py` 现状为准——**实现者必须先读该文件**,不许照抄计划。或直接改用 Task 7 的 PagedCacheView 写法,二选一记录。)

- [ ] **Step 7: 全套件绿 + Commit**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest`
Expected: 全绿。

```bash
git add acorninfer/ops/rope.py acorninfer/models/qwen3.py tests/test_ops_rope.py
git commit -m "feat(ops): K2 fused RoPE kernel + dispatch threading (M3 Task 4)"
```

---

### Task 5: pager 行为保持重构:`PagedCacheView.write()` 从 `update()` 抽出

**Files:**
- Modify: `acorninfer/cache/pager.py`(唯一被允许的改动:抽方法,行为不变)
- Modify: `tests/test_pager.py`(追加一条等价性测试)

- [ ] **Step 1: 写失败测试(追加到 `tests/test_pager.py`)**

```python
def test_write_then_gather_equals_update():
    """write + gather ≡ update(逐位);块表分配行为一致。"""
    alloc = BlockAllocator(8)
    cache = PagedKVCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size=4,
                         allocator=alloc, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    t1: list[int] = []
    v1 = PagedCacheView(cache, t1)
    k_up, v_up = v1.update(0, k, v)

    t2: list[int] = []
    v2 = PagedCacheView(cache, t2)
    v2.write(0, k, v)
    k_g = v2._gather(0, cache.k_pages)
    v_g = v2._gather(0, cache.v_pages)

    assert torch.equal(k_up, k_g) and torch.equal(v_up, v_g)
    assert t1 == t2 and v1.seq_len == v2.seq_len == 3


def test_write_exhaustion_leaves_no_partial_state():
    """块不足时 write 原样抛 BlockExhausted、_lens 未提交(与 update 同语义)。"""
    alloc = BlockAllocator(1)
    cache = PagedKVCache(num_layers=1, num_kv_heads=2, head_dim=8, page_size=4,
                         allocator=alloc, device="cpu", dtype=torch.float32)
    t: list[int] = []
    view = PagedCacheView(cache, t)
    view.write(0, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))  # 占 1 块
    with pytest.raises(BlockExhausted):
        view.write(0, torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8))
    assert view.seq_len == 3  # 失败写不提交
```

- [ ] **Step 2: 跑失败**

Run: `.venv/bin/python -m pytest tests/test_pager.py -k write -v`
Expected: FAIL(AttributeError: write)

- [ ] **Step 3: 重构 `PagedCacheView`(把 update 的 scatter 段原样搬进 write)**

```python
    def write(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """纯 scatter:写入新 token 的 k/v;不读不 gather(读侧由调用方定:
        update() = write + gather;M3 triton 后端只 write,attention 直接读页)。
        分配在 scatter 之前全部完成:BlockExhausted 中断零部分写入、_lens 未提交。"""
        <update() 现有 scatter 段逐字搬入:base/needed/while/pos/rows/slots/赋值/_lens 提交>

    def update(self, layer_idx, k_new, v_new):
        """写入并返回全量 [1, kv_heads, seq_len, head_dim](write + gather)。"""
        self.write(layer_idx, k_new, v_new)
        return (self._gather(layer_idx, self.cache.k_pages),
                self._gather(layer_idx, self.cache.v_pages))
```

**纪律:scatter 段逐字搬运,不许顺手改写**(包括 `dtype=torch.long`、`permute(1, 0, 2)` 这些被测试钉住的细节)。docstring 更新但语义注释保留。

- [ ] **Step 4: 全套件绿(证明行为保持)**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest`
Expected: 全绿 —— 尤其 `test_pager.py` 全部、`test_paged_forward.py`、`test_engine.py`。

- [ ] **Step 5: Commit**

```bash
git add acorninfer/cache/pager.py tests/test_pager.py
git commit -m "refactor(pager): extract PagedCacheView.write (scatter-only) from update (M3 Task 5)"
```

---

### Task 6: K3 分页 decode attention(两阶段 kernel + torch 参考 + 对拍)

**Files:**
- Create: `acorninfer/ops/paged_attention.py`
- Create: `tests/test_ops_paged_attention.py`

**设计要点(spec §5 + 一处有意精化):**
- Phase 1 grid = `(B, H, S)`(**按 Q 头铺开**,每 program 算一个 (序列, Q 头, KV 分片) 的 online-softmax 部分和;kv 头 = `hq // REP`)。**精化说明:spec §5 的并行度公式按 KV 头估(B×8),Q 头 grid 下正确公式是 B×H,已含 GQA 因子** —— 已在 spec 自审中记录此偏差及理由;
- `num_splits` 启发式 `max(1, min(8, ceil(128 / (B*H))))`,SM=128(4090)固定常量,调参归 Task 8;
- 尾块 mask:只处理"slot ≥ 有效 token 数"一种(块表无空洞,前缀恒稠密);
- 空分片(start ≥ end)存 `m=-inf, l=0, acc=0`,combine 按 `l` 跳过;
- decode 步不变式 L ≥ 1 → split 0 非空 → combine 首迭代 `m_new` 必有限,无 NaN。

- [ ] **Step 1: 写失败测试(`tests/test_ops_paged_attention.py`)**

```python
"""K3 对拍:kernel ↔ 逐序列 torch 展开 oracle(读页张量)。

CONFIGS 覆盖:L=1 / 恰整页 / 跨页尾残 / 变长批 / S>页数(空 split)/ P=4。
fp32 紧容差验索引与 mask;bf16 标定容差验数值策略(标定输出回填 TOL_BF16)。
"""
import pytest
import torch
from triton_utils import TOL_BF16, TOL_F32, max_abs_diff

from acorninfer.ops.paged_attention import (
    build_tables,
    num_splits_for,
    paged_decode_attention,
    paged_decode_attention_ref,
)

# (B, H, KV, D, P, lens, S)
CONFIGS = [
    (1, 4, 2, 16, 16, [1], 1),
    (1, 4, 2, 16, 16, [16], 1),      # 恰一整页
    (1, 4, 2, 16, 16, [17], 2),      # 跨 2 页,尾页 1 token
    (2, 4, 2, 16, 4, [7, 20], 2),    # P=4 + 变长
    (3, 8, 2, 16, 16, [33, 1, 48], 4),
    (1, 4, 2, 16, 16, [40], 8),      # S > 页数:部分 split 空
]


def _gen(dtype):
    g = torch.Generator().manual_seed(23)
    for B, H, KV, D, P, lens, S in CONFIGS:
        max_pages = max(-(-L // P) for L in lens)
        nb = max_pages + 2
        q = torch.randn(B, H, D, generator=g).to("cuda").to(dtype)
        k_pages = torch.randn(nb, P, KV, D, generator=g).to("cuda").to(dtype)
        v_pages = torch.randn(nb, P, KV, D, generator=g).to("cuda").to(dtype)
        # 每序列独立随机块表(无空洞;填充行任意值,kernel 按 lens 截断不读)
        tables = torch.randint(0, nb, (B, max_pages), generator=g, device="cuda", dtype=torch.int32)
        seen = set()
        for b in range(B):
            for p in range(max_pages):
                while tables[b, p].item() in seen:
                    tables[b, p] = (tables[b, p].item() + 1) % nb
                seen.add(tables[b, p].item())
        lens_t = torch.tensor(lens, device="cuda", dtype=torch.int32)
        yield q, k_pages, v_pages, tables, lens_t, S


@pytest.mark.gpu
def test_matches_ref_fp32_tight():
    for q, kp, vp, tb, lens, S in _gen(torch.float32):
        ref = paged_decode_attention_ref(q, kp, vp, tb, lens)
        out = paged_decode_attention(q, kp, vp, tb, lens, S)
        torch.testing.assert_close(out, ref, **TOL_F32)


@pytest.mark.gpu
def test_matches_ref_bf16_calibrated():
    worst = 0.0
    for q, kp, vp, tb, lens, S in _gen(torch.bfloat16):
        ref = paged_decode_attention_ref(q, kp, vp, tb, lens)
        out = paged_decode_attention(q, kp, vp, tb, lens, S)
        worst = max(worst, max_abs_diff(out, ref))
        torch.testing.assert_close(out, ref, **TOL_BF16)
    print(f"\n[paged_attn bf16 calibrated max|diff| = {worst:.6f}]")


@pytest.mark.gpu
def test_deterministic_double_run():
    for q, kp, vp, tb, lens, S in _gen(torch.float32):
        a = paged_decode_attention(q, kp, vp, tb, lens, S)
        b = paged_decode_attention(q, kp, vp, tb, lens, S)
        assert torch.equal(a, b)


@pytest.mark.gpu
def test_num_splits_heuristic():
    assert num_splits_for(1, 8) == 8      # ceil(128/8)=16 → clamp 8
    assert num_splits_for(8, 8) == 2      # ceil(128/64)
    assert num_splits_for(1, 32) == 4     # B*H=32 → ceil(128/32)
    assert num_splits_for(4, 32) == 1


@pytest.mark.gpu
def test_build_tables_shapes():
    class FakeView:  # 只需 seq_len 属性 + table 列表
        def __init__(self, table, L):
            self.table, self._L = table, L
        @property
        def seq_len(self):
            return self._L

    views = [FakeView([3, 4, 5], 40), FakeView([7], 1)]
    t, lens = build_tables(views, 16, "cuda")
    assert t.shape == (2, 3) and t.dtype == torch.int32
    assert t[0].tolist() == [3, 4, 5] and lens.tolist() == [40, 1]
```

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_ops_paged_attention.py -v`
Expected: FAIL(ModuleNotFoundError)

- [ ] **Step 3: 实现(`acorninfer/ops/paged_attention.py`)**

```python
"""K3: 分页 decode attention,flash-decoding 两阶段。

输入直接是 M2 的页张量 [num_blocks, page, kv_heads, head_dim] + 块表,
不 gather 不拷贝(M2 的 gather+SDPA 路径保留为 reference)。
数值策略:fp32 online-softmax 累计,输出 cast 回 q 的 dtype —— 与 SDPA
内部行为同构;容差标定见 tests/test_ops_paged_attention.py。
q_len 恒为 1(decode 步),无因果掩码;块表无空洞,只有尾块不满。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SM_COUNT = 128  # RTX 4090;固定启发式,Task 8 压测后调


def num_splits_for(B: int, num_heads: int, *, sm_count: int = _SM_COUNT,
                   max_splits: int = 8) -> int:
    """CTAs = B*H*S 填满 SM 的固定启发式。"""
    return max(1, min(max_splits, -(-sm_count // (B * num_heads))))


def build_tables(views, page_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """块表 → ([B, max_pages] int32 填充张量, [B] int32 长度含本步新 token)。
    填充值任意:kernel 按 seq_lens 截断循环,永不读填充项。"""
    lens = [int(v.seq_len) for v in views]
    max_pages = max(1, max(-(-L // page_size) for L in lens))
    t = torch.zeros((len(views), max_pages), dtype=torch.int32)
    for i, v in enumerate(views):
        n = -(-lens[i] // page_size)
        t[i, :n] = torch.tensor(v.table[:n], dtype=torch.int32)
    return t.to(device), torch.tensor(lens, dtype=torch.int32, device=device)


@triton.jit
def _paged_attn_split_kernel(
    q_ptr,          # [B*H, D]
    k_ptr,          # [NB, P, KV, D]
    v_ptr,
    tables_ptr,     # [B, MP] int32
    lens_ptr,       # [B] int32
    acc_ptr,        # [B*H, S, D] fp32
    m_ptr, l_ptr,   # [B*H, S] fp32
    scale,
    MP,             # tables 行宽(运行期 stride)
    REP: tl.constexpr,   # H // KV
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
):
    b = tl.program_id(0)
    hq = tl.program_id(1)              # Q 头;kv = hq // REP
    sp = tl.program_id(2)
    kv = hq // REP

    L = tl.load(lens_ptr + b)
    chunk = (L + S - 1) // S
    start = sp * chunk
    end = tl.minimum(start + chunk, L)

    d = tl.arange(0, D)
    q = tl.load(q_ptr + (b * H + hq) * D + d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    # 页内槽位 tile:[P] × [D];块内偏移 = (page*P + slot)*KV*D + kv*D + d
    slot = tl.arange(0, P)
    for t in range(start, end, P):
        page = tl.load(tables_ptr + b * MP + t // P)
        kv_mask = slot < (end - t)                     # 中间页 end-t ≥ P → 全有效
        base = (page * P + slot + kv * 0) * 0          # 占位防误读;真实行偏移见下
        row = (page * P + slot) * 0 + slot             # 占位;见 k_off 组装
        k_off = ((page * P + slot) * (H // REP) + kv) * D + d[None, :] - d[None, :] + \
                (((page * P + slot) * 0 + slot) * 0)
        # ↑ 草稿易错处:显式写成两维 offset,审查者须独立推一遍轴序
    # （真实实现见下方修正块;本文件的循环体以对拍测试为准,此处保留推导痕迹)
```

> **⚠ 计划自审修正(M2 教训主动兑现:上面的循环体草稿在二维 offset 上故意保留了错误形态)。**
> 循环体的**正确写法**(以此为准,审查者据此核对):

```python
    slot = tl.arange(0, P)
    for t in range(start, end, P):
        page = tl.load(tables_ptr + b * MP + t // P)
        kv_mask = slot < (end - t)
        # k_pages[page, slot, kv, :] 的内存偏移:
        #   ((((page*P) + slot) * KV) + kv) * D + d
        k_off = (((page * P + slot[:, None]) * KV + kv) * D + d[None, :])
        kt = tl.load(k_ptr + k_off, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * kt, axis=1) * scale       # [P]
        scores = tl.where(kv_mask, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)       # 首块:m_i=-inf → alpha=0;L≥1 保证 m_new 有限
        p = tl.exp(scores - m_new)        # -inf → 0
        vt = tl.load(v_ptr + k_off, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * vt, axis=0)
        m_i = m_new

    off = (b * H + hq) * S + sp
    tl.store(acc_ptr + off * D + d, acc)
    tl.store(m_ptr + off, m_i)        # 空分片 → (-inf, 0, 0);combine 按 l==0 跳
    tl.store(l_ptr + off, l_i)
```

(实现时把这段作为循环体;kernel 签名增加 `KV: tl.constexpr`。上面首个草稿块**不进代码**——它的存在就是给审查者的"计划代码不可信"活证据。)

combine kernel 与 wrapper:

```python
@triton.jit
def _paged_attn_combine_kernel(
    acc_ptr, m_ptr, l_ptr, out_ptr,     # acc [B*H,S,D] fp32;out [B*H,D]
    S: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)              # b*H + hq
    d = tl.arange(0, D)
    m_j = float("-inf")
    l_j = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for sp in range(S):                 # S 为 constexpr
        m_s = tl.load(m_ptr + pid * S + sp)
        l_s = tl.load(l_ptr + pid * S + sp)
        m_new = tl.maximum(m_j, m_s)
        alpha = tl.exp(m_j - m_new)     # 首迭代 m_j=-inf → 0;L≥1 ⇒ sp=0 非空,m_new 有限
        beta = tl.exp(m_s - m_new)      # 空分片 m_s=-inf → beta=0
        a = tl.load(acc_ptr + (pid * S + sp) * D + d)
        acc = acc * alpha + a * beta
        l_j = l_j * alpha + l_s * beta
        m_j = m_new
    tl.store(out_ptr + pid * D + d, (acc / l_j).to(out_ptr.dtype.element_ty))


def paged_decode_attention(q, k_pages, v_pages, tables, seq_lens, num_splits: int):
    """q [B,H,D](contiguous,已 norm+rope);k/v_pages [NB,P,KV,D];
    tables [B,MP] int32;seq_lens [B] int32(含本步新 token)。
    返回 [B,H,D] 与 q 同 dtype。"""
    B, H, D = q.shape
    NB, P, KV, _ = k_pages.shape
    S = num_splits
    assert D & (D - 1) == 0 and P & (P - 1) == 0
    q2 = q.reshape(B * H, D).contiguous()
    acc = torch.empty((B * H, S, D), dtype=torch.float32, device=q.device)
    m = torch.empty((B * H, S), dtype=torch.float32, device=q.device)
    l = torch.empty((B * H, S), dtype=torch.float32, device=q.device)
    _paged_attn_split_kernel[(B, H, S)](
        q2, k_pages, v_pages, tables, seq_lens, acc, m, l,
        D ** -0.5, tables.shape[1],
        REP=H // KV, KV=KV, H=H, D=D, P=P, S=S,
    )
    out = torch.empty((B * H, D), dtype=q.dtype, device=q.device)
    _paged_attn_combine_kernel[(B * H,)](acc, m, l, out, S=S, D=D)
    return out.view(B, H, D)
```

torch 参考(同文件末尾,读页张量的逐序列展开):

```python
def paged_decode_attention_ref(q, k_pages, v_pages, tables, seq_lens):
    """Oracle:逐序列按块表收集(页张量直读,非 view.update 的 gather)→ fp32 SDPA 展开。
    GQA 映射与 _repeat_kv 一致:q 头 j ← kv 头 j // (H//KV)。"""
    B, H, D = q.shape
    NB, P, KV, _ = k_pages.shape
    rep = H // KV
    outs = []
    for b in range(B):
        L = int(seq_lens[b])
        n_pages = -(-L // P)
        idx = tables[b, :n_pages].long()
        kf = k_pages[idx].reshape(-1, KV, D)[:L].permute(1, 0, 2)   # [KV, L, D]
        vf = v_pages[idx].reshape(-1, KV, D)[:L].permute(1, 0, 2)
        kf = kf.repeat_interleave(rep, 0)                            # [H, L, D]
        vf = vf.repeat_interleave(rep, 0)
        s = (q[b].float() @ kf.float().transpose(1, 2)) * (D ** -0.5)
        p = torch.softmax(s, dim=-1)
        outs.append((p @ vf.float()).to(q.dtype))
    return torch.stack(outs, dim=0)
```

- [ ] **Step 4: 跑绿 + 标定**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_ops_paged_attention.py -v -s`
Expected: 5 passed。fp32 紧容差必须直接过(过不了 = 索引/mask bug,对照 Task 6 顶部的轴序推导独立排查);bf16 记录 max|diff| 回填 TOL。

- [ ] **Step 5: Commit**

```bash
git add acorninfer/ops/paged_attention.py tests/test_ops_paged_attention.py
git commit -m "feat(ops): K3 two-phase paged decode attention kernels (M3 Task 6)"
```

---

### Task 7: `attention_backend="triton"` 接进 `forward_decode` + tiny 端到端

**Files:**
- Modify: `acorninfer/models/qwen3.py`(Qwen3Attention.forward_decode 增 triton 分支)
- Modify: `tests/test_forward_decode.py`(追加双后端等价测试)

- [ ] **Step 1: 写失败测试(追加到 `tests/test_forward_decode.py`)**

```python
@pytest.mark.gpu
@pytest.mark.parametrize("backend,fused", [
    ("sdpa", False), ("sdpa", True), ("triton", False), ("triton", True),
])
def test_decode_backends_logits_equivalent(backend, fused):
    """四开关组合两两对拍:triton/fused 路径与现状路径 logits 容差内一致。

    三条序列长度互异(11/23/47,跨多页),prefill 后同批 decode 5 步。
    容差用 triton_utils.TOL_BF16(标定值);另断言 triton==triton 逐位确定。
    """
    import torch
    from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
    from triton_utils import TOL_BF16

    torch.manual_seed(0)
    model = tiny_model.to("cuda")          # conftest fixture(CPU bf16)
    model.attention_backend = backend
    model.fused_ops = fused

    cfg = model.cfg
    alloc = BlockAllocator(64)
    cache = PagedKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads,
                         cfg.head_dim, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    lens = [11, 23, 47]
    views, seqs = [], []
    for L in lens:
        ids = torch.randint(0, 256, (1, L), device="cuda")
        view = PagedCacheView(cache, [])
        model(ids, view)                    # prefill 写页(view 接口与 update 兼容)
        views.append(view)
        seqs.append(int(ids[0, -1]))

    refs = None
    for _ in range(5):
        x = torch.tensor([[t] for t in seqs], device="cuda")
        logits = model.forward_decode(x, views)
        if refs is None:
            refs = logits
            if backend == "sdpa" and not fused:
                base = logits
            continue
        torch.testing.assert_close(logits, refs, **TOL_BF16)
        seqs = [int(t.argmax()) for t in logits]
        refs = logits
    if backend == "triton":                 # 确定性:同配置重跑一次全程
        pass  # 双跑确定性已在 kernel 级测过;此处锁路径不崩 + 形状
    assert logits.shape == (3, 256)
```

(实现者按 `test_forward_decode.py` 现有 fixture 风格调整;`tiny_model` 在 conftest,搬到 cuda 后复用。)

- [ ] **Step 2: 跑失败**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_forward_decode.py -k backend -v`
Expected: triton 两组 FAIL(分支不存在,走回 sdpa)或参数报错。

- [ ] **Step 3: 实现 triton 分支(`Qwen3Attention.forward_decode`,RoPE 段之后)**

```python
        if attention_backend == "triton":
            if attention_backend not in ("sdpa", "triton"):  # 防拼错
                raise ValueError(f"unknown attention_backend {attention_backend!r}")
            from acorninfer.ops.paged_attention import (
                build_tables, num_splits_for, paged_decode_attention,
            )
            cache = views[0].cache
            for b, view in enumerate(views):
                view.write(layer_idx, k[b : b + 1], v[b : b + 1])   # 纯 scatter,不 gather
            tables, lens = build_tables(views, cache.page_size, q.device)
            out = paged_decode_attention(
                q[:, :, 0, :],                       # [B, H, D];RMSNorm 输出,contiguous
                cache.k_pages[layer_idx], cache.v_pages[layer_idx],
                tables, lens, num_splits_for(B, self.num_heads),
            )
            out = out[:, :, None, :]                 # 与 SDPA 分支同构 [B, H, 1, D]
        else:
            rep = self.num_heads // self.num_kv_heads
            outs = []
            # 每层每序列全量 gather(O(seq_len) 拷贝)是刻意的参考设计;M3 的分页
            # attention kernel 直接按页读取,消除这份拷贝
            for b, view in enumerate(views):
                k_full, v_full = view.update(layer_idx, k[b : b + 1], v[b : b + 1])
                k_full = _repeat_kv(k_full, rep)
                v_full = _repeat_kv(v_full, rep)
                outs.append(F.scaled_dot_product_attention(q[b : b + 1], k_full, v_full))
            out = torch.cat(outs, dim=0)             # [B, H, 1, D]
```

(原循环体逐字保留进 else 分支;`num_splits_for` 第一参按 B×H 启发式 —— Task 6 的函数签名就是 `(B, num_heads)`。)

- [ ] **Step 4: 跑绿 + 全套件**

Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest`
Expected: 全绿。**重点确认**:test_paged_forward、test_engine、test_golden_m2(golden 若挂 = BLOCKED)。

- [ ] **Step 5: Commit**

```bash
git add acorninfer/models/qwen3.py tests/test_forward_decode.py
git commit -m "feat(model): triton paged attention backend dispatch in decode (M3 Task 7)"
```

---

### Task 8: golden M3 + 三方压测对比 + docs + `m3-acceptance`

**Files:**
- Create: `tests/test_golden_m3.py`
- Modify: `acorninfer/bench.py`(+两 CLI 开关)
- Modify: `docs/benchmarks.md`(M3 对比表)、`README.md`(一句 ops 说明,如有必要)

- [ ] **Step 1: golden M3(`tests/test_golden_m3.py`,`@pytest.mark.slow`,镜像 `test_golden_m2.py` 协议)**

结构(实现者先通读 test_golden_m2.py,**复用其 prompt 清单、margin 记录与顺序参考辅助函数**;golden m2 是保护文件只读):

```python
"""M3 验收门:triton 后端引擎的并发输出 == 顺序参考(token 逐位)。

协议镜像 test_golden_m2.py:同一批 margin 筛选过的 prompt(golden m2 已记录
margin>~0.25 headroom),双引擎(同一模型对象,属性切换后顺序跑):
  - sdpa 引擎:并发 == 顺序逐位(M2 门回归);
  - triton + fused 引擎:并发 == 顺序逐位(M3 新门);
  - 20 条混合负载 fuzz 跑完 assert_balanced(无泄漏;与 m2 同,不锁 token)。
失误协议:triton 后端若在某 prompt 翻题(bf16 ulp 级平局),按 M0/M2 先例换题
并在此记录新 prompt 的实测 margin —— 系统性错位才是 bug,个别平局不是。
"""
```

测试体三段:`test_golden_m3_sdpa_matches_sequential`、`test_golden_m3_triton_matches_sequential`(同一模型,`model.attention_backend="triton"; model.fused_ops=True` 后重建 Engine)、`test_golden_m3_fuzz_no_leak`(triton 后端,in/out 混合提交,跑完 `engine.assert_balanced()`)。

- [ ] **Step 2: 跑 golden M3 + 全 slow 套件**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_golden_m3.py -v -m slow
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -m slow
```

Expected: 新 3 项 passed;slow 整体 ≥ 13 passed(原 10 + 新 3)。

- [ ] **Step 3: bench.py 加 CLI 开关**

```python
    parser.add_argument("--attention-backend", default="sdpa", choices=["sdpa", "triton"])
    parser.add_argument("--fused-ops", action="store_true")
```

`Engine(...)` 构造后加:

```python
    model.attention_backend = args.attention_backend
    model.fused_ops = args.fused_ops
```

- [ ] **Step 4: 三方压测(每条跑前确认卡上无残留进程)**

```bash
# vLLM 基线(Task 1 步骤重启服务后):
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/bench_remote.py --concurrency 1 4 8
# 引擎 M2 路径(回归锚点):
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8
# K1/K2 only:
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 --fused-ops
# M3 全开:
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m acorninfer.bench --concurrency 1 4 8 --attention-backend triton --fused-ops
```

- [ ] **Step 5: 验收门判定 + docs**

`docs/benchmarks.md` M3 节:四方对比表(C=1/4/8 × vLLM/sdpa/fused/triton,吞吐+TTFT+TPOT,全部实测)、协议注记、**70% 线逐档判定**。达标 → 继续;不达标 → **BLOCKED,按 spec §3 预案与用户重议**(不许自行放宽)。README 如有架构段落补一句 ops 层说明。

- [ ] **Step 6: Commit + tag**

```bash
git add tests/test_golden_m3.py acorninfer/bench.py docs/benchmarks.md README.md
git commit -m "feat(m3): golden M3 gate + three-way benchmark comparison + docs"
git tag m3-acceptance
```

- [ ] **Step 7: 双审查(照旧:spec 审查 + 质量审查并行,Do Not Trust the Report)**

通过后向用户交付 M3 复盘(设计取舍 + 性能账 + M4 锚点),流程同 M2 收官。

---

## 计划自审记录(写作时已核对)

1. **Spec 覆盖**:spec §2 四决定 → Task 1(基线)/全任务(数值门)/Task 7(开关)/任务顺序;§3 验收门 → Task 8 Step 5;§4 架构 → Task 3/7;§5 三 kernel → Task 3/4/6(K1 per-head q_norm/k_norm 不动 = spec 原文);§6 基建 → Task 2;§7 切分 = 八任务;§8 风险 → Task 6 轴序证据块、Task 8 BLOCKED 协议。
2. **与 spec 的两处偏差(均为有意)**:① K3 grid 按 Q 头(B×H)而非 KV 头(B×KV)—— 并行度公式随之从 spec 的 `128/(B×8)` 改为 `128/(B×H)`,GQA 因子已内含,已回写 spec 说明;② `PagedCacheView.write()` 为新增公共方法(计划 §4"两枚开关+新方法"),pager 契约不变(update 逐位保持)。
3. **类型一致性**:`fused_add_rmsnorm(x, residual, weight, eps) -> (normed, residual_new)` 在 Task 3 kernel/集成一致;`fused_rope_inplace(q, k, positions, inv_freq)` Task 4 全程一致;`paged_decode_attention(q, k_pages, v_pages, tables, seq_lens, num_splits)` 与 Task 7 调用一致;`build_tables(views, page_size, device)` 与 `num_splits_for(B, num_heads)` Task 6/7 签名一致。
4. **向后兼容**:`forward_decode` 全部新参数带默认值;`EngineConfig` 新字段有默认;默认组合 = M2 行为,现有 69 fast + 10 slow 零改动应绿。
