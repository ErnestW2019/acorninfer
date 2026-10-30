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


@pytest.mark.gpu
def test_tiny_model_fused_ops_roundtrip():
    """tiny 模型端到端:fused_ops 开关不改变 logits 超出 bf16 标定容差。

    decode 路径(计划首选):prefill 一条 8-token 序列建页(page_size=4 覆盖
    多页),再 decode 2 步;fused_ops False/True 各跑一遍。模型权重相同,开关
    切换不改变权重;两个 run 各用全新 PagedCacheView(互不串页),步末手动还块。
    """
    import torch
    from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
    from acorninfer.config import Qwen3Config
    from acorninfer.models.qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    alloc = BlockAllocator(32)
    cache = PagedKVCache(
        model.cfg.num_hidden_layers,
        model.cfg.num_key_value_heads,
        model.cfg.head_dim,
        4,  # page_size=4:8-token prefill 跨 2 页
        alloc,
        "cuda",
        torch.bfloat16,
    )
    torch.manual_seed(1)
    prompt = torch.randint(0, 256, (1, 8), device="cuda")
    steps = torch.randint(0, 256, (2, 1), device="cuda")

    runs = []
    for fused in (False, True):
        model.fused_ops = fused
        view = PagedCacheView(cache)
        decode_logits = []
        with torch.inference_mode():
            model(prompt, view)  # prefill 建页
            for i in range(steps.shape[0]):
                decode_logits.append(model.forward_decode(steps[i : i + 1], [view]))
        runs.append(torch.stack(decode_logits))  # [2, 1, vocab]
        for blk in view.table:
            alloc.free(blk)

    torch.testing.assert_close(runs[1], runs[0], **TOL_BF16)
    alloc.assert_balanced()
