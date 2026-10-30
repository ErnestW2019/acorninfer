"""K2 fused_rope 对拍:镜像 torch 路径 q*cos + rotate_half(q)*sin 的三段 bf16 取整,
目标 bitwise 相等(bf16 取整 RNE 与 torch 一致);对不上再退 assert_close 并记录。"""
import pytest
import torch
from triton_utils import TOL_BF16, TOL_F32, max_abs_diff

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
    """2026-10-30 标定:bf16 逐位相等(4 case 全 0 mismatch,triton 3.7.1/sm89)。

    取整步必须走 inline PTX(见 acorninfer/ops/rope.py 模块 docstring):
    直接 `.to(act)` 取整会被 triton elementwise 降级改写成单次取整的 bf16
    原生 fma(~3-13% 元素差 1 ulp)。若未来 triton 升级破坏屏障,本测试
    退级到 TOL_BF16 放行并打印实测 max|diff| 供按协议回填标定。
    标定命令:`CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_ops_rope.py -v -s`
    """
    worst = 0.0
    for x, pos, inv_freq in _gen(torch.bfloat16):
        ref = fused_rope_ref(x, pos, inv_freq)
        q = x.clone()
        k = x.clone()
        fused_rope_inplace(q, k, pos, inv_freq.to("cuda"))
        worst = max(worst, max_abs_diff(q, ref))
        try:
            torch.testing.assert_close(q, ref, rtol=0, atol=0)  # 目标:逐位
        except AssertionError:
            torch.testing.assert_close(q, ref, **TOL_BF16)      # 退级:标定容差
        assert torch.equal(k, q)  # 同输入同 kernel,q/k 必一致
    print(f"\n[rope bf16 calibrated max|diff| = {worst:.6f}]")  # 标定输出,回填 TOL;逐位绿时为 0.0


# ---- 集成:tiny 模型 decode 路径(复用 K1 的 env helper;计划里的 M1 KVCache
# 变体已作废 —— 单序列 forward 不经过 forward_decode,测不到 fused rope)----

from test_ops_rmsnorm import _tiny_decode_env  # noqa: E402
from acorninfer.cache.pager import PagedCacheView  # noqa: E402


@pytest.mark.gpu
def test_tiny_model_fused_rope_decode_matches_torch():
    """fused_ops 开关在真实 decode 路径上 logits 一致(超出 bf16 标定容差必炸)。

    prefill 8-token 建页(torch forward 路径,两层跨 2 页),fused False/True
    各 decode 2 步。fused 步的 cos/sin=None、kernel 内重算,与 torch 路径的
    逐位对拍(单元级)叠加,钉住端到端等价。
    """
    model, cache, alloc = _tiny_decode_env()
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


@pytest.mark.gpu
def test_rope_switch_is_wired(monkeypatch):
    """守卫:fused rope 没接线(attention 内忘了调 kernel 或 positions 没透传)必炸。

    镜像 test_ops_rmsnorm 的 rmsnorm 守卫:attention 方法体内局部
    `from acorninfer.ops.rope import fused_rope_inplace` 每次调用重新解析
    模块属性,monkeypatch 可命中。一次调用同时转 q/k(wrapper 内处理),
    fused decode 一步应恰 = len(model.layers) 次。
    """
    import acorninfer.ops.rope as rope_mod

    model, cache, alloc = _tiny_decode_env()
    torch.manual_seed(1)
    prompt = torch.randint(0, 256, (1, 8), device="cuda")
    step = torch.randint(0, 256, (1, 1), device="cuda")
    view = PagedCacheView(cache)
    real = rope_mod.fused_rope_inplace
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(rope_mod, "fused_rope_inplace", spy)
    with torch.inference_mode():
        model(prompt, view)  # prefill 走 torch forward,不碰 fused rope
        assert calls["n"] == 0
        model.fused_ops = True
        model.forward_decode(step, [view])
        assert calls["n"] == len(model.layers)
        model.fused_ops = False
        calls["n"] = 0
        model.forward_decode(step, [view])
        assert calls["n"] == 0  # 反向:开关关掉后不得再进 fused
    for blk in view.table:
        alloc.free(blk)
