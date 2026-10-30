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
    x_ptr, res_in_ptr, res_out_ptr, weight_ptr, normed_ptr,
    n_cols, eps,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(res_in_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
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
    """融合 add + RMSNorm(行内单 kernel)。语义契约(与 torch 路径逐操作同构,
    oracle 见本文件 fused_add_rmsnorm_ref):

        fused_add_rmsnorm(x, residual, weight, eps) -> (normed, residual_new)
        residual is None    : residual_new = x(同值新张量),     normed = rmsnorm(x)
        residual is not None: residual_new = round(residual + x), normed = rmsnorm(residual_new)

    加法在 fp32 opmath 后按输入 dtype 取整(torch bf16 逐元素加法行为);
    归一对取整后的残差在 fp32 进行、末尾 cast 回输入 dtype(镜像
    models.qwen3.RMSNorm)。

    - x: [..., H],任意前置维度(内部展平 [rows, H]);non-contiguous 允许
      (reshape 按需拷贝);residual: None 或与 x 同形状同 dtype。
    - weight: [H];eps:方差稳定项。
    - 返回 (normed, residual_new):均新分配、与 x 同形状,绝不原地写 x 或
      residual。
    - dtype:bf16 / fp16 / fp32 均可。
    """
    *lead, H = x.shape
    rows = x.numel() // H
    x2 = x.reshape(rows, H)
    normed = torch.empty_like(x2)
    res_out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(H)
    # HAS_RESIDUAL=False 时 res_in_ptr 传 x2 作占位(编译期常量剪掉,不读);
    # res_out 恒为新分配并被写入,绝不落笔调用方的 x。
    _fused_add_rmsnorm_kernel[(rows,)](
        x2, x2 if residual is None else residual.reshape(rows, H), res_out, weight, normed,
        H, eps, BLOCK=BLOCK, HAS_RESIDUAL=residual is not None,
        num_warps=8,
    )
    return normed.view(*lead, H), res_out.view(*lead, H)


def fused_add_rmsnorm_ref(x, residual, weight, eps):
    """Oracle。residual is None → res = x;否则 res = residual + x(输入 dtype 加法)。"""
    if residual is None:
        res = x
    else:
        res = residual + x
    xf = res.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
    return normed, res.to(x.dtype)
