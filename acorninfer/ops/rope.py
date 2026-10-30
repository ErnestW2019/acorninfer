"""K2: fused RoPE —— kernel 内由 positions×inv_freq 算 cos/sin 并就地旋转。

布局 = HF Qwen3 的 half-rotate(chunk 前/后半),_rotate_half 语义:
    y1 = x1*c - x2*s;  y2 = x2*c + x1*s
数值镜像 torch 路径的三段 bf16 取整:cos/sin fp32 算完先 cast 激活 dtype;
x*cos、x2*sin 各自取整后,和/差再取整(RNE 与 torch 一致 → 目标逐位相等)。

取整步必须走 inline PTX(_round_via_act),不能直接 `.to(act)`:实测(2026-10-30,
triton 3.7.1 / sm89)Triton 的 elementwise 降级会把 fp32 opmath 链
(ext→fmul→trunc→ext→fsub→trunc)整体改写成 bf16 原生 fmul/fsub(随后
NVPTX 再缩成 fma.rn.bf16),中间取整步被静默删除 —— 单次取整 vs torch 的
逐段取整差 1 ulp bf16(LLIR 可见 `fmul bfloat` 替代了显式 fptrunc 链)。
inline asm 对所有 pass 不透明,强制生成独立 cvt.rn 指令,取整段数与 torch
逐段对齐。tl.cos/tl.sin(fp32)与 torch 逐位一致(__nv_cosf 同源,已对拍)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _round_via_act(x, ACT_BF16: tl.constexpr, ACT_F16: tl.constexpr):
    """fp32 值按激活 dtype 取整(RNE)后仍以 fp32 返回(值已落在激活格点)。

    等价于 `x.to(act).to(tl.float32)`,但取整+回扩在同一 asm 块内完成:
    单独 `.to()` 会被 Triton 的 elementwise 降级把整条 opmath 链改写成
    bf16 原生单次取整运算(实测删掉中间取整步,见模块 docstring);asm 块
    对所有 pass 不透明,取整段数与 torch 逐段对齐。恒返 fp32 —— Triton
    统一 jit 函数各 return 分支的类型。fp32 激活时为 mov.f32 恒等屏障:
    挡住 LLVM 把相邻 fmul+fsub 收缩成 fma.f32(torch 逐元素路径无收缩)。
    """
    if ACT_BF16:
        return tl.inline_asm_elementwise(
            """{
            .reg .b16 tmp;
            cvt.rn.bf16.f32 tmp, $1;
            cvt.f32.bf16 $0, tmp;
            }""",
            "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1,
        )
    if ACT_F16:
        return tl.inline_asm_elementwise(
            """{
            .reg .b16 tmp;
            cvt.rn.f16.f32 tmp, $1;
            cvt.f32.f16 $0, tmp;
            }""",
            "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1,
        )
    return tl.inline_asm_elementwise(
        "mov.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _rope_inplace_kernel(
    x_ptr,          # [B*H, D] contiguous
    pos_ptr,        # [B] int
    inv_freq_ptr,   # [D/2] fp32
    H: tl.constexpr,
    D: tl.constexpr,
    ACT_BF16: tl.constexpr,
    ACT_F16: tl.constexpr,
):
    row = tl.program_id(0)          # b * H + h
    b = row // H
    half: tl.constexpr = D // 2
    i = tl.arange(0, half)
    pos = tl.load(pos_ptr + b).to(tl.float32)
    freq = pos * tl.load(inv_freq_ptr + i)                  # [half] fp32
    act = x_ptr.dtype.element_ty
    cos = _round_via_act(tl.cos(freq), ACT_BF16, ACT_F16)           # 镜像:先 cast 再乘
    sin = _round_via_act(tl.sin(freq), ACT_BF16, ACT_F16)
    base = x_ptr + row * D
    x1 = tl.load(base + i).to(tl.float32)
    x2 = tl.load(base + half + i).to(tl.float32)
    t1 = _round_via_act(x1 * cos, ACT_BF16, ACT_F16)                # 三段取整镜像 torch
    t2 = _round_via_act(x2 * sin, ACT_BF16, ACT_F16)
    t3 = _round_via_act(x2 * cos, ACT_BF16, ACT_F16)
    t4 = _round_via_act(x1 * sin, ACT_BF16, ACT_F16)
    tl.store(base + i, (t1 - t2).to(act))
    tl.store(base + half + i, (t3 + t4).to(act))


def fused_rope_inplace(q, k, positions, inv_freq):
    """q/k: [B, H, 1, D] 且 contiguous(必须);就地旋转。positions [B] int;
    inv_freq [D/2](fp32/bf16 均可,内部 .float() —— 镜像 RotaryEmbedding.forward
    的 `inv_freq.float()`,模块级 .to(dtype) 会把 buffer cast 成 bf16)。
    dtype:bf16/fp32 已对拍;fp16 走同一 cvt.rn 机制(未单独对拍)。
    """
    B, H, s, D = q.shape
    assert s == 1 and D % 2 == 0 and (D & (D - 1)) == 0, "D must be power-of-2"
    assert q.is_contiguous() and k.is_contiguous()
    act_bf16 = q.dtype == torch.bfloat16
    act_f16 = q.dtype == torch.float16
    inv_freq = inv_freq.float()
    _rope_inplace_kernel[(B * H,)](
        q.view(B * H, D), positions, inv_freq, H=H, D=D,
        ACT_BF16=act_bf16, ACT_F16=act_f16,
    )
    KB = k.shape[0] * k.shape[1]
    _rope_inplace_kernel[(KB,)](
        k.view(KB, D), positions, inv_freq, H=k.shape[1], D=D,
        ACT_BF16=act_bf16, ACT_F16=act_f16,
    )


def fused_rope_ref(x, positions, inv_freq):
    """Oracle:x [B,H,1,D] → 旋转后同形状。逐操作镜像 Qwen3Attention 的 torch 分支。"""
    pos = positions.to(device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq.float().to(x.device))       # [B, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)[:, None, None, :]
    sin = emb.sin().to(x.dtype)[:, None, None, :]
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin
