"""RTN weight-only 量化数学(纯 torch,无 triton/GPU 依赖;spec §2/§4)。

格式定案(spec §2,测试即契约 tests/test_quantize.py):
- W8 per-channel 对称(通道 = 输出行):scale fp16 [N] = max|w[n,:]|/127
  (clamp 下限 EPS=1e-8),q int8 [N,K] ∈ [-127,127]。免校准数据(RTN 只看
  权重自身,这是选它而非 GPTQ 的首要原因)。
- W4 per-group 非对称,g=128:per (输出通道, 组) scale/zp fp16 [N, K/128]
  (scale=(max−min)/15, zp=min),q uint8 [N,K] ∈ [0,15];打包 uint8 [N,K/2]
  (byte = q_even | q_odd<<4,**低 nibble = 偶数列**,kernel 端 unpack 同约定)。
- 形状守门:K/N 非 128 倍数 → ValueError(spec §2 零特例口径;生产五档
  q/o (4096,4096)、kv (1024,4096)、gate/up (12288,4096)、down (4096,12288)、
  lm_head (151936,4096) 全为 128 倍数,零特例)。

dequant_w8/dequant_w4 为 **fp32 参考实现**(kernel TOL 对拍锚,Task 1 标定);
数值界:|dequant − w| ≤ s/2 + |q|·s·2⁻¹¹(scale/zp 存 fp16 的表示误差)
+ eps,tests/test_quantize.py 已钉。
"""
from __future__ import annotations

import torch

EPS = 1e-8           # scale 下限 clamp(全零行/常值组不除零)
_SHAPE_MULT = 128    # 零特例口径:N/K 必须 128 的倍数


def _guard_shape(w: torch.Tensor) -> tuple[int, int]:
    if w.dim() != 2:
        raise ValueError(f"w 须 2D [N,K],got shape={tuple(w.shape)}")
    n, k = w.shape
    if k % _SHAPE_MULT != 0:
        raise ValueError(f"K={k} 非 {_SHAPE_MULT} 的倍数(零特例口径)")
    if n % _SHAPE_MULT != 0:
        raise ValueError(f"N={n} 非 {_SHAPE_MULT} 的倍数(零特例口径)")
    return n, k


def quantize_w8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W8 per-channel 对称 RTN:w [N,K](bf16)→ (q int8 [N,K], scale fp16 [N])。

    scale = max|w[n,:]| / 127(clamp_min EPS);q = round(w/scale) clamp ±127
    (amax/127 下 |w/scale| ≤ 127,clamp 只兜 fp32 除法末位)。
    """
    n, k = _guard_shape(w)
    w32 = w.float()
    scale = w32.abs().amax(dim=1).clamp_min(EPS) / 127.0
    q = torch.round(w32 / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale.half()


def quantize_w4(
    w: torch.Tensor, group: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W4 per-group 非对称 RTN:g128 缺省。

    w [N,K] → (q uint8 [N,K] ∈ [0,15], scale fp16 [N,K/g], zp fp16 [N,K/g])。
    scale = (max−min)/15(clamp_min EPS,常值组不除零),zp = min,
    q = clamp(round((w−zp)/scale), 0, 15)。
    """
    n, k = _guard_shape(w)
    if k % group != 0:
        raise ValueError(f"K={k} 不被 group={group} 整除")
    wg = w.float().view(n, k // group, group)
    wmax, wmin = wg.amax(dim=2), wg.amin(dim=2)
    scale = ((wmax - wmin) / 15.0).clamp_min(EPS)
    zp = wmin
    q = torch.round((wg - zp[..., None]) / scale[..., None]).clamp_(0, 15)
    return q.to(torch.uint8).view(n, k), scale.half(), zp.half()


def pack_w4(q: torch.Tensor) -> torch.Tensor:
    """双 nibble 打包:q uint8 [N,K] ∈ [0,15] → uint8 [N,K/2]。

    **byte = q_even | q_odd << 4(低 nibble = 偶数列)** —— kernel 端 unpack
    (shift/mask 与 tl.interleave 两法)同此约定,测试钉死。
    """
    if q.dim() != 2 or q.shape[1] % 2 != 0:
        raise ValueError(f"pack_w4 须 2D 且 K 偶数,got shape={tuple(q.shape)}")
    qq = q.reshape(q.shape[0], q.shape[1] // 2, 2)
    return (qq[..., 0] | (qq[..., 1] << 4)).contiguous()


def unpack_w4(p: torch.Tensor) -> torch.Tensor:
    """pack_w4 逆:偶数列 = 低 nibble(p & 0xF),奇数列 = 高 nibble(p >> 4)。"""
    if p.dim() != 2:
        raise ValueError(f"unpack_w4 须 2D,got shape={tuple(p.shape)}")
    n, kh = p.shape
    out = torch.empty(n, kh * 2, dtype=torch.uint8, device=p.device)
    out[:, 0::2] = p & 0xF
    out[:, 1::2] = p >> 4
    return out


def dequant_w8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 参考实现(TOL 对拍锚):q int8 [N,K] × scale fp16 [N] → fp32 [N,K]。"""
    return q.float() * scale.float()[:, None]


def dequant_w4(q: torch.Tensor, scale: torch.Tensor,
               zp: torch.Tensor) -> torch.Tensor:
    """fp32 参考实现(TOL 对拍锚):q uint8 [N,K](未打包)→ fp32 [N,K]。

    组索引 = 列号 // group(group = K // scale.shape[1]);打包存储先过
    unpack_w4(低 nibble = 偶数列约定)。
    """
    n, k = q.shape
    g = scale.shape[1]
    if k % g != 0:
        raise ValueError(f"K={k} 与 scale 组数 {g} 不相容")
    rep = k // g
    s = scale.float().repeat_interleave(rep, dim=1)
    z = zp.float().repeat_interleave(rep, dim=1)
    return q.float() * s + z
