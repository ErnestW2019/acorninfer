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

# 接线目标(M7 Task 2,spec §4):末段名 ∈ 8 类的 nn.Linear;embed/norm/KV/
# q_norm/k_norm/rope 零触碰。末段名匹配(name.split(".")[-1])防跨层误伤。
QUANT_ATTRS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
})


def apply_quant(model, mode: str) -> None:
    """装载期现场量化(spec §4 单存储):walk named_modules,末段名命中 8 类的
    nn.Linear → quantize_w8 → weight.data 换 int8 存储(T1b data 换存储先例;
    bf16 原存储随引用释放),scale/分派标记挂模块 attr:
    - `_q_scale`:fp16 [N] per-channel 行广播(quant_gemm 核内消费);
    - `_m7_quant`:分派标记,dispatch_linear 读它分流(none = 无标记 → 旧路径)。

    mode "none":不 walk、模块零触碰(none 硬门的静态面)。
    mode "w8a16":纯 torch 数学,CPU 可跑(单测不依赖 GPU);cls 五档键由
    调用点按 (N,K) 查表(quant_gemm._CLS_BY_SHAPE),walk 期不打标。
    mode "w4a16":段 B 门控,显式 NotImplementedError(构造期硬失败,不静默;
    EngineConfig 校验面仍收 w4a16 字符串,spec §4 CLI 面)。
    cls 形状守门:五档全为 128 倍数,quantize_w8 零特例守门即装载期守门——
    非 8B 形状模型走 quant 在首个目标层显式 ValueError。
    装载耗时(R5 预算 120s)print 进装载账(quant_gemm.configure 同款先例);
    逐层转换、bf16 随换随放,峰值 ≈ 全量 bf16 + 单层 fp32 瞬态(~2.5GB)。
    """
    if mode == "none":
        return
    if mode != "w8a16":
        raise NotImplementedError(
            f"apply_quant: mode {mode!r} 未接线(w4a16 为段 B 门控,M7 段 A "
            f"只落地 w8a16)")
    import time

    t0 = time.perf_counter()
    hit = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in QUANT_ATTRS and isinstance(mod, torch.nn.Linear):
            q, s = quantize_w8(mod.weight.data)
            # int8 data 换存储前必须 requires_grad_(False):torch 禁止把
            # requires_grad 的 .data 换成非浮点(plan 草稿注③ 实测命中;
            # 引擎全推理无 backward,int8 权重本就不可训,语义无损)
            mod.weight.requires_grad_(False)
            mod.weight.data = q          # int8 Parameter data 换存储
            mod._q_scale = s             # fp16 [N]
            mod._m7_quant = "w8a16"      # dispatch_linear 分派标记
            hit += 1
    if hit == 0:
        raise ValueError(
            "apply_quant 未命中任何量化目标(8 类投影 + lm_head;"
            "模型结构或 walk 面有误)")
    print(f"W8 APPLY modules={hit} s={time.perf_counter() - t0:.2f}", flush=True)


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
