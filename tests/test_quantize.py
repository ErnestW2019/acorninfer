"""M7 Task 0:RTN 量化数学单测(纯 CPU,无 GPU/triton 依赖)。

契约(spec §2 格式定案):
- W8 per-channel 对称:scale fp16 [N] = max|w[n,:]|/127(clamp 下限 EPS=1e-8),
  q int8 [N,K] ∈ [-127,127];
- W4 g128 非对称:per (输出通道, 组) scale/zp fp16 [N, K/g](scale=(max−min)/15,
  zp=min),q uint8 [N,K] ∈ [0,15];
- 打包:byte = q_even | q_odd<<4(**低 nibble = 偶数列**,约定钉死);
- 形状守门:K/N 非 128 倍数 → ValueError(spec §2 零特例口径的守门;
  生产五档 N/K 全为 128 倍数:4096/1024/12288/151936 × 4096/12288)。

理论界(测试断言依据,比 plan 草稿的 scale/2+eps 更精确,偏离已记录):
- W8:|dequant(q)−w| ≤ s/2(round-to-nearest 半步界)
      + |q|·|s_fp16−s_fp32| ≤ 127·s·2⁻¹¹(half 尾数 11 位,scale 存 fp16
      的表示误差;dequant 用的是存储后的 fp16 scale)
      + 1e-6(fp32 除法末位噪声兜底)。
- W4(组内):≤ s/2 + 15·s·2⁻¹¹ + |zp|·2⁻¹¹ + 1e-6(同理,|q|≤15、zp 亦存 fp16)。
"""
import pytest
import torch

from acorninfer.quant.quantize import (
    EPS,
    dequant_w4,
    dequant_w8,
    pack_w4,
    quantize_w4,
    quantize_w8,
    unpack_w4,
)


def _bf16_randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(torch.bfloat16)


def test_w8_roundtrip_within_theoretical_bound():
    """w8:|dequant(q,s) − w| ≤ s/2 + 127·s·2⁻¹¹ + eps(逐元素,CPU 随机张量)。"""
    w = _bf16_randn(256, 4096)
    w[0].fill_(0.0)          # 全零行:amax=0 → scale clamp 到 EPS
    w[1].fill_(0.5)          # 常值行:q 应精确落在 ±127
    q, s = quantize_w8(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float16
    assert q.shape == w.shape and s.shape == (256,)
    assert q.abs().max() <= 127
    # EPS clamp 路径(全零行):fp32 数学域 scale = clamp(amax,1e-8)=1e-8,但
    # **fp16 存储不可表示 1e-8**(half 最小次正规 ≈ 5.96e-8)→ 存储后为 0.0。
    # 无损:该行 q ≡ 0,dequant 精确为 0(spec ε 的职责是 fp32 域防除零,
    # 存储行为如实钉住,报告记录此发现)。
    fp32_scale_row0 = w[0].float().abs().amax().clamp_min(EPS) / 127.0
    assert float(fp32_scale_row0) == pytest.approx(EPS / 127.0, rel=1e-5)
    assert float(s[0]) == 0.0
    dq = dequant_w8(q, s)
    assert dq.dtype == torch.float32
    bound = s.float() / 2 + s.float() * 127 * (2 ** -11) + 1e-6
    err = (dq - w.float()).abs()
    assert bool((err <= bound[:, None]).all()), \
        f"超界 {(err - bound[:, None]).max().item()}"


def test_w4_roundtrip_within_theoretical_bound():
    """w4 g128:|dequant − w| ≤ s/2 + 15·s·2⁻¹¹ + |zp|·2⁻¹¹ + eps(逐元素,组内)。"""
    w = _bf16_randn(256, 4096)
    w[0].fill_(0.0)          # 全组常值:scale=(max−min)/15=0 → clamp EPS
    w[1, :128].fill_(-2.0)   # 单组常值
    q, s, zp = quantize_w4(w)
    assert q.dtype == torch.uint8 and s.dtype == torch.float16
    assert zp.dtype == torch.float16
    assert q.shape == w.shape and s.shape == (256, 32) and zp.shape == (256, 32)
    assert int(q.max()) <= 15 and int(q.min()) >= 0
    dq = dequant_w4(q, s, zp)
    assert dq.dtype == torch.float32
    bound_g = (s.float() / 2 + s.float() * 15 * (2 ** -11)
               + zp.float().abs() * (2 ** -11) + 1e-6)      # [N, K/g]
    err = (dq - w.float()).abs().view(256, 4096 // 128, 128)
    assert bool((err <= bound_g[..., None]).all()), \
        f"超界 {(err - bound_g[..., None]).max().item()}"


def test_pack_w4_roundtrip():
    """pack/unpack round-trip;低 nibble = 偶数列(约定钉死,注释写明)。"""
    q = torch.randint(0, 16, (256, 4096), dtype=torch.uint8)
    p = pack_w4(q)
    assert p.dtype == torch.uint8 and p.shape == (256, 2048)
    # 约定钉死:byte = q_even | q_odd << 4 → 低 nibble 是偶数列
    q2 = torch.tensor([[1, 2, 3, 4, 15, 0]], dtype=torch.uint8)
    p2 = pack_w4(q2)
    assert p2.tolist() == [[0x21, 0x43, 0x0F]]   # 0x21 = 1 | 2<<4,低 nibble=列 0
    assert torch.equal(unpack_w4(p2), q2)
    # 全量 round-trip:unpack(pack(q)) == q(偶列 = 低 nibble,p & 0xF;奇列 = p >> 4)
    assert torch.equal(unpack_w4(p), q)
    # 打包形态下的 dequant 与未打包一致(参考实现吃 unpack 后的 q)
    w = _bf16_randn(256, 4096, seed=3)
    qw, s, zp = quantize_w4(w)
    assert torch.equal(
        dequant_w4(unpack_w4(pack_w4(qw)), s, zp), dequant_w4(qw, s, zp))


def test_shape_guards():
    """K/N 非 128 倍数 → ValueError(两函数统一守门);group 缺省 128。"""
    ok = _bf16_randn(256, 4096)
    quantize_w8(ok)
    quantize_w4(ok)                       # 缺省 group=128 → scale [N, K/128]
    _, s, _ = quantize_w4(ok)
    assert s.shape == (256, 32)
    _, s64, _ = quantize_w4(ok, group=64)  # 显式 group 生效
    assert s64.shape == (256, 64)
    bad_k = _bf16_randn(256, 4000)         # K 非 128 倍数
    bad_n = _bf16_randn(250, 4096)         # N 非 128 倍数
    for fn in (quantize_w8, quantize_w4):
        with pytest.raises(ValueError):
            fn(bad_k)
        with pytest.raises(ValueError):
            fn(bad_n)
    with pytest.raises(ValueError):        # group 不整除 K
        quantize_w4(ok, group=300)
