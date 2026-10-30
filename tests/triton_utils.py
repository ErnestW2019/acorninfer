"""Triton kernel 测试基建(先于一切 kernel —— spec §6)。

两级容差协议:
- fp32 输入 + TOL_F32(紧):只容忍归约序差异;索引/mask/边界错了必炸;
  TOL_F32 亦可按 kernel 标定(大 K 归约序漂移可合理超过 1e-5),规则同
  bf16:按实测写死并注释标定命令与日期。
- bf16 输入 + TOL_BF16(标定):先以 INIT 值跑通,再按实测 max|diff| 收紧写死,
  注释记录标定命令与日期。不允许拍脑袋容差。

导入约定:pytest 默认 import-mode=prepend 会把 tests/ 目录插入 sys.path,
kernel 测试文件直接 `from triton_utils import ...`。
"""

import torch

TOL_F32 = dict(rtol=1e-5, atol=1e-5)

# bf16 初值(宽松;各 kernel 标定后收紧):bf16 ulp ≈ 2^-8 相对,跨路径双重
# 取整放宽到 ~2%。标定命令见各测试 docstring。
# 标定:2026-10-30 K1 fused_add_rmsnorm 首标,bit-exact(max|diff|=0.0,32 组
# x/residual/case 全零);命令 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest
# tests/test_ops_rmsnorm.py -v -s`。按实测收紧到 1e-2(为后续 kernel 的归约序
# 漂移留 ~2-3 ulp 余量;新 kernel 落地时按协议再标定、再收紧)。
TOL_BF16 = dict(rtol=1e-2, atol=1e-2)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    if a.numel() == 0:
        return 0.0
    return (a.float() - b.float()).abs().max().item()
