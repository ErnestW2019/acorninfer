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
