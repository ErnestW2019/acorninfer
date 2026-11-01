"""M7 Task 1 对拍:W8 dequant-GEMM kernel(acorninfer/ops/quant_gemm.py)。

fp32 参考锚 = `quantize.dequant_w8(q, s)` 后 fp32 matmul(quantize.py 参考实现)。
kernel 侧 dequant(int8→fp32×scale→**bf16**→tl.dot)的 w 舍入是本 kernel 数值域
的真实组成 —— 与 decode_gemm TOL_BF16_GEMM(kernel/ref 同吃 bf16 w,纯归约序,
atol 1e-2)口径不同,M7 段 0 QSPIKE_CHECK 先例;误差 = Σ_k x_k·Δw_k(Δw 为
bf16 舍入 ~2⁻⁹ 相对),随 K 增大 → **逐类标定写死**(命令/日期见下),spike 级
atol=1.0 不得复用(段 0 裁决:K=12288 的 down 类 atol 余量薄,单独标定)。

标定(TOL_W8,2026-11-01):命令
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/opencode/m7a/calib_tol_w8.py
(真实分布权重 randn→quantize_w8,五档 × M∈{1,7,8,16} × 段 0 胜出 cfg,seed
同本文件 _gen;实测逐类,选值紧比 ≤0.38 = 绑定元素 ≥2.6× 余量):

  cls      K      maxrel(>1) maxabs(≤1) maxrel(>10) maxdiff@|ref|  选值(rtol,atol) 紧比  atol/maxabs
  qo       4096   0.263      0.468      0.038       1.024@271      (2e-2, 1.5)      0.322 3.21×
  kv       4096   0.237      0.315      0.033       0.917@138      (2e-2, 1.5)      0.232 4.76×
  gate_up  4096   0.317      0.468      0.044       1.024@271      (2e-2, 1.5)      0.322 3.21×
  down    12288   0.488      0.714      0.067       1.641@312      (2e-2, 2.0)      0.380 2.80×
  lm_head  4096   0.354      0.496      0.045       1.308@259      (2e-2, 1.5)      0.368 3.02×

  紧比 = max|diff|/(atol+rtol·|ref|)(assert_close 判据全分布复算,绑定元素余量
  的精确口径;maxrel(>1) 大值是 |ref|≈1 处 atol 覆盖区的相对误差尾,不单独选
  rtol)。down K=12288 单独标定:实测 maxabs 0.714 / maxdiff 1.641,spike 级
  (2e-2, 1.0) 紧比 0.711(余量 1.4×,段 0 警示证实)→ 写死 (2e-2, 2.0)。

capture 硬门:graph 捕获含 kernel(含 down sk2 的 split-K 双发射)重放
torch.equal 逐位(M6 T5 / M7 段 0 QSPIKE_CAP 协议);launch 路径
set_sync_debug_mode(2) 零 host 同步。

大 M 桶(Task 2,2026-11-02):kernel `_w8_gemm_kernel_m`(pid_m 瓦片轴 × BM
32/64/128,sk1 定死——大 M 网格 N//BN × cdiv(M,BM) 并行自足);桶 s(17–64)/
m(65–256)/l(>256),装载期 autotune(`configure()`,代表 M = 桶上界
{64,256,1024})。容差沿用 TOL_W8 逐类表:误差机制 = 核内 bf16 舍入 × K 点积,
逐元素界与 M 无关(M 只改瓦片并行,不改归约链);大 M 标定复算确认(命令/
日期见 test_w8_gemm_large_m_matches_fp32_reference docstring,选值不变)。
显式 cfg 在大 M 桶解释为 (BM,BN,BK,warps,stages)(BM 首位,与 decode 桶
(BN,BK,warps,stages,SK) 字段序不同——M regime 决定解释,见 quant_gemm docstring)。
"""
import pytest
import torch

from acorninfer.ops import quant_gemm
from acorninfer.quant.quantize import dequant_w8, quantize_w8

CLASSES = quant_gemm.CLASSES      # 五档生产形状(q/o 共 (4096,4096) 等)
MS = [1, 7, 8, 16]

# 段 0 胜出配置(BN, BK, warps, stages, SPLIT_K):主形态 bn64_bk128_w4_s4_sk1
# 五档全胜 + down sk2(docs/benchmarks.md M7 段 0 控制器裁决)。测试显式 cfg,
# 不触发装载期 autotune(autotune 仅 configure(),Task 2 生产接线调用)。
CFG_MAIN = (64, 128, 4, 4, 1)
CFG_DOWN_SK2 = (64, 128, 4, 4, 2)

# 逐类标定容差(2026-11-01 标定后写死;实测表与选值依据见模块 docstring)。
TOL_W8 = {
    "qo": dict(rtol=2e-2, atol=1.5),
    "kv": dict(rtol=2e-2, atol=1.5),
    "gate_up": dict(rtol=2e-2, atol=1.5),
    "down": dict(rtol=2e-2, atol=2.0),
    "lm_head": dict(rtol=2e-2, atol=1.5),
}

_W_CACHE: dict = {}               # (cls) -> (q, s, wref fp32);逐类只建一次


def _gen(cls, m):
    """真实分布权重(段 0 口径):randn → quantize_w8;q/s/wref 逐类缓存。

    权重 CPU generator(种子固定,跨运行确定;decode_gemm _gen 同款),
    x 逐 (cls, m) 独立种子。
    """
    if cls not in _W_CACHE:
        n, k = CLASSES[cls]
        g = torch.Generator().manual_seed(11)
        w = torch.randn(n, k, generator=g).to("cuda").to(torch.bfloat16)
        q, s = quantize_w8(w)
        wref = dequant_w8(q, s)   # fp32 [N,K] 参考锚(存储后 fp16 scale 同源)
        _W_CACHE[cls] = (q, s, wref)
        del w
    q, s, wref = _W_CACHE[cls]
    k = CLASSES[cls][1]
    gx = torch.Generator().manual_seed(23 + m)
    x = torch.randn(m, k, generator=gx).to("cuda").to(torch.bfloat16)
    return x, q, s, wref


# ---------- 数值对拍(fp32 dequant 参考,逐类 TOL) ----------

@pytest.mark.gpu
@pytest.mark.parametrize("cls", list(CLASSES))
@pytest.mark.parametrize("m", MS)
def test_w8_gemm_matches_fp32_reference(cls, m):
    x, q, s, wref = _gen(cls, m)
    ref = x.float() @ wref.T
    cfg = CFG_DOWN_SK2 if cls == "down" else CFG_MAIN
    out = quant_gemm.quant_gemm(x, q, s, cls, cfg=cfg)
    assert out.dtype == torch.bfloat16 and out.shape == (m, CLASSES[cls][0])
    torch.testing.assert_close(out.float(), ref, **TOL_W8[cls])


# ---------- split-K 确定性(无 atomic) ----------

@pytest.mark.gpu
def test_w8_splitk_deterministic():
    """down sk2(段 0 定档):同输入两次 launch 输出逐位同。"""
    x, q, s, _ = _gen("down", 8)
    o1 = quant_gemm.quant_gemm(x, q, s, "down", cfg=CFG_DOWN_SK2)
    o2 = quant_gemm.quant_gemm(x, q, s, "down", cfg=CFG_DOWN_SK2)
    assert torch.equal(o1, o2)


# ---------- capture 重放逐位(M6 T5 / 段 0 QSPIKE_CAP 协议) ----------

@pytest.mark.gpu
@pytest.mark.parametrize("cls,cfg", [("down", CFG_DOWN_SK2), ("qo", CFG_MAIN)])
def test_w8_capture_replay_bitwise(cls, cfg):
    """graph 捕获含 kernel(down 含 split-K 双发射):重放逐位 + eager 对照逐位。"""
    x, q, s, _ = _gen(cls, 8)
    n = CLASSES[cls][0]
    quant_gemm.quant_gemm(x, q, s, cls, cfg=cfg)   # 预编译 + split workspace 就位
    torch.cuda.synchronize()
    xs, qs, ss = x.clone(), q.clone(), s.clone()   # 静态缓冲(捕获期地址冻结)
    out = torch.empty(8, n, dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            quant_gemm.quant_gemm(xs, qs, ss, cls, cfg=cfg, out=out)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        quant_gemm.quant_gemm(xs, qs, ss, cls, cfg=cfg, out=out)
    g.replay()
    torch.cuda.synchronize()
    r1 = out.clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(r1, out)                    # 重放确定性(逐位)
    out_e = torch.empty_like(out)
    quant_gemm.quant_gemm(xs, qs, ss, cls, cfg=cfg, out=out_e)
    torch.cuda.synchronize()
    assert torch.equal(r1, out_e)                  # eager 同输入逐位对照


# ---------- dispatch launch 纪律 ----------

@pytest.mark.gpu
def test_dispatch_no_host_sync():
    """查表 launch 零 host 同步(set_sync_debug_mode(2) 出错即抛);表查路径与
    显式 cfg 路径同 cfg 逐位同。表为 mock 装载(autotune 仅装载期,不进测试)。"""
    quant_gemm.reset()
    x, q, s, _ = _gen("down", 8)
    quant_gemm._state["configs"]["down"] = CFG_DOWN_SK2
    quant_gemm.quant_gemm(x, q, s, "down")          # 预分配 workspace(捕获前)
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode(2)
        out = quant_gemm.quant_gemm(x, q, s, "down")
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    out2 = quant_gemm.quant_gemm(x, q, s, "down", cfg=CFG_DOWN_SK2)
    assert torch.equal(out, out2)
    quant_gemm.reset()


# ---------- 大 M 桶(Task 2:prefill/extend;桶边界 + 数值 + 捕获) ----------

# 桶代表 cfg(显式,绕表;BM 首位):s→BM32 / m→BM64 / l→BM64,均 autotune
# 候选面内(生产 cfg 由装载期 autotune 实测定,测试钉显式 cfg——T1 先例)
CFG_LARGE = {
    "s": (32, 64, 128, 4, 3),
    "m": (64, 64, 128, 4, 3),
    "l": (64, 64, 64, 4, 4),
}
LARGE_MS = (17, 64, 256, 257)   # 跨 s/m/l 三桶 + 边界(65 见 bucket_for 单测)


def test_large_m_bucket_boundaries():
    """桶边界契约(CPU 纯逻辑):大 M 三桶上界 64/256(d 桶边界见注记期测)。"""
    assert quant_gemm.bucket_for(66) == "m"
    assert quant_gemm.bucket_for(256) == "m"
    assert quant_gemm.bucket_for(257) == "l"
    assert quant_gemm.bucket_for(1024) == "l"


@pytest.mark.gpu
@pytest.mark.parametrize("cls", list(CLASSES))
@pytest.mark.parametrize("m", LARGE_MS)
def test_w8_gemm_large_m_matches_fp32_reference(cls, m):
    """大 M 对拍:fp32 dequant 参考逐类 TOL(TOL_W8 沿用;标定复算见下)。

    标定复算(2026-11-02):CUDA_VISIBLE_DEVICES=1 .venv/bin/python
    /tmp/opencode/m7b/calib_tol_w8_m.py —— 五档 × M∈{64,256,1024} × 桶代表
    cfg,误差分布与 M≤16 桶同源(逐元素界 M 无关):maxabs(qo/kv/gate_up/
    lm_head ≤0.57、down ≤0.84)均在 TOL_W8 原表覆盖内,紧比 ≤0.43(绑定元素
    ≥2.3× 余量;T1 桶 ≤0.38),选值不变、零新桶容差。
    """
    x, q, s, wref = _gen(cls, m)
    ref = x.float() @ wref.T
    bucket = quant_gemm.bucket_for(m)
    out = quant_gemm.quant_gemm(x, q, s, cls, cfg=CFG_LARGE[bucket])
    assert out.dtype == torch.bfloat16 and out.shape == (m, CLASSES[cls][0])
    torch.testing.assert_close(out.float(), ref, **TOL_W8[cls])


@pytest.mark.gpu
def test_w8_capture_replay_bitwise_large_m():
    """大 M kernel graph 捕获重放逐位(pid_m 瓦片轴;T5 协议;sk1 单发射)。"""
    x, q, s, _ = _gen("gate_up", 64)
    n = CLASSES["gate_up"][0]
    quant_gemm.quant_gemm(x, q, s, "gate_up", cfg=CFG_LARGE["s"])  # 预编译
    torch.cuda.synchronize()
    xs, qs, ss = x.clone(), q.clone(), s.clone()
    out = torch.empty(64, n, dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            quant_gemm.quant_gemm(xs, qs, ss, "gate_up", cfg=CFG_LARGE["s"], out=out)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        quant_gemm.quant_gemm(xs, qs, ss, "gate_up", cfg=CFG_LARGE["s"], out=out)
    g.replay()
    torch.cuda.synchronize()
    r1 = out.clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(r1, out)
    out_e = torch.empty_like(out)
    quant_gemm.quant_gemm(xs, qs, ss, "gate_up", cfg=CFG_LARGE["s"], out=out_e)
    torch.cuda.synchronize()
    assert torch.equal(r1, out_e)


@pytest.mark.gpu
def test_large_m_routing_contract():
    """大 M 路由契约:quant_gemm 入口按 M 分流(M≤16 decode 表 / M>16 大 M 表);
    大 M 表未配置 → 显式 RuntimeError(不静默 KeyError);显式 cfg 绕表可用。"""
    quant_gemm.reset()
    x, q, s, _ = _gen("qo", 17)
    # 未配置:显式报错(防静默 KeyError / 误用 decode 表)
    with pytest.raises(RuntimeError, match="configure"):
        quant_gemm.quant_gemm(x, q, s, "qo")
    # 显式 cfg(BM 首位)绕表可用
    out = quant_gemm.quant_gemm(x, q, s, "qo", cfg=CFG_LARGE["s"])
    assert out.shape == (17, CLASSES["qo"][0])
    quant_gemm.reset()


# ---------- 分派契约(CPU 纯逻辑) ----------

def test_classes_shape_contract():
    """五档 cls 键与生产形状契约(q/o 共、gate/up 共;quantize.py 同口径)。"""
    assert quant_gemm.CLASSES == {
        "qo": (4096, 4096), "kv": (1024, 4096), "gate_up": (12288, 4096),
        "down": (4096, 12288), "lm_head": (151936, 4096),
    }


def test_bucket_guard_m_gt16_prefill_note():
    """(Task 2 已落地,本注记期契约退役;NotImplementedError 不得恢复。)
    大 M 桶路由/数值/捕获契约移交 test_large_m_* 三测;d/s/m/l 全边界在此。"""
    assert quant_gemm.bucket_for(1) == "d"
    assert quant_gemm.bucket_for(16) == "d"
    assert quant_gemm.bucket_for(17) == "s"
    assert quant_gemm.bucket_for(64) == "s"
    assert quant_gemm.bucket_for(65) == "m"
