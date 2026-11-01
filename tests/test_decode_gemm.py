"""M6 Task 5 对拍:decode GEMM triton kernel(+ per-(类, M-bucket) dispatch 表)。

两类:gate_up = 引擎真实 per-call 形状 (N=12288, K=4096)(gate_proj / up_proj
各一次;spike 的 N=24576 为融合口径,生产接线为逐调用分派)、down = (N=4096,
K=12288)。M 维:tl.dot 张量核路径 BLOCK_M=16(M≤16 全部 pad 到 16)。

TOL 协议(tests/triton_utils.py):bf16 输入先以 INIT 宽松值跑通,再按实测
收紧写死,注释记录标定命令与日期。K=4096/12288 大 K 归约序漂移(cuBLAS TN
vs triton 异序 fp32 累加)在 bf16 输出上表现为大 |ref| 位的 1–2 ulp 翻转,
绝对差随幅值走 → 容差用 rtol+atol 组合(torch.testing.assert_close),非纯
绝对差(逐元素算子的 max_abs_diff 口径在此不适用,如实注记)。

标定(TOL_BF16_GEMM,2026-11-01):命令
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/opencode/m6t5/calib_tol.py
(gate_up/down × M∈{1,7,8,16} × 各自 tuned cfg,seed=11):实测
maxrel(|ref|>1) = 7.752e-3(≈2 bf16 ulp)、maxabs(|ref|≤1) = 0.0039、
max|diff| = 2.0(|ref|≈258);(rtol=1e-2, atol=1e-2) 下最紧约束比 0.758
(余量仅 1.3×)→ 写死 rtol=2e-2(2.6× 实测)、atol=1e-2(2.6× 实测),
复算最紧比 ≈0.39。真错(索引/mask/越界)差 O(10) 起,不可漏网。
"""
import pytest
import torch
import torch.nn.functional as F
from triton_utils import TOL_BF16  # noqa: F401 (协议常量,本文件用 per-kernel 标定值)

from acorninfer.ops import decode_gemm

# 引擎真实 per-call 形状(单卡 8B)
CLASSES = {"gate_up": (12288, 4096), "down": (4096, 12288)}
MS = [1, 7, 8, 16]

# bf16 标定容差(2026-11-01 标定后写死;依据见模块 docstring)
TOL_BF16_GEMM = dict(rtol=2e-2, atol=1e-2)

# 测试用显式 config(BN, BK, warps, stages, split_k);split-K 路径单独 case
CFG_GATE_UP = (128, 64, 4, 3, 1)
CFG_DOWN_SPLIT4 = (128, 64, 4, 4, 4)


def _gen(cls, m):
    N, K = CLASSES[cls]
    g = torch.Generator().manual_seed(11)
    x = torch.randn(m, K, generator=g).to("cuda").to(torch.bfloat16)
    w = torch.randn(N, K, generator=g).to("cuda").to(torch.bfloat16)
    return x, w


# ---------- 数值对拍(bf16 标定) ----------

@pytest.mark.gpu
@pytest.mark.parametrize("cls", ["gate_up", "down"])
@pytest.mark.parametrize("m", MS)
def test_triton_matches_cublas_bf16(cls, m):
    x, w = _gen(cls, m)
    ref = F.linear(x, w)
    out = decode_gemm.triton_gemm(x, w, cls, cfg=CFG_GATE_UP if cls == "gate_up"
                                  else CFG_DOWN_SPLIT4)
    torch.testing.assert_close(out, ref, **TOL_BF16_GEMM)


@pytest.mark.gpu
@pytest.mark.parametrize("cls", ["down", "gate_up"])
def test_splitk_path_matches_cublas_m1(cls):
    """split-K 路径单独 case(含 M=1):强制 SPLIT_K=4 对拍 cuBLAS TN。"""
    x, w = _gen(cls, 1)
    ref = F.linear(x, w)
    N, K = CLASSES[cls]
    assert K % (4 * 64) == 0
    out = decode_gemm.triton_gemm(x, w, cls, cfg=(128, 64, 4, 4, 4))
    torch.testing.assert_close(out, ref, **TOL_BF16_GEMM)


@pytest.mark.gpu
def test_splitk_deterministic_bitwise():
    """确定性 second-pass 归约:同输入两次 launch 逐位同(无 atomic)。"""
    x, w = _gen("down", 8)
    o1 = decode_gemm.triton_gemm(x, w, "down", cfg=CFG_DOWN_SPLIT4)
    o2 = decode_gemm.triton_gemm(x, w, "down", cfg=CFG_DOWN_SPLIT4)
    assert torch.equal(o1, o2)


# ---------- dispatch 表(查表逻辑,mock 实测仪,不实测) ----------

def test_bucket_lookup():
    assert decode_gemm.M_BUCKETS == (1, 2, 4, 8)
    # 与 graph_runner.BUCKETS 同源(禁改文件的常量;测试侧钉住一致)
    from acorninfer.engine.graph_runner import BUCKETS
    assert decode_gemm.M_BUCKETS == BUCKETS
    assert decode_gemm._bucket_for(1) == 1
    assert decode_gemm._bucket_for(2) == 2
    assert decode_gemm._bucket_for(3) == 4
    assert decode_gemm._bucket_for(7) == 8
    assert decode_gemm._bucket_for(8) == 8
    assert decode_gemm._bucket_for(9) is None   # 超最大桶 → 回 cuBLAS


def test_dispatch_table_routing(monkeypatch):
    """查表单测:表内 triton → launch 路径;表内 cublas / 无表 / layout none → F.linear。"""
    decode_gemm.reset()
    x, w = _gen("gate_up", 8)
    ref = F.linear(x, w)
    sentinel = ref + 1.0     # 可区分值
    decode_gemm._state["layout"] = "triton"
    decode_gemm._state["configs"]["gate_up"] = CFG_GATE_UP
    decode_gemm._state["active"] = {"gate_up"}
    for b in decode_gemm.M_BUCKETS:
        decode_gemm._state["table"][("gate_up", b)] = "cublas"
    decode_gemm._state["table"][("gate_up", 8)] = "triton"
    monkeypatch.setattr(decode_gemm, "_launch", lambda x2, w, cls: sentinel)
    # bucket 8(M=7/8)→ triton launch
    assert decode_gemm.linear_dispatch(x, w, "gate_up") is sentinel
    # bucket 1(M=1)→ cuBLAS 原径(逐字节 F.linear)
    x1, _ = _gen("gate_up", 1)
    out1 = decode_gemm.linear_dispatch(x1, w, "gate_up")
    assert torch.equal(out1, F.linear(x1, w))
    # 未注册类 → cuBLAS
    assert torch.equal(decode_gemm.linear_dispatch(x1, w, "qkv"), F.linear(x1, w))
    # layout none 硬门:表在也走 F.linear
    decode_gemm._state["layout"] = "none"
    assert torch.equal(decode_gemm.linear_dispatch(x, w, "gate_up"), ref)
    decode_gemm.reset()


def test_configure_valve_logic(monkeypatch):
    """接入阀:triton ≤ 0.95 × cuBLAS 最优变体 才接入该类;类内逐桶择优。"""
    decode_gemm.reset()
    monkeypatch.setattr(decode_gemm, "_autotune_config", lambda *a, **k: CFG_GATE_UP)
    monkeypatch.setattr(decode_gemm, "_warmup_table", lambda: None)
    # 实测仪 mock:gate_up 全桶 triton 赢 6%(>5% 阀,接);down 全桶赢 1%(<5%,不接)
    def fake_measure(cls, *, N, K, buckets, device, dtype, verbose):
        tri = {m: 100.0 for m in buckets}
        if cls == "gate_up":
            per = {m: {"tn": 106.0, "nn": 107.0, "pad16": 108.0} for m in buckets}
        else:
            tri = {m: 110.0 for m in buckets}
            per = {m: {"tn": 111.0, "nn": 111.5, "pad16": 112.0} for m in buckets}
        cub = {"tn": {m: v["tn"] for m, v in per.items()},
               "nn": {m: v["nn"] for m, v in per.items()},
               "pad16": {m: v["pad16"] for m, v in per.items()}}
        return tri, cub
    monkeypatch.setattr(decode_gemm, "_measure_class", fake_measure)
    report = decode_gemm.configure(
        "triton", {"gate_up": CLASSES["gate_up"], "down": CLASSES["down"]},
        verbose=False)
    assert report["classes"]["gate_up"]["adopted"] is True
    assert report["classes"]["down"]["adopted"] is False
    for b in decode_gemm.M_BUCKETS:
        assert decode_gemm._state["table"][("gate_up", b)] == "triton"
        # 未过阀类:表行记录逐桶原始择优(down 行 tri=110 < tn=111 → triton),
        # 但 active/adopted 门不放行 → linear_dispatch 落 F.linear(硬门在门不在行)
        assert decode_gemm._state["active"] == {"gate_up"}
        assert decode_gemm._state["adopted"] == {"gate_up"}
    assert "down" not in decode_gemm._state["active"]
    decode_gemm.reset()


def test_configure_valve_partial_buckets(monkeypatch):
    """类内逐桶择优:部分桶 triton 输给 TN 时该桶留 cuBLAS(允许部分接入)。"""
    decode_gemm.reset()
    monkeypatch.setattr(decode_gemm, "_autotune_config", lambda *a, **k: CFG_GATE_UP)
    monkeypatch.setattr(decode_gemm, "_warmup_table", lambda: None)
    tri = {1: 200.0, 2: 100.0, 4: 100.0, 8: 100.0}       # M=1 输
    per = {m: {"tn": 106.0, "nn": 106.5, "pad16": 107.0}
           for m in decode_gemm.M_BUCKETS}
    cub = {"tn": {m: v["tn"] for m, v in per.items()},
           "nn": {m: v["nn"] for m, v in per.items()},
           "pad16": {m: v["pad16"] for m, v in per.items()}}
    monkeypatch.setattr(decode_gemm, "_measure_class",
                        lambda cls, **k: (tri, cub))
    report = decode_gemm.configure("triton", {"gate_up": CLASSES["gate_up"]},
                                   verbose=False)
    assert report["classes"]["gate_up"]["adopted"] is True
    assert decode_gemm._state["table"][("gate_up", 1)] == "cublas"
    assert decode_gemm._state["table"][("gate_up", 8)] == "triton"
    decode_gemm.reset()


# ---------- launch 路径纪律 ----------

@pytest.mark.gpu
def test_launch_zero_host_sync():
    """CUDA-graph 兼容硬约束:launch 路径零 host 同步(set_sync_debug_mode 硬门)。"""
    decode_gemm.reset()
    x, w = _gen("down", 8)
    decode_gemm.triton_gemm(x, w, "down", cfg=CFG_DOWN_SPLIT4)   # 预编译 + 预分配
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(2)    # 出错即抛:synchronize/.item() 类调用禁止
    try:
        out = decode_gemm.triton_gemm(x, w, "down", cfg=CFG_DOWN_SPLIT4)
        decode_gemm._state["layout"] = "triton"
        decode_gemm._state["configs"]["down"] = CFG_DOWN_SPLIT4
        decode_gemm._state["table"][("down", 8)] = "triton"
        decode_gemm._state["active"] = {"down"}
        out2 = decode_gemm.linear_dispatch(x, w, "down")
    finally:
        torch.cuda.set_sync_debug_mode(prev)
        decode_gemm.reset()
    torch.cuda.synchronize()
    assert torch.equal(out, out2)


@pytest.mark.gpu
def test_graph_capture_with_triton_launch():
    """graph 捕获含 triton launch:捕获成功 + 重放成功 + 重放结果 = eager 同输入。"""
    decode_gemm.reset()
    x, w = _gen("down", 8)
    cfg = CFG_DOWN_SPLIT4
    decode_gemm.triton_gemm(x, w, "down", cfg=cfg)   # 捕获前 warmup(编译/工作区就位)
    xs = torch.zeros_like(x)
    ws = torch.zeros_like(w)
    out = torch.empty(8, w.shape[0], dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    xs.copy_(x)
    ws.copy_(w)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            decode_gemm.triton_gemm(xs, ws, "down", cfg=cfg, out=out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        decode_gemm.triton_gemm(xs, ws, "down", cfg=cfg, out=out)
    g.replay()
    torch.cuda.synchronize()
    r1 = out.clone()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(r1, out)                       # 重放确定性
    ref = F.linear(x, w)
    torch.testing.assert_close(r1, ref, **TOL_BF16_GEMM)
    decode_gemm.reset()


# 生产接线硬门:两类均未过 ≥5% 阀(见模块 docstring / docs M6 段 5)→
# 不接线。EngineConfig 无 gemm_layout 字段(已随 no-op 收尾回退),
# linear_dispatch 无生产调用点;layout="none" 直通语义由
# test_dispatch_table_routing 的 none 分支覆盖。
