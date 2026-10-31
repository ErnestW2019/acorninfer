"""TP=2 eager decode 对拍(M5 Task 3,双卡;单卡环境整模块 skip)。

协议:tiny 模型随机权重(seed 0)先存单文件 checkpoint;两个 spawn 子进程
file:// init NCCL,各 rank 经 load_safetensors_sharded 结构化分片加载
(端到端打新加载器),各建 PagedKVCache(每卡 num_kv_heads/world);跑
强制输入流的 decode 步(三段:B=1 → B=2(row1 延迟加入,变长 KV)→
row0 单独续跑),每步落 logits + argmax;页表/长度快照落盘。

对拍门(见文件尾容差注释):
- 父进程内建 tiny 全宽模型(bf16, cuda:0)跑同一协议作 TP=1 参照;
- 逐步 logits 标定 TOL;参照 top-2 差距 >= margin 的步,TP argmax 逐位一致;
- 两 rank 的 token/页表/长度快照逐位一致(SPMD 决策全同),页表还须与
  参照逐位一致。

为何是强制输入流而非贪婪采样链:bf16 logits 存在精确平局(实测参照
step4 row1 top-2 gap = 0.0),TP 的切分归约噪声(~1 ulp)在平局步必然
可能翻转 argmax 并毒化整条采样链 —— 与 M4 decode 对拍"喂预生成 token
防平局分叉"同源。强制流下每步独立,平局步不放大。
"""

import json
import multiprocessing as mp

import pytest
import torch
from safetensors.torch import save_file

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import Qwen3Config
from acorninfer.models.qwen3 import Qwen3ForCausalLM

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="TP=2 需要 2 张 CUDA 卡"
)

FORCED_SEED = 42   # 强制输入流种子(两侧同一生成器,逐 token 一致)
P1_STEPS = 3       # 段1:B=1(row0 独行起步)
P2_STEPS = 40      # 段2:B=2(row1 在 KV=3 处加入,双行变长;KV 最长 43,跨 11 页 @page4)
P3_STEPS = 5       # 段3:row0 单独续跑(B 回落 1)


def _forced_inputs():
    """预生成输入 token 流(CPU 生成器,三段顺序抽样;两侧逐位一致)。

    布局 [B, steps]:每步取 [:, t:t+1] 即 forward_decode 的 [B, 1] 输入面。
    """
    g = torch.Generator().manual_seed(FORCED_SEED)
    vocab = Qwen3Config.tiny().vocab_size
    p1 = torch.randint(0, vocab, (1, P1_STEPS), generator=g)
    p2 = torch.randint(0, vocab, (2, P2_STEPS), generator=g)
    p3 = torch.randint(0, vocab, (1, P3_STEPS), generator=g)
    return p1, p2, p3


def _decode_protocol(model, device):
    """TP=1 / TP=2 共用的强制流 decode 协议(结果全 JSON 化)。"""
    cfg = model.cfg
    kv_heads = getattr(model, "num_kv_heads_local", cfg.num_key_value_heads)
    alloc = BlockAllocator(64)
    cache = PagedKVCache(
        cfg.num_hidden_layers, kv_heads, cfg.head_dim,
        4, alloc, device, torch.bfloat16,
    )
    views = [PagedCacheView(cache) for _ in range(2)]
    inputs = _forced_inputs()
    result = {}
    with torch.inference_mode():
        # 段1:B=1
        out = {"tokens": [], "logits": []}
        for t in range(P1_STEPS):
            cur = inputs[0][:, t:t + 1].to(device=device, dtype=torch.long)
            logits = model.forward_decode(cur, views[:1])
            out["tokens"].append([int(x) for x in logits.argmax(dim=-1)])
            out["logits"].append(logits.float().cpu().tolist())
            cur = inputs[0][t:t + 1].to(device=device, dtype=torch.long)
        result["p1"] = out
        # 段2:B=2(row1 从 KV=3 起步)
        out = {"tokens": [], "logits": []}
        for t in range(P2_STEPS):
            cur = inputs[1][:, t:t + 1].to(device=device, dtype=torch.long)
            logits = model.forward_decode(cur, views)
            out["tokens"].append([int(x) for x in logits.argmax(dim=-1)])
            out["logits"].append(logits.float().cpu().tolist())
        result["p2"] = out
        # 段3:row0 单独续跑
        out = {"tokens": [], "logits": []}
        for t in range(P3_STEPS):
            cur = inputs[2][:, t:t + 1].to(device=device, dtype=torch.long)
            logits = model.forward_decode(cur, views[:1])
            out["tokens"].append([int(x) for x in logits.argmax(dim=-1)])
            out["logits"].append(logits.float().cpu().tolist())
        result["p3"] = out

    result["tables"] = [list(v.table) for v in views]
    # 键全 str 化:子进程结果走 JSON 往返(int 键会被转 str),参照侧保持同形
    result["lens"] = {str(i): {str(k): v for k, v in v._lens.items()}
                      for i, v in enumerate(views)}
    return result


def _tp_worker(rank, world, init_file, ckpt_dir, out_json):
    """子进程入口(spawn 可 pickle):NCCL setup → 分片加载 → 协议 → 落盘。"""
    import acorninfer.distributed as dist_tp
    from acorninfer.models.qwen3_tp import Qwen3TPForCausalLM, build_shard_fn
    from acorninfer.weights import load_safetensors_sharded

    dist_tp.setup(rank, world, f"file://{init_file}")
    try:
        cfg = Qwen3Config.tiny()
        shard_fn = build_shard_fn(cfg, rank, world)
        state = load_safetensors_sharded(
            ckpt_dir, shard_fn, rank, world,
            device=f"cuda:{rank}", dtype=torch.bfloat16,
        )
        model = Qwen3TPForCausalLM(cfg, rank, world).to(
            device=f"cuda:{rank}", dtype=torch.bfloat16)
        model.load_state_dict(state, strict=True)
        result = _decode_protocol(model, f"cuda:{rank}")
        result["rank"] = rank
        with open(out_json, "w") as f:
            json.dump(result, f)
        torch.distributed.barrier()
    finally:
        dist_tp.destroy()


@pytest.fixture
def ckpt_and_ref(tmp_path):
    """单文件 checkpoint(全量 seed0 tiny)→ [(checkpoint 路径, TP=1 参照)]。"""
    torch.manual_seed(0)
    full = Qwen3ForCausalLM(Qwen3Config.tiny())
    sd = {k: v.contiguous() for k, v in full.state_dict().items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    ref = _decode_protocol(full.to(device="cuda:0", dtype=torch.bfloat16), "cuda:0")
    return tmp_path / "model.safetensors", ref


def _logits_tensor(result, phase):
    return torch.tensor(result[phase]["logits"], dtype=torch.float32)


def test_tp2_eager_decode_matches_full(ckpt_and_ref, tmp_path):
    """TP=2:逐步 logits TOL;非平局步 argmax 逐位;rank 间与页表全同。"""
    ckpt, ref = ckpt_and_ref
    world = 2
    ctx = mp.get_context("spawn")
    init_file = tmp_path / "dist_init"
    procs = []
    for rank in range(world):
        p = ctx.Process(
            target=_tp_worker,
            args=(rank, world, str(init_file), str(ckpt.parent),
                  str(tmp_path / f"tp_rank{rank}.json")),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=600)
        assert p.exitcode == 0, f"rank exitcode={p.exitcode}"

    tp0 = json.load(open(tmp_path / "tp_rank0.json"))
    tp1 = json.load(open(tmp_path / "tp_rank1.json"))

    # rank 间 SPMD 全同:argmax / 页表 / 长度逐位一致
    for phase in ("p1", "p2", "p3"):
        assert tp0[phase]["tokens"] == tp1[phase]["tokens"], f"{phase} rank 间 argmax 分叉"
    assert tp0["tables"] == tp1["tables"]
    assert tp0["lens"] == tp1["lens"]
    # 页表与参照逐位一致(分片 KV 不得导致 rank 侧决策异于全宽)
    assert tp0["tables"] == ref["tables"]
    assert tp0["lens"] == ref["lens"]

    excluded = []
    for phase, steps in (("p1", P1_STEPS), ("p2", P2_STEPS), ("p3", P3_STEPS)):
        got = _logits_tensor(tp0, phase)          # [steps, B, vocab]
        want = _logits_tensor(ref, phase)
        # 逐步 logits TOL(标定常数见文件尾)
        torch.testing.assert_close(got, want, **_TP_LOGITS_TOL)
        # 非平局步 argmax 逐位:参照 top-2 差距 >= margin 的步才断言
        top2 = want.topk(2, dim=-1).values
        gap = top2[..., 0] - top2[..., 1]         # [steps, B]
        tp_tok = torch.tensor(tp0[phase]["tokens"], dtype=torch.long)
        ref_tok = want.argmax(dim=-1)
        strong = gap >= _TP_TIE_MARGIN
        assert torch.equal(tp_tok[strong], ref_tok[strong]), (
            f"{phase}: 强 gap 步 argmax 分叉(非平局步不应翻)"
        )
        weak = (~strong).sum().item()
        if weak:
            excluded.append(f"{phase}:{weak} 步")
    # 平局防护报告(不计红):margin 内的步只受 TOL 约束
    if excluded:
        print(f"\n[平局防护] margin={_TP_TIE_MARGIN} 内未断言 argmax 的步: {excluded}")


# ---- 容差标定(TOL 协议;按实测写死,再触发即红)----
# TP=2 与 TP=1 的数值差来源(结构性,不可消除):o/down 行并行两路 K/2 部分
# GEMM + bf16 allreduce,对比单次 K 全长 GEMM 的归约序差;列切 GEMM(qkv/
# gate/up/lm_head)K 维不变,同序无差。噪声经残差流复合。
# 标定:2026-10-31,强制流协议(seed0 tiny,本文件 _decode_protocol 全协议步
# × 每步全 [B,256] logits,TP=2 双卡对单卡参照逐位取 max|diff|;测量脚本
# 不入库,口径即本文件协议):
# 实测 ALL max|diff| = 0.009766(≈1.25 bf16 ulp @|logit|≈2);argmax 分叉 0;
# gap<0.08 的平局位 21 个。
_TP_LOGITS_TOL = dict(rtol=1.6e-2, atol=1.2e-2)  # 实测 0.009766 之上留 ~20% 裕量;真 bug(错头分片)差 O(1) 必红
_TP_TIE_MARGIN = 8e-2                             # ≈ 8 × 实测 max|diff|;参照 top-2 差距低于此的步不主张 argmax 逐位
