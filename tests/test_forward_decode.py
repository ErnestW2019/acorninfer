"""批量 decode 正确性:
- 单序列 forward_decode == 现有 forward 路径(逐位;同形状同内核);
- 批量 forward_decode 的 token 序列 == 逐序列单独 decode(token 序列是真正的门;
  logits 允许批处理内核的 ulp 级差异,但 tiny CPU 模型上预期逐位一致)。
"""

import pytest
import torch

from triton_utils import TOL_BF16, max_abs_diff

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache


@pytest.fixture
def env(tiny_model):
    alloc = BlockAllocator(32)
    cache = PagedKVCache(
        tiny_model.cfg.num_hidden_layers,
        tiny_model.cfg.num_key_value_heads,
        tiny_model.cfg.head_dim,
        4,
        alloc,
        "cpu",
        torch.bfloat16,
    )
    return tiny_model, cache, alloc


def test_decode_single_matches_forward_path(env):
    model, cache, alloc = env
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 5))
    with torch.inference_mode():
        ref_view = PagedCacheView(cache)
        ref = model(ids[:, :4], ref_view)          # 旧路径 prefill
        ref = model(ids[:, 4:], ref_view)          # 旧路径 decode 一步
        v = PagedCacheView(cache)
        _ = model(ids[:, :4], v)                    # 同一 prefill
        new = model.forward_decode(ids[:, 4:], [v])  # [1, vocab]
    assert torch.equal(new[0], ref[0, -1])
    for b in v.table:
        alloc.free(b)
    for b in ref_view.table:
        alloc.free(b)


def test_decode_batch_matches_individual_token_sequences(env):
    model, cache, alloc = env
    torch.manual_seed(1)
    lens = [3, 5, 5]  # 两条等长(同位置的常见 bench 形态)+ 一条异长
    prompts = [torch.randint(0, 256, (1, n)) for n in lens]
    views_indiv = [PagedCacheView(cache) for _ in range(3)]
    views_batch = [PagedCacheView(cache) for _ in range(3)]
    indiv_seqs = [[] for _ in range(3)]
    batch_seqs = [[] for _ in range(3)]
    with torch.inference_mode():
        for p, vi, vb in zip(prompts, views_indiv, views_batch):
            model(p[:, :-1], vi)   # prefill(留 1 个 token 做 decode 输入)
            model(p[:, :-1], vb)
        for _ in range(4):
            # 逐序列:自己上一步的 token 喂自己
            for i, (p, vi) in enumerate(zip(prompts, views_indiv)):
                prev = p[0, -1] if not indiv_seqs[i] else indiv_seqs[i][-1]
                logits = model(torch.tensor([[int(prev)]]), vi)
                indiv_seqs[i].append(int(logits[0, -1].argmax()))
            # 批量:forward_decode 一次吃 B 条
            prevs = [p[0, -1] if not batch_seqs[i] else batch_seqs[i][-1]
                     for i, p in enumerate(prompts)]
            x = torch.tensor([[int(t)] for t in prevs])
            logits = model.forward_decode(x, views_batch)
            for i in range(3):
                batch_seqs[i].append(int(logits[i].argmax()))
    assert indiv_seqs == batch_seqs
    for v in views_indiv + views_batch:
        for blk in v.table:
            alloc.free(blk)


@pytest.mark.gpu
@pytest.mark.parametrize("backend,fused", [
    ("sdpa", False), ("sdpa", True), ("triton", False), ("triton", True),
])
def test_decode_backends_logits_equivalent(tiny_model, backend, fused):
    """四开关组合对拍:每组合全新 allocator/cache/views、同 ids prefill、同串
    强制 token decode 5 步,逐步与 (sdpa, False) baseline 的 logits 比对 TOL_BF16。

    平局分叉防护:decode 输入是预生成的同一串 token,不是 argmax 链 ——
    argmax 链会在 bf16 平局处分叉,使各组合第 2 步以后比较的是不同上下文
    的 logits(假红/假绿都可能)。
    triton 组合另跑一遍全程(全新 views)断言逐位相等(kernel 确定性)。
    """
    model = tiny_model.to("cuda")
    torch.manual_seed(0)
    lens = [11, 23, 47]  # 异长且均跨多页、尾页不对齐;47+5=52 是下方 alloc 容量算式的输入
    prompts = [torch.randint(0, 256, (1, n), device="cuda") for n in lens]
    decode_tokens = torch.randint(0, 256, (5, 3, 1), device="cuda")

    def run_combo():
        # 全新 allocator/cache/views:页槽不串运行,views 消费后手动还块
        alloc = BlockAllocator(64)  # 3 序列 × ceil(52/4)=13 页,富余
        cache = PagedKVCache(
            model.cfg.num_hidden_layers,
            model.cfg.num_key_value_heads,
            model.cfg.head_dim,
            4,
            alloc,
            "cuda",
            torch.bfloat16,
        )
        views = [PagedCacheView(cache) for _ in range(3)]
        with torch.inference_mode():
            for p, v in zip(prompts, views):
                model(p, v)  # prefill:逐条同 ids,长度互异
            steps = [model.forward_decode(decode_tokens[t], views) for t in range(5)]
        for v in views:
            for blk in v.table:
                alloc.free(blk)
        alloc.assert_balanced()
        return torch.stack(steps)  # [5, 3, vocab]

    model.attention_backend = "sdpa"
    model.fused_ops = False
    baseline = run_combo()  # (sdpa, False)

    model.attention_backend = backend
    model.fused_ops = fused
    steps = run_combo()
    assert steps.shape == (5, 3, 256)
    for t in range(5):
        torch.testing.assert_close(steps[t], baseline[t], **TOL_BF16)
    worst = max(max_abs_diff(steps[t], baseline[t]) for t in range(5))
    print(f"\n[{backend} fused={fused} vs sdpa baseline max|diff| = {worst:.6f}]")  # 标定输出
    if backend == "triton":
        again = run_combo()
        assert torch.equal(steps, again)  # kernel 确定性
