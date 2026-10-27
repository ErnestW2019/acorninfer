"""批量 decode 正确性:
- 单序列 forward_decode == 现有 forward 路径(逐位;同形状同内核);
- 批量 forward_decode 的 token 序列 == 逐序列单独 decode(token 序列是真正的门;
  logits 允许批处理内核的 ulp 级差异,但 tiny CPU 模型上预期逐位一致)。
"""

import pytest
import torch

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
