"""DecodeGraphRunner 对拍:graph 路径 vs eager 路径。

门(双环境独立 —— 同 seed 造同权重的两套 model/cache/allocator,页表互不污染):
1. 4 序列(桶 4 满批)× 6 步:每步 argmax token 逐位;logits assert_close(TOL_BF16;
   缩批步两路 B 不同(3 vs 桶 4 的 GEMM 形状差)允许低比特漂移,token 不许翻);
2. step 3(0 起)退休 2 序列 + 同 seed 新序列入批:缩批 4→3(桶仍 4)后 row 3 转为
   padding,但残留 row-3 序列的真 (page,slot);新序列经 allocator LIFO 恰好
   拿回该页 —— replay 的 padding scatter 若不每步重置回 scratch,会写花新
   序列刚拿到的页(静默 KV 腐败;本门钉死该回归);
3. step 4 再退 1 序列 → 桶 4→2 切换,余下序列仍逐位;
4. 每 runner 步后 views 的 _lens 与 eager 步后逐层相等(_commit_lens 契约);
5. 收尾:显式 free 全部测试页后,runner 侧 alloc 恰 1 块 live(scratch 常驻),
   eager 侧 assert_balanced 零 live。
"""
import pytest
import torch
from types import SimpleNamespace

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import Qwen3Config
from acorninfer.engine.graph_runner import DecodeGraphRunner
from acorninfer.models.qwen3 import Qwen3ForCausalLM
from triton_utils import TOL_BF16

# LENS 各跨不同页数;LENS[3]=7 使残留 scatter 位 (slot 2) 落在新序列已 prefill 的区间内,回归门才咬得到
LENS = [5, 9, 13, 7]


def _env(seed=0):
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).to("cuda", torch.bfloat16).eval()
    model.attention_backend = "triton"
    model.fused_ops = True
    alloc = BlockAllocator(64)
    cache = PagedKVCache(2, 2, 16, page_size=4, allocator=alloc,
                         device="cuda", dtype=torch.bfloat16)
    return model, alloc, cache


def _prefill(model, cache, L, seed):
    # 草稿笔误修正:torch.Generator() 默认 CPU,randint(device="cuda") 拒用 ——
    # 显式造 cuda generator(draft bug;双环境同 seed,对拍语义不变)。
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, 256, (1, L), generator=g, device="cuda")
    view = PagedCacheView(cache, [])
    model(ids, view)
    return view, int(ids[0, -1])


@pytest.mark.gpu
def test_runner_matches_eager_and_bucket_switch():
    model, alloc, cache = _env()
    runner = DecodeGraphRunner(model, cache, alloc)
    views, tokens = map(list, zip(*[_prefill(model, cache, L, 100 + i)
                                    for i, L in enumerate(LENS)]))

    # eager 参考环境(同权重同 prompt,独立页表)
    model2, alloc2, cache2 = _env()
    views2, _ = map(list, zip(*[_prefill(model2, cache2, L, 100 + i)
                                for i, L in enumerate(LENS)]))

    for step in range(6):
        if step == 3:
            # 缩批 4→3(桶仍 4)+ 页立即复用(引擎路径:退休即 free,allocator
            # LIFO 立刻把页发给下一个 prefill)。row 3 的序列退休,其末次
            # (page,slot) 残留为 padding 行;先 free row 2、后 free row 3 是
            # 刻意的 —— LIFO 令 row 3 残留页成为新序列首块,回归门才咬得到。
            # 两环境同步退休/入批,对拍保持同构。
            rv_pad = views.pop()   # row 3 序列
            rv_mid = views.pop()   # row 2 序列
            for b in rv_mid.table:
                alloc.free(b)
            for b in rv_pad.table:
                alloc.free(b)
            tokens.pop()
            tokens.pop()
            nv, nt = _prefill(model, cache, 7, 200)
            views.append(nv)
            tokens.append(nt)
            # eager 环境同步(无 runner/padding,仅保对拍同构)
            rv2_pad = views2.pop()
            rv2_mid = views2.pop()
            for b in rv2_mid.table:
                alloc2.free(b)
            for b in rv2_pad.table:
                alloc2.free(b)
            nv2, _ = _prefill(model2, cache2, 7, 200)
            views2.append(nv2)
        if step == 4:  # 纯缩批 3→2:桶切换 4→2;退休页即刻 free,保收尾收支平衡
            rv = views.pop()
            for b in rv.table:
                alloc.free(b)
            rv2 = views2.pop()
            for b in rv2.table:
                alloc2.free(b)
            tokens.pop()
        x = torch.tensor([[t] for t in tokens], device="cuda")
        eager_out = model2.forward_decode(x, views2)
        stubs = [SimpleNamespace(generated_ids=[t]) for t in tokens]
        graph_tokens = runner.decode_batch(stubs, views)
        torch.testing.assert_close(runner.last_logits[: len(views)], eager_out, **TOL_BF16)
        assert graph_tokens == [int(r.argmax()) for r in eager_out]
        for v, v2 in zip(views, views2):
            assert v._lens == v2._lens  # _commit_lens 契约:全层逐层等长
        tokens = graph_tokens

    for v in views:
        for b in v.table:
            alloc.free(b)
    for v in views2:
        for b in v.table:
            alloc2.free(b)
    assert alloc.num_live == 1  # 仅 runner 常驻 scratch;其余零 live
    alloc2.assert_balanced()
