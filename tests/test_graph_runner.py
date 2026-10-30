"""DecodeGraphRunner 对拍:graph 路径 vs eager 路径。

门(双环境独立 —— 同 seed 造同权重的两套 model/cache/allocator,页表互不污染):
1. 3 序列(桶 4)× 6 步:每步 argmax token 逐位;logits assert_close(TOL_BF16;
   前 3 步两路 B 不同(3 vs 桶 4 的 GEMM 形状差)允许低比特漂移,token 不许翻);
2. 第 4 步起丢 1 序列(模拟 EOS 退休)→ 桶 4→2 切换,余下序列仍逐位;
3. 每 runner 步后 views 的 _lens 与 eager 步后逐层相等(_commit_lens 契约);
4. 收尾:显式 free 全部测试页后,runner 侧 alloc 恰 1 块 live(scratch 常驻),
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

LENS = [5, 9, 13]


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

    retired = []  # 退休视图的页由测试自行 free(引擎外无 scheduler 代劳)
    for step in range(6):
        if step == 3:  # 模拟 EOS 退休:批 3→2,桶 4→2
            retired.append((alloc, views.pop()))
            retired.append((alloc2, views2.pop()))
            tokens.pop()  # 草稿笔误修正:退休序列的下一步 token 同步丢弃(view i ↔ token i 对齐)
        x = torch.tensor([[t] for t in tokens], device="cuda")
        eager_out = model2.forward_decode(x, views2)
        stubs = [SimpleNamespace(generated_ids=[t]) for t in tokens]
        graph_tokens = runner.decode_batch(stubs, views)
        torch.testing.assert_close(runner.last_logits[: len(views)], eager_out, **TOL_BF16)
        assert graph_tokens == [int(r.argmax()) for r in eager_out]
        for v, v2 in zip(views, views2):
            assert v._lens == v2._lens  # _commit_lens 契约:全层逐层等长
        tokens = graph_tokens

    for a, v in retired + [(alloc, v) for v in views] + [(alloc2, v) for v in views2]:
        for b in v.table:
            a.free(b)
    assert alloc.num_live == 1  # 仅 runner 常驻 scratch;其余零 live
    alloc2.assert_balanced()
