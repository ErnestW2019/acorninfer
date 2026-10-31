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


@pytest.mark.gpu
@pytest.mark.parametrize("fused", [True, False])
def test_forward_decode_prepared_bundle_bitwise(tiny_model, monkeypatch, fused):
    """prepared 四件套路径 ≡ 自 prepare 路径(逐位);且传入 prepared 时
    forward_decode 内部零 ensure 调用 —— 否则 graph 捕获会冻结 CPU 源地址。

    关键时序:pre-write 状态(pre/长度/scatter 目标)必须在参考步**之前**
    取 —— decode 一步会提交 _lens(L → L+1),之后同状态无法再复现。
    """
    import acorninfer.cache.pager as pager_mod
    from acorninfer.ops.paged_attention import build_tables

    model = tiny_model.to("cuda")
    torch.manual_seed(0)
    model.attention_backend = "triton"
    model.fused_ops = fused
    alloc = BlockAllocator(32)  # 2 序列:ceil(5/4)+ceil(9/4)+2 追加位 = 7 页,富余
    cache = PagedKVCache(
        model.cfg.num_hidden_layers,
        model.cfg.num_key_value_heads,
        model.cfg.head_dim,
        4,
        alloc,
        "cuda",
        torch.bfloat16,
    )
    views, tokens = [], []
    with torch.inference_mode():
        for L in (5, 9):
            ids = torch.randint(0, 256, (1, L), device="cuda")
            view = PagedCacheView(cache)
            model(ids, view)  # prefill:写入 L 个 token 的 KV
            views.append(view)
            tokens.append(int(ids[0, -1]))

        # pre 状态与 scatter 目标必须在参考步**之前**取(理由见 docstring 关键时序)。
        pre = [int(v.seq_len) for v in views]
        targets = [v.ensure_append_capacity(0) for v in views]
        x = torch.tensor([[t] for t in tokens], device="cuda")
        ref = model.forward_decode(x, views)  # eager 自 prepare

        # runner 同款图外 prepare → prepared(四件套,build_tables 返回 (tables, lens))
        calls = {"n": 0}
        real_ensure = pager_mod.PagedCacheView.ensure_append_capacity

        def spy_ensure(self, layer_idx):
            calls["n"] += 1
            return real_ensure(self, layer_idx)

        monkeypatch.setattr(pager_mod.PagedCacheView, "ensure_append_capacity", spy_ensure)
        tables, lens = build_tables(views, 4, "cuda", lens_override=[p + 1 for p in pre])
        prepared = (
            torch.tensor(pre, device="cuda"),
            tables,
            lens,
            (torch.tensor([p for p, _ in targets], device="cuda", dtype=torch.long),
             torch.tensor([s for _, s in targets], device="cuda", dtype=torch.long)),
        )
        out = model.forward_decode(x, views, prepared=prepared)
    assert calls["n"] == 0  # 测试自己的 ensure 在 spy 装配前;prepared 路径内部必须零调用
    assert torch.equal(out, ref)  # 与 eager 自 prepare 逐位一致
    for v in views:
        for blk in v.table:
            alloc.free(blk)
    alloc.assert_balanced()


@pytest.mark.gpu
@pytest.mark.parametrize("fused", [True, False])
def test_forward_prefill_batch_matches_sequential(tiny_model, fused):
    """批量 prefill(M4.1)vs 逐条 forward:per-row 末位 logits 紧容差一致;
    KV 写入逐行等长且页表一致(pad 永不入 KV、行间零串扰);fused 开关与
    attention_backend 均不影响 prefill 分支(现状即如此),参数化 + 非默认
    后端取值钉住这一解耦。

    参考腿逐条走现有 forward(num_logits_to_keep=1),独立 allocator/cache
    (页表互不污染);两腿 allocator 同构,LIFO 分配序一致 → 页表可逐位对拍。
    """
    model = tiny_model.to("cuda")
    torch.manual_seed(0)
    model.attention_backend = "triton"   # prefill 与后端解耦;取一个非默认值
    model.fused_ops = fused
    lens = [5, 9, 3]
    gen = torch.Generator(device="cuda").manual_seed(11)
    prompts = [torch.randint(0, 256, (1, L), generator=gen, device="cuda")
               for L in lens]
    Lmax = max(lens)
    ids = torch.zeros(3, Lmax, dtype=torch.long, device="cuda")  # 右 padding
    # 批量腿
    alloc = BlockAllocator(32)
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
    for i, p in enumerate(prompts):
        ids[i, : lens[i]] = p
    with torch.inference_mode():
        batch_logits = model.forward_prefill_batch(ids, views, real_lens=lens)
    # 逐条参考腿(独立 cache/views,页表互不污染)
    alloc2 = BlockAllocator(32)
    cache2 = PagedKVCache(
        model.cfg.num_hidden_layers,
        model.cfg.num_key_value_heads,
        model.cfg.head_dim,
        4,
        alloc2,
        "cuda",
        torch.bfloat16,
    )
    ref, ref_views = [], []
    with torch.inference_mode():
        for i, p in enumerate(prompts):
            v = PagedCacheView(cache2)
            ref.append(model(p, v, num_logits_to_keep=1)[0, -1])
            ref_views.append(v)
    ref = torch.stack(ref)
    assert batch_logits.shape == (3, 256)
    # 容差标定(TOL 协议):2026-10-31,命令 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python
    # -m pytest tests/test_forward_decode.py -k prefill_batch -v -s` +
    # 探针脚本打印逐元素 slack,实测 max|diff| = 0.0078125(= 2^-7,恰 1 个 bf16 ulp;
    # argmax 处 ref=-1.039,batch=-1.031;407/768 元素有 ulp 级差异 —— 合批 GEMM 与
    # [1,L] GEMM 选核/归约序差异的固有舍入)。按协议取实测 2-3 倍:atol=6e-3
    # (≈0.77× 实测 max|diff|,覆盖小 |ref| 元素的 ≥1 ulp)+ rtol=1.6e-2(≈2 bf16
    # ulp 的相对界,覆盖大 |ref|);对 |ref|<0.67(本 logits 分布均值 0.46)严于
    # 全局 TOL_BF16(1e-2/1e-2)。最小 slack 0.0025,配置/驱动变化后按协议重标定。
    tol = dict(rtol=1.6e-2, atol=6e-3)
    torch.testing.assert_close(batch_logits, ref, **tol)
    worst = max_abs_diff(batch_logits, ref)
    print(f"\n[forward_prefill_batch fused={fused} vs sequential max|diff| = {worst:.6f}]")  # 标定输出
    for i in range(3):
        assert views[i]._lens == ref_views[i]._lens   # 逐层 KV 长度一致(pad 未入 KV)
        assert views[i].table == ref_views[i].table   # 页表一致(无跨行写)
    for v in views:
        for blk in v.table:
            alloc.free(blk)
    for v in ref_views:
        for blk in v.table:
            alloc2.free(blk)
    alloc.assert_balanced()
    alloc2.assert_balanced()
