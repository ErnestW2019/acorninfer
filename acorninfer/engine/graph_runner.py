"""Decode CUDA graph runner(M4):批大小分桶预捕获 + 静态缓冲重放。

静态化契约(与 eager 逐位的根据):
- 图体 = model.forward_decode(ids, views, prepared=bundle)。bundle 四件
  (positions/tables/lens/scatter)全部来自静态缓冲,每步重放前原位覆写;
- 图内 python(局部导入、asserts、num_splits_for、scatter_decode 的
  _lens 提交)只在捕获期执行一次,重放不执行 —— _lens 提交由 runner 在
  prepare 阶段代行(_commit_lens),捕获期那次与代行同值,无副作用;
- padding 行:views 尾部补 dummy view;scatter 索引指向 scratch 块(init
  期从 allocator 常驻分配,永不读);输出由调用方丢弃;
- 捕获在 runner 构造期完成(全 dummy views):请求路径零捕获抖动;捕获前
  3 次 side-stream warmup 与正式步计算同值(确定性 + 同 (page,slot) 写同
  k/v 的幂等性)。
预期用途:attention_backend="triton"、全 greedy(temp=0)、B ≤ 8;
不满足时由 Executor 分流回 eager。

scratch 契约:init 期从 allocator 常驻分配 1 块,永不释放 —— KV 排空后的
收支平衡判据因此为"恰 1 块 live"(Engine.assert_balanced 按此实现,Task 4);
本 runner 单测按同一契约以 num_live == 1 收尾。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from acorninfer.cache.pager import PagedCacheView
from acorninfer.ops.paged_attention import build_tables

BUCKETS = (1, 2, 4, 8)
WARMUP_STEPS = 3


@dataclass
class _BucketState:
    bucket: int
    ids: torch.Tensor        # [B, 1] long
    positions: torch.Tensor  # [B] long
    tables: torch.Tensor     # [B, W] int32;W = allocator.total(单序列页数上界)
    lens: torch.Tensor       # [B] int32
    pages: torch.Tensor      # [B] long
    slots: torch.Tensor      # [B] long
    views: list              # [B] dummy PagedCacheView(捕获期 views 实参)
    graph: torch.cuda.CUDAGraph
    logits: torch.Tensor | None  # [B, vocab] float32;捕获后由图冻结


class DecodeGraphRunner:
    def __init__(self, model, cache, allocator):
        assert getattr(model, "attention_backend", "sdpa") == "triton", \
            "DecodeGraphRunner 仅支持 triton 后端"
        self.model = model
        self.cache = cache
        self.allocator = allocator
        self.num_layers = model.cfg.num_hidden_layers
        self.page_size = cache.page_size
        self.device = next(model.parameters()).device
        self.scratch = allocator.allocate()  # 常驻;padding 行 KV 落点
        self.states = {b: self._build_state(b) for b in BUCKETS}
        for st in self.states.values():
            self._capture(st)

    # ---- 构造期:状态与捕获 ----

    def _build_state(self, bucket: int) -> _BucketState:
        dev = self.device
        st = _BucketState(
            bucket=bucket,
            ids=torch.zeros(bucket, 1, dtype=torch.long, device=dev),
            positions=torch.zeros(bucket, dtype=torch.long, device=dev),
            tables=torch.zeros(bucket, self.allocator.total, dtype=torch.int32, device=dev),
            lens=torch.ones(bucket, dtype=torch.int32, device=dev),
            pages=torch.full((bucket,), self.scratch, dtype=torch.long, device=dev),
            slots=torch.zeros(bucket, dtype=torch.long, device=dev),
            views=[PagedCacheView(self.cache, [self.scratch]) for _ in range(bucket)],
            graph=torch.cuda.CUDAGraph(),
            logits=None,
        )
        return st

    def _capture(self, st: _BucketState) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(WARMUP_STEPS):
                self.model.forward_decode(st.ids, st.views, prepared=self._bundle(st))
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(st.graph):
            st.logits = self.model.forward_decode(st.ids, st.views, prepared=self._bundle(st))

    def _bundle(self, st: _BucketState):
        return (st.positions, st.tables, st.lens, (st.pages, st.slots))

    # ---- 每步路径 ----

    def decode_batch(self, seqs, views):
        """与 Executor.decode_batch 同参;seqs 仅取 generated_ids[-1]。"""
        B = len(views)
        assert 1 <= B <= BUCKETS[-1]
        st = self.states[min(b for b in BUCKETS if b >= B)]
        targets = [v.ensure_append_capacity(0) for v in views]
        pre = [int(v.seq_len) for v in views]
        new_len = [p + 1 for p in pre]
        st.pages[:B].copy_(torch.tensor([p for p, _ in targets], dtype=torch.long))
        st.slots[:B].copy_(torch.tensor([s for _, s in targets], dtype=torch.long))
        st.positions[:B].copy_(torch.tensor(pre, dtype=torch.long))
        st.lens[:B].copy_(torch.tensor(new_len, dtype=torch.int32))
        t, _ = build_tables(views, self.page_size, self.device, lens_override=new_len)
        st.tables.zero_()
        st.tables[:B, : t.shape[1]].copy_(t)
        st.tables[B:, 0] = self.scratch  # padding 行的常量场(zero_ 后重写)
        st.ids[:B].copy_(torch.tensor([[s.generated_ids[-1]] for s in seqs], dtype=torch.long))
        self._commit_lens(views, new_len)
        st.graph.replay()
        self.last_logits = st.logits
        return [int(row.argmax()) for row in st.logits[:B]]

    def _commit_lens(self, views, new_len) -> None:
        # pager 私有状态的图外代行(pager 本里程碑禁改):与 scatter_decode 的
        # per-layer 提交同值同义;重放体内无 python,提交只能发生在图外。
        for v, n in zip(views, new_len):
            for li in range(self.num_layers):
                v._lens[li] = n
