"""Decode CUDA graph runner(M4):批大小分桶预捕获 + 静态缓冲重放。

静态化契约(与 eager 逐位的根据):
- 图体 = model.forward_decode(ids, views, prepared=bundle)。bundle 四件
  (positions/tables/lens/scatter)全部来自静态缓冲,每步重放前原位覆写;
- 图内 python(局部导入、asserts、num_splits_for、scatter_decode 的
  _lens 提交)只在捕获期执行一次,重放不执行 —— _lens 提交由 runner 在
  prepare 阶段代行(_commit_lens),捕获期那次与代行同值,无副作用;
- padding 行:views 尾部补 dummy view;scatter 索引指向 scratch 块(init
  期从 allocator 常驻分配,永不读);输出由调用方丢弃;缩批后旧 padding 行
  残留的真 (page,slot) 必须每步压回 scratch(decode_batch 重置),否则引擎
  复用退休页时 replay scatter 会写花活序列 KV;
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
    # 扁平 GPU 静态块(M4.1):按 dtype 两块,保引用防 GC;下面六个静态字段
    # 全是它们的切片视图(共享存储 = graph 捕获地址,warmup/capture/replay
    # 三处地址一致)。布局:long = ids|positions|pages|slots 各 bucket 段;
    # int32 = lens 一段 + tables [bucket, W] 一段。
    flat_long: torch.Tensor            # [4*bucket] long
    flat_i32: torch.Tensor             # [bucket*(1+W)] int32
    ids: torch.Tensor        # [B, 1] long = flat_long[0:bucket] 视图
    positions: torch.Tensor  # [B] long     = flat_long[bucket:2b] 视图
    tables: torch.Tensor     # [B, W] int32 = flat_i32[bucket:] 视图;W = allocator.total
    lens: torch.Tensor       # [B] int32    = flat_i32[0:bucket] 视图
    pages: torch.Tensor      # [B] long     = flat_long[2b:3b] 视图
    slots: torch.Tensor      # [B] long     = flat_long[3b:4b] 视图
    views: list              # [B] dummy PagedCacheView(捕获期 views 实参)
    graph: torch.cuda.CUDAGraph
    logits: torch.Tensor | None  # [B, vocab] float32;捕获后由图冻结
    # pinned CPU 镜像(M4.1):与 flat 同构布局,每步 CPU 侧填满(含 tables
    # 构建与 padding 重写),然后每 dtype 一次 H2D —— 6-7 次小传输 → 2 次
    pin_long: torch.Tensor | None = None      # [4*bucket] long 镜像
    pin_i32: torch.Tensor | None = None       # [bucket*(1+W)] int32 镜像
    pin_tables: torch.Tensor | None = None    # [B, W] = pin_i32[bucket:] 视图


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
        self.last_logits = None

    # ---- 构造期:状态与捕获 ----

    def _build_state(self, bucket: int) -> _BucketState:
        dev = self.device
        W = self.allocator.total
        # 扁平 GPU 块:静态字段 = 切片视图(共享存储 = graph 捕获地址)。
        # GPU 侧初值逐字段对齐重构前(pages=scratch / lens=1 / 其余 0):draft
        # 只种 pinned 镜像、GPU 侧留全零,warmup/capture 的 scatter 落点会打到
        # 池内真 0 页而非 scratch,违背模块 docstring 的 scratch 契约 —— 故
        # GPU/镜像两侧同种(捕获期 KV 写落点与旧实现逐字节一致)。
        flat_long = torch.zeros(4 * bucket, dtype=torch.long, device=dev)
        flat_i32 = torch.zeros(bucket * (1 + W), dtype=torch.int32, device=dev)
        flat_long[2 * bucket:3 * bucket].fill_(self.scratch)  # pages 初值 = scratch
        flat_i32[0:bucket].fill_(1)                           # lens 初值 = 1
        st = _BucketState(
            bucket=bucket,
            flat_long=flat_long,
            flat_i32=flat_i32,
            ids=flat_long[0:bucket].view(bucket, 1),
            positions=flat_long[bucket:2 * bucket],
            pages=flat_long[2 * bucket:3 * bucket],
            slots=flat_long[3 * bucket:4 * bucket],
            lens=flat_i32[0:bucket],
            tables=flat_i32[bucket:].view(bucket, W),
            views=[PagedCacheView(self.cache, [self.scratch]) for _ in range(bucket)],
            graph=torch.cuda.CUDAGraph(),
            logits=None,
        )
        # pinned CPU 镜像(同构布局);平台不支持 pinned 时回退普通内存。
        #(draft 用 torch.empty:ids/positions/slots/lens 的 padding 区成为
        # 未初始化内存,首发 H2D 即上传垃圾 —— embed 越界查表 / scatter slot
        # 越界写页风险,且违背逐字节契约。必须零初始化。)
        st.pin_long = self._pinned(4 * bucket, torch.long)
        st.pin_i32 = self._pinned(bucket * (1 + W), torch.int32)
        st.pin_long[2 * bucket:3 * bucket].fill_(self.scratch)  # padding pages 常量场
        st.pin_i32[0:bucket].fill_(1)                           # padding lens 恒 1
        st.pin_tables = st.pin_i32[bucket:].view(bucket, W)
        st.pin_tables[:, 0].fill_(self.scratch)                 # padding 行 tables 第 0 列
        return st

    @staticmethod
    def _pinned(numel: int, dtype) -> torch.Tensor:
        try:
            return torch.zeros(numel, dtype=dtype, pin_memory=True)
        except RuntimeError:   # 平台/驱动不支持 pinned:回退(收益略降,正确性不变)
            return torch.zeros(numel, dtype=dtype)

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
        assert len(seqs) == B, "seqs/views 必须一一对应"
        st = self.states[min(b for b in BUCKETS if b >= B)]
        bk = st.bucket
        targets = [v.ensure_append_capacity(0) for v in views]
        pre = [int(v.seq_len) for v in views]
        new_len = [p + 1 for p in pre]
        # CPU 镜像填充(pinned):真行 [:B];每 dtype 一块扁平镜像,末尾单发
        st.pin_long[0:B] = torch.tensor([s.generated_ids[-1] for s in seqs],
                                        dtype=torch.long)
        st.pin_long[bk:bk + B] = torch.tensor(pre, dtype=torch.long)
        st.pin_long[2 * bk:2 * bk + B] = torch.tensor([p for p, _ in targets],
                                                      dtype=torch.long)
        st.pin_long[3 * bk:3 * bk + B] = torch.tensor([s for _, s in targets],
                                                      dtype=torch.long)
        st.pin_i32[0:B] = torch.tensor(new_len, dtype=torch.int32)
        st.pin_tables.zero_()                                # 全清(B 真行 + padding)
        build_tables(views, self.page_size, self.device,
                     lens_override=new_len, out=st.pin_tables)   # 直填 CPU 镜像
        # padding 行每步压回常量场(draft 声称"init 常量、[:B] 之外不动"不成立:
        # 缩批后 [B:] 残留上一步真 (page,slot),不压回则 replay scatter 写花
        # 已退休复用的页 —— 对拍门 step 3/4 专钉此回归;旧代码逐字段 fill_/
        # zero_ 的镜像侧同职责。lens[B:] 恒 init 1(写真行只切 [:B]),ids/
        # positions[B:] 陈旧无害(旧代码同样不重置,字节写历史一致)。
        st.pin_long[2 * bk + B:3 * bk].fill_(self.scratch)
        st.pin_long[3 * bk + B:4 * bk].zero_()
        st.pin_tables[B:, 0] = self.scratch
        # 单发 H2D(每 dtype 一次;切片视图地址 = 捕获地址,replay 原位可见)
        st.flat_long.copy_(st.pin_long)
        st.flat_i32.copy_(st.pin_i32)
        self._commit_lens(views, new_len)
        st.graph.replay()
        # 图输出缓冲原址覆写:仅本步有效;parity 测试的对拍观察点
        self.last_logits = st.logits
        # 合批 argmax:B 次 D2H 同步 → 1 次(图外残差清理,M4.1 Task 1)
        return [int(t) for t in st.logits[:B].argmax(dim=1).tolist()]

    def _commit_lens(self, views, new_len) -> None:
        # pager 私有状态的图外代行(pager 本里程碑禁改):与 scatter_decode 的
        # per-layer 提交同值同义;重放体内无 python,提交只能发生在图外。
        for v, n in zip(views, new_len):
            for li in range(self.num_layers):
                v._lens[li] = n
