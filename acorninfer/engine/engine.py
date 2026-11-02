"""Engine:请求生命周期 + step 循环。单线程契约(同一时刻一个 step 调用方)。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from acorninfer.cache.pager import BlockAllocator, PagedCacheView, PagedKVCache
from acorninfer.config import QWEN3_EOS_IDS, EngineConfig
from acorninfer.engine.executor import Executor
from acorninfer.engine.graph_runner import DecodeGraphRunner
from acorninfer.engine.scheduler import Scheduler
from acorninfer.engine.sequence import RequestStatus, Sequence
from acorninfer.models.qwen3 import Qwen3ForCausalLM, from_pretrained
from acorninfer.quant.quantize import apply_quant
from acorninfer.sampler import SamplingParams
from acorninfer.tokenizer import Tokenizer


@dataclass
class TokenOutput:
    seq_id: int
    token_id: int
    finished: bool
    stop_reason: str | None = None


class Engine:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        tokenizer: Tokenizer | None,
        *,
        num_blocks: int = 1024,
        page_size: int = 16,
        max_running: int = 8,
        watermark_blocks: int = 8,
        chunked_budget: int | None = None,
        tp=None,
        spec_ngram: int = 0,
        spec_num: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tp = tp  # TPLink | None:None = 单卡原路径(TP=1 逐字节硬门)
        if spec_ngram:
            # spec 直构校验(镜像 EngineConfig.__post_init__;直接构造 Engine
            # 的测试/脚本不走 config,构造期爆掉而非静默错路径)
            if not (1 <= spec_ngram <= 8) or not (1 <= spec_num <= 8):
                raise ValueError(
                    f"spec_ngram must be 1..8 and spec_num 1..8 when enabled, "
                    f"got spec_ngram={spec_ngram}, spec_num={spec_num}")
            if model.attention_backend != "triton":
                raise ValueError(
                    f"spec decode requires attention_backend='triton', "
                    f"got {model.attention_backend!r}")
            if getattr(model, "decode_graph", False):
                raise ValueError(
                    "spec × decode_graph 互斥(v1 eager verify only,spec §2.4)")
        self.spec_ngram = spec_ngram
        self.spec_num = spec_num
        # M8 Task 4 接受率计量(引擎侧步级计数;spec 关闭时恒零值、零开销,
        # 缺省路径逐字节不变):verify_steps = verify 腿步数(seq 级计),
        # accepted = Σ a(greedy_accept 接受数),committed = 实际入流 token
        # (eos 截断可 < a+1),hist = committed/步 分布(键 ≤ m 有界)。
        self.spec_stats = {"verify_steps": 0, "accepted": 0, "committed": 0,
                           "hist": {}}
        self._proposers: dict[int, "object"] = {}  # seq_id → NgramProposer(spec 模式)
        cfg = model.cfg
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.allocator = BlockAllocator(num_blocks)
        self.cache = PagedKVCache(
            cfg.num_hidden_layers,
            # TP:每卡独立 cache 按 local kv 头数构造(pager/kv 零改动;
            # 单卡模型无此属性 → cfg 原值,逐字节不变)
            getattr(model, "num_kv_heads_local", cfg.num_key_value_heads),
            cfg.head_dim,
            page_size,
            self.allocator,
            device,
            dtype,
        )
        self.scheduler = Scheduler(self.allocator, page_size, max_running,
                                   watermark_blocks, chunked_budget)
        if watermark_blocks >= num_blocks:
            raise ValueError(
                f"watermark_blocks ({watermark_blocks}) must be < num_blocks ({num_blocks})"
            )
        self.graph_runner = None
        if getattr(model, "decode_graph", False):
            self.graph_runner = DecodeGraphRunner(model, self.cache, self.allocator)
        self.executor = Executor(model, self.graph_runner, tp=tp)
        self.page_size = page_size
        self.seqs: dict[int, Sequence] = {}
        self._views: dict[int, PagedCacheView] = {}
        self._next_id = 0

    @classmethod
    def from_config(cls, cfg: EngineConfig, tokenizer: Tokenizer) -> "Engine":
        """EngineConfig → Engine(from_pretrained 全量模型,单卡路径)。

        与 server._build_engine 为双构造路径:那边内联本逻辑并多出
        chunked_budget / --tp 分支 —— 改一处必查另一处。
        """
        model = from_pretrained(cfg.model_dir, cfg.device, cfg.dtype)
        # attention_backend/decode_graph 由 from_config 透传到 model,decode 路径
        # 已生效(Task 7);decode_graph 必须在 Engine 构造前置位(graph runner
        # 在 __init__ 捕获各桶)
        model.attention_backend = cfg.attention_backend
        model.fused_ops = cfg.fused_ops
        model.decode_graph = cfg.decode_graph
        if cfg.quant != "none":
            # M7 段 A:装载后现场量化(spec §4 单存储)+ kernel 表装载期 autotune
            # —— 均在 Engine 构造(graph 捕获)之前;quant×tp 互斥在 server.
            # _build_engine 构造期把守(from_config 是单卡路径,无 tp 面)
            apply_quant(model, cfg.quant)
            from acorninfer.ops import quant_gemm  # triton:仅 quant 路径(局部导入同 qwen3 先例)
            quant_gemm.configure(device=cfg.device)
        return cls(
            model,
            tokenizer,
            num_blocks=cfg.num_blocks,
            page_size=cfg.page_size,
            max_running=cfg.max_running,
            watermark_blocks=cfg.watermark_blocks,
            spec_ngram=cfg.spec_ngram,
            spec_num=cfg.spec_num,
        )

    def submit(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        *,
        max_tokens: int,
        eos_ids: frozenset[int] | None = None,
    ) -> int:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        # 拒绝阈值必须是"可调度容量"(total - watermark)而非 total:准入预算
        # 按 free - watermark 结算,落在两者之间的 prompt 会被接受却永远排在
        # 队首饿死(has_pending 永不清零,server/bench 步循环卡死)
        capacity = self.allocator.total - self.scheduler.watermark
        if -(-len(prompt_ids) // self.page_size) > capacity:
            raise ValueError(
                f"prompt needs more KV blocks than the schedulable capacity "
                f"({capacity} = {self.allocator.total} blocks - watermark); "
                f"it could never be admitted"
            )
        seq = Sequence(
            seq_id=self._next_id,
            prompt_ids=list(prompt_ids),
            sampling=sampling,
            max_tokens=max_tokens,
            eos_ids=frozenset(eos_ids) if eos_ids is not None else QWEN3_EOS_IDS,
        )
        self._next_id += 1
        self.seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    @torch.inference_mode()
    def step(self) -> list[TokenOutput]:
        chunks, decodes = self.scheduler.schedule()
        outputs: list[TokenOutput] = []
        try:
            # 按 start 分流(M5):start==0 且全量 → 既有 prefill 路径
            # (byte-identical,零回归面);start==0 截断 → forward_prefill_batch
            # (更短的 prefill,数值同源);start>0 → forward_extend 续读。
            built = []
            part: list[tuple[Sequence, PagedCacheView, int, int]] = []
            ext: list[tuple[Sequence, PagedCacheView, int, int]] = []
            for chunk in chunks:
                seq, start, length = chunk
                if start == 0:
                    # prefill (重)起点:M4.1 同款无条件新建视图 —— 被抢占者的
                    # 旧视图 _lens 残留(_release 只清共享块表),复用会脏写
                    view = PagedCacheView(self.cache, seq.block_table)
                    self._views[seq.seq_id] = view
                else:
                    # 续读(cursor>0):视图由首块建立;KeyError 即不变式破坏,宁可响
                    view = self._views[seq.seq_id]
                if start == 0 and length == len(seq.all_ids):
                    built.append((seq, view))
                elif start == 0:
                    part.append((seq, view, start, length))
                else:
                    ext.append((seq, view, start, length))
            if len(built) == 1:
                # N=1 原路径:byte-identical(零回归面)
                seq, view = built[0]
                token = self.executor.prefill(seq, view)
                seq.num_pages = len(view.table)
                seq.prefill_cursor = len(seq.all_ids)
                self._record(seq, token, outputs)
            elif built:
                # M4.1:同批多序列一次批量 prefill(调度器本就一步全准入)
                tokens = self.executor.prefill_batch(
                    [s for s, _ in built], [v for _, v in built])
                for (seq, view), token in zip(built, tokens):
                    seq.num_pages = len(view.table)
                    seq.prefill_cursor = len(seq.all_ids)
                    self._record(seq, token, outputs)
            if part:
                # 预算截断的首块(cursor==0):走批量 prefill,实长=length;
                # 截断块必不完成(start+length < len(all_ids)),不发 token
                lens = [ln for (_, _, _, ln) in part]
                ids = torch.zeros(len(part), max(lens), dtype=torch.long,
                                  device=self.executor.device)
                views = []
                for i, (seq, view, _, ln) in enumerate(part):
                    ids[i, :ln] = torch.tensor(seq.prompt_ids[:ln],
                                               dtype=torch.long,
                                               device=ids.device)
                    views.append(view)
                self.model.forward_prefill_batch(ids, views, real_lens=lens)
                for seq, view, start, ln in part:
                    seq.num_pages = len(view.table)
                    seq.prefill_cursor = start + ln
            if ext:
                # 续读块(cursor>0):仅 prompt 完成的行发 token(extend 内判定)
                tokens = self.executor.extend(
                    [s for s, _, _, _ in ext],
                    [v for _, v, _, _ in ext],
                    [(s, st, ln) for s, _, st, ln in ext],
                )
                for (seq, view, start, ln), token in zip(ext, tokens):
                    seq.num_pages = len(view.table)
                    seq.prefill_cursor = start + ln
                    if token is not None:
                        self._record(seq, token, outputs)
            if decodes:
                if self.spec_ngram > 0:
                    self._spec_step(decodes, outputs)
                else:
                    views = [self._views[s.seq_id] for s in decodes]
                    tokens = self.executor.decode_batch(decodes, views)
                    for seq, token in zip(decodes, tokens):
                        seq.num_pages = len(self._views[seq.seq_id].table)
                        self._record(seq, token, outputs)
        finally:
            # 退休放在 finally:即使前向抛异常,已 FINISHED 的序列也必须在本步
            # 出场 —— 否则僵尸序列下轮会被再次 decode(毒循环)
            for seq in [s for s in self.scheduler.running if s.is_finished]:
                self.scheduler.running.remove(seq)
                seq.status = RequestStatus.FINISHED
                self._views.pop(seq.seq_id)  # 视图丢弃;块经共享块表释放
                self._proposers.pop(seq.seq_id, None)  # 草稿器随退休弃(spec 模式)
                for b in seq.block_table:
                    self.allocator.free(b)
                seq.block_table.clear()
                seq.num_pages = 0
        if self.tp is not None:
            # 决策漂移哨兵(R3,每步):rank0 广播页表快照,rank1 逐位比对;
            # 漂移即硬退(调度决策输入全同 → 页表必须全同)
            self.tp.check_views(self._views)
        return outputs

    def _spec_step(self, decodes: list[Sequence], outputs: list[TokenOutput]) -> None:
        """spec verify 步(M8 Task 3,spec §2.2):decodes 全批按序分流。

        - 草稿:proposers[seq_id].propose()(engine 侧 dict;调度器不动)。
          生命周期:首进 verify 腿惰性建(context = all_ids;抢占重算后
          context 失配自动重建 —— 不变式 context == seq.all_ids),退休弃
          (step finally + drop 兜底)。
        - 降级 m=1(引擎级,正确性优先):①剩余生成预算 max_tokens −
          len(generated) < m(窗口尾段的 eos/length 截断交久经考验的单
          token 路径,末步数值 = decode 路径原样);②非贪心采样(temperature
          >0;spec §2.4 采样域数学不成立 —— 按 seq 降级由 decode_batch 既有
          逐行 _sample 正确采样,不做静默 greedy)。降级 seq 走 decode 腿,
          与 verify 腿同一步并存(批量统一性让位于正确性)。
        - 验证:verify_batch → argmax(dim=-1) → 逐 seq greedy_accept →
          发射 token 逐个过 _record(eos/length 终止即断,其后窗口位作废)
          → proposer.extend(实际提交段)→ set_lens × 全层回 len+已发数
          (死槽 [len', len+m) 下步复写,spec §2.3)→ num_pages 同步 →
          退休检查照 step finally。
        """
        m = self.spec_num + 1
        verify_leg: list[Sequence] = []
        decode_leg: list[Sequence] = []
        drafts_by_seq: dict[int, list[int]] = {}
        for seq in decodes:
            remaining = seq.max_tokens - len(seq.generated_ids)
            if remaining < m or seq.sampling.temperature != 0:
                decode_leg.append(seq)
                continue
            p = self._proposers.get(seq.seq_id)
            if p is None or p.context != seq.all_ids:
                # 惰性建/失配重建:索引重扫 O(len) 仅发生在 admit 与抢占
                # 重算后(verify 步首尚无新接受 token,键 = 末 n 位已提交
                # 上下文,与 Task 0 草稿注一致)
                from acorninfer.spec.ngram import NgramProposer
                p = NgramProposer(list(seq.all_ids), self.spec_ngram,
                                  self.spec_num)
                self._proposers[seq.seq_id] = p
            drafts_by_seq[seq.seq_id] = p.propose()
            verify_leg.append(seq)
        if decode_leg:
            # 降级腿:纯 decode 路径原样(与 spec_ngram==0 分支逐字同构)
            views = [self._views[s.seq_id] for s in decode_leg]
            tokens = self.executor.decode_batch(decode_leg, views)
            for seq, token in zip(decode_leg, tokens):
                seq.num_pages = len(self._views[seq.seq_id].table)
                self._record(seq, token, outputs)
        if verify_leg:
            views = [self._views[s.seq_id] for s in verify_leg]
            lens0 = [int(v.seq_len) for v in views]   # pre-write(ensure 不提交)
            drafts = [drafts_by_seq[s.seq_id] for s in verify_leg]
            logits = self.executor.verify_batch(verify_leg, views, drafts)
            next_ids = logits.argmax(dim=-1).tolist()  # [Bv, m](单次 D2H)
            from acorninfer.spec.ngram import greedy_accept
            for i, seq in enumerate(verify_leg):
                view = self._views[seq.seq_id]
                seq.num_pages = len(view.table)
                a, bonus = greedy_accept(next_ids[i], drafts[i])
                # 发射 = argmax 链前缀 a 位 + bonus(位置序);截断到剩余预算
                emitted = (next_ids[i][:a] + [bonus])[: seq.max_tokens - len(seq.generated_ids)]
                n0 = len(seq.generated_ids)
                for t in emitted:
                    self._record(seq, t, outputs)   # eos/length 终止判定在 _record
                    if seq.is_finished:
                        break                        # 终止 token 已记,其后窗口位作废
                committed = seq.generated_ids[n0:]   # 实际提交段(含终止 token)
                # 接受账(M8 Task 4):a = greedy 接受数;committed = 实际
                # 入流 token(eos 中断可 < a+1,如实分列;hist 键 ≤ m 有界)
                st = self.spec_stats
                st["verify_steps"] += 1
                st["accepted"] += a
                st["committed"] += len(committed)
                st["hist"][len(committed)] = st["hist"].get(len(committed), 0) + 1
                self._proposers[seq.seq_id].extend(committed)
                # KV 回退:接受 a → lens = len+a+1;截断时 = len+已发数。
                # 死槽 [len', lens0+m) 由下步 ensure_n/scatter 复写(T1 契约)
                for layer in range(self.cache.num_layers):
                    view.set_lens(layer, lens0[i] + len(committed))

    def drop(self, seq_id: int) -> None:
        """消费完结果后释放序列记录(bench/server 负责调用,防 seqs 无界增长)。"""
        seq = self.seqs.get(seq_id)
        if seq is not None and not seq.is_finished:
            raise ValueError(f"seq {seq_id} is still pending; cannot drop")
        self.seqs.pop(seq_id, None)
        self._proposers.pop(seq_id, None)  # 防御性(退休已清;drop 面兜底)

    def _record(self, seq: Sequence, token_id: int, outputs: list[TokenOutput]) -> None:
        seq.generated_ids.append(token_id)
        now = time.perf_counter()
        if seq.first_token_at is None:
            seq.first_token_at = now
        seq.last_token_at = now
        finished, reason = False, None
        if token_id in seq.eos_ids:
            finished, reason = True, "stop"
        elif len(seq.generated_ids) >= seq.max_tokens:
            finished, reason = True, "length"
        if finished:
            seq.stop_reason = reason
            seq.status = RequestStatus.FINISHED
        outputs.append(TokenOutput(seq.seq_id, token_id, finished, seq.stop_reason))

    def has_pending(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def assert_balanced(self) -> None:
        if self.graph_runner is None:
            self.allocator.assert_balanced()
        else:
            # graph runner 常驻 1 块 scratch(padding 行 KV 落点,永不读):
            # 排空后恰它一块 live,其余任何 live 块 = KV 泄漏
            if self.allocator.num_live != 1:
                raise AssertionError(
                    f"expected exactly 1 live block (graph runner scratch), "
                    f"got {self.allocator.num_live}")
