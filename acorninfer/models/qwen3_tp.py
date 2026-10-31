"""Qwen3 TP=2 变体(M5):词并行 embed / 分头 attention / 分片 MLP /
词并行 lm_head。TP 显式构造才启用,单卡 qwen3.py 零触碰。

结构照 qwen3 同名模块镜像(模块命名与全宽 state_dict 一一对应,分片加载
strict=True);GEMM 保持与全宽参考同构的分立投影(对拍逐位门优先)。decode
双分支:sdpa eager(逐序列 update+SDPA,与 TP=1 参照同构)+ triton 分页核
(M5 Task 4,prepared 契约,过既有 DecodeGraphRunner 捕获 —— allreduce/
all-gather 在模型前向内随图捕获,段 0 尖刀已证)。prefill_batch/extend 走
逐行 update + SDPA(与 fused/backend 选择器解耦,同 qwen3 现状)。

数值差来源(对拍门持续钉住):列/行切分 GEMM 的归约序差 + triton-vs-SDPA
kernel 差(graph 对拍),均由 TOL 协议覆盖;token 流由 margin 门 + 强制流
协议钉住(bf16 精确平局会翻 argmax,贪婪链逐位断言不允许)。

正确性不变式:两 rank 的调度/输入 SPMD 全同 → KV 页表两卡逐位一致;
allreduce 在两 rank 产出同一结果(NCCL 确定性)→ hidden 状态两卡一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from acorninfer.distributed import all_gather_logits
from acorninfer.models.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from acorninfer.models.qwen3 import (
    RMSNorm,
    RotaryEmbedding,
    _apply_rope,
    _repeat_kv,
    _rotate_half,
)


class Qwen3TPAttention(nn.Module):
    """每卡 num_heads/world 个 q 头 + num_key_value_heads/world 个 kv 头。"""

    def __init__(self, cfg: Qwen3Config, rank: int, world: int):
        super().__init__()
        assert cfg.num_attention_heads % world == 0
        assert cfg.num_key_value_heads % world == 0
        self.num_heads = cfg.num_attention_heads // world
        self.num_kv_heads = cfg.num_key_value_heads // world
        self.head_dim = cfg.head_dim
        hidden = cfg.hidden_size
        self.q_proj = ColumnParallelLinear(hidden, cfg.num_attention_heads * self.head_dim, rank, world)
        self.k_proj = ColumnParallelLinear(hidden, cfg.num_key_value_heads * self.head_dim, rank, world)
        self.v_proj = ColumnParallelLinear(hidden, cfg.num_key_value_heads * self.head_dim, rank, world)
        self.o_proj = RowParallelLinear(cfg.num_attention_heads * self.head_dim, hidden, rank, world)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)   # 复制不切
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, hidden, cos, sin, cache=None, layer_idx=0):
        """单序列 prefill/decode 通用前向(M5 Task 4:engine N=1 prefill 走
        model(ids, view) —— 镜像 qwen3 attention.forward:cache.update 全量
        gather,is_causal=(s>1),非空 cache 多 token 显式右下对齐掩码);
        o_proj allreduce。"""
        b, s, _ = hidden.shape
        q = self.q_proj(hidden).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = _apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)  # k/v 变为全量 [1, kv, past+s, d]
        past_len = k.shape[2] - s

        rep = self.num_heads // self.num_kv_heads
        k = _repeat_kv(k, rep)
        v = _repeat_kv(v, rep)

        if past_len == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(s > 1))
        elif s == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            q_pos = torch.arange(past_len, past_len + s, device=q.device)
            kv_pos = torch.arange(past_len + s, device=q.device)
            mask = q_pos[:, None] >= kv_pos[None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)

    def forward_decode(self, hidden, cos, sin, views, layer_idx, *,
                       attention_backend="sdpa", tables=None, lens=None,
                       scatter=None):
        """decode: hidden [B, 1, hidden],cos/sin [B, 1, 1, D];o_proj 就地
        allreduce(各 rank partial 之和 = 全宽 o 输出)。

        triton 后端(M5 Task 4,graph 捕获必需):与 qwen3.forward_decode
        同一语义 —— tables/lens/scatter(prepared 四件套)传入时单次合批
        scatter;None 时逐序列 write + build_tables。分页核按每 rank 头数
        寻址(REP=H_local//KV_local),cache 本身就是每卡独立构造的。
        SDPA 分支:逐序列 update + SDPA(与 TP=1 参照同构)。

        警告:rope 内联应用是本文件三份拷贝之一(decode/prefill_batch/
        extend,形状约定不同)——改公式需本文件内三处 + qwen3 四处同步;
        golden 门 + TP 对拍是安全网。
        """
        B, s, _ = hidden.shape
        q = self.q_proj(hidden).view(B, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if attention_backend == "triton":
            from acorninfer.ops.paged_attention import (
                build_tables, num_splits_for, paged_decode_attention,
            )
            cache = views[0].cache   # 契约:全部 views 共享同一 cache(引擎单一实例保证)
            if tables is not None:
                pages, slots = scatter
                cache.scatter_decode(views, layer_idx, k, v, pages, slots)
            else:
                for b, view in enumerate(views):
                    view.write(layer_idx, k[b: b + 1], v[b: b + 1])
                tables, lens = build_tables(views, cache.page_size, q.device)
            out = paged_decode_attention(
                q[:, :, 0, :],                       # [B, H_local, D]
                cache.k_pages[layer_idx], cache.v_pages[layer_idx],
                tables, lens, num_splits_for(B, self.num_heads),
            )
            out = out[:, :, None, :]                 # 与 SDPA 分支同构
        else:
            rep = self.num_heads // self.num_kv_heads
            outs = []
            for b, view in enumerate(views):
                k_full, v_full = view.update(layer_idx, k[b: b + 1], v[b: b + 1])
                k_full = _repeat_kv(k_full, rep)
                v_full = _repeat_kv(v_full, rep)
                outs.append(F.scaled_dot_product_attention(q[b: b + 1], k_full, v_full))
            out = torch.cat(outs, dim=0)             # [B, H_local, 1, D]
        out = out.transpose(1, 2).reshape(B, s, -1)
        return self.o_proj(out)

    def forward_prefill_batch(self, hidden, cos, sin, views, layer_idx, *,
                              real_lens):
        """批量 prefill(M5 Task 4):hidden [N, Lmax, D](右 padding),逐行
        update 截断 [:L] + causal SDPA(镜像 qwen3 同名);o_proj allreduce。"""
        N, Lmax, _ = hidden.shape
        q = self.q_proj(hidden).view(N, Lmax, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        rep = self.num_heads // self.num_kv_heads
        outs = torch.zeros(N, self.num_heads, Lmax, self.head_dim,
                           dtype=hidden.dtype, device=hidden.device)
        for b, view in enumerate(views):
            L = real_lens[b]
            k_full, v_full = view.update(layer_idx, k[b: b + 1, :, :L],
                                         v[b: b + 1, :, :L])
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
            outs[b, :, :L] = F.scaled_dot_product_attention(
                q[b: b + 1, :, :L], k_full, v_full, is_causal=(L > 1))[0]
        out = outs.transpose(1, 2).reshape(N, Lmax, -1)
        return self.o_proj(out)

    def forward_extend(self, hidden, cos, sin, views, layer_idx, *,
                       starts, chunk_lens):
        """chunk 续读(M5 Task 4):cursor>0 追加前向;逐行 update(追加
        [:L])+ 显式右下对齐 causal 掩码 tril(diagonal=past)——与
        qwen3.forward_extend 同一语义;o_proj allreduce。"""
        N, Lc_max, _ = hidden.shape
        q = self.q_proj(hidden).view(N, Lc_max, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(N, Lc_max, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(N, Lc_max, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        rep = self.num_heads // self.num_kv_heads
        outs = torch.zeros(N, self.num_heads, Lc_max, self.head_dim,
                           dtype=hidden.dtype, device=hidden.device)
        for b, view in enumerate(views):
            L = chunk_lens[b]
            past = starts[b]
            k_full, v_full = view.update(layer_idx, k[b: b + 1, :, :L],
                                         v[b: b + 1, :, :L])
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
            mask = torch.ones(L, past + L, dtype=torch.bool,
                              device=hidden.device).tril(diagonal=past)[None, None]
            outs[b, :, :L] = F.scaled_dot_product_attention(
                q[b: b + 1, :, :L], k_full, v_full, attn_mask=mask)[0]
        out = outs.transpose(1, 2).reshape(N, Lc_max, -1)
        return self.o_proj(out)


class Qwen3TPMLP(nn.Module):
    """gate/up 列并行(partial)+ down 行并行(就地 allreduce)。"""

    def __init__(self, cfg: Qwen3Config, rank: int, world: int):
        super().__init__()
        hidden, inter = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = ColumnParallelLinear(hidden, inter, rank, world)
        self.up_proj = ColumnParallelLinear(hidden, inter, rank, world)
        self.down_proj = RowParallelLinear(inter, hidden, rank, world)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * F.silu(self.gate_proj(x)))


class Qwen3TPBlock(nn.Module):
    def __init__(self, cfg: Qwen3Config, rank: int, world: int):
        super().__init__()
        self.self_attn = Qwen3TPAttention(cfg, rank, world)
        self.mlp = Qwen3TPMLP(cfg, rank, world)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, hidden, cos, sin, cache=None, layer_idx=0):
        hidden = hidden + self.self_attn(hidden, cos, sin, cache, layer_idx)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def forward_decode(self, hidden, cos, sin, views, layer_idx, *,
                       attention_backend="sdpa", tables=None, lens=None,
                       scatter=None):
        if attention_backend == "triton":
            hidden = hidden + self.self_attn.forward_decode(
                self.input_layernorm(hidden), cos, sin, views, layer_idx,
                attention_backend=attention_backend,
                tables=tables, lens=lens, scatter=scatter,
            )
        else:
            hidden = hidden + self.self_attn.forward_decode(
                self.input_layernorm(hidden), cos, sin, views, layer_idx,
            )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def forward_prefill_batch(self, hidden, cos, sin, views, layer_idx, *,
                              real_lens):
        hidden = hidden + self.self_attn.forward_prefill_batch(
            self.input_layernorm(hidden), cos, sin, views, layer_idx,
            real_lens=real_lens,
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def forward_extend(self, hidden, cos, sin, views, layer_idx, *,
                       starts, chunk_lens):
        hidden = hidden + self.self_attn.forward_extend(
            self.input_layernorm(hidden), cos, sin, views, layer_idx,
            starts=starts, chunk_lens=chunk_lens,
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class Qwen3TPForCausalLM(nn.Module):
    """TP=2 causal LM:词并行 embed(allreduce)+ 分片层 + 词并行 lm_head
    (all-gather,模型内封装,对外与 TP=1 同一返回面 [B, vocab] fp32)。"""

    def __init__(self, cfg: Qwen3Config, rank: int, world: int):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, rank, world)
        self.layers = nn.ModuleList(
            Qwen3TPBlock(cfg, rank, world) for _ in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(cfg.hidden_size, cfg.vocab_size, rank, world)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)   # 复制不切
        # 每卡 kv 头数:KV cache 按此构造(pager/kv 零改动,每卡独立 cache)
        self.num_kv_heads_local = self.layers[0].self_attn.num_kv_heads

    def forward(self, input_ids: torch.Tensor, cache=None, *,
                num_logits_to_keep: int | None = None) -> torch.Tensor:
        """单序列前向(M5 Task 4):engine N=1 prefill 路径
        (executor.prefill 调 model(ids, view, num_logits_to_keep=1))。
        语义镜像 Qwen3ForCausalLM.forward:位置 = arange(cache.seq_len, +s),
        is_causal=(s>1);返回 [1, s_keep, vocab](float32,all-gather 全词表)。"""
        if num_logits_to_keep is not None and num_logits_to_keep < 1:
            raise ValueError("num_logits_to_keep must be >= 1")
        s = input_ids.shape[1]
        past_len = 0 if cache is None else cache.seq_len
        positions = torch.arange(past_len, past_len + s, device=input_ids.device)
        dtype = self.embed_tokens.weight.dtype
        cos, sin = self.rope(positions, dtype, input_ids.device)
        hidden = self.embed_tokens(input_ids)            # 词并行 + allreduce
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, i)
        hidden = self.norm(hidden)
        if num_logits_to_keep is not None:
            hidden = hidden[:, -num_logits_to_keep:, :]
        return all_gather_logits(self.lm_head(hidden)).float()  # [1, s_keep, vocab]

    def forward_decode(self, input_ids: torch.Tensor, views, *,
                       prepared=None) -> torch.Tensor:
        """批量 decode:input_ids [B, 1] + 每序列 PagedCacheView ->
        [B, vocab](float32,全词表)。

        sdpa eager 分支:语义对齐 Qwen3ForCausalLM.forward_decode(每序列
        位置 = 各自 cache.seq_len;新 token K/V 写入各自分页块)。
        triton 分支(M5 Task 4,graph 捕获必需):契约同 qwen3 —— prepared
        四件套 (positions, tables, lens, (pages, slots)) 由 DecodeGraphRunner
        图外准备传入(prepare 的 CPU 构建进图会让重放读冻结源);None 时自
        prepare(ensure_append_capacity + scatter + build_tables),eager
        triton decode 亦可用。allreduce/all-gather 在模型前向内,随图捕获
        (段 0 尖刀已证)。
        """
        B, s = input_ids.shape
        if s != 1:
            raise ValueError(f"decode 步 s 必须为 1, got {s}")
        backend = getattr(self, "attention_backend", "sdpa")
        if prepared is not None:
            positions, tables, lens, scatter = prepared
        else:
            positions = torch.tensor([v.seq_len for v in views],
                                     device=input_ids.device)
            tables = lens = scatter = None
            if backend == "triton":
                from acorninfer.ops.paged_attention import build_tables
                page_size = views[0].cache.page_size
                targets = [v.ensure_append_capacity(0) for v in views]
                scatter = (
                    torch.tensor([p for p, _ in targets], device=input_ids.device,
                                 dtype=torch.long),
                    torch.tensor([sl for _, sl in targets], device=input_ids.device,
                                 dtype=torch.long),
                )
                tables, lens = build_tables(
                    views, page_size, input_ids.device,
                    lens_override=[int(v.seq_len) + 1 for v in views],
                )
        dtype = self.embed_tokens.weight.dtype
        cos, sin = self.rope(positions, dtype, input_ids.device)   # [B, head_dim]
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
        hidden = self.embed_tokens(input_ids)            # 词并行 + allreduce
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_decode(hidden, cos, sin, views, i,
                                          attention_backend=backend,
                                          tables=tables, lens=lens,
                                          scatter=scatter)
        hidden = self.norm(hidden)
        partial = self.lm_head(hidden[:, 0, :])          # [B, vocab/world]
        return all_gather_logits(partial).float()        # [B, vocab]

    def forward_prefill_batch(self, input_ids: torch.Tensor, views, *,
                              real_lens) -> torch.Tensor:
        """批量 prefill(M5 Task 4):input_ids [N, Lmax](右 padding)+
        real_lens → 各行末真实位 logits [N, vocab](float32,全词表)。
        语义镜像 Qwen3ForCausalLM.forward_prefill_batch(KV 写入经各自
        views[i],pad 永不入 KV);embed allreduce / lm_head all-gather。"""
        N, Lmax = input_ids.shape
        if len(views) != N or len(real_lens) != N:
            raise ValueError(
                f"views/real_lens 必须与 batch 维一一对应: got N={N}, "
                f"len(views)={len(views)}, len(real_lens)={len(real_lens)}"
            )
        if min(real_lens) < 1 or max(real_lens) > Lmax:
            raise ValueError(
                f"real_lens 必须落在 [1, Lmax] 内: got "
                f"min={min(real_lens)}, max={max(real_lens)}, Lmax={Lmax}"
            )
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        pos = torch.zeros(N * Lmax, dtype=torch.long, device=device)
        for i, L in enumerate(real_lens):
            pos[i * Lmax: i * Lmax + L] = torch.arange(L, device=device)
        cos, sin = self.rope(pos, dtype, device)      # [N*Lmax, head_dim]
        cos = cos.view(N, Lmax, -1)
        sin = sin.view(N, Lmax, -1)
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_prefill_batch(hidden, cos, sin, views, i,
                                                 real_lens=real_lens)
        hidden = self.norm(hidden)
        last = torch.tensor([L - 1 for L in real_lens], device=device)
        hidden = hidden[torch.arange(N, device=device), last]  # [N, hidden]
        return all_gather_logits(self.lm_head(hidden)).float()  # [N, vocab]

    def forward_extend(self, input_ids: torch.Tensor, views, *,
                       starts: list[int], chunk_lens: list[int]) -> torch.Tensor:
        """chunk 续读前向(M5 Task 4):input_ids [N, Lc_max](右 padding)+
        starts[i](各行已完成 token数)+ chunk_lens[i](本块真实长度)→
        各行本块末位(start+len-1)logits [N, vocab](float32)。语义镜像
        Qwen3ForCausalLM.forward_extend(views[i] 的 KV 长度必须恰为 starts[i])。"""
        N, Lc_max = input_ids.shape
        if len(views) != N or len(starts) != N or len(chunk_lens) != N:
            raise ValueError(
                f"views/starts/chunk_lens 必须与 batch 维一一对应: got N={N}, "
                f"len(views)={len(views)}, len(starts)={len(starts)}, "
                f"len(chunk_lens)={len(chunk_lens)}"
            )
        if min(chunk_lens) < 1 or max(chunk_lens) > Lc_max or min(starts) < 0:
            raise ValueError(
                f"chunk_lens 必须落在 [1, Lc_max] 且 starts 非负: got "
                f"min(len)={min(chunk_lens)}, max(len)={max(chunk_lens)}, "
                f"Lc_max={Lc_max}, min(start)={min(starts)}"
            )
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        pos = torch.zeros(N * Lc_max, dtype=torch.long, device=device)
        for i, (st, L) in enumerate(zip(starts, chunk_lens)):
            pos[i * Lc_max: i * Lc_max + L] = torch.arange(st, st + L, device=device)
        cos, sin = self.rope(pos, dtype, device)      # [N*Lc_max, head_dim]
        cos = cos.view(N, Lc_max, -1)
        sin = sin.view(N, Lc_max, -1)
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_extend(hidden, cos, sin, views, i,
                                          starts=starts, chunk_lens=chunk_lens)
        hidden = self.norm(hidden)
        # 末位 token = 绝对位 start+len-1,在 pad 张量中位于第 len-1 列(局部坐标)
        last = torch.tensor([L - 1 for L in chunk_lens], device=device)
        hidden = hidden[torch.arange(N, device=device), last]  # [N, hidden]
        return all_gather_logits(self.lm_head(hidden)).float()  # [N, vocab]


def build_shard_fn(cfg: Qwen3Config, rank: int, world: int):
    """全宽 state_dict → 本 rank 分片的逐名映射闭包(结构化切分,禁止裸对半)。

    - embed/lm_head:词并行行切(V/world);
    - q/k/v_proj:按头列切(分离投影,头主续行,连续段即 rank 头段;
      融合布局的等价切分见 parallel.shard_qkv);
    - o_proj:K 维列切;q_norm/k_norm 复制;
    - gate/up_proj:inter 行切(inter/world;融合布局见 parallel.shard_gate_up);
    - down_proj:K 维列切;layer norm / norm 复制。
    """
    hd = cfg.head_dim

    def shard_fn(name: str, t: torch.Tensor, r: int, w: int):
        from acorninfer.models.parallel import shard_cols, shard_rows

        if name in ("embed_tokens.weight", "lm_head.weight"):
            return shard_rows(t, r, w)
        if name in ("norm.weight",) or name.endswith("layernorm.weight"):
            return t.clone()                       # 复制不切
        if name.endswith("self_attn.q_norm.weight") or name.endswith("self_attn.k_norm.weight"):
            return t.clone()
        if name.endswith("self_attn.q_proj.weight"):
            n_loc = cfg.num_attention_heads // w * hd
            return t[r * n_loc:(r + 1) * n_loc, :].contiguous()   # 头主续行 → 连续段即按头
        if name.endswith("self_attn.k_proj.weight") or name.endswith("self_attn.v_proj.weight"):
            n_loc = cfg.num_key_value_heads // w * hd
            return t[r * n_loc:(r + 1) * n_loc, :].contiguous()
        if name.endswith("self_attn.o_proj.weight"):
            return shard_cols(t, r, w)
        if name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
            return shard_rows(t, r, w)
        if name.endswith("mlp.down_proj.weight"):
            return shard_cols(t, r, w)
        raise KeyError(f"未知的 TP 分片目标: {name}")

    return shard_fn
