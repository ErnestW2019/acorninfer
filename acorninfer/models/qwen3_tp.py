"""Qwen3 TP=2 变体(M5 Task 3):词并行 embed / 分头 attention / 分片 MLP /
词并行 lm_head。TP 显式构造才启用,单卡 qwen3.py 零触碰。

结构照 qwen3 同名模块镜像(模块命名与全宽 state_dict 一一对应,分片加载
strict=True);attention 核取 SDPA eager 分支(q_len=1 无需掩码,逐序列
update+SDPA 与 Qwen3ForCausalLM.forward_decode 的 sdpa 分支同一语义)——
对拍门是 token 流逐位一致,选与参照同构的 torch 路径,避免引入 triton
kernel 的第二类数值源。数值差来源只剩列/行切分 GEMM 的归约序差(logits
TOL 覆盖;token 流由双卡对拍钉住)。

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

    def forward_decode(self, hidden, cos, sin, views, layer_idx):
        """decode: hidden [B, 1, hidden],cos/sin [B, 1, 1, D];o_proj 就地
        allreduce(各 rank partial 之和 = 全宽 o 输出)。

        警告:rope 内联应用是第 5 份拷贝(qwen3 的 forward/_apply_rope、
        forward_decode、forward_prefill_batch、forward_extend 各一份,形状
        约定不同)——改公式需五处同步;golden 门 + 本文件 TP 对拍是安全网。
        """
        B, s, _ = hidden.shape
        q = self.q_proj(hidden).view(B, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        rep = self.num_heads // self.num_kv_heads
        outs = []
        for b, view in enumerate(views):
            k_full, v_full = view.update(layer_idx, k[b: b + 1], v[b: b + 1])
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
            outs.append(F.scaled_dot_product_attention(q[b: b + 1], k_full, v_full))
        out = torch.cat(outs, dim=0)                 # [B, H_local, 1, D]
        out = out.transpose(1, 2).reshape(B, s, -1)
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

    def forward_decode(self, hidden, cos, sin, views, layer_idx):
        hidden = hidden + self.self_attn.forward_decode(
            self.input_layernorm(hidden), cos, sin, views, layer_idx,
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

    def forward_decode(self, input_ids: torch.Tensor, views) -> torch.Tensor:
        """批量 decode:input_ids [B, 1] + 每序列 PagedCacheView ->
        [B, vocab](float32,全词表)。语义对齐 Qwen3ForCausalLM.forward_decode
        的 sdpa eager 分支(s 必须为 1;每序列位置 = 各自 cache.seq_len;
        新 token K/V 写入各自分页块);TP eager 路径,无 prepared/graph 面。
        """
        B, s = input_ids.shape
        if s != 1:
            raise ValueError(f"decode 步 s 必须为 1, got {s}")
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        positions = torch.tensor([v.seq_len for v in views], device=device,
                                 dtype=torch.long)
        cos, sin = self.rope(positions, dtype, device)   # [B, head_dim]
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
        hidden = self.embed_tokens(input_ids)            # 词并行 + allreduce
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_decode(hidden, cos, sin, views, i)
        hidden = self.norm(hidden)
        partial = self.lm_head(hidden[:, 0, :])          # [B, vocab/world]
        return all_gather_logits(partial).float()        # [B, vocab]


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
