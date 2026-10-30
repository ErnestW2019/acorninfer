"""Qwen3 dense model, pure-PyTorch reference implementation (no KV cache).

Numerics deliberately mirror HF transformers Qwen3:
- RMSNorm computes in float32 and casts back to the input dtype;
- RoPE cos/sin are computed in float32 and cast to the activation dtype
  before the elementwise multiply;
- QK-Norm normalizes each head over head_dim;
- SDPA with is_causal=True and the default softmax scale (head_dim ** -0.5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from acorninfer.config import Qwen3Config
from acorninfer.weights import load_safetensors


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype, device: torch.device):
        # positions: [q_len] float/int tensor of absolute positions.
        pos = positions.to(device=device, dtype=torch.float32)
        # `.to(dtype=...)` on the module casts this buffer to bf16; force fp32 back.
        freqs = torch.outer(pos, self.inv_freq.float().to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin  # [q_len, head_dim]


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [batch, heads, seq, head_dim]; cos/sin: [seq, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


def _repeat_kv(x: torch.Tensor, repeat: int) -> torch.Tensor:
    # [batch, kv_heads, seq, dim] -> [batch, kv_heads * repeat, seq, dim]
    if repeat == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, repeat, s, d).reshape(b, h * repeat, s, d)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, hidden, cos, sin, cache=None, layer_idx=0):
        # cos/sin 由 Qwen3ForCausalLM.rope 预计算,全层共享
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

        k = _repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = _repeat_kv(v, self.num_heads // self.num_kv_heads)

        if past_len == 0:
            # prefill:cache 空时左上对齐 == 右下对齐,flash 友好
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(s > 1))
        elif s == 1:
            # decode:单 query,新 K/V 已在 cache 里,全可见 == 正确掩码
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # 非空 cache + 多 token:显式右下对齐掩码
            q_pos = torch.arange(past_len, past_len + s, device=q.device)
            kv_pos = torch.arange(past_len + s, device=q.device)
            mask = q_pos[:, None] >= kv_pos[None, :]  # True = 允许 attend
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)

    def forward_decode(self, hidden, cos, sin, views, layer_idx, *,
                       positions=None, inv_freq=None, fused_ops=False,
                       attention_backend="sdpa"):
        """批量 decode: hidden [B, 1, hidden], cos/sin [B, 1, 1, D]。

        GEMM 批量化(continuous batching 的收益所在);attention 逐序列
        (KV 长度不同,q_len=1 无需掩码 —— M1 分支 2 的推广)。

        fused_ops=True 时走 K2 fused RoPE:需要 positions[B] + inv_freq[D/2],
        且 cos/sin 必须为 None(否则 torch 路径的 cos/sin 与 kernel 内重算
        会二选一,传双份是调用方 bug)。
        attention_backend="triton" 时走 K3 分页 decode attention:各序列
        write 进页(纯 scatter)后 build_tables,kernel 按块表直读页张量、
        不 gather;lens 含本步新 token,与 SDPA 分支 update() 的可见性
        语义一致(都 attend 到含新 token 的全序列)。
        """
        if attention_backend not in ("sdpa", "triton"):
            raise ValueError(
                f"attention_backend must be 'sdpa'|'triton', got {attention_backend!r}"
            )
        B, s, _ = hidden.shape
        q = self.q_proj(hidden).view(B, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE:每个序列自己的位置;cos/sin 已是 [B,1,1,D],不走 _apply_rope(形状约定不同)
        if fused_ops:
            if positions is None or inv_freq is None or cos is not None:
                raise ValueError("fused_ops requires positions+inv_freq and no cos/sin")
            from acorninfer.ops.rope import fused_rope_inplace
            fused_rope_inplace(q, k, positions, inv_freq)   # 就地;q/k decode 布局 contiguous
        else:
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

        if attention_backend == "triton":
            from acorninfer.ops.paged_attention import (
                build_tables, num_splits_for, paged_decode_attention,
            )
            cache = views[0].cache
            for b, view in enumerate(views):
                view.write(layer_idx, k[b : b + 1], v[b : b + 1])   # 纯 scatter,不 gather
            tables, lens = build_tables(views, cache.page_size, q.device)
            out = paged_decode_attention(
                q[:, :, 0, :],                       # [B, H, D];RMSNorm 输出,contiguous
                cache.k_pages[layer_idx], cache.v_pages[layer_idx],
                tables, lens, num_splits_for(B, self.num_heads),
            )
            out = out[:, :, None, :]                 # 与 SDPA 分支同构 [B, H, 1, D]
        else:
            rep = self.num_heads // self.num_kv_heads
            outs = []
            # 每层每序列全量 gather(O(seq_len) 拷贝)是刻意的参考设计;M3 的分页
            # attention kernel 直接按页读取,消除这份拷贝
            for b, view in enumerate(views):
                k_full, v_full = view.update(layer_idx, k[b : b + 1], v[b : b + 1])
                k_full = _repeat_kv(k_full, rep)
                v_full = _repeat_kv(v_full, rep)
                outs.append(F.scaled_dot_product_attention(q[b : b + 1], k_full, v_full))
            out = torch.cat(outs, dim=0)             # [B, H, 1, D]
        out = out.transpose(1, 2).reshape(B, s, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, hidden, cos, sin, cache=None, layer_idx=0):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, cache, layer_idx)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def forward_decode(self, hidden, cos, sin, views, layer_idx, *,
                       positions=None, inv_freq=None, fused_ops=False,
                       attention_backend="sdpa"):
        if not fused_ops:
            hidden = hidden + self.self_attn.forward_decode(
                self.input_layernorm(hidden), cos, sin, views, layer_idx,
                positions=positions, inv_freq=inv_freq, fused_ops=fused_ops,
                attention_backend=attention_backend,
            )
            hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
            return hidden
        from acorninfer.ops.rmsnorm import fused_add_rmsnorm  # 局部导入,避免 torch 路径依赖 ops
        normed, residual = fused_add_rmsnorm(
            hidden, None, self.input_layernorm.weight, self.input_layernorm.eps
        )
        hidden = self.self_attn.forward_decode(
            normed, cos, sin, views, layer_idx,
            positions=positions, inv_freq=inv_freq, fused_ops=fused_ops,
            attention_backend=attention_backend,
        )
        normed, residual = fused_add_rmsnorm(
            hidden, residual, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return residual + self.mlp(normed)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        # M3 后端开关(EngineConfig/bench/测试直接赋值;默认 = M2 行为)
        self.attention_backend = "sdpa"
        self.fused_ops = False

    def forward(
        self,
        input_ids: torch.Tensor,
        cache=None,
        num_logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """input_ids: [batch, seq] -> logits: [batch, seq, vocab] (float32).

        cache=None: 全量前向(M0 oracle 路径)。传入 KVCache 时为增量路径:
        位置由 cache.seq_len 推导,prefill(cache 空)与 decode(cache 非空)共用。
        num_logits_to_keep: 只对最后 k 个位置算 lm_head(prefill 生成只需末位;
        None = 全位置,M0 路径不变)。k 超过 seq_len 时静默返回全量(镜像
        Python 切片语义);k < 1 直接拒绝(切片 `[-0:]` 会悄悄变成全量)。
        """
        if num_logits_to_keep is not None and num_logits_to_keep < 1:
            raise ValueError("num_logits_to_keep must be >= 1")
        s = input_ids.shape[1]
        past_len = 0 if cache is None else cache.seq_len
        positions = torch.arange(past_len, past_len + s, device=input_ids.device)
        dtype = self.embed_tokens.weight.dtype
        cos, sin = self.rope(positions, dtype, input_ids.device)
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, cache, i)
        hidden = self.norm(hidden)
        if num_logits_to_keep is not None:
            hidden = hidden[:, -num_logits_to_keep:, :]
        return self.lm_head(hidden).float()

    def forward_decode(self, input_ids: torch.Tensor, views) -> torch.Tensor:
        """批量 decode:input_ids [B, 1] + 每序列的 PagedCacheView -> [B, vocab]。

        契约:views[i] 与 input_ids[i] 一一对应;s 必须为 1(decode 步)。
        每序列位置 = 各自 cache.seq_len(互不相同);新 token 的 K/V 在
        attention 内写入各自分页块。
        """
        if input_ids.shape[1] != 1:
            # s>1 会静默腐败:cos/sin 广播成同位置、update 多写 token
            raise ValueError("forward_decode expects [B, 1] input_ids (decode step)")
        positions = torch.tensor([v.seq_len for v in views], device=input_ids.device)
        if self.fused_ops:
            # K2:cos/sin 不再预算,kernel 内由 positions×inv_freq 重算
            # (inv_freq 是 fp32 buffer,模块 .to(dtype) 后变 bf16,wrapper 内 .float() 还原)
            cos = sin = None
            inv_freq = self.rope.inv_freq
        else:
            dtype = self.embed_tokens.weight.dtype
            cos, sin = self.rope(positions, dtype, input_ids.device)  # [B, head_dim]
            cos = cos[:, None, None, :]
            sin = sin[:, None, None, :]
            inv_freq = None
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_decode(
                hidden, cos, sin, views, i,
                positions=positions, inv_freq=inv_freq,
                fused_ops=self.fused_ops,
                attention_backend=self.attention_backend,
            )
        hidden = self.norm(hidden)
        return self.lm_head(hidden).float()[:, 0, :]  # [B, vocab]


def map_hf_state_dict(hf_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF checkpoint names -> our module names (strip the `model.` prefix)."""
    return {name.removeprefix("model."): t for name, t in hf_tensors.items()}


def from_pretrained(
    model_dir: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3ForCausalLM:
    """Load a Qwen3 checkpoint dir.

    CPU build -> strict assign load -> single .to(device) move keeps peak VRAM
    at one weight copy (init-on-GPU + assign would peak at ~2x on a 24GB card).
    """
    cfg = Qwen3Config.from_json(f"{model_dir}/config.json")
    model = Qwen3ForCausalLM(cfg)
    mapped = map_hf_state_dict(load_safetensors(model_dir, device="cpu"))
    model.load_state_dict(mapped, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()
