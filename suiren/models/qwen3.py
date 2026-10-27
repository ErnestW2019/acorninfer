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

from suiren.config import Qwen3Config
from suiren.weights import load_safetensors


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

    def forward(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        # `.to(dtype=...)` on the module casts this buffer to bf16; force fp32 back.
        freqs = torch.outer(pos, self.inv_freq.float().to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin  # [seq_len, head_dim]


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
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        b, s, _ = hidden.shape
        q = self.q_proj(hidden).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rope(s, hidden.dtype, hidden.device)
        q, k = _apply_rope(q, k, cos, sin)

        k = _repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = _repeat_kv(v, self.num_heads // self.num_kv_heads)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, -1)
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

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [batch, seq] -> logits: [batch, seq, vocab] (float32)."""
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return self.lm_head(hidden).float()


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
