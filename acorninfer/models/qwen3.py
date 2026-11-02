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


def dispatch_linear(mod, x):
    """线性层分派(M7 段 A 接线;全部投影/lm_head 调用点的唯一入口):
    - none(缺省,模块无 `_m7_quant` 标记):`mod(x)` —— nn.Linear.__call__
      旧路径逐字节(F.linear),none 硬门的运行时面;
    - w8a16:`quant_gemm(x, int8 权重, scale, cls)`,cls 按 (N,K) 查五档表
      (`_CLS_BY_SHAPE`,host dict 一次查 ~0.1µs,graph 捕获期定死零 host
      同步;walk 期不打标 → apply_quant 保持纯 torch/CPU 可测,cls 形状
      不匹配显式报错零静默);
    输出 view 回 x 领先维形状(调用点 .view()/.float() 语义不变;quant_gemm
    内部 2D 化,quant 路径等效 m6_ab layout harness 的 make_forward 形态)。
    其他标记值(w4a16 等)= 显式报错(walk 侧 NotImplementedError 之后
    的第二道守门)。
    """
    mode = getattr(mod, "_m7_quant", None)
    if mode is None:
        return mod(x)
    if mode != "w8a16":
        raise ValueError(f"dispatch_linear: unsupported _m7_quant={mode!r}")
    from acorninfer.ops.quant_gemm import _CLS_BY_SHAPE, quant_gemm
    w = mod.weight.data
    cls = _CLS_BY_SHAPE.get(tuple(w.shape))
    if cls is None:
        raise ValueError(
            f"dispatch_linear: quantized linear shape {tuple(w.shape)} "
            f"不在五档 CLASSES(apply_quant 只对生产 8B 形状有效)")
    out = quant_gemm(x, w, mod._q_scale, cls)
    return out.view(*x.shape[:-1], w.shape[0])


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
        q = dispatch_linear(self.q_proj, hidden).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = dispatch_linear(self.k_proj, hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = dispatch_linear(self.v_proj, hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
        return dispatch_linear(self.o_proj, out)

    def forward_decode(self, hidden, cos, sin, views, layer_idx, *,
                       positions=None, inv_freq=None, fused_ops=False,
                       attention_backend="sdpa",
                       tables=None, lens=None, scatter=None):
        """批量 decode: hidden [B, 1, hidden], cos/sin [B, 1, 1, D]。

        GEMM 批量化(continuous batching 的收益所在);attention 逐序列
        (KV 长度不同,q_len=1 无需掩码 —— M1 分支 2 的推广)。

        fused_ops=True 时走 K2 fused RoPE:需要 positions[B] + inv_freq[D/2],
        且 cos/sin 必须为 None(否则 torch 路径的 cos/sin 与 kernel 内重算
        会二选一,传双份是调用方 bug)。
        attention_backend="triton" 时走 K3 分页 decode attention:各序列
        write 进页(纯 scatter)后按块表直读页张量、不 gather;lens 含本步
        新 token,与 SDPA 分支 update() 的可见性语义一致(都 attend 到含
        新 token 的全序列)。
        tables/lens/scatter(M3 Task 9):顶层步首一次准备好的块表 / 长度 /
        各序列追加位 (pages, slots);非 None 时本层只做单次合批 scatter,
        不再逐序列 write + build_tables(那曾是每层每步的固定 CPU 开销)。
        None 时回退自建,保持独立调用可用性。
        """
        if attention_backend not in ("sdpa", "triton"):
            raise ValueError(
                f"attention_backend must be 'sdpa'|'triton', got {attention_backend!r}"
            )
        B, s, _ = hidden.shape
        q = dispatch_linear(self.q_proj, hidden).view(B, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = dispatch_linear(self.k_proj, hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = dispatch_linear(self.v_proj, hidden).view(B, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
            cache = views[0].cache  # 契约:全部 views 共享同一 PagedKVCache(引擎单一 cache 实例保证);
            # 混用多 cache 会静默读错页 —— kernel 只按 views[0] 的页张量寻址(SDPA 分支对此容忍,triton 分支不容)
            assert (tables is None) == (lens is None) == (scatter is None), (
                "tables/lens/scatter 必须整体传入或整体为 None"
            )
            if tables is not None:
                # M3 Task 9 合批路径:步首已完成容量分配与建表,此处单次
                # scatter + _lens 提交(原每层 B 次小发射收拢为 2 次)
                pages, slots = scatter
                cache.scatter_decode(views, layer_idx, k, v, pages, slots)
            else:
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
        return dispatch_linear(self.o_proj, out)

    def forward_verify(self, hidden, cos, sin, views, layer_idx, *,
                       tables=None, lens=None, scatter=None):
        """verify 步 m 位窗口(spec §2.2,M8 Task 3)。hidden [B, m, hidden];
        cos/sin [B, m, head_dim](位置 = len+arange(m),Model 级预算)。

        镜像 forward_decode 的 triton 分支结构,三处推广:
        - KV 写 = scatter_verify(M8 Task 1):[B,kv,m,D] 一次散写于步首
          ensure_append_capacity_n 目标(pages/slots [B,m]),_lens 提交 +m;
        - attention = paged_verify_attention(M8 Task 2):q [B,m,H,D],读侧
          lens = len+m,行 j 可见集 = 历史 len + 窗口前 j+1 位(窗口内因果);
        - GEMM 全位宽:qkv/o/mlp/lm_head 按 [B*m] 行合批(投影分派复用
          dispatch_linear —— none 旧路径逐字节 / w8a16 quant 路径;spec_num
          ≤ 8 → m ≤ 16 落 s 桶,spec × quant 兼容)。
        投影/归一化/rope 公式与 forward_decode 逐字同源(位置广播形状不同:
        cos/sin [B,1,m,D] 对 [B,H,m,D])。engine 路径恒传 tables/lens/scatter
        (步首 prepare,同 forward_decode 合批契约);None 时本方法无自 prepare
        (verify 只在引擎 spec 分支调用,独立调用方须自带 —— 显式报错防静默
        错写)。"""
        if tables is None or lens is None or scatter is None:
            raise ValueError(
                "forward_verify requires tables/lens/scatter from a step-start "
                "prepare (engine spec branch); no self-prepare fallback")
        B, m, _ = hidden.shape
        q = dispatch_linear(self.q_proj, hidden).view(B, m, self.num_heads, self.head_dim).transpose(1, 2)
        k = dispatch_linear(self.k_proj, hidden).view(B, m, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = dispatch_linear(self.v_proj, hidden).view(B, m, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # rope 公式与 forward_decode 非融合分支逐字同源;广播 [B,1,m,D]
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        from acorninfer.ops.paged_attention import num_splits_for
        from acorninfer.ops.paged_verify import paged_verify_attention
        cache = views[0].cache  # 契约同 forward_decode:全部 views 共享单一 cache
        pages, slots = scatter
        cache.scatter_verify(views, layer_idx, k, v, pages, slots)
        # q 转置 [B,H,m,D]→[B,m,H,D](kernel 行序 (b,j,h));输出已是
        # (b,j,h,d) 连续布局,直接 reshape 合头(行内 = 头拼接,与 decode
        # 的 [B,H,1,D]→transpose→reshape 行序逐位同构)——decode 的
        # transpose(1,2) 在此不可照抄:kernel 返回 (b,j,h,d) 而非
        # (b,h,j,d),转置会把 h 变外层轴,reshape 按逻辑序 h-major 复制,
        # m>1 且 H>1 时 o_proj 输入行 = 多窗口位×单头的错序混合(A1
        # 无损性硬门抓出的原始缺陷,勿改回)。
        out = paged_verify_attention(
            q.transpose(1, 2),
            cache.k_pages[layer_idx], cache.v_pages[layer_idx],
            tables, lens, num_splits_for(B * m, self.num_heads),
        )
        out = out.reshape(B, m, -1)
        return dispatch_linear(self.o_proj, out)

    def forward_prefill_batch(self, hidden, cos, sin, views, layer_idx, *,
                              real_lens):
        """批量 prefill(M4.1):hidden [N, Lmax, D](右 padding),cos/sin
        [N, Lmax, head_dim],real_lens[i] = 第 i 行真实长度。

        大 GEMM 合批:qkv/o 按 [N, Lmax] 一次算(pad 行垃圾值只在自己行内
        传播);KV 写入与 attention 逐行 —— 与单序列 forward 的 prefill 分支
        (cache 空,is_causal=(s>1))完全同一语义:update 截断 [:L],pad 永不
        入 KV,逐行 SDPA 天然行间隔离。本方法不接 fused_ops /
        attention_backend 选择器:prefill 批量化与 decode 后端开关解耦。

        警告:rope 公式与 forward(_apply_rope)/forward_decode 各有一份内联
        拷贝(形状约定不同)—— 改公式需三处同步;golden 门是安全网。
        """
        N, Lmax, _ = hidden.shape
        q = dispatch_linear(self.q_proj, hidden).view(N, Lmax, self.num_heads, self.head_dim).transpose(1, 2)
        k = dispatch_linear(self.k_proj, hidden).view(N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = dispatch_linear(self.v_proj, hidden).view(N, Lmax, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # 批量 RoPE(逐 token):cos/sin [N,1,Lmax,D] 广播 [N,H,Lmax,D];
        # pad 位算出的垃圾值随 [:L] 截断丢弃
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        rep = self.num_heads // self.num_kv_heads
        outs = torch.zeros(N, self.num_heads, Lmax, self.head_dim,
                           dtype=hidden.dtype, device=hidden.device)
        for b, view in enumerate(views):
            L = real_lens[b]
            k_full, v_full = view.update(layer_idx, k[b : b + 1, :, :L],
                                         v[b : b + 1, :, :L])
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
            # 镜像单序列 prefill 分支:is_causal=(s>1);L==1 单 query 全可见
            # 与 causal 等价(同 forward 的 s==1 注释)
            outs[b, :, :L] = F.scaled_dot_product_attention(
                q[b : b + 1, :, :L], k_full, v_full, is_causal=(L > 1))[0]
        out = outs.transpose(1, 2).reshape(N, Lmax, -1)
        return dispatch_linear(self.o_proj, out)

    def forward_extend(self, hidden, cos, sin, views, layer_idx, *,
                       starts, chunk_lens):
        """chunk 续读(M5):cursor>0 的追加前向。hidden [N, Lc_max, D](右
        padding),starts[b] = 该行已完成 token 数(past),chunk_lens[b] = 本
        块真实长度。

        逐行 update + SDPA 与 prefill 分支同一结构,差异仅三点:
        - KV 追加的是 chunk(k/v 截断 [:L]),update 返回追加后的全量
          k_full/v_full( past+L );
        - 位置从 starts[b] 起(cos/sin 由上层按行 arange(start, start+L) 算好);
        - 掩码显式右下对齐 bottom-right causal(past>0 时 is_causal 语义错误):
          query i(全局位 past+i)可见 key 0..past+i,即 tril(diagonal=past)。
        SDPA 用默认 scale(head_dim ** -0.5),与全部既有路径一致。

        警告:rope 公式与 forward(_apply_rope)/forward_decode/
        forward_prefill_batch 各有一份内联拷贝(这是第 4 份,形状约定不同)
        —— 改公式需四处同步;golden 门是安全网。
        """
        N, Lc_max, _ = hidden.shape
        q = dispatch_linear(self.q_proj, hidden).view(N, Lc_max, self.num_heads, self.head_dim).transpose(1, 2)
        k = dispatch_linear(self.k_proj, hidden).view(N, Lc_max, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = dispatch_linear(self.v_proj, hidden).view(N, Lc_max, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
            k_full, v_full = view.update(layer_idx, k[b : b + 1, :, :L],
                                         v[b : b + 1, :, :L])
            k_full = _repeat_kv(k_full, rep)
            v_full = _repeat_kv(v_full, rep)
            mask = torch.ones(L, past + L, dtype=torch.bool,
                              device=hidden.device).tril(diagonal=past)[None, None]
            outs[b, :, :L] = F.scaled_dot_product_attention(
                q[b : b + 1, :, :L], k_full, v_full, attn_mask=mask)[0]
        out = outs.transpose(1, 2).reshape(N, Lc_max, -1)
        return dispatch_linear(self.o_proj, out)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dispatch_linear(
            self.down_proj,
            F.silu(dispatch_linear(self.gate_proj, x)) * dispatch_linear(self.up_proj, x),
        )


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
                       attention_backend="sdpa",
                       tables=None, lens=None, scatter=None):
        if not fused_ops:
            hidden = hidden + self.self_attn.forward_decode(
                self.input_layernorm(hidden), cos, sin, views, layer_idx,
                positions=positions, inv_freq=inv_freq, fused_ops=fused_ops,
                attention_backend=attention_backend,
                tables=tables, lens=lens, scatter=scatter,
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
            tables=tables, lens=lens, scatter=scatter,
        )
        normed, residual = fused_add_rmsnorm(
            hidden, residual, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return residual + self.mlp(normed)

    def forward_prefill_batch(self, hidden, cos, sin, views, layer_idx, *,
                              real_lens):
        """批量 prefill(M4.1):RMSNorm 逐 token,天然支持 [N, Lmax] 合批;
        残差结构与单序列 forward 一致(torch 路径,无 fused 分支 —— prefill
        批量化与 fused_ops 解耦)。"""
        hidden = hidden + self.self_attn.forward_prefill_batch(
            self.input_layernorm(hidden), cos, sin, views, layer_idx,
            real_lens=real_lens,
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def forward_extend(self, hidden, cos, sin, views, layer_idx, *,
                       starts, chunk_lens):
        """chunk 续读(M5):RMSNorm 逐 token,天然支持 [N, Lc_max] 合批;
        残差结构与 prefill 分支一致(torch 路径,无 fused 分支 —— 与
        fused_ops/attention_backend 解耦,同 prefill 现状)。"""
        hidden = hidden + self.self_attn.forward_extend(
            self.input_layernorm(hidden), cos, sin, views, layer_idx,
            starts=starts, chunk_lens=chunk_lens,
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def forward_verify(self, hidden, cos, sin, views, layer_idx, *,
                       tables=None, lens=None, scatter=None):
        """verify 步(M8 Task 3):残差结构照 forward_decode 非融合分支
        (RMSNorm 逐 token,[B,m] 合批;spec v1 与 fused_ops 解耦 —— verify
        不接 K1/K2,spec × fused 组合下 verify 腿走 torch rope/残差路径)。"""
        hidden = hidden + self.self_attn.forward_verify(
            self.input_layernorm(hidden), cos, sin, views, layer_idx,
            tables=tables, lens=lens, scatter=scatter,
        )
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
        return dispatch_linear(self.lm_head, hidden).float()

    def forward_decode(self, input_ids: torch.Tensor, views, *, prepared=None) -> torch.Tensor:
        """批量 decode:input_ids [B, 1] + 每序列的 PagedCacheView -> [B, vocab]。

        契约:views 非空(B ≥ 1);views[i] 与 input_ids[i] 一一对应;
        s 必须为 1(decode 步)。
        每序列位置 = 各自 cache.seq_len(互不相同);新 token 的 K/V 在
        attention 内写入各自分页块。
        triton 后端(M3 Task 9):层循环前步首一次 prepare —— 各序列容量确保
        (跨页分新页)+ 块表/长度(含本步新 token = pre-write + 1)+ scatter
        目标一次构建并随层下传,消除原每层每步的 build_tables 重建与逐序列
        write 小发射;RoPE 位置仍为 pre-write seq_len(容量确保不提交长度)。
        prepared(M4,可选):graph 捕获路径的四件套
        (positions, tables, lens, (pages, slots)),由调用方(DecodeGraphRunner)
        在图外完成原 prepare 后传入 —— prepare 的 CPU torch.tensor 构建 + H2D
        若在图内执行,重放会从捕获期冻结的源地址拷贝陈旧数据。其语义与自
        prepare(prepared=None)逐位一致(单测钉死);prepared 仅对 triton 后端
        有意义(sdpa 分支忽略 tables/lens/scatter);None(默认)= 自 prepare,
        既有 eager 执行序不变。
        dtype 契约:positions/scatter = long,tables/lens = build_tables 产物
        (int32);与图外静态缓冲逐字段一致,勿自选 dtype(单测只钉值,不钉 dtype)。
        rope 派生不进四件套:positions 已在设备上,fused 路径纯引用 inv_freq
        buffer、非融合路径为纯 GPU kernel,图内执行可安全捕获,故仍由本函数完成。
        """
        if input_ids.shape[1] != 1:
            # s>1 会静默腐败:cos/sin 广播成同位置、update 多写 token
            raise ValueError("forward_decode expects [B, 1] input_ids (decode step)")
        if prepared is not None:
            # M4 graph 捕获路径:四件套契约、dtype、图外理由见 docstring prepared 契约。
            positions, tables, lens, scatter = prepared
        else:
            positions = torch.tensor([v.seq_len for v in views], device=input_ids.device)
        # rope 派生与 prepare 无数据依赖,位置不动:prepared=None 时执行序与
        # 改前逐语句相同;prepared 路径用传入 positions 走同一派生。
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
        if prepared is None:
            tables = lens = scatter = None
            if self.attention_backend == "triton":
                from acorninfer.ops.paged_attention import build_tables
                # M3 Task 9:步首一次 prepare(层循环之前),替代原每步 36 层 ×
                # (build_tables CPU 构造 + H2D 同步,加 B 次逐序列 write 小算子)。
                page_size = views[0].cache.page_size
                targets = [v.ensure_append_capacity(0) for v in views]  # 跨页者在此分新页
                scatter = (
                    torch.tensor([p for p, _ in targets],
                                 device=input_ids.device, dtype=torch.long),
                    torch.tensor([s for _, s in targets],
                                 device=input_ids.device, dtype=torch.long),
                )
                # lens 含本步新 token(pre-write + 1):旧路径 build_tables 在
                # layer-0 write 之后取 seq_len,得值与此逐位相同;块表行宽与
                # 零填充由 build_tables 自身构造,kernel 输入逐位不变。
                tables, lens = build_tables(
                    views, page_size, input_ids.device,
                    lens_override=[int(v.seq_len) + 1 for v in views],
                )
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_decode(
                hidden, cos, sin, views, i,
                positions=positions, inv_freq=inv_freq,
                fused_ops=self.fused_ops,
                attention_backend=self.attention_backend,
                tables=tables, lens=lens, scatter=scatter,
            )
        hidden = self.norm(hidden)
        return dispatch_linear(self.lm_head, hidden).float()[:, 0, :]  # [B, vocab]

    def forward_prefill_batch(self, input_ids: torch.Tensor, views, *,
                              real_lens) -> torch.Tensor:
        """批量 prefill(M4.1):input_ids [N, Lmax](右 padding,pad 位值任意)
        + real_lens[i](第 i 行真实长度)-> 各行末真实位 logits [N, vocab]
        (float32)。

        大 GEMM 合批 embed/RMSNorm/qkv/o/MLP(pad 行垃圾值只在其行内传播,
        逐行 attention 天然隔离);KV 写入与 attention 逐行(view.update 截断
        [:L])= 单序列 forward 的 prefill 分支同一语义;末层只对各行
        real_lens[i]-1 位置过 lm_head。与 fused_ops/attention_backend 解耦
        (prefill 分支现状即不接后端选择器)。
        契约:views[i] 与 input_ids 第 i 行一一对应(real_lens[i] 同行);
        KV 写入经各自 views[i],行间零串扰。
        """
        N, Lmax = input_ids.shape
        if len(views) != N or len(real_lens) != N:
            raise ValueError(
                f"views/real_lens 必须与 batch 维一一对应: got N={N}, "
                f"len(views)={len(views)}, len(real_lens)={len(real_lens)}"
            )
        if min(real_lens) < 1 or max(real_lens) > Lmax:
            # L==0 无末位可取;L>Lmax 会静默越界/截断错误
            raise ValueError(
                f"real_lens 必须落在 [1, Lmax] 内: got "
                f"min={min(real_lens)}, max={max(real_lens)}, Lmax={Lmax}"
            )
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        # 逐行位置(pad 位 0);展平为一次 rope 调用
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
        return dispatch_linear(self.lm_head, hidden).float()

    def forward_extend(self, input_ids: torch.Tensor, views, *,
                       starts: list[int], chunk_lens: list[int]) -> torch.Tensor:
        """chunk 续读前向(M5):input_ids [N, Lc_max](右 padding,pad 位值
        任意)+ starts[i](该行已完成 token 数 = view 当前 KV 长度)+
        chunk_lens[i](本块真实长度)-> 各行本块末位(start+len-1)logits
        [N, vocab](float32)。

        大 GEMM 合批 embed/RMSNorm/qkv/o/MLP 与 prefill_batch 同款(pad 行
        垃圾值只在其行内传播);KV 追加与 attention 逐行(views[i].update 截断
        [:len_i])—— 追加后 view KV 长度 = starts[i]+len_i;末层只对各行
        start+len-1 位置过 lm_head。与 fused_ops/attention_backend 解耦。
        契约:views[i]/starts[i]/chunk_lens[i] 与 input_ids 第 i 行一一对应;
        views[i] 的 KV 长度必须恰为 starts[i](引擎 cursor 同步不变式)。
        """
        N, Lc_max = input_ids.shape
        if len(views) != N or len(starts) != N or len(chunk_lens) != N:
            raise ValueError(
                f"views/starts/chunk_lens 必须与 batch 维一一对应: got N={N}, "
                f"len(views)={len(views)}, len(starts)={len(starts)}, "
                f"len(chunk_lens)={len(chunk_lens)}"
            )
        if min(chunk_lens) < 1 or max(chunk_lens) > Lc_max or min(starts) < 0:
            # L==0 无末位可取;L>Lc_max / start<0 会静默越界
            raise ValueError(
                f"chunk_lens 必须落在 [1, Lc_max] 且 starts 非负: got "
                f"min(len)={min(chunk_lens)}, max(len)={max(chunk_lens)}, "
                f"Lc_max={Lc_max}, min(start)={min(starts)}"
            )
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        # 逐行绝对位置 pad 位 0;展平为一次 rope 调用
        pos = torch.zeros(N * Lc_max, dtype=torch.long, device=device)
        for i, (s, L) in enumerate(zip(starts, chunk_lens)):
            pos[i * Lc_max: i * Lc_max + L] = torch.arange(s, s + L, device=device)
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
        return dispatch_linear(self.lm_head, hidden).float()

    def forward_verify(self, input_ids: torch.Tensor, views) -> torch.Tensor:
        """spec verify 前向(M8 Task 3,spec §2.2):input_ids [B, m]
        (m = spec_num+1 全批统一;行 = [已提交末位] + K 个草稿)+
        每序列 PagedCacheView -> 全窗口 logits [B, m, vocab](float32)。

        镜像 forward_decode 的 triton 合批结构,步首一次 prepare:
        - 各序列 ensure_append_capacity_n(·, m) —— 跨页者在此分新页
          (返回 m 个游标 (page,slot) 随层下传给 scatter_verify);
        - build_tables(lens_override=[len+m]):块表/长度含本步全部窗口位
          (kernel 读侧契约);与 decode 的 lens_override=[len+1] 同源;
        - 位置 = len + arange(m)(镜像 forward_decode 的 pre-write
          seq_len 推导;ensure 家族不提交长度,seq_len 仍为 pre-write);
          展平 [B*m] 一次 rope 调用(M8 verify 不接 fused_ops:spec v1
          eager-only,torch rope 路径;
        - lm_head 全 m 位 -> [B, m, vocab](接受逻辑在引擎侧逐行 argmax)。
        KV 写入在层循环内 scatter_verify(M8 Task 1);attention =
        paged_verify_attention(M8 Task 2,行 j 可见历史+窗口前缀)。
        契约:views 非空且与 input_ids 行一一对应;attention_backend 必须
        已是 triton(verify kernel 只有 triton 路径)。
        """
        B, m = input_ids.shape
        if len(views) != B:
            raise ValueError(
                f"views 必须与 batch 维一一对应: got B={B}, len(views)={len(views)}")
        if self.attention_backend != "triton":
            raise ValueError(
                f"forward_verify requires attention_backend='triton', "
                f"got {self.attention_backend!r}")
        from acorninfer.ops.paged_attention import build_tables
        device = input_ids.device
        # 步首 prepare(层循环之前,同 forward_decode Task 9 形态;layer 0
        # 保证 —— 层间锁步,table 共享、ensure 按各层等长 lens 推导)
        targets = [v.ensure_append_capacity_n(0, m) for v in views]
        scatter = (
            torch.tensor([[p for p, _ in t] for t in targets],
                         device=device, dtype=torch.long),
            torch.tensor([[s for _, s in t] for t in targets],
                         device=device, dtype=torch.long),
        )
        tables, lens = build_tables(
            views, views[0].cache.page_size, device,
            lens_override=[int(v.seq_len) + m for v in views],
        )
        # 位置 = len + arange(m);展平 [B*m] 一次 rope(与 forward_decode
        # 的 [B] 单位推导同源;pad 无 —— verify 全批统一 m,零 padding)
        positions = torch.cat(
            [torch.arange(int(v.seq_len), int(v.seq_len) + m, device=device)
             for v in views])
        dtype = self.embed_tokens.weight.dtype
        cos, sin = self.rope(positions, dtype, device)      # [B*m, head_dim]
        cos = cos.view(B, m, -1)
        sin = sin.view(B, m, -1)
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_verify(
                hidden, cos, sin, views, i,
                tables=tables, lens=lens, scatter=scatter,
            )
        hidden = self.norm(hidden)
        return dispatch_linear(self.lm_head, hidden).float()  # [B, m, vocab]


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
