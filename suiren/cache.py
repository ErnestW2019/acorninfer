"""Naive per-layer KV cache for a single sequence (batch=1, M1).

Storage is concat-on-append: O(n) copy per token, O(n^2) per generation.
Intentionally naive -- correct and simple. M2's paged cache replaces the
storage behind the same update()/seq_len interface.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, num_layers: int):
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers

    @property
    def seq_len(self) -> int:
        """Cached length (read from layer 0; all layers must stay in lockstep)."""
        return 0 if self.k[0] is None else int(self.k[0].shape[2])

    def update(
        self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this layer's new K/V; returns the full cached K/V.

        k_new/v_new: [1, kv_heads, new_len, head_dim]
        returns:     [1, kv_heads, past_len + new_len, head_dim]
        """
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = torch.cat((self.k[layer_idx], k_new), dim=2)
            self.v[layer_idx] = torch.cat((self.v[layer_idx], v_new), dim=2)
        return self.k[layer_idx], self.v[layer_idx]
