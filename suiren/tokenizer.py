"""Thin wrapper around the HF tokenizers library.

The engine's own tokenization path; `transformers` is never imported here.
"""

from __future__ import annotations

from tokenizers import Tokenizer as _HFTokenizer


class Tokenizer:
    def __init__(self, tokenizer_json: str):
        self._tk = _HFTokenizer.from_file(tokenizer_json)

    def encode(self, text: str) -> list[int]:
        """Text to token ids; no special tokens are added."""
        return self._tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        """Token ids to text, skipping special tokens."""
        return self._tk.decode(ids, skip_special_tokens=True)
