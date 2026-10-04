"""Prompt token accounting.

Uses ``tiktoken`` (cl100k_base) when installed for exact BPE counts; otherwise
falls back to a deterministic estimator calibrated to BPE behaviour on English
and structured text (~4 characters per token, with punctuation-heavy text
counted per symbol run). Both paths are deterministic, so benchmark ratios are
reproducible.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Callable

_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


@lru_cache(maxsize=1)
def _counter() -> tuple[str, Callable[[str], int]]:
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        return "tiktoken/cl100k_base", lambda text: len(enc.encode(text))
    except Exception:  # noqa: BLE001 - optional dependency, any failure means fallback
        def estimate(text: str) -> int:
            total = 0
            for piece in _PIECES.findall(text):
                if piece.isalpha():
                    total += max(1, math.ceil(len(piece) / 4))
                elif piece.isdigit():
                    total += max(1, math.ceil(len(piece) / 3))
                else:
                    total += 1
            return total

        return "heuristic/bpe-approx", estimate


def tokenizer_name() -> str:
    return _counter()[0]


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return _counter()[1](text)
