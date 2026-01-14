from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_PATTERN.split(text.strip()) if part.strip()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
