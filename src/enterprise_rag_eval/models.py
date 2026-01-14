from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceDocument:
    id: str
    title: str
    domain: str
    path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    domain: str
    text: str
    token_count: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    bm25_score: float
    semantic_score: float


@dataclass(frozen=True)
class SyntheticQuestion:
    id: str
    question: str
    expected_answer: str
    source_chunk_id: str
    source_document_id: str
