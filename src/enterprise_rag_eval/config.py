from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Configuration for semantic chunk creation."""

    max_tokens: int = Field(default=140, ge=40)
    overlap_tokens: int = Field(default=24, ge=0)
    min_tokens: int = Field(default=20, ge=1)


class RetrievalConfig(BaseModel):
    """Hybrid retrieval weights."""

    top_k: int = Field(default=4, ge=1)
    bm25_weight: float = Field(default=0.55, ge=0, le=1)
    semantic_weight: float = Field(default=0.45, ge=0, le=1)


class EvalThresholds(BaseModel):
    """Minimum quality thresholds for CI."""

    faithfulness: float = 0.65
    answer_relevancy: float = 0.45
    context_precision: float = 0.30
    recall_at_3: float = 0.55


class HarnessConfig(BaseModel):
    docs_path: Path = Path("data/raw")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)
