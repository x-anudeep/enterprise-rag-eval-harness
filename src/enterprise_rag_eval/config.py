from __future__ import annotations

from pathlib import Path
from typing import Literal

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


class VectorStoreConfig(BaseModel):
    """Configuration for local and production vector stores."""

    provider: Literal["local", "pgvector", "pinecone"] = "local"
    collection_name: str = "healthcare_compliance_chunks"
    dimensions: int = Field(default=96, ge=8)
    postgres_dsn: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index: str | None = None


class RerankerConfig(BaseModel):
    """Configuration for the cross-encoder reranking stage."""

    enabled: bool = True
    top_n: int = Field(default=3, ge=1)


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
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)
