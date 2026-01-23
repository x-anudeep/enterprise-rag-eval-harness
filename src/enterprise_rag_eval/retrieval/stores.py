from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from enterprise_rag_eval.models import Chunk


@dataclass(frozen=True)
class VectorRecord:
    id: str
    values: list[float]
    chunk: Chunk
    metadata: dict[str, str]


@dataclass(frozen=True)
class VectorSearchResult:
    record: VectorRecord
    score: float


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None:
        """Persist vector records."""

    def search(self, vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Return nearest vector records."""


class InMemoryVectorStore:
    """Production-shaped local vector store for tests and offline demos."""

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self.records[record.id] = record

    def search(self, vector: list[float], top_k: int) -> list[VectorSearchResult]:
        results = [
            VectorSearchResult(record=record, score=_cosine(vector, record.values))
            for record in self.records.values()
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


class PgVectorStore:
    """pgvector adapter boundary.

    Phase 2 keeps this dependency-optional: callers can inspect SQL and batches locally, while
    production code can pass an open psycopg connection and execute the same statements.
    """

    def __init__(self, table_name: str = "rag_chunks", connection: object | None = None) -> None:
        self.table_name = table_name
        self.connection = connection

    def create_table_sql(self, dimensions: int) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            "id TEXT PRIMARY KEY, document_id TEXT NOT NULL, title TEXT NOT NULL, "
            "domain TEXT NOT NULL, text TEXT NOT NULL, embedding VECTOR"
            f"({dimensions}), metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb);"
        )

    def to_rows(
        self,
        records: list[VectorRecord],
    ) -> list[tuple[str, str, str, str, str, list[float], dict[str, str]]]:
        return [
            (
                record.id,
                record.chunk.document_id,
                record.chunk.title,
                record.chunk.domain,
                record.chunk.text,
                record.values,
                record.metadata,
            )
            for record in records
        ]

    def upsert(self, records: list[VectorRecord]) -> None:
        if self.connection is None:
            raise RuntimeError("PgVectorStore.upsert requires a psycopg connection.")
        rows = self.to_rows(records)
        with self.connection.cursor() as cursor:
            cursor.executemany(
                f"""
                INSERT INTO {self.table_name}
                    (id, document_id, title, domain, text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )

    def search(self, vector: list[float], top_k: int) -> list[VectorSearchResult]:
        raise NotImplementedError(
            "Use HybridRetriever local search until pgvector hydration is enabled."
        )


class PineconeVectorStore:
    """Pinecone adapter boundary with testable payload construction."""

    def __init__(self, index_name: str, index: object | None = None) -> None:
        self.index_name = index_name
        self.index = index

    def to_vectors(self, records: list[VectorRecord]) -> list[dict[str, object]]:
        return [
            {
                "id": record.id,
                "values": record.values,
                "metadata": {
                    **record.metadata,
                    "document_id": record.chunk.document_id,
                    "title": record.chunk.title,
                    "domain": record.chunk.domain,
                    "text": record.chunk.text,
                },
            }
            for record in records
        ]

    def upsert(self, records: list[VectorRecord]) -> None:
        if self.index is None:
            raise RuntimeError("PineconeVectorStore.upsert requires an initialized Pinecone index.")
        self.index.upsert(vectors=self.to_vectors(records))

    def search(self, vector: list[float], top_k: int) -> list[VectorSearchResult]:
        raise NotImplementedError(
            "Use HybridRetriever local search until Pinecone hydration is enabled."
        )


def build_vector_records(chunks: list[Chunk], vectors: list[list[float]]) -> list[VectorRecord]:
    return [
        VectorRecord(
            id=chunk.id,
            values=vector,
            chunk=chunk,
            metadata={
                **chunk.metadata,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "domain": chunk.domain,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm))
