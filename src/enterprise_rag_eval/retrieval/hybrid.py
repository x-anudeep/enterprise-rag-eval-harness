from __future__ import annotations

import math
from collections import Counter

from enterprise_rag_eval.config import RetrievalConfig
from enterprise_rag_eval.models import Chunk, RetrievalResult
from enterprise_rag_eval.retrieval.embeddings import HashEmbeddingModel
from enterprise_rag_eval.retrieval.reranker import CrossEncoderReranker
from enterprise_rag_eval.retrieval.stores import InMemoryVectorStore, build_vector_records
from enterprise_rag_eval.text import tokenize


class HybridRetriever:
    """Deterministic BM25 plus hash-vector semantic retriever for local CI."""

    def __init__(
        self,
        chunks: list[Chunk],
        config: RetrievalConfig | None = None,
        embedding_model: HashEmbeddingModel | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.chunks = chunks
        self.config = config or RetrievalConfig()
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.reranker = reranker
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.document_frequency = self._document_frequency()
        self.avg_doc_len = self._average_length()
        self.vectors = [self.embedding_model.embed(chunk.text) for chunk in chunks]
        self.vector_store = InMemoryVectorStore()
        self.vector_store.upsert(build_vector_records(chunks, self.vectors))

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        query_vector = self.embedding_model.embed(query)
        scored: list[RetrievalResult] = []
        bm25_scores = [self._bm25(query, index) for index in range(len(self.chunks))]
        semantic_by_id = {
            result.record.id: result.score
            for result in self.vector_store.search(query_vector, top_k=len(self.chunks))
        }
        normalized_bm25 = self._normalize(bm25_scores)

        for chunk, bm25_score in zip(self.chunks, normalized_bm25, strict=True):
            semantic_score = semantic_by_id.get(chunk.id, 0.0)
            score = (
                self.config.bm25_weight * bm25_score
                + self.config.semantic_weight * semantic_score
            )
            scored.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                )
            )

        ranked = sorted(scored, key=lambda result: result.score, reverse=True)
        if self.reranker:
            return self.reranker.rerank(query, ranked[: max(top_k or self.config.top_k, 10)])
        return ranked[: top_k or self.config.top_k]

    def _document_frequency(self) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for terms in self.term_frequencies:
            frequency.update(set(terms))
        return frequency

    def _average_length(self) -> float:
        if not self.term_frequencies:
            return 0.0
        return sum(sum(terms.values()) for terms in self.term_frequencies) / len(
            self.term_frequencies
        )

    def _bm25(self, query: str, index: int) -> float:
        terms = self.term_frequencies[index]
        doc_len = sum(terms.values())
        if not terms or not self.avg_doc_len:
            return 0.0

        k1 = 1.5
        b = 0.75
        score = 0.0
        for term in tokenize(query):
            tf = terms[term]
            if tf == 0:
                continue
            idf = math.log(1 + (len(self.chunks) - self.document_frequency[term] + 0.5) / (
                self.document_frequency[term] + 0.5
            ))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)
            score += idf * numerator / denominator
        return score

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        high = max(scores)
        if high == 0:
            return [0.0 for _ in scores]
        return [score / high for score in scores]
