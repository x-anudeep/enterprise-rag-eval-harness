from __future__ import annotations

import hashlib
import math
from collections import Counter

from enterprise_rag_eval.config import RetrievalConfig
from enterprise_rag_eval.models import Chunk, RetrievalResult
from enterprise_rag_eval.text import tokenize


class HybridRetriever:
    """Deterministic BM25 plus hash-vector semantic retriever for local CI."""

    def __init__(self, chunks: list[Chunk], config: RetrievalConfig | None = None) -> None:
        self.chunks = chunks
        self.config = config or RetrievalConfig()
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.document_frequency = self._document_frequency()
        self.avg_doc_len = self._average_length()
        self.vectors = [self._embed(chunk.text) for chunk in chunks]

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        query_vector = self._embed(query)
        scored: list[RetrievalResult] = []
        bm25_scores = [self._bm25(query, index) for index in range(len(self.chunks))]
        semantic_scores = [self._cosine(query_vector, vector) for vector in self.vectors]
        normalized_bm25 = self._normalize(bm25_scores)

        for chunk, bm25_score, semantic_score in zip(
            self.chunks, normalized_bm25, semantic_scores, strict=True
        ):
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

        return sorted(scored, key=lambda result: result.score, reverse=True)[
            : top_k or self.config.top_k
        ]

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
    def _embed(text: str, dimensions: int = 96) -> list[float]:
        vector = [0.0] * dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % dimensions
            sign = 1 if digest[2] % 2 == 0 else -1
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return max(0.0, sum(a * b for a, b in zip(left, right, strict=True)))

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        high = max(scores)
        if high == 0:
            return [0.0 for _ in scores]
        return [score / high for score in scores]
