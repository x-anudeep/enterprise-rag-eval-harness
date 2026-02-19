from __future__ import annotations

from enterprise_rag_eval.config import RerankerConfig
from enterprise_rag_eval.models import RetrievalResult
from enterprise_rag_eval.text import split_sentences, tokenize


class CrossEncoderReranker:
    """Deterministic cross-encoder stand-in for ranking tests.

    A real cross-encoder can replace this class by preserving the same ``rerank`` method.
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not self.config.enabled:
            return results
        query_terms = set(tokenize(query))
        rescored = [
            (
                self._cross_score(query_terms, result.chunk.text) + result.score * 0.25,
                result,
            )
            for result in results
        ]
        return [
            result
            for _, result in sorted(rescored, key=lambda item: item[0], reverse=True)[
                : self.config.top_n
            ]
        ]

    @staticmethod
    def _cross_score(query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        best_overlap = 0.0
        for sentence in split_sentences(text):
            sentence_terms = set(tokenize(sentence))
            overlap = len(query_terms & sentence_terms) / len(query_terms)
            best_overlap = max(best_overlap, overlap)
        return best_overlap
