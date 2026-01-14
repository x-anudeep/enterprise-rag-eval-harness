from __future__ import annotations

from enterprise_rag_eval.models import RetrievalResult
from enterprise_rag_eval.text import split_sentences, tokenize


class ExtractiveAnswerGenerator:
    """Select the highest-overlap context sentence as a deterministic baseline answer."""

    def answer(self, question: str, contexts: list[RetrievalResult]) -> str:
        question_terms = set(tokenize(question))
        best_sentence = ""
        best_score = -1
        for result in contexts:
            for sentence in split_sentences(result.chunk.text):
                sentence_terms = set(tokenize(sentence))
                score = len(question_terms & sentence_terms)
                if score > best_score:
                    best_sentence = sentence
                    best_score = score
        return best_sentence or "No answer found in the retrieved context."
