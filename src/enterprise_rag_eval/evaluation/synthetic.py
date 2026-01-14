from __future__ import annotations

from enterprise_rag_eval.models import Chunk, SyntheticQuestion
from enterprise_rag_eval.text import split_sentences, tokenize


class SyntheticQAGenerator:
    """Generate deterministic QA cases from source chunks."""

    def generate(self, chunks: list[Chunk], limit: int | None = None) -> list[SyntheticQuestion]:
        questions: list[SyntheticQuestion] = []
        for chunk in chunks:
            answer = self._representative_sentence(chunk.text)
            if not answer:
                continue
            topic = self._topic(answer)
            questions.append(
                SyntheticQuestion(
                    id=f"qa-{len(questions) + 1:04d}",
                    question=f"What does the {chunk.domain} document say about {topic}?",
                    expected_answer=answer,
                    source_chunk_id=chunk.id,
                    source_document_id=chunk.document_id,
                )
            )
            if limit and len(questions) >= limit:
                break
        return questions

    @staticmethod
    def _representative_sentence(text: str) -> str:
        candidates = [
            sentence for sentence in split_sentences(text) if len(tokenize(sentence)) >= 8
        ]
        return candidates[0] if candidates else ""

    @staticmethod
    def _topic(sentence: str) -> str:
        stopwords = {
            "the",
            "and",
            "that",
            "with",
            "must",
            "shall",
            "should",
            "from",
            "this",
            "have",
            "into",
            "when",
        }
        terms = [term for term in tokenize(sentence) if term not in stopwords and len(term) > 3]
        return " ".join(terms[:4]) or "the policy requirement"
