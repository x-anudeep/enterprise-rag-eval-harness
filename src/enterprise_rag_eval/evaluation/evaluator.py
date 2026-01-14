from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from enterprise_rag_eval.config import EvalThresholds
from enterprise_rag_eval.generation import ExtractiveAnswerGenerator
from enterprise_rag_eval.models import RetrievalResult, SyntheticQuestion
from enterprise_rag_eval.retrieval import HybridRetriever
from enterprise_rag_eval.text import jaccard, tokenize


@dataclass(frozen=True)
class EvaluationCaseResult:
    question_id: str
    question: str
    answer: str
    expected_answer: str
    retrieved_chunk_ids: list[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    recall_at_3: float
    latency_ms: float


@dataclass(frozen=True)
class EvaluationReport:
    metrics: dict[str, float]
    passed: bool
    case_results: list[EvaluationCaseResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": self.metrics,
            "passed": self.passed,
            "case_results": [asdict(result) for result in self.case_results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class RagEvaluator:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: ExtractiveAnswerGenerator | None = None,
        thresholds: EvalThresholds | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or ExtractiveAnswerGenerator()
        self.thresholds = thresholds or EvalThresholds()

    def evaluate(self, questions: list[SyntheticQuestion]) -> EvaluationReport:
        results = [self._evaluate_case(question) for question in questions]
        metrics = self._aggregate(results)
        passed = self._passed(metrics)
        return EvaluationReport(metrics=metrics, passed=passed, case_results=results)

    def _evaluate_case(self, question: SyntheticQuestion) -> EvaluationCaseResult:
        started = time.perf_counter()
        contexts = self.retriever.search(question.question, top_k=3)
        answer = self.generator.answer(question.question, contexts)
        latency_ms = (time.perf_counter() - started) * 1000

        context_text = " ".join(result.chunk.text for result in contexts)
        retrieved_ids = [result.chunk.id for result in contexts]
        return EvaluationCaseResult(
            question_id=question.id,
            question=question.question,
            answer=answer,
            expected_answer=question.expected_answer,
            retrieved_chunk_ids=retrieved_ids,
            faithfulness=self._faithfulness(answer, context_text),
            answer_relevancy=jaccard(tokenize(answer), tokenize(question.expected_answer)),
            context_precision=self._context_precision(contexts, question.expected_answer),
            recall_at_3=1.0 if question.source_chunk_id in retrieved_ids else 0.0,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _faithfulness(answer: str, context_text: str) -> float:
        answer_terms = set(tokenize(answer))
        context_terms = set(tokenize(context_text))
        if not answer_terms:
            return 0.0
        return len(answer_terms & context_terms) / len(answer_terms)

    @staticmethod
    def _context_precision(contexts: list[RetrievalResult], expected_answer: str) -> float:
        expected_terms = tokenize(expected_answer)
        if not contexts:
            return 0.0
        scored = [
            jaccard(tokenize(result.chunk.text), expected_terms)
            for result in contexts
        ]
        return max(scored)

    @staticmethod
    def _aggregate(results: list[EvaluationCaseResult]) -> dict[str, float]:
        if not results:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "recall_at_3": 0.0,
                "latency_ms_p50": 0.0,
                "estimated_cost_usd": 0.0,
            }
        return {
            "faithfulness": statistics.fmean(result.faithfulness for result in results),
            "answer_relevancy": statistics.fmean(result.answer_relevancy for result in results),
            "context_precision": statistics.fmean(result.context_precision for result in results),
            "recall_at_3": statistics.fmean(result.recall_at_3 for result in results),
            "latency_ms_p50": statistics.median(result.latency_ms for result in results),
            "estimated_cost_usd": 0.0,
        }

    def _passed(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["faithfulness"] >= self.thresholds.faithfulness
            and metrics["answer_relevancy"] >= self.thresholds.answer_relevancy
            and metrics["context_precision"] >= self.thresholds.context_precision
            and metrics["recall_at_3"] >= self.thresholds.recall_at_3
        )
