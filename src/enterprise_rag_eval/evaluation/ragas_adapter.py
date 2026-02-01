from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from enterprise_rag_eval.evaluation.evaluator import EvaluationCaseResult


def to_ragas_record(result: EvaluationCaseResult) -> dict[str, object]:
    """Convert a local evaluation case to the shape expected by RAGAS datasets."""

    return {
        "question": result.question,
        "answer": result.answer,
        "contexts": result.contexts,
        "ground_truth": result.expected_answer,
        "metadata": {
            "question_id": result.question_id,
            "retrieved_chunk_ids": result.retrieved_chunk_ids,
            "local_metrics": {
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "context_precision": result.context_precision,
                "recall_at_3": result.recall_at_3,
            },
            "guardrails": asdict(result.guardrail),
        },
    }


def write_ragas_jsonl(results: list[EvaluationCaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(to_ragas_record(result), sort_keys=True) for result in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
