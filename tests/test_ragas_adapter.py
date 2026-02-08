import json

from enterprise_rag_eval.evaluation.evaluator import EvaluationCaseResult
from enterprise_rag_eval.evaluation.ragas_adapter import to_ragas_record, write_ragas_jsonl
from enterprise_rag_eval.guardrails import GuardrailResult


def _case() -> EvaluationCaseResult:
    return EvaluationCaseResult(
        question_id="qa-0001",
        question="What does the policy say about audit controls?",
        answer="Technical safeguards include audit controls.",
        expected_answer="Technical safeguards include audit controls.",
        contexts=["Technical safeguards include audit controls and transmission security."],
        retrieved_chunk_ids=["doc::chunk-001"],
        faithfulness=1.0,
        answer_relevancy=1.0,
        context_precision=0.8,
        recall_at_3=1.0,
        guardrail=GuardrailResult(passed=True, violations=[]),
        latency_ms=1.2,
    )


def test_ragas_record_shape_contains_required_fields() -> None:
    record = to_ragas_record(_case())

    assert record["question"]
    assert record["answer"]
    assert record["contexts"]
    assert record["ground_truth"]
    assert record["metadata"]["guardrails"]["passed"] is True


def test_write_ragas_jsonl(tmp_path) -> None:
    path = tmp_path / "ragas.jsonl"

    write_ragas_jsonl([_case()], path)

    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["metadata"]["question_id"] == "qa-0001"
