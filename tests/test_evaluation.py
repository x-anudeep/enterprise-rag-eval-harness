from pathlib import Path

from enterprise_rag_eval.config import HarnessConfig
from enterprise_rag_eval.pipeline import run_local_evaluation


def test_local_evaluation_generates_passing_report() -> None:
    _, chunks, questions, report = run_local_evaluation(
        HarnessConfig(docs_path=Path("data/raw")),
        qa_limit=9,
    )

    assert chunks
    assert len(questions) == 9
    assert report.passed
    assert report.metrics["faithfulness"] >= 0.65
    assert report.metrics["recall_at_3"] >= 0.55
