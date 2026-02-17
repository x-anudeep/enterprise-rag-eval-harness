from pathlib import Path

from enterprise_rag_eval.config import HarnessConfig
from enterprise_rag_eval.pipeline import run_local_evaluation
from enterprise_rag_eval.tracking import compare_metric_regressions, load_metrics


def test_pipeline_writes_markdown_summary_and_cost_metrics(tmp_path) -> None:
    config = HarnessConfig(docs_path=Path("data/raw"))
    config.ragas.export_path = tmp_path / "ragas.jsonl"
    config.reporting.markdown_path = tmp_path / "summary.md"

    _, _, _, report = run_local_evaluation(config, qa_limit=4)

    assert report.metrics["estimated_cost_usd"] > 0
    assert report.metrics["input_tokens"] > 0
    assert report.metrics["output_tokens"] > 0
    assert "Enterprise RAG Evaluation Summary" in config.reporting.markdown_path.read_text(
        encoding="utf-8"
    )


def test_regression_comparison_flags_metric_drift() -> None:
    findings = compare_metric_regressions(
        baseline_metrics={"faithfulness": 0.9, "latency_ms_p50": 100.0},
        current_metrics={"faithfulness": 0.85, "latency_ms_p50": 140.0},
        tolerance=0.02,
    )

    assert {finding.metric for finding in findings if finding.regressed} == {
        "faithfulness",
        "latency_ms_p50",
    }


def test_load_metrics_reads_report_json(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text('{"metrics": {"faithfulness": 1.0}}', encoding="utf-8")

    assert load_metrics(path)["faithfulness"] == 1.0
