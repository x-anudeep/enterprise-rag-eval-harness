from __future__ import annotations

from pathlib import Path

from enterprise_rag_eval.evaluation.evaluator import EvaluationReport
from enterprise_rag_eval.tracking.costs import CostEstimate


def write_markdown_report(
    report: EvaluationReport,
    cost: CostEstimate,
    path: Path,
    title: str = "Enterprise RAG Evaluation Summary",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"**Status:** {'PASS' if report.passed else 'FAIL'}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {metric} | {value:.4f} |" for metric, value in report.metrics.items())
    lines.extend(
        [
            "",
            "## Cost And Latency",
            "",
            f"- Input tokens: {cost.input_tokens}",
            f"- Output tokens: {cost.output_tokens}",
            f"- Estimated cost: ${cost.estimated_cost_usd:.6f}",
            f"- Median latency: {report.metrics['latency_ms_p50']:.2f} ms",
            "",
            "## Slowest Cases",
            "",
            "| Question | Latency ms | Guardrails |",
            "| --- | ---: | --- |",
        ]
    )
    slowest = sorted(report.case_results, key=lambda result: result.latency_ms, reverse=True)[:5]
    for result in slowest:
        question = result.question.replace("|", "\\|")
        guardrails = "pass" if result.guardrail.passed else ", ".join(result.guardrail.violations)
        lines.append(f"| {question} | {result.latency_ms:.2f} | {guardrails} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
