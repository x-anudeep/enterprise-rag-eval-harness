from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from enterprise_rag_eval.config import HarnessConfig
from enterprise_rag_eval.pipeline import run_local_evaluation
from enterprise_rag_eval.server import create_app

app = typer.Typer(help="Enterprise RAG evaluation harness.")
console = Console()


@app.callback()
def main() -> None:
    """Run local retrieval and evaluation workflows."""


@app.command("run-eval")
def run_eval(
    docs: Annotated[
        Path,
        typer.Option(help="Root folder of source documents."),
    ] = Path("data/raw"),
    output: Annotated[
        Path,
        typer.Option(help="JSON report path."),
    ] = Path("reports/eval.json"),
    qa_limit: Annotated[int, typer.Option(help="Maximum synthetic QA cases.")] = 20,
    rerank: Annotated[bool, typer.Option(help="Enable cross-encoder reranking stage.")] = True,
    ragas_export: Annotated[
        Path,
        typer.Option(help="RAGAS-compatible JSONL export path."),
    ] = Path("reports/ragas_dataset.jsonl"),
    summary: Annotated[
        Path,
        typer.Option(help="Markdown summary report path."),
    ] = Path("reports/eval_summary.md"),
    guardrails: Annotated[bool, typer.Option(help="Enable healthcare guardrail checks.")] = True,
) -> None:
    config = HarnessConfig(docs_path=docs)
    config.reranker.enabled = rerank
    config.ragas.export_path = ragas_export
    config.reporting.markdown_path = summary
    config.guardrails.enabled = guardrails
    documents, chunks, questions, report = run_local_evaluation(config, qa_limit=qa_limit)
    report.write_json(output)

    table = Table(title="RAG Evaluation Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for metric, value in report.metrics.items():
        table.add_row(metric, f"{value:.4f}")
    console.print(table)
    console.print(
        f"Loaded {len(documents)} documents, built {len(chunks)} chunks, "
        f"generated {len(questions)} QA cases."
    )
    if not report.passed:
        raise typer.Exit(code=1)


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="Dashboard host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Dashboard port.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload server on code changes.")] = False,
) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, reload=reload)
