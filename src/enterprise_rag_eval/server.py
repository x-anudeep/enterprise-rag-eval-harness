from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from enterprise_rag_eval.config import HarnessConfig
from enterprise_rag_eval.generation import ExtractiveAnswerGenerator
from enterprise_rag_eval.guardrails import HealthcareGuardrails
from enterprise_rag_eval.ingestion import DocumentLoader, SemanticChunker
from enterprise_rag_eval.pipeline import run_local_evaluation
from enterprise_rag_eval.retrieval import CrossEncoderReranker, HashEmbeddingModel, HybridRetriever


class SearchResponse(BaseModel):
    query: str
    answer: str
    guardrails: dict[str, object]
    latency_ms: float
    contexts: list[dict[str, object]]


class EvaluationResponse(BaseModel):
    documents: int
    chunks: int
    questions: int
    passed: bool
    metrics: dict[str, float]
    cases: list[dict[str, object]]


class CorpusSummary(BaseModel):
    documents: int
    chunks: int
    domains: dict[str, int]
    total_characters: int


class ReportManifest(BaseModel):
    json_report: str
    ragas_export: str
    markdown_summary: str
    summary_available: bool


class UploadResponse(BaseModel):
    id: str
    title: str
    domain: str
    path: str
    characters: int


def create_app(config: HarnessConfig | None = None) -> FastAPI:
    settings = config or HarnessConfig()
    app = FastAPI(title="Enterprise RAG Evaluation Harness")
    static_dir = files("enterprise_rag_eval").joinpath("frontend")

    @lru_cache(maxsize=1)
    def corpus():
        documents = DocumentLoader(settings.docs_path).load()
        chunks = SemanticChunker(settings.chunking).chunk(documents)
        retriever = HybridRetriever(
            chunks,
            settings.retrieval,
            embedding_model=HashEmbeddingModel(settings.vector_store.dimensions),
            reranker=CrossEncoderReranker(settings.reranker) if settings.reranker.enabled else None,
        )
        return documents, chunks, retriever

    def clear_runtime_cache() -> None:
        corpus.cache_clear()

    def evaluate(qa_limit: int = 20) -> EvaluationResponse:
        documents, chunks, questions, report = run_local_evaluation(settings, qa_limit=qa_limit)
        return EvaluationResponse(
            documents=len(documents),
            chunks=len(chunks),
            questions=len(questions),
            passed=report.passed,
            metrics=report.metrics,
            cases=[
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "answer": result.answer,
                    "expected_answer": result.expected_answer,
                    "retrieved_chunk_ids": result.retrieved_chunk_ids,
                    "guardrails": {
                        "passed": result.guardrail.passed,
                        "violations": result.guardrail.violations,
                    },
                    "latency_ms": result.latency_ms,
                    "faithfulness": result.faithfulness,
                    "answer_relevancy": result.answer_relevancy,
                    "context_precision": result.context_precision,
                    "recall_at_3": result.recall_at_3,
                }
                for result in report.case_results
            ],
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "enterprise-rag-eval-harness"}

    @app.get("/api/overview", response_model=EvaluationResponse)
    def overview() -> EvaluationResponse:
        return evaluate()

    @app.post("/api/evaluations", response_model=EvaluationResponse)
    def run_evaluation(qa_limit: int = Query(default=20, ge=1, le=100)) -> EvaluationResponse:
        return evaluate(qa_limit=qa_limit)

    @app.get("/api/corpus", response_model=CorpusSummary)
    def corpus_summary() -> CorpusSummary:
        documents, chunks, _ = corpus()
        domains: dict[str, int] = {}
        for document in documents:
            domains[document.domain] = domains.get(document.domain, 0) + 1
        return CorpusSummary(
            documents=len(documents),
            chunks=len(chunks),
            domains=domains,
            total_characters=sum(len(document.text) for document in documents),
        )

    @app.get("/api/documents")
    def documents(domain: str | None = None) -> list[dict[str, object]]:
        source_documents, chunks, _ = corpus()
        if domain:
            source_documents = [
                document
                for document in source_documents
                if document.domain.lower() == domain.lower()
            ]
        chunk_counts = {
            document.id: len([chunk for chunk in chunks if chunk.document_id == document.id])
            for document in source_documents
        }
        return [
            {
                "id": document.id,
                "title": document.title,
                "domain": document.domain,
                "path": document.path,
                "characters": len(document.text),
                "chunks": chunk_counts[document.id],
            }
            for document in source_documents
        ]

    @app.get("/api/documents/{document_id}")
    def document_detail(document_id: str) -> dict[str, object]:
        source_documents, chunks, _ = corpus()
        document = next((item for item in source_documents if item.id == document_id), None)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "id": document.id,
            "title": document.title,
            "domain": document.domain,
            "path": document.path,
            "text": document.text,
            "chunks": [
                {
                    "id": chunk.id,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                }
                for chunk in chunks
                if chunk.document_id == document.id
            ],
        }

    @app.post("/api/documents/upload", response_model=UploadResponse)
    async def upload_document(
        file: Annotated[UploadFile, File()],
        domain: Annotated[str, Form()] = "uploaded",
    ) -> UploadResponse:
        filename = _safe_filename(file.filename or "uploaded.txt")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md"}:
            raise HTTPException(
                status_code=400,
                detail="Upload text or Markdown files only.",
            )

        raw = await file.read()
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded document must be UTF-8 text.",
            ) from exc
        if not text:
            raise HTTPException(status_code=400, detail="Uploaded document is empty.")

        safe_domain = _safe_filename(domain).replace(".", "_") or "uploaded"
        target_dir = settings.docs_path / safe_domain
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = _unique_path(target_dir / filename)
        target_path.write_text(text + "\n", encoding="utf-8")
        clear_runtime_cache()

        document_id = target_path.relative_to(settings.docs_path).with_suffix("").as_posix()
        return UploadResponse(
            id=document_id.replace("/", "__"),
            title=target_path.stem.replace("_", " ").title(),
            domain=safe_domain.replace("_", " "),
            path=str(target_path),
            characters=len(text),
        )

    @app.get("/api/search", response_model=SearchResponse)
    def search(q: str = Query(min_length=3, max_length=240)) -> SearchResponse:
        _, _, retriever = corpus()
        started = perf_counter()
        contexts = retriever.search(q, top_k=3)
        answer = ExtractiveAnswerGenerator().answer(q, contexts)
        guardrail = HealthcareGuardrails(settings.guardrails).validate(answer, contexts)
        latency_ms = (perf_counter() - started) * 1000
        if not contexts:
            raise HTTPException(status_code=404, detail="No context found")
        return SearchResponse(
            query=q,
            answer=answer,
            guardrails={"passed": guardrail.passed, "violations": guardrail.violations},
            latency_ms=latency_ms,
            contexts=[
                {
                    "chunk_id": result.chunk.id,
                    "title": result.chunk.title,
                    "domain": result.chunk.domain,
                    "score": result.score,
                    "bm25_score": result.bm25_score,
                    "semantic_score": result.semantic_score,
                    "text": result.chunk.text,
                }
                for result in contexts
            ],
        )

    @app.get("/api/reports", response_model=ReportManifest)
    def reports() -> ReportManifest:
        return ReportManifest(
            json_report=str(Path("reports/eval.json")),
            ragas_export=str(settings.ragas.export_path),
            markdown_summary=str(settings.reporting.markdown_path),
            summary_available=settings.reporting.markdown_path.exists(),
        )

    @app.get("/api/reports/summary", response_class=PlainTextResponse)
    def report_summary() -> str:
        if not settings.reporting.markdown_path.exists():
            run_local_evaluation(settings, qa_limit=20)
        return settings.reporting.markdown_path.read_text(encoding="utf-8")

    app.mount("/assets", StaticFiles(directory=Path(str(static_dir))), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(str(static_dir.joinpath("index.html"))))

    return app


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "uploaded.txt"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to create a unique upload path.")


app = create_app()
