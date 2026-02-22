from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
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

    @app.get("/api/overview", response_model=EvaluationResponse)
    def overview() -> EvaluationResponse:
        documents, chunks, questions, report = run_local_evaluation(settings, qa_limit=20)
        return EvaluationResponse(
            documents=len(documents),
            chunks=len(chunks),
            questions=len(questions),
            passed=report.passed,
            metrics=report.metrics,
            cases=[
                {
                    "question": result.question,
                    "answer": result.answer,
                    "expected_answer": result.expected_answer,
                    "retrieved_chunk_ids": result.retrieved_chunk_ids,
                    "guardrails": {
                        "passed": result.guardrail.passed,
                        "violations": result.guardrail.violations,
                    },
                    "latency_ms": result.latency_ms,
                }
                for result in report.case_results
            ],
        )

    @app.get("/api/documents")
    def documents() -> list[dict[str, object]]:
        source_documents, chunks, _ = corpus()
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

    app.mount("/assets", StaticFiles(directory=Path(str(static_dir))), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(str(static_dir.joinpath("index.html"))))

    return app
