from __future__ import annotations

from enterprise_rag_eval.config import HarnessConfig
from enterprise_rag_eval.evaluation import RagEvaluator, SyntheticQAGenerator, write_ragas_jsonl
from enterprise_rag_eval.guardrails import HealthcareGuardrails
from enterprise_rag_eval.ingestion import DocumentLoader, SemanticChunker
from enterprise_rag_eval.retrieval import CrossEncoderReranker, HashEmbeddingModel, HybridRetriever


def run_local_evaluation(config: HarnessConfig, qa_limit: int | None = None):
    documents = DocumentLoader(config.docs_path).load()
    chunks = SemanticChunker(config.chunking).chunk(documents)
    retriever = HybridRetriever(
        chunks,
        config.retrieval,
        embedding_model=HashEmbeddingModel(config.vector_store.dimensions),
        reranker=CrossEncoderReranker(config.reranker) if config.reranker.enabled else None,
    )
    questions = SyntheticQAGenerator().generate(chunks, limit=qa_limit)
    report = RagEvaluator(
        retriever,
        guardrails=HealthcareGuardrails(config.guardrails),
        thresholds=config.thresholds,
    ).evaluate(questions)
    if config.ragas.enabled:
        write_ragas_jsonl(report.case_results, config.ragas.export_path)
    return documents, chunks, questions, report
