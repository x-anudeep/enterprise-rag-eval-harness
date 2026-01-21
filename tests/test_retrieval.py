from pathlib import Path

from enterprise_rag_eval.ingestion import DocumentLoader, SemanticChunker
from enterprise_rag_eval.retrieval import HybridRetriever


def test_hybrid_retriever_returns_relevant_hipaa_context() -> None:
    documents = DocumentLoader(Path("data/raw")).load()
    chunks = SemanticChunker().chunk(documents)
    retriever = HybridRetriever(chunks)

    results = retriever.search("What safeguards protect electronic health information?", top_k=3)

    assert results
    assert any("safeguards" in result.chunk.text.lower() for result in results)
    assert results[0].score >= results[-1].score
