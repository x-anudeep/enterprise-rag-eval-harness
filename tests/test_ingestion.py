from pathlib import Path

from enterprise_rag_eval.config import ChunkingConfig
from enterprise_rag_eval.ingestion import DocumentLoader, SemanticChunker


def test_loader_reads_domain_documents() -> None:
    documents = DocumentLoader(Path("data/raw")).load()

    assert len(documents) >= 6
    assert {document.domain for document in documents} >= {"hipaa", "clinical trials", "fda"}


def test_semantic_chunker_preserves_source_metadata() -> None:
    documents = DocumentLoader(Path("data/raw")).load()
    chunks = SemanticChunker(ChunkingConfig(max_tokens=60, overlap_tokens=10)).chunk(documents)

    assert chunks
    assert all(chunk.document_id for chunk in chunks)
    assert all(chunk.metadata["source_path"].endswith(".txt") for chunk in chunks)
