from enterprise_rag_eval.models import Chunk
from enterprise_rag_eval.retrieval import (
    HashEmbeddingModel,
    InMemoryVectorStore,
    PgVectorStore,
    PineconeVectorStore,
    build_vector_records,
)


def _chunk() -> Chunk:
    return Chunk(
        id="doc::chunk-001",
        document_id="doc",
        title="Security Rule",
        domain="hipaa",
        text="Technical safeguards include access controls and audit controls.",
        token_count=8,
        metadata={"source_path": "security.txt"},
    )


def test_in_memory_vector_store_returns_nearest_record() -> None:
    embedder = HashEmbeddingModel(dimensions=32)
    chunk = _chunk()
    records = build_vector_records([chunk], [embedder.embed(chunk.text)])
    store = InMemoryVectorStore()

    store.upsert(records)
    results = store.search(embedder.embed("audit controls"), top_k=1)

    assert results[0].record.id == chunk.id
    assert results[0].score > 0


def test_pinecone_payload_contains_chunk_metadata() -> None:
    embedder = HashEmbeddingModel(dimensions=16)
    chunk = _chunk()
    record = build_vector_records([chunk], [embedder.embed(chunk.text)])[0]
    payload = PineconeVectorStore("rag-index").to_vectors([record])

    assert payload[0]["id"] == chunk.id
    assert payload[0]["metadata"]["document_id"] == "doc"
    assert payload[0]["metadata"]["text"] == chunk.text


def test_pgvector_adapter_builds_rows_and_table_sql() -> None:
    embedder = HashEmbeddingModel(dimensions=16)
    chunk = _chunk()
    record = build_vector_records([chunk], [embedder.embed(chunk.text)])[0]
    store = PgVectorStore(table_name="compliance_chunks")

    assert "VECTOR(16)" in store.create_table_sql(dimensions=16)
    assert store.to_rows([record])[0][0] == chunk.id
