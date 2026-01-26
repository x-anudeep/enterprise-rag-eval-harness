# Phase 2 Retrieval Layer

Phase 2 turns the local baseline into a production-shaped retrieval system while keeping CI fully
offline.

## Delivered

- Shared embedding model interface using deterministic hash vectors for local runs.
- In-memory vector store with the same upsert/search shape as production stores.
- pgvector adapter boundary with table DDL and row construction.
- Pinecone adapter boundary with upsert payload construction.
- Cross-encoder reranker interface with a deterministic lexical implementation.
- Expanded healthcare compliance corpus for retrieval regression coverage.

## Production Hook Points

- Replace `HashEmbeddingModel` with a hosted embedding provider.
- Initialize `PineconeVectorStore` with a real Pinecone index object.
- Initialize `PgVectorStore` with a psycopg connection and hydrate search results.
- Replace `CrossEncoderReranker` scoring with a real cross-encoder model.

The local implementations are deliberately deterministic so evaluation runs remain stable in CI.
