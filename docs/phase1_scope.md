# Phase 1 Scope

Phase 1 establishes a local RAG baseline that can run in CI without cloud credentials.

## Delivered

- Domain-organized ingestion for healthcare compliance text files.
- Sentence-aware semantic chunking with configurable overlap.
- Deterministic BM25 plus hash-vector semantic retrieval.
- Extractive generation baseline for reproducible tests.
- Synthetic QA generation from ground-truth chunks.
- Lightweight evaluation metrics aligned to RAGAS concepts.
- GitHub Actions workflow that runs linting, tests, and the eval gate.

## Deferred

- Pinecone and pgvector persistence.
- Cross-encoder re-ranking.
- External LLM answer generation.
- Full RAGAS metric execution.
- Guardrails AI runtime policies.
- Latency and cost dashboards.
