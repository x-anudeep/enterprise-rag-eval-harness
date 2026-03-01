# Dashboard

The web dashboard provides a dark-mode operational view of the RAG harness.

## Views

- Overview metrics for faithfulness, relevancy, precision, recall, safety, latency, and cost.
- Live retrieval where a user can ask compliance questions and inspect the retrieved contexts.
- Pipeline view that shows ingestion, chunking, retrieval, reranking, generation, and evaluation.
- Evaluation cases with guardrail status and per-case quality scores.
- Corpus browser with domain filtering and document detail.
- Report preview for the generated Markdown evaluation summary.

## Backend APIs

- `GET /api/health`
- `GET /api/overview`
- `POST /api/evaluations?qa_limit=20`
- `GET /api/corpus`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `GET /api/search?q=...`
- `GET /api/reports`
- `GET /api/reports/summary`
