# Enterprise RAG + Evaluation Harness

Automated accuracy testing, multi-source retrieval, and RAG-style quality metrics for
healthcare compliance documents.

Phase 1 builds a local, reproducible baseline over HIPAA policies, clinical trial protocols,
and FDA guidance snippets. It includes ingestion, semantic chunking, BM25 plus semantic hybrid
retrieval, deterministic answer generation, synthetic QA generation, lightweight quality metrics,
and a CI gate.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
rag-harness run-eval --docs data/raw --output reports/phase1_eval.json
pytest
```

## Phase Plan

1. **Phase 1 - Local baseline:** ingestion, chunking, local hybrid retriever, synthetic QA, metrics,
   tests, and CI quality gate.
2. **Phase 2 - Production retrieval:** Pinecone and pgvector adapters, cross-encoder reranking,
   and larger benchmark corpora.
3. **Phase 3 - RAGAS + guardrails:** RAGAS integration, Guardrails AI policies, and metric
   calibration.
4. **Phase 4 - Ops hardening:** latency and cost telemetry, dashboards, regression reports, and
   deployment documentation.

## Current Architecture

```text
Document ingestion -> semantic chunking -> local hybrid index
      -> answer generation -> synthetic QA -> evaluation metrics -> CI gate
```

## Metrics

The Phase 1 evaluator reports:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `recall_at_3`
- `latency_ms_p50`
- `estimated_cost_usd`

These local metrics are intentionally deterministic so CI can block regressions before adding
external model providers.
