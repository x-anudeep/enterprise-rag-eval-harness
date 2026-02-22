# Enterprise RAG + Evaluation Harness

Automated accuracy testing, multi-source retrieval, and RAG-style quality metrics for
healthcare compliance documents.

The harness builds a local, reproducible baseline over HIPAA policies, clinical trial protocols,
and FDA guidance snippets. It includes ingestion, semantic chunking, BM25 plus semantic hybrid
retrieval, deterministic reranking, answer generation, synthetic QA generation, guardrail checks,
RAGAS-compatible exports, lightweight quality metrics, and a CI gate.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
rag-harness run-eval --docs data/raw --output reports/eval.json
pytest
```

## Web Dashboard

```bash
rag-harness serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` to inspect the corpus, retrieval results, generated answers,
quality metrics, safety checks, cost estimates, and report exports in a dark-mode UI.

## Current Architecture

```text
Document ingestion -> semantic chunking -> vector store adapter
      -> hybrid retrieval -> reranking -> answer generation
      -> synthetic QA -> evaluation metrics -> CI gate
```

## Metrics

The evaluator reports:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `recall_at_3`
- `safety_pass_rate`
- `latency_ms_p50`
- `estimated_cost_usd`
- `input_tokens`
- `output_tokens`

These local metrics are intentionally deterministic so CI can block regressions before adding
external model providers.

## Optional Retrieval Backends

The retrieval layer includes dependency-optional adapter boundaries for Pinecone and pgvector.
Local tests validate payload construction and SQL shape without requiring cloud credentials:

```bash
pip install -e ".[pinecone]"
pip install -e ".[postgres]"
```

## RAGAS And Guardrails

The harness writes a RAGAS-compatible JSONL dataset during every eval run:

```bash
rag-harness run-eval \
  --docs data/raw \
  --output reports/eval.json \
  --ragas-export reports/ragas_dataset.jsonl
```

Optional extras are available when connecting the local deterministic harness to the real packages:

```bash
pip install -e ".[ragas,guardrails]"
```

## Operational Reports

Every eval run writes:

- JSON metrics and per-case results via `--output`
- RAGAS-compatible JSONL via `--ragas-export`
- Markdown summary with cost, latency, and slowest cases via `--summary`
