# Benchmark Corpus

The repository includes a deterministic synthetic benchmark corpus for local retrieval and
evaluation testing.

## Purpose

The generated files provide enough document volume to exercise ingestion, chunking, hybrid
retrieval, reranking, metrics, CI gates, and the dashboard without requiring paid APIs or
restricted healthcare source material.

## Composition

- HIPAA-style synthetic operating procedures
- FDA-style synthetic quality guidance
- Clinical-trial-style synthetic protocol procedures

These documents are synthetic test fixtures. They are useful for scale and regression testing, but
they should not be represented as official regulatory guidance.

## Regeneration

```bash
python scripts/generate_benchmark_corpus.py
```
