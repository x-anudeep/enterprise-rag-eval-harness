# Phase 3 Quality Layer

Phase 3 adds evaluation and safety controls around the retrieval pipeline.

## Delivered

- RAGAS-compatible JSONL export with questions, answers, contexts, ground truth, local metrics,
  and guardrail metadata.
- Deterministic healthcare guardrails that block direct identifiers and unsupported answers.
- Safety pass rate metric included in the CI quality gate.
- Optional dependency extras for `ragas` and `guardrails-ai`.
- Tests for guardrail violations, RAGAS record shape, and pipeline exports.

## Production Hook Points

- Replace the local JSONL export with a `datasets.Dataset` object for direct RAGAS execution.
- Swap `HealthcareGuardrails` internals with Guardrails AI validators while preserving the
  `validate(answer, contexts)` contract.
- Add calibrated metric thresholds once model-backed generation replaces the extractive baseline.
