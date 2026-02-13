# Phase 4 Operations

Phase 4 turns the harness into an operational evaluation workflow.

## Delivered

- Token-based cost estimation for every evaluation run.
- Markdown evaluation summary with pass/fail status, metrics, cost, latency, and slowest cases.
- Regression comparison helpers for baseline-versus-current metric tracking.
- CI artifact upload for JSON, JSONL, and Markdown reports.
- Deployment runbook for local, CI, and production retrieval configurations.

## Runbook

1. Install dependencies with `pip install -e ".[dev]"`.
2. Run `rag-harness run-eval --docs data/raw --output reports/eval.json`.
3. Review `reports/eval_summary.md` for metric drift, slow cases, and guardrail failures.
4. Export `reports/ragas_dataset.jsonl` into RAGAS when model-backed scoring is enabled.
5. Promote changes only when quality gates pass and regressions are within tolerance.

## Production Readiness Notes

- Store JSON reports from the main branch as baselines.
- Compare pull request reports against the latest accepted baseline.
- Track p50 latency and estimated cost per run before swapping in hosted LLM providers.
- Keep guardrail failures as release blockers for healthcare compliance demos.
