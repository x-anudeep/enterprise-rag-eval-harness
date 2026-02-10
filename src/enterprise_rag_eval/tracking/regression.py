from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegressionFinding:
    metric: str
    baseline: float
    current: float
    delta: float
    regressed: bool


def compare_metric_regressions(
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
    tolerance: float = 0.02,
) -> list[RegressionFinding]:
    findings: list[RegressionFinding] = []
    for metric, baseline in baseline_metrics.items():
        if metric not in current_metrics:
            continue
        current = current_metrics[metric]
        delta = current - baseline
        lower_is_better = metric.startswith("latency") or metric.endswith("cost_usd")
        regressed = delta > tolerance if lower_is_better else delta < -tolerance
        findings.append(
            RegressionFinding(
                metric=metric,
                baseline=baseline,
                current=current,
                delta=delta,
                regressed=regressed,
            )
        )
    return findings


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["metrics"].items()}
