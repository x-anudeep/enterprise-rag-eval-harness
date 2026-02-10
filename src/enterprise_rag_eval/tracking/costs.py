from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag_eval.config import CostConfig
from enterprise_rag_eval.evaluation.evaluator import EvaluationCaseResult
from enterprise_rag_eval.text import tokenize


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class CostTracker:
    """Estimate local evaluation cost from prompt, context, and answer tokens."""

    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()

    def estimate(self, results: list[EvaluationCaseResult]) -> CostEstimate:
        input_tokens = 0
        output_tokens = 0
        for result in results:
            input_tokens += len(tokenize(result.question))
            input_tokens += sum(len(tokenize(context)) for context in result.contexts)
            output_tokens += len(tokenize(result.answer))

        cost = (
            input_tokens / 1000 * self.config.input_cost_per_1k_tokens
            + output_tokens / 1000 * self.config.output_cost_per_1k_tokens
            + self.config.fixed_eval_overhead_usd
        )
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 8),
        )
