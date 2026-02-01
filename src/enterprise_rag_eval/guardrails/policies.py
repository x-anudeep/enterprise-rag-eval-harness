from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_rag_eval.config import GuardrailsConfig
from enterprise_rag_eval.models import RetrievalResult


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    violations: list[str]


class HealthcareGuardrails:
    """Deterministic healthcare compliance guardrails for local CI.

    These checks mirror the policy surface that Guardrails AI would enforce in production while
    keeping Phase 3 free of external model calls.
    """

    DIRECT_IDENTIFIER_PATTERNS = {
        "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "medical_record_number": re.compile(r"\bMRN[:\s-]*\d{5,}\b", re.IGNORECASE),
    }

    def __init__(self, config: GuardrailsConfig | None = None) -> None:
        self.config = config or GuardrailsConfig()

    def validate(self, answer: str, contexts: list[RetrievalResult]) -> GuardrailResult:
        if not self.config.enabled:
            return GuardrailResult(passed=True, violations=[])

        violations: list[str] = []
        if self.config.block_direct_identifiers:
            violations.extend(self._direct_identifier_violations(answer))
        if self.config.require_citation_context and not self._answer_supported(answer, contexts):
            violations.append("answer_not_supported_by_retrieved_context")

        return GuardrailResult(passed=not violations, violations=violations)

    def _direct_identifier_violations(self, answer: str) -> list[str]:
        return [
            f"direct_identifier:{name}"
            for name, pattern in self.DIRECT_IDENTIFIER_PATTERNS.items()
            if pattern.search(answer)
        ]

    @staticmethod
    def _answer_supported(answer: str, contexts: list[RetrievalResult]) -> bool:
        normalized_answer = " ".join(answer.lower().split())
        if not normalized_answer or normalized_answer.startswith("no answer found"):
            return False
        return any(
            normalized_answer in " ".join(result.chunk.text.lower().split())
            for result in contexts
        )
