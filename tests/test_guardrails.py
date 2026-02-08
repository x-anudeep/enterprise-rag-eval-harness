from enterprise_rag_eval.config import GuardrailsConfig
from enterprise_rag_eval.guardrails import HealthcareGuardrails
from enterprise_rag_eval.models import Chunk, RetrievalResult


def _context(text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            id="ctx",
            document_id="doc",
            title="Policy",
            domain="hipaa",
            text=text,
            token_count=10,
        ),
        score=1.0,
        bm25_score=1.0,
        semantic_score=1.0,
    )


def test_guardrails_block_direct_identifiers() -> None:
    guardrails = HealthcareGuardrails()

    result = guardrails.validate(
        "Patient email jane.doe@example.com should not be exposed.",
        [_context("Patient email jane.doe@example.com should not be exposed.")],
    )

    assert not result.passed
    assert "direct_identifier:email" in result.violations


def test_guardrails_block_unsupported_answers() -> None:
    guardrails = HealthcareGuardrails()

    result = guardrails.validate(
        "The policy requires daily public breach notices.",
        [_context("The policy requires notification without unreasonable delay.")],
    )

    assert not result.passed
    assert "answer_not_supported_by_retrieved_context" in result.violations


def test_guardrails_can_be_disabled() -> None:
    guardrails = HealthcareGuardrails(GuardrailsConfig(enabled=False))

    result = guardrails.validate("MRN: 123456", [])

    assert result.passed
    assert result.violations == []
