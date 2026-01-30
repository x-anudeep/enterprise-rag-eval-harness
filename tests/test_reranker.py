from enterprise_rag_eval.config import RerankerConfig
from enterprise_rag_eval.models import Chunk, RetrievalResult
from enterprise_rag_eval.retrieval import CrossEncoderReranker


def test_reranker_promotes_best_sentence_overlap() -> None:
    weak = RetrievalResult(
        chunk=Chunk(
            id="weak",
            document_id="doc",
            title="Other",
            domain="fda",
            text="Manufacturers keep records for inspections.",
            token_count=5,
        ),
        score=0.9,
        bm25_score=0.9,
        semantic_score=0.0,
    )
    strong = RetrievalResult(
        chunk=Chunk(
            id="strong",
            document_id="doc",
            title="HIPAA",
            domain="hipaa",
            text="Technical safeguards include audit controls and transmission security.",
            token_count=8,
        ),
        score=0.4,
        bm25_score=0.4,
        semantic_score=0.0,
    )

    reranked = CrossEncoderReranker(RerankerConfig(top_n=2)).rerank(
        "Which technical safeguards include audit controls?",
        [weak, strong],
    )

    assert reranked[0].chunk.id == "strong"
