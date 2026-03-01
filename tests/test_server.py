from fastapi.testclient import TestClient

from enterprise_rag_eval.server import create_app


def test_dashboard_index_loads() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "RAG Eval Harness" in response.text


def test_overview_api_returns_metrics() -> None:
    client = TestClient(create_app())

    response = client.get("/api/overview")
    payload = response.json()

    assert response.status_code == 200
    assert payload["documents"] >= 9
    assert payload["metrics"]["faithfulness"] >= 0.65
    assert payload["passed"] is True


def test_health_and_corpus_apis_return_runtime_state() -> None:
    client = TestClient(create_app())

    health = client.get("/api/health")
    corpus = client.get("/api/corpus")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert corpus.status_code == 200
    assert corpus.json()["domains"]["hipaa"] >= 3


def test_evaluation_api_can_refresh_limited_cases() -> None:
    client = TestClient(create_app())

    response = client.post("/api/evaluations", params={"qa_limit": 3})
    payload = response.json()

    assert response.status_code == 200
    assert payload["questions"] == 3
    assert len(payload["cases"]) == 3


def test_search_api_returns_answer_and_contexts() -> None:
    client = TestClient(create_app())

    response = client.get(
        "/api/search",
        params={"q": "What safeguards protect electronic health information?"},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["answer"]
    assert payload["guardrails"]["passed"] is True
    assert payload["contexts"]


def test_document_detail_and_reports_are_available() -> None:
    client = TestClient(create_app())

    documents = client.get("/api/documents").json()
    detail = client.get(f"/api/documents/{documents[0]['id']}")
    manifest = client.get("/api/reports")
    summary = client.get("/api/reports/summary")

    assert detail.status_code == 200
    assert detail.json()["chunks"]
    assert manifest.status_code == 200
    assert summary.status_code == 200
    assert "Enterprise RAG Evaluation Summary" in summary.text
