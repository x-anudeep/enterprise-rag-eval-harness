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
