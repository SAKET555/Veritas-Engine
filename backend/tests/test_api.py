from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(main, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    with TestClient(main.app) as c:
        yield c


def observe(c: TestClient, **body) -> dict:  # type: ignore[no-untyped-def]
    r = c.post("/api/observe", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_health_on_empty_ledger(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["ledger_records"] == 0 and body["llm_model"] == "mock-grounded-reader-v1"


def test_observe_text_and_structured(client: TestClient) -> None:
    out = observe(client, text="Alice lives in Mumbai", timestamp="2026-01-01T00:00:00Z")
    assert out["outcomes"][0]["decision"] == "INSERTED"
    out = observe(client, subject="Alice", predicate="lives_in", object="London", timestamp="2026-02-01T00:00:00Z")
    assert out["outcomes"][0]["decision"] == "SUPERSEDED"
    assert [m["mutation_type"] for m in out["outcomes"][0]["mutations"]] == ["INVALIDATE", "INSERT"]


@pytest.mark.parametrize(
    "body",
    [{}, {"subject": "Alice", "predicate": "lives_in"}, {"text": "gibberish"}, {"text": "x", "confidence": 2}],
)
def test_observe_validation_errors(client: TestClient, body: dict) -> None:
    assert client.post("/api/observe", json=body).status_code == 422


def test_graph_current_point_in_time_and_history(client: TestClient) -> None:
    observe(client, text="Alice lives in Mumbai", timestamp="2026-01-01T00:00:00Z")
    observe(client, text="Alice moved to London", timestamp="2026-02-01T00:00:00Z")

    current = client.get("/api/graph").json()
    assert [e["edge"]["target_id"] for e in current["edges"]] == ["london"]

    jan = client.get("/api/graph", params={"timestamp": "2026-01-15T00:00:00Z"}).json()
    assert [e["edge"]["target_id"] for e in jan["edges"]] == ["mumbai"]

    hist = client.get("/api/graph", params={"timestamp": "2026-03-01T00:00:00Z", "include_history": True}).json()
    assert {e["edge"]["target_id"]: e["valid_at_query_time"] for e in hist["edges"]} == {"mumbai": False, "london": True}

    timeline = client.get("/api/timeline").json()["timestamps"]
    assert timeline == ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"]


def test_audit_verify_and_provenance(client: TestClient) -> None:
    observe(client, text="Alice lives in Mumbai", timestamp="2026-01-01T00:00:00Z")
    observe(client, text="Alice moved to London", timestamp="2026-02-01T00:00:00Z")
    audit = client.get("/api/audit").json()
    assert audit["count"] == 3 and audit["head_hash"] == audit["records"][-1]["record_hash"]
    assert client.get("/api/audit", params={"entity_id": "mumbai"}).json()["count"] == 2
    assert client.get("/api/audit/verify").json()["valid"]
    assert len(client.get("/api/provenance/alice").json()["entries"]) == 3
    assert client.get("/api/provenance/nobody").status_code == 404


def test_query_endpoint(client: TestClient) -> None:
    observe(client, text="Alice lives in Mumbai", timestamp="2026-01-01T00:00:00Z")
    observe(client, text="Alice moved to London", timestamp="2026-02-01T00:00:00Z")
    r = client.post("/api/query", json={"question": "Where does Alice live?"})
    assert r.status_code == 200
    body = r.json()
    assert "London" in body["answer"] and body["grounded"]
    assert body["metrics"]["pruned_edges"] >= 1
    past = client.post(
        "/api/query", json={"question": "Where does Alice live?", "as_of": "2026-01-15T00:00:00Z"}
    ).json()
    assert "Mumbai" in past["answer"] and "London" not in past["answer"]
    assert client.post("/api/query", json={"question": ""}).status_code == 422


def test_state_survives_restart(client: TestClient, tmp_path: Path) -> None:
    observe(client, text="Alice lives in Mumbai", timestamp="2026-01-01T00:00:00Z")
    observe(client, text="Alice moved to London", timestamp="2026-02-01T00:00:00Z")
    before = client.get("/api/graph", params={"include_history": True}).json()["edges"]
    with TestClient(main.app) as restarted:  # lifespan reloads + replays the same ledger file
        after = restarted.get("/api/graph", params={"include_history": True}).json()["edges"]
    assert after == before


def test_seed_and_reset(client: TestClient, tmp_path: Path) -> None:
    seeded = client.post("/api/demo/seed").json()
    assert seeded["ledger_records"] > 50 and seeded["decisions"]["SUPERSEDED"] >= 5
    assert client.get("/api/audit/verify").json()["valid"]
    reset = client.post("/api/reset").json()
    assert reset["archived_to"] and Path(reset["archived_to"]).exists()
    assert client.get("/api/health").json()["ledger_records"] == 0


def test_cors_headers(client: TestClient) -> None:
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
