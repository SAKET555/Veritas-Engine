"""Multi-tenancy, API keys, metrics, readiness and the alias endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(main, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    def factory(**env: str) -> TestClient:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return TestClient(main.app)

    return factory


@pytest.fixture
def client(make_client) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    with make_client() as c:
        yield c


def observe(c: TestClient, text: str, tenant: str | None = None, ts: str = "2026-01-01T00:00:00Z",
            **headers: str) -> dict:
    if tenant:
        headers["X-Tenant-ID"] = tenant
    r = c.post("/api/observe", json={"text": text, "timestamp": ts}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def test_tenants_are_isolated(client: TestClient, tmp_path: Path) -> None:
    observe(client, "Alice lives in Mumbai")
    observe(client, "Alice lives in Oslo", tenant="acme")
    default = client.get("/api/graph").json()["edges"]
    acme = client.get("/api/graph", headers={"X-Tenant-ID": "acme"}).json()["edges"]
    assert [e["edge"]["target_id"] for e in default] == ["mumbai"]
    assert [e["edge"]["target_id"] for e in acme] == ["oslo"]
    assert (tmp_path / "tenants" / "acme" / "audit.jsonl").exists()
    assert client.get("/api/health", headers={"X-Tenant-ID": "acme"}).json()["tenant"] == "acme"
    listed = {t["id"] for t in client.get("/api/tenants").json()["tenants"]}
    assert listed == {"default", "acme"}
    assert client.get("/api/audit/verify", headers={"X-Tenant-ID": "acme"}).json()["valid"]


def test_invalid_tenant_rejected(client: TestClient) -> None:
    assert client.get("/api/graph", headers={"X-Tenant-ID": "../etc"}).status_code == 400
    assert client.get("/api/graph", headers={"X-Tenant-ID": "A" * 80}).status_code == 400


def test_tenant_limit(make_client) -> None:  # type: ignore[no-untyped-def]
    with make_client(MAX_TENANTS="2") as c:
        assert c.get("/api/graph", headers={"X-Tenant-ID": "one"}).status_code == 200
        assert c.get("/api/graph", headers={"X-Tenant-ID": "two"}).status_code == 429


def test_api_keys(make_client) -> None:  # type: ignore[no-untyped-def]
    with make_client(VERITAS_API_KEYS="admin-key,acme-key:acme") as c:
        assert c.get("/api/health").status_code == 200  # probes stay public
        assert c.get("/api/ready").status_code == 200
        assert c.get("/api/graph").status_code == 401
        assert c.get("/api/graph", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/api/graph", headers={"Authorization": "Bearer admin-key"}).status_code == 200
        observe(c, "Alice lives in Oslo", **{"X-API-Key": "acme-key"})  # bound key → tenant acme
        acme = c.get("/api/graph", headers={"X-API-Key": "admin-key", "X-Tenant-ID": "acme"}).json()["edges"]
        assert [e["edge"]["target_id"] for e in acme] == ["oslo"]
        assert c.get("/api/graph", headers={"X-API-Key": "acme-key", "X-Tenant-ID": "other"}).status_code == 403
        assert c.get("/api/tenants", headers={"X-API-Key": "acme-key"}).status_code == 403
        assert c.get("/api/tenants", headers={"X-API-Key": "admin-key"}).status_code == 200


def _sample(text: str, series: str) -> float:
    for line in text.splitlines():
        if line.startswith(series + " "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_and_readiness(client: TestClient) -> None:
    observe_series = 'veritas_http_requests_total{method="POST",route="/api/observe",status="200"}'
    inserted_series = 'veritas_reconciliation_decisions_total{decision="INSERTED",tenant="default"}'
    before = client.get("/api/metrics").text  # counters are process-wide: compare deltas
    observe(client, "Alice lives in Mumbai")
    client.post("/api/query", json={"question": "Where does Alice live?"})
    assert client.get("/api/ready").json()["ready"]
    text = client.get("/api/metrics").text
    assert _sample(text, observe_series) - _sample(before, observe_series) == 1
    assert _sample(text, inserted_series) - _sample(before, inserted_series) == 1
    assert 'veritas_tenant_state{kind="ledger_records",tenant="default"} 1' in text
    assert "veritas_http_request_duration_seconds_bucket" in text
    assert 'veritas_query_context_tokens_total{kind="sent",tenant="default"}' in text


def test_alias_endpoints(client: TestClient) -> None:
    observe(client, "Robert Smith lives in Oslo")
    r = client.post("/api/entities/alias", json={"alias": "Bob", "canonical": "Robert Smith"})
    assert r.status_code == 200 and r.json()["mutation_type"] == "ALIAS"
    assert client.post("/api/entities/alias", json={"alias": "Bob", "canonical": "Bob"}).status_code == 422
    observe(client, "Bob moved to Bergen", ts="2026-02-01T00:00:00Z")
    edges = client.get("/api/graph").json()["edges"]
    assert [(e["edge"]["source_id"], e["edge"]["target_id"]) for e in edges] == [("robert-smith", "bergen")]
    listed = client.get("/api/entities/aliases").json()
    assert listed["mode"] == "explicit" and listed["aliases"][0]["canonical_id"] == "robert-smith"
    trail = client.get("/api/provenance/robert-smith").json()["entries"]
    assert any(e["record"]["mutation_type"] == "ALIAS" for e in trail)
    with TestClient(main.app) as restarted:  # aliases survive a restart via ledger replay
        assert restarted.get("/api/entities/aliases").json()["aliases"][0]["alias"] == "Bob"
    client.post("/api/reset")
    assert client.get("/api/entities/aliases").json()["aliases"] == []


def test_query_date_detection_is_opt_in(client: TestClient) -> None:
    client.post("/api/observe", json={"text": "Alice lives in Mumbai", "timestamp": "2026-01-01T00:00:00Z"})
    client.post("/api/observe", json={"text": "Alice moved to London", "timestamp": "2026-02-01T00:00:00Z"})
    plain = client.post("/api/query", json={"question": "Where did Alice live on January 15, 2026?"}).json()
    assert plain["as_of"] is None
    dated = client.post("/api/query", json={"question": "Where did Alice live on January 15, 2026?",
                                            "detect_as_of": True}).json()
    assert dated["as_of"].startswith("2026-01-15T12:00") and "Mumbai" in dated["answer"]
