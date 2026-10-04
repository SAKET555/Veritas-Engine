"""A/B Lab service + endpoint tests (mock model; no Ollama required)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import ab_service
import main
from ab_service import ABJobManager, ABRunRequest, ObservationError, ObservationIn, QueryIn, suggest_queries, to_scripted

PRIYA = [
    {"text": "Priya lives in Pune"},
    {"text": "Priya works at Zoho"},
    {"text": "Priya moved to Bengaluru"},
    {"text": "Priya left Zoho for Freshworks"},
    {"text": "Ravi lives in Delhi"},
]


def obs(rows: list[dict]) -> list[ObservationIn]:
    return [ObservationIn(**r) for r in rows]


def wait_for(manager: ABJobManager, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = manager.get(job_id).view()
        if view["status"] in ("done", "error", "cancelled"):
            return view
        time.sleep(0.05)
    raise AssertionError("A/B job did not finish")


def test_missing_timestamps_follow_list_order() -> None:
    scripted = to_scripted(obs([{"text": "A lives in X"}, {"text": "A moved to Y"},
                                {"text": "A moved to Z", "timestamp": "2030-01-01T00:00:00Z"}, {"text": "B lives in Q"}]))
    times = [o.timestamp for o in scripted]
    assert times[0] < times[1] < times[2] < times[3]
    assert (times[1] - times[0]).total_seconds() == 3600


def test_unparseable_rows_are_reported_by_index() -> None:
    with pytest.raises(ObservationError) as info:
        to_scripted(obs([{"text": "Priya lives in Pune"}, {"text": "hello there"}, {"text": "blah"}]))
    assert [e["index"] for e in info.value.errors] == [1, 2]


def test_suggestions_have_ground_truth_from_reconciliation() -> None:
    result = suggest_queries(obs(PRIYA), max_questions=10)
    by_q = {q["question"]: q for q in result["queries"]}
    assert by_q["Where does Priya live now?"] == {
        "question": "Where does Priya live now?", "expected": "Bengaluru", "wrong": ["Pune"], "category": "changed"}
    assert by_q["Where did Priya work before Freshworks?"]["expected"] == "Zoho"
    assert by_q["Where does Ravi live now?"]["category"] == "stable"
    assert result["available"] == {"changed": 2, "historical": 2, "stable": 1}


def test_suggestions_respect_rejected_rumours() -> None:
    rows = [{"text": "Priya lives in Pune", "confidence": 0.95},
            {"text": "Priya lives in Goa", "source": "hearsay", "confidence": 0.3}]
    result = suggest_queries(obs(rows))
    # The rumour never became state, so there is no "changed" question claiming Goa.
    assert all(q["expected"] != "Goa" for q in result["queries"])


def test_suggestion_budget_interleaves_categories() -> None:
    result = suggest_queries(obs(PRIYA), max_questions=3)
    assert len(result["queries"]) == 3
    assert {q["category"] for q in result["queries"]} >= {"changed", "historical"}


def test_job_runs_streams_and_writes_report(tmp_path: Path) -> None:
    manager = ABJobManager(tmp_path)
    queries = [QueryIn(**q) for q in suggest_queries(obs(PRIYA))["queries"]]
    job = manager.submit(ABRunRequest(observations=obs(PRIYA), queries=queries, provider="mock", methods=["A1", "B2"]))
    view = wait_for(manager, job.id)
    assert view["status"] == "done", view["error"]
    assert view["progress"] == {"done": len(queries) * 2, "total": len(queries) * 2}
    assert set(view["summary"]) == {"A1", "B2"}
    assert view["summary"]["B2"]["accuracy"] == 1.0  # the mock reader can parse VERITAS's compact facts
    assert "chain intact" in view["config"]["ledger_verification"]
    assert "prompt" in view["rows"][0] and "system_prompt" not in view["rows"][0]
    report_dir = Path(view["report_dir"])
    assert (report_dir / "report.json").exists() and (report_dir / "report.md").exists()


def test_only_one_job_at_a_time_and_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ABJobManager(tmp_path)
    real_run = ab_service.ABRunner.run

    def slow_run(self, on_row=None, should_stop=None, quiet=False):  # type: ignore[no-untyped-def]
        def slow_stop() -> bool:
            time.sleep(0.05)
            return bool(should_stop and should_stop())
        return real_run(self, on_row=on_row, should_stop=slow_stop, quiet=quiet)

    monkeypatch.setattr(ab_service.ABRunner, "run", slow_run)
    request = ABRunRequest(observations=obs(PRIYA), queries=[QueryIn(question="Where does Priya live now?",
                                                                       expected="Bengaluru", wrong=["Pune"])],
                           provider="mock", repeats=5)
    job = manager.submit(request)
    with pytest.raises(ab_service.BusyError):
        manager.submit(request)
    manager.cancel(job.id)
    view = wait_for(manager, job.id)
    assert view["status"] == "cancelled"
    assert view["progress"]["done"] < view["progress"]["total"]
    assert manager.submit(request).id != job.id  # free again


def test_request_validation() -> None:
    with pytest.raises(ValueError):
        ABRunRequest(observations=obs(PRIYA), queries=[QueryIn(question="q", expected="a")], methods=["Z"])
    q = QueryIn(question="q", expected="a", wrong=[" Pune ", "", "  "])
    assert q.wrong == ["Pune"]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(main, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    with TestClient(main.app) as c:
        yield c


def test_lab_endpoints(client: TestClient) -> None:
    preset = client.get("/api/ab/preset").json()
    assert len(preset["observations"]) > 50 and preset["queries"][0]["expected"] == "Munich"

    bad = client.post("/api/ab/suggest-queries", json={"observations": [{"text": "nonsense words"}]})
    assert bad.status_code == 422 and bad.json()["detail"]["errors"][0]["index"] == 0

    suggested = client.post("/api/ab/suggest-queries", json={"observations": PRIYA}).json()["queries"]
    started = client.post("/api/ab/runs", json={"observations": PRIYA, "queries": suggested, "provider": "mock"})
    assert started.status_code == 200
    job_id = started.json()["id"]
    for _ in range(200):
        view = client.get(f"/api/ab/runs/{job_id}").json()
        if view["status"] == "done":
            break
        time.sleep(0.05)
    assert view["status"] == "done"
    assert client.get("/api/ab/runs").json()["runs"][0]["id"] == job_id
    assert client.get("/api/ab/runs/nope").status_code == 404


def test_dashboard_observations_are_journaled_and_survive_restart(client: TestClient) -> None:
    client.post("/api/observe", json={"text": "Zed lives in Oslo", "source": "sensor"})
    client.post("/api/observe", json={"subject": "Zed", "predicate": "works_at", "object": "Nokia"})
    journal = client.get("/api/observations").json()
    assert journal["count"] == 2
    assert journal["observations"][1]["text"] == "Zed | works_at | Nokia"
    assert journal["observations"][0]["source"] == "sensor"
    with TestClient(main.app) as restarted:
        assert restarted.get("/api/observations").json()["count"] == 2
        restarted.post("/api/reset")
        assert restarted.get("/api/observations").json()["count"] == 0
