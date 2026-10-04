"""A/B harness tests with a deterministic fake SLM (no Ollama required)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import ab_eval
from ab_eval import (
    AB_QUERIES,
    ABQuery,
    ABRunner,
    AppendOnlyHistory,
    VectorRetrieval,
    VeritasMemory,
    extract_answer,
    grade,
    required_context,
    scaled_trajectory,
    summarize,
)
from agent import LLMClient, LLMCompletion, LLMError

ACME = ABQuery("Where is Acme currently headquartered?", "Munich", ("Berlin",), "changed")


class LastMentionReader(LLMClient):
    """Fake SLM: answers with the last city mentioned for Acme's HQ, i.e. 'latest wins' in prompt order.

    It also records prompts and reports prompt tokens as len(prompt)//4, like a tokenizer would.
    """

    def __init__(self, fail_first: int = 0, crash_first: int = 0) -> None:
        self.prompts: list[tuple[str, str]] = []
        self.fail_first = fail_first
        self.crash_first = crash_first

    @property
    def model_name(self) -> str:
        return "fake-slm"

    def complete(self, system: str, user: str) -> LLMCompletion:
        self.prompts.append((system, user))
        if self.crash_first > 0:
            self.crash_first -= 1
            raise LLMError('Ollama /api/chat returned 500: {"error":"CUDA error: an illegal instruction"}')
        if self.fail_first > 0:
            self.fail_first -= 1
            raise LLMError('Ollama /api/chat returned 500: {"error":"prediction aborted, token repeat limit reached"}')
        cities = re.findall(r"Acme(?: is headquartered in| -headquartered_in->| moved its headquarters to) (\w+)", user)
        answer = cities[-1] if cities else "UNKNOWN"
        cites = " ".join(f"[M:{m}]" for m in re.findall(r"M:([0-9a-f]{8})", user)[:1])
        return LLMCompletion(f"ANSWER: {answer}\nEXPLANATION: from memory {cites}", self.model_name,
                             prompt_tokens=(len(system) + len(user)) // 4, completion_tokens=12,
                             timings_ms={"total": 10.0, "prompt_eval": 4.0, "generation": 6.0, "tokens_per_s": 50.0})


@pytest.mark.parametrize(
    ("text", "answer", "ok"),
    [
        ("ANSWER: Munich\nEXPLANATION: x", "Munich", True),
        ("**Answer:** Munich", "Munich", True),
        ("  answer :  Munich, Germany  ", "Munich, Germany", True),
        ("Acme is in Munich.", "Acme is in Munich.", False),
        ("", "", False),
    ],
)
def test_extract_answer(text: str, answer: str, ok: bool) -> None:
    assert extract_answer(text) == (answer, ok)


def test_grading_rejects_hedged_stale_answers() -> None:
    assert grade(ACME, "ANSWER: Munich").correct
    assert not grade(ACME, "ANSWER: Berlin").correct
    hedged = grade(ACME, "ANSWER: Berlin or Munich")
    assert not hedged.correct and hedged.stale_values == ["Berlin"]
    # Stale values mentioned only in the explanation do not count against the answer.
    assert grade(ACME, "ANSWER: Munich\nEXPLANATION: moved from Berlin").correct
    assert not grade(ACME, "ANSWER: Munichberg").correct


def test_scaled_trajectory_grows_with_drift() -> None:
    base, scaled = scaled_trajectory(1), scaled_trajectory(3)
    assert len(scaled) > len(base) + 90
    assert any("Agent047 moved to" in o.text or "Agent046 moved to" in o.text for o in scaled)
    phases = [o.phase for o in scaled]
    assert phases == sorted(phases, key=["T0", "T1", "T2"].index)


def make_runner(tmp_path: Path, client: LLMClient, **kw) -> tuple[ABRunner, list]:  # type: ignore[no-untyped-def]
    methods = [AppendOnlyHistory(), VectorRetrieval(top_k=4), VeritasMemory(tmp_path / "ledger.jsonl", client)]
    runner = ABRunner(client, methods, [ACME, AB_QUERIES[-1]], **kw)
    runner.ingest(scaled_trajectory(1))
    return runner, methods


def test_methods_see_same_stream_but_build_different_memory(tmp_path: Path) -> None:
    runner, (history, rag, veritas) = make_runner(tmp_path, LastMentionReader())
    h, r, v = (m.build(ACME.question) for m in (history, rag, veritas))
    assert "Acme is headquartered in Berlin" in h.user and "Munich" in h.user  # append-only keeps both
    assert "Berlin" not in v.context.split("## Mutation audit trail")[0]  # VERITAS drops the superseded fact
    assert "Munich" in v.context
    assert len(h.user) > len(v.user) > 0
    assert len(r.context.splitlines()) == 1 + 4
    assert veritas.ledger.verify().valid


def test_runner_end_to_end_rows_and_summary(tmp_path: Path) -> None:
    client = LastMentionReader()
    runner, methods = make_runner(tmp_path, client, repeats=2)
    rows = runner.run()
    assert len(rows) == 2 * 2 * 3
    acme_b = [r for r in rows if r.method == "B" and r.query == ACME.question]
    assert all(r.correct for r in acme_b) and all(r.extra["citations_verified"] for r in acme_b)
    # Cold-cache measurement: each request starts with a distinct id.
    first_lines = [system.splitlines()[0] for system, _ in client.prompts]
    assert len(set(first_lines)) == len(first_lines)

    summary = summarize(rows, methods)
    assert summary["A1"]["prompt_tokens_source"] == "model"
    assert summary["A1"]["prompt_tokens_mean"] > summary["B"]["prompt_tokens_mean"]
    assert summary["B"]["citation_verified_rate"] > 0
    assert set(summary["B"]["accuracy_by_category"]) == {"changed", "stable"}


def test_method_order_rotates_per_query(tmp_path: Path) -> None:
    client = LastMentionReader()
    runner, _ = make_runner(tmp_path, client)
    runner.run()
    leaders = [client.prompts[0][0], client.prompts[3][0]]
    assert "Memory mode: accumulated" in leaders[0]
    assert "Memory mode: retrieved" in leaders[1]


def test_degenerate_generation_is_recorded_not_fatal(tmp_path: Path) -> None:
    runner, methods = make_runner(tmp_path, LastMentionReader(fail_first=1))
    rows = runner.run()
    failed = [r for r in rows if r.timings_ms.get("generation_error")]
    assert len(failed) == 1 and not failed[0].correct
    assert summarize(rows, methods)[failed[0].method]["generation_errors"] == 1


def test_runner_crash_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab_eval.time, "sleep", lambda _s: None)
    runner, _ = make_runner(tmp_path, LastMentionReader(crash_first=1))
    rows = runner.run()
    assert rows[0].extra.get("runner_crash_retries") == 1
    assert not any(r.timings_ms.get("generation_error") for r in rows)


def test_persistent_crash_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab_eval.time, "sleep", lambda _s: None)
    runner, _ = make_runner(tmp_path, LastMentionReader(crash_first=10), crash_retries=2)
    with pytest.raises(LLMError, match="CUDA"):
        runner.run()


def test_required_context_fits_longest_prompt(tmp_path: Path) -> None:
    runner, methods = make_runner(tmp_path, LastMentionReader())
    ctx = required_context(methods, AB_QUERIES, max_tokens=160)
    assert ctx % 1024 == 0 and ctx >= 2048


def test_cli_with_mock_provider_writes_reports(tmp_path: Path) -> None:
    assert ab_eval.main(["--provider", "mock", "--no-warmup", "--out", str(tmp_path), "--methods", "A1,B"]) == 0
    (run_dir,) = list(tmp_path.iterdir())
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["config"]["methods"] == ["A1", "B"]
    assert len(report["rows"]) == len(AB_QUERIES) * 2
    assert "chain intact" in report["config"]["ledger_verification"]
    assert (run_dir / "report.md").read_text(encoding="utf-8").startswith("# VERITAS A/B evaluation")
    assert (run_dir / "veritas_ledger.jsonl").exists()
