"""Synthetic dataset, shared reader and the offline temporal benchmark."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import temporal_bench
from baselines import Fact
from datasets import CATEGORIES, generate
from extraction import extract_triples
from temporal_bench import evaluate, read


@pytest.fixture(scope="module")
def dataset():  # type: ignore[no-untyped-def]
    return generate(people=16, sessions=5, max_questions=36, seed=3)


def test_generator_is_deterministic(dataset) -> None:  # type: ignore[no-untyped-def]
    again = generate(people=16, sessions=5, max_questions=36, seed=3)
    assert [o.text for o in again.observations] == [o.text for o in dataset.observations]
    assert [q.question for q in again.questions] == [q.question for q in dataset.questions]
    assert generate(people=16, sessions=5, max_questions=36, seed=4).observations != dataset.observations


def test_every_observation_parses(dataset) -> None:  # type: ignore[no-untyped-def]
    for obs in dataset.observations:
        assert extract_triples(obs.text, timestamp=obs.timestamp)


def test_ground_truth_is_independent_of_rumors(dataset) -> None:  # type: ignore[no-untyped-def]
    for slot, rumors in dataset.rumors.items():
        true_values = {iv.value for iv in dataset.truth[slot]}
        assert not set(rumors) & true_values
    hearsay = [o for o in dataset.observations if o.source == "hearsay"]
    assert hearsay, "the generator must emit rumors"
    for q in dataset.questions:
        assert q.answer not in q.wrong


def test_truth_intervals_are_contiguous(dataset) -> None:  # type: ignore[no-untyped-def]
    for history in dataset.truth.values():
        for a, b in zip(history, history[1:]):
            assert a.end == b.start
        assert history[-1].end is None


def test_question_categories_are_covered(dataset) -> None:  # type: ignore[no-untyped-def]
    assert {q.category for q in dataset.questions} == set(CATEGORIES)
    for q in dataset.questions:
        if q.category == "point-in-time":
            assert q.as_of is not None and str(q.as_of.year) in q.question


def test_reader_rules() -> None:
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    day = lambda n: t + timedelta(days=n)  # noqa: E731
    from datasets import TemporalQuestion

    timed = [Fact("A", "lives_in", "X", day(0), rank=0), Fact("A", "lives_in", "Y", day(10), rank=1),
             Fact("A", "lives_in", "Z", day(20), rank=2)]
    cur = TemporalQuestion("q", "Z", (), "changed", "A", ("lives_in",), "current")
    assert read(timed, cur)[0].object == "Z"
    at = TemporalQuestion("q", "Y", (), "point-in-time", "A", ("lives_in",), "as_of", as_of=day(15))
    assert read(timed, at)[0].object == "Y"
    prev = TemporalQuestion("q", "Y", (), "historical", "A", ("lives_in",), "previous", before="Z")
    assert read(timed, prev)[0].object == "Y"
    untimed = [Fact("A", "lives_in", "X", rank=0), Fact("A", "lives_in", "Y", rank=1)]
    assert read(untimed, cur)[0].object == "Y"  # no time: the last one listed
    closed = [Fact("A", "lives_in", "X", day(0), day(10), presented_as_current=False),
              Fact("A", "lives_in", "Y", day(10))]
    assert read(closed, cur)[0].object == "Y"
    hop = [Fact("A", "works_at", "Acme", day(0)), Fact("Acme", "headquartered_in", "Oslo", day(0))]
    multi = TemporalQuestion("q", "Oslo", (), "multi-hop", "A", ("works_at", "headquartered_in"), "current")
    fact, support = read(hop, multi)
    assert fact.object == "Oslo" and len(support) == 2


def test_benchmark_veritas_beats_every_baseline(dataset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    report = evaluate(dataset, fsync=False, workdir=tmp_path)
    summary = report["summary"]
    assert report["graph"]["ledger_valid"]
    for key in ("veritas", "veritas_optimized"):
        assert summary[key]["accuracy"] == 1.0
        assert summary[key]["verifiable_rate"] == 1.0
        assert summary[key]["contradictory_context_rate"] == 0.0
        assert summary[key]["stale_answer_rate"] == 0.0
    for key in ("full_history", "vector_rag", "hybrid_rag", "static_graph_rag", "overwrite_memory"):
        assert summary[key]["accuracy"] < 1.0
        assert summary[key]["verifiable_rate"] == 0.0
    # Rumors are what audit-free memories get wrong.
    assert summary["overwrite_memory"]["accuracy_by_category"]["contested"] < 1.0
    assert summary["full_history"]["context_tokens_mean"] > 5 * summary["veritas"]["context_tokens_mean"]


def test_cli_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_json, out_md, ds = tmp_path / "r.json", tmp_path / "r.md", tmp_path / "ds.json"
    code = temporal_bench.main(["--people", "8", "--sessions", "4", "--questions", "12", "--no-fsync",
                                "--json", str(out_json), "--markdown", str(out_md), "--export-dataset", str(ds)])
    assert code == 0 and out_json.exists() and ds.exists()
    assert "| Accuracy |" in out_md.read_text(encoding="utf-8")
    assert "VERITAS temporal benchmark" in capsys.readouterr().out


def test_scaling_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert temporal_bench.main(["--scaling", "6,12", "--sessions", "3", "--questions", "6", "--no-fsync"]) == 0
    assert "SCALING" in capsys.readouterr().out
