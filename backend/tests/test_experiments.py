"""Robustness experiments: statistics, ablations, sweep and the adversarial generator option."""

from __future__ import annotations

from pathlib import Path

import pytest

import experiments
from datasets import generate
from reconciliation import authority_of
from temporal_bench import evaluate
from tests.conftest import Stack, obs


def test_wilson_interval() -> None:
    lo, hi = experiments.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = experiments.wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)


def test_default_dataset_unchanged_by_lie_option() -> None:
    a = generate(people=10, sessions=4, max_questions=12, seed=5)
    b = generate(people=10, sessions=4, max_questions=12, seed=5, lie_rate=0.0)
    assert [o.text for o in a.observations] == [o.text for o in b.observations]
    lied = generate(people=10, sessions=4, max_questions=12, seed=5, lie_rate=0.5)
    assert sum(o.source == "system_of_record" for o in lied.observations) > \
        sum(o.source == "system_of_record" for o in a.observations)


def test_engine_authority_option(stack: Stack) -> None:
    assert stack.reconciler.authority is authority_of
    stack.ingest(obs("Alice", "lives_in", "London", day=0))
    stack.reconciler.authority = lambda _source: 1.0  # every source equal
    out = stack.ingest(obs("Alice", "lives_in", "Paris", day=1, confidence=0.9, source="hearsay"))[0]
    assert out.decision == "SUPERSEDED"  # without authority weighting the rumor wins


def test_ablations_isolate_each_mechanism(tmp_path: Path) -> None:
    ds = generate(people=12, sessions=4, max_questions=24, seed=1)
    report = evaluate(ds, memories=experiments.ablation_memories(tmp_path), fsync=False)
    s = report["summary"]
    assert s["veritas"]["accuracy"] == 1.0
    assert s["no_authority"]["accuracy_by_category"]["contested"] < 1.0
    assert s["no_pruning"]["accuracy"] == 1.0
    assert s["no_pruning"]["context_tokens_mean"] > 5 * s["veritas"]["context_tokens_mean"]
    assert s["optimized_pruning"]["accuracy"] == 1.0


def test_cli_runs_every_experiment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "exp.json"
    assert experiments.main(["all", "--seeds", "1", "--people", "8", "--sessions", "3", "--questions", "12",
                             "--json", str(out)]) == 0
    text = capsys.readouterr().out
    for title in ("MULTI-SEED", "ABLATIONS", "PRUNING SWEEP", "ADVERSARIAL"):
        assert title in text
    assert out.exists()
