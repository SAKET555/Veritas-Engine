"""The additional A/B methods (A3–A5, B3) and the synthetic dataset option."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ab_eval
import ab_service
from ab_eval import ALL_METHODS, DEFAULT_METHODS, ABRunner, build_methods, scaled_trajectory
from agent import MockLLMClient


@pytest.fixture
def methods(tmp_path: Path):  # type: ignore[no-untyped-def]
    available, _ = build_methods(MockLLMClient(), tmp_path / "ledger.jsonl")
    runner = ABRunner(MockLLMClient(), list(available.values()), [])
    runner.ingest(scaled_trajectory(1), [available[k] for k in ("A1", "A2", "A3", "A4", "A5", "B")])
    return available


def test_registry_and_defaults() -> None:
    assert set(DEFAULT_METHODS) == {"A1", "A2", "B", "B2"}
    assert set(ALL_METHODS) == set(DEFAULT_METHODS) | {"A3", "A4", "A5", "B3"}
    assert ab_service.ABRunRequest.model_fields["methods"].default_factory() == list(DEFAULT_METHODS)


def test_new_baseline_prompts(methods) -> None:  # type: ignore[no-untyped-def]
    q = "Where does Alice live now?"
    hybrid = methods["A3"].build(q)
    assert "BM25 + dense" in hybrid.context and hybrid.system == ab_eval.RAG_SYSTEM
    graph = methods["A4"].build(q)
    assert "Alice -lives_in-> London" in graph.context and "Alice -lives_in-> Mumbai" in graph.context
    memory = methods["A5"].build(q)
    assert "Alice lives in: London" in memory.context and "Mumbai" not in memory.context
    assert memory.system == ab_eval.MEMORY_SYSTEM


def test_optimized_veritas_shares_the_graph_and_reads_dates(methods) -> None:  # type: ignore[no-untyped-def]
    b3 = methods["B3"]
    assert b3.graph is methods["B"].graph
    dated = b3.build("Where did Alice live on February 1, 2026?")
    assert "as_of 2026-02-01" in dated.context and "Mumbai" in dated.context
    plain = methods["B"].build("Where did Alice live on February 1, 2026?")
    assert "as_of now" in plain.context
    assert b3.post_process(dated, "ANSWER: Mumbai")["citations_verified"] is False  # no citation given


def test_cli_with_synthetic_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = ab_eval.main(["--provider", "mock", "--no-warmup", "--dataset", "synthetic", "--people", "6",
                         "--sessions", "3", "--questions", "6", "--methods", "A3,A4,A5,B,B3",
                         "--out", str(tmp_path)])
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["config"]["dataset"]["name"] == "synthetic"
    assert set(report["summary"]) == {"A3", "A4", "A5", "B", "B3"}
    assert "B3 VERITAS (optimized" in capsys.readouterr().out
