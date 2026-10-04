from __future__ import annotations

import pytest

import benchmark


def test_hashed_encoder_is_deterministic_and_normalized() -> None:
    enc = benchmark.HashedDenseEncoder()
    a, b = enc.encode("Alice lives in London"), enc.encode("Alice lives in London")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)


def test_benchmark_end_to_end(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    report = benchmark.run("hashed", top_k=6, verbose=False)
    tkg = report["summary"]["tkg"]
    assert tkg["epistemic_consistency"] == 1.0
    assert tkg["stale_in_context_rate"] == 0.0
    assert tkg["contradiction_rate"] == 0.0
    assert tkg["verifiable_rate"] == 1.0
    assert report["ledger"]["valid"]
    assert report["summary"]["mean_pruned_graph_tokens"] < 0.1 * report["summary"]["full_graph_tokens"]
    # The temporally blind baseline must surface stale facts on this trajectory.
    assert report["summary"]["vector_rag"]["stale_in_context_rate"] > 0
    assert "Verifiable TKG" in capsys.readouterr().out
