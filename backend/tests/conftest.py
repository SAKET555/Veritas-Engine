"""Shared fixtures. Tests import backend modules flat, exactly as uvicorn does."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_engine import TemporalGraphEngine  # noqa: E402
from models import TripleObservation  # noqa: E402
from pruning import SubgraphPruner  # noqa: E402
from reconciliation import AuditLedger, ReconciliationEngine  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(days: float) -> datetime:
    return T0 + timedelta(days=days)


def obs(subject: str, predicate: str, obj: str, *, day: float = 0, confidence: float = 0.9,
        source: str = "user_statement", prior: str | None = None) -> TripleObservation:
    return TripleObservation(
        subject=subject, predicate=predicate, object=obj, timestamp=at(day),
        confidence=confidence, source=source, expected_prior_object=prior,
    )


class Stack:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.graph = TemporalGraphEngine()
        self.ledger = AuditLedger(path)
        self.reconciler = ReconciliationEngine(self.graph, self.ledger)
        self.pruner = SubgraphPruner(self.graph, self.ledger)

    def ingest(self, *observations: TripleObservation):
        return [self.reconciler.ingest(o) for o in observations]


@pytest.fixture
def stack(tmp_path: Path) -> Stack:
    return Stack(tmp_path / "audit.jsonl")
