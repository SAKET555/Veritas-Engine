"""Opt-in retrieval options: ego-local scoring, hub damping and question-date detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from datasets import generate
from extraction import extract_triples
from models import PruningParams
from pruning import extract_as_of
from tests.conftest import Stack, obs


@pytest.mark.parametrize("question,expected", [
    ("Where did Asha live on March 3, 2025?", datetime(2025, 3, 3, 12, tzinfo=timezone.utc)),
    ("as of 2026-03-01?", datetime(2026, 3, 1, 12, tzinfo=timezone.utc)),
    ("at 2026-03-01T09:30Z", datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)),
    ("on the 15th of Sept 2024", datetime(2024, 9, 15, 12, tzinfo=timezone.utc)),
    ("Where does Alice live?", None),
    ("between 2025-01-01 and 2025-02-01", None),  # two dates: ambiguous
    ("Feb 30, 2025", None),
])
def test_extract_as_of(question: str, expected: datetime | None) -> None:
    assert extract_as_of(question) == expected


def test_ego_mode_keeps_exactly_the_same_facts(stack: Stack) -> None:
    ds = generate(people=30, sessions=5, max_questions=40, seed=11)
    for o in ds.observations:
        for t in extract_triples(o.text, confidence=o.confidence, source=o.source, timestamp=o.timestamp):
            stack.reconciler.ingest(t)
    questions = [q.question for q in ds.questions] + ["Tell me everything", "Who lives in Lisbon?"]
    for q in questions:
        for history in (None, True):
            full = stack.pruner.prune(q, PruningParams(include_history=history))
            ego = stack.pruner.prune(q, PruningParams(include_history=history, candidate_mode="ego"))
            assert [e.id for e in ego.kept_edges] == [e.id for e in full.kept_edges], q
            assert ego.text == full.text
            assert len(ego.scored) + ego.unscored == len(full.scored) + full.unscored
            assert full.unscored == 0


def test_hub_damping_suppresses_hub_neighbours(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "London", day=0), obs("Alice", "works_at", "Acme", day=0))
    for i in range(30):
        stack.ingest(obs(f"Person{i}", "lives_in", "London", day=1, confidence=0.99, source="sensor"),
                     obs(f"Person{i}", "works_at", "Acme", day=1, confidence=0.99, source="sensor"))
    q = "Where does Alice live and who else is there?"
    loose = stack.pruner.prune(q, PruningParams())
    damped = stack.pruner.prune(q, PruningParams(hub_damping=0.5))
    assert ("alice", "lives_in", "london") in {(e.source_id, e.relation, e.target_id) for e in damped.kept_edges}
    assert len(damped.kept_edges) < len(loose.kept_edges)
    # A hub that *is* the question's subject is never damped.
    london = stack.pruner.prune("Who lives in London?", PruningParams(hub_damping=0.5))
    assert len([e for e in london.kept_edges if e.target_id == "london"]) >= 20


def test_defaults_are_unchanged() -> None:
    p = PruningParams()
    assert p.candidate_mode == "all" and p.hub_damping == 0.0
