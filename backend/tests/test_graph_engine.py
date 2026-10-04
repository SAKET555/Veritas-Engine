from __future__ import annotations

import pytest

from graph_engine import GraphEngineError, TemporalGraphEngine
from tests.conftest import at


@pytest.fixture
def engine() -> TemporalGraphEngine:
    g = TemporalGraphEngine()
    g.upsert_entity("Alice", "Person", first_seen=at(0))
    g.upsert_entity("Mumbai", "Location", first_seen=at(0))
    g.upsert_entity("London", "Location", first_seen=at(0))
    return g


def test_add_creates_open_interval(engine: TemporalGraphEngine) -> None:
    edge = engine.add_temporal_edge("alice", "lives_in", "mumbai", 0.9, "prov-1", at(0))
    assert edge.t_start == at(0)
    assert edge.t_end is None
    assert edge.is_active
    assert engine.stats() == {"entities": 3, "edges_total": 1, "edges_active": 1, "edges_historical": 0}


def test_invalidate_closes_interval_but_keeps_history(engine: TemporalGraphEngine) -> None:
    edge = engine.add_temporal_edge("alice", "lives_in", "mumbai", 0.9, "p", at(0))
    prior, updated = engine.invalidate_edge(edge.id, at(10), "moved")
    assert prior.t_end is None
    assert updated.t_end == at(10)
    assert updated.invalidation_rationale == "moved"
    assert updated.superseded_at is not None
    assert engine.has_edge(edge.id)
    assert engine.get_entity("mumbai") is not None
    assert engine.stats()["edges_active"] == 0
    assert engine.stats()["edges_historical"] == 1


def test_invalidate_rejects_double_close_and_backwards_end(engine: TemporalGraphEngine) -> None:
    edge = engine.add_temporal_edge("alice", "lives_in", "mumbai", 0.9, "p", at(5))
    with pytest.raises(GraphEngineError, match="before its t_start"):
        engine.invalidate_edge(edge.id, at(1), "bad")
    engine.invalidate_edge(edge.id, at(10), "ok")
    with pytest.raises(GraphEngineError, match="already closed"):
        engine.invalidate_edge(edge.id, at(11), "again")


def test_unknown_edge_raises(engine: TemporalGraphEngine) -> None:
    with pytest.raises(GraphEngineError):
        engine.get_edge("missing")


def test_point_in_time_half_open_interval(engine: TemporalGraphEngine) -> None:
    old = engine.add_temporal_edge("alice", "lives_in", "mumbai", 0.9, "p", at(0))
    engine.invalidate_edge(old.id, at(10), "moved")
    engine.add_temporal_edge("alice", "lives_in", "london", 0.9, "p", at(10))

    def targets(day: float) -> set[str]:
        return {v for _, v in engine.query_point_in_time(at(day)).edges()}

    assert targets(-1) == set()
    assert targets(0) == {"mumbai"}  # t_start inclusive
    assert targets(9.99) == {"mumbai"}
    assert targets(10) == {"london"}  # t_end exclusive
    assert targets(500) == {"london"}


def test_point_in_time_excludes_isolated_nodes(engine: TemporalGraphEngine) -> None:
    engine.add_temporal_edge("alice", "lives_in", "london", 0.9, "p", at(10))
    sub = engine.query_point_in_time(at(11))
    assert set(sub.nodes) == {"alice", "london"}


def test_active_subgraph_fast_path(engine: TemporalGraphEngine) -> None:
    old = engine.add_temporal_edge("alice", "lives_in", "mumbai", 0.9, "p", at(0))
    engine.invalidate_edge(old.id, at(10), "moved")
    new = engine.add_temporal_edge("alice", "lives_in", "london", 0.9, "p", at(10))
    active = engine.get_active_subgraph()
    assert [k for _, _, k in active.edges(keys=True)] == [new.id]


def test_bitemporal_as_known_at() -> None:
    from models import TemporalEdge

    recorded, invalidation_learned = at(1), at(30)
    closed = TemporalEdge(
        source_id="alice", relation="lives_in", target_id="mumbai", t_start=at(0), t_end=at(10),
        confidence_score=0.9, provenance_id="p", recorded_at=recorded, superseded_at=invalidation_learned,
    )
    visible = TemporalGraphEngine.is_visible
    # Before the invalidation was learned, the system still believed Mumbai held on day 20.
    assert visible(closed, at(20), at(15))
    # Once learned (and with no system-time filter), day 20 is outside the valid interval.
    assert not visible(closed, at(20), at(30))
    assert not visible(closed, at(20), None)
    # Before the fact was recorded at all, nothing is visible.
    assert not visible(closed, at(5), at(0.5))
    assert visible(closed, at(5), at(2))


def test_parallel_edges_between_same_pair(engine: TemporalGraphEngine) -> None:
    a = engine.add_temporal_edge("alice", "lives_in", "london", 0.9, "p", at(0))
    engine.invalidate_edge(a.id, at(5), "left")
    b = engine.add_temporal_edge("alice", "lives_in", "london", 0.9, "p", at(20))
    assert {e.id for e in engine.edges_for("alice", "lives_in")} == {a.id, b.id}
    assert [e.id for e in engine.active_edges_for("alice", "lives_in")] == [b.id]


def test_timeline_is_sorted_distinct(engine: TemporalGraphEngine) -> None:
    old = engine.add_temporal_edge("alice", "lives_in", "mumbai", 0.9, "p", at(0))
    engine.invalidate_edge(old.id, at(10), "moved")
    engine.add_temporal_edge("alice", "lives_in", "london", 0.9, "p", at(10))
    assert engine.timeline() == [at(0), at(10)]


def test_upsert_entity_refines_but_never_downgrades(engine: TemporalGraphEngine) -> None:
    engine.upsert_entity("Acme")
    assert engine.get_entity("acme").entity_type == "Entity"
    engine.upsert_entity("Acme", "Organization")
    assert engine.get_entity("acme").entity_type == "Organization"
    engine.upsert_entity("Acme", "Entity")
    assert engine.get_entity("acme").entity_type == "Organization"


def test_remove_edge_keeps_indexes_consistent(engine: TemporalGraphEngine) -> None:
    edge = engine.add_temporal_edge("alice", "lives_in", "mumbai", 0.9, "p", at(0))
    engine.remove_edge(edge.id)
    assert not engine.has_edge(edge.id)
    assert engine.edges_for("alice", "lives_in") == []
    assert engine.stats()["edges_active"] == 0
