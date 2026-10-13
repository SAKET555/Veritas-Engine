from __future__ import annotations

import json

import pytest

from graph_engine import TemporalGraphEngine
from reconciliation import AuditLedger, ReconciliationEngine
from tests.conftest import Stack, at, obs


def test_first_value_inserts(stack: Stack) -> None:
    (out,) = stack.ingest(obs("Alice", "lives_in", "Mumbai"))
    assert out.decision == "INSERTED"
    assert [m.mutation_type for m in out.mutations] == ["INSERT"]
    assert out.mutations[0].prior_state_snapshot is None
    assert stack.graph.get_entity("alice").entity_type == "Person"
    assert stack.graph.get_entity("mumbai").entity_type == "Location"


def test_newer_conflict_supersedes_atomically(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai", day=0))
    (out,) = stack.ingest(obs("Alice", "lives_in", "London", day=10, prior="Mumbai"))
    assert out.decision == "SUPERSEDED"
    assert [m.mutation_type for m in out.mutations] == ["INVALIDATE", "INSERT"]
    assert len({m.transaction_id for m in out.mutations}) == 1

    edges = stack.graph.edges_for("alice", "lives_in")
    mumbai, london = edges
    assert mumbai.t_end == at(10)
    assert mumbai.invalidated_by == london.id
    assert london.t_start == at(10) and london.t_end is None
    assert "matches the ledger" in out.mutations[0].deterministic_rationale


def test_functional_invariant_one_active_value(stack: Stack) -> None:
    for day, city in enumerate(["Mumbai", "London", "Lisbon", "Berlin"]):
        stack.ingest(obs("Alice", "lives_in", city, day=day * 10))
    active = stack.graph.active_edges_for("alice", "lives_in")
    assert [e.target_id for e in active] == ["berlin"]
    assert len(stack.graph.edges_for("alice", "lives_in")) == 4


def test_identical_fact_is_reinforced_with_noisy_or(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "London", confidence=0.8))
    (out,) = stack.ingest(obs("Alice", "lives_in", "London", day=3, confidence=0.5, source="sensor"))
    assert out.decision == "REINFORCED"
    edge = stack.graph.get_edge(out.active_edge_id)
    assert edge.confidence_score == pytest.approx(1 - 0.2 * 0.5)
    assert out.mutations[0].mutation_type == "RECONCILE"
    assert len(edge.corroborating_provenance) == 1
    assert edge.t_start == at(0)


def test_low_authority_candidate_is_rejected_and_logged(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "London", confidence=0.95))
    (out,) = stack.ingest(obs("Alice", "lives_in", "Paris", day=1, confidence=0.5, source="hearsay"))
    assert out.decision == "RETAINED_INCUMBENT"
    assert [e.target_id for e in stack.graph.active_edges_for("alice", "lives_in")] == ["london"]
    assert stack.graph.get_entity("paris") is None
    record = out.mutations[0]
    assert record.mutation_type == "RECONCILE"
    assert record.prior_state_snapshot == record.new_state_snapshot
    assert "Rejected candidate" in record.deterministic_rationale


def test_equal_valid_time_requires_strictly_higher_authority(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "London", day=5, confidence=0.9))
    (tie,) = stack.ingest(obs("Alice", "lives_in", "Paris", day=5, confidence=0.9))
    assert tie.decision == "RETAINED_INCUMBENT"
    (win,) = stack.ingest(obs("Alice", "lives_in", "Paris", day=5, confidence=0.9, source="system_of_record"))
    assert win.decision == "SUPERSEDED"


def test_late_arriving_history_is_backfilled(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai", day=10))
    (out,) = stack.ingest(obs("Alice", "lives_in", "Pune", day=0))
    assert out.decision == "RETROACTIVE_INSERT"
    pune = stack.graph.get_edge(out.active_edge_id)
    assert (pune.t_start, pune.t_end) == (at(0), at(10))
    assert [e.target_id for e in stack.graph.active_edges_for("alice", "lives_in")] == ["mumbai"]


def test_late_arriving_contradiction_of_history_is_rejected(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai", day=0))
    stack.ingest(obs("Alice", "lives_in", "London", day=10))
    (out,) = stack.ingest(obs("Alice", "lives_in", "Pune", day=5))
    assert out.decision == "RETAINED_INCUMBENT"
    assert stack.graph.get_entity("pune") is None


def test_multivalued_predicate_never_conflicts(stack: Stack) -> None:
    stack.ingest(obs("Alice", "knows", "Bob"))
    (out,) = stack.ingest(obs("Alice", "knows", "Carol", day=1))
    assert out.decision == "INSERTED"
    assert {e.target_id for e in stack.graph.active_edges_for("alice", "knows")} == {"bob", "carol"}


def test_unknown_predicate_defaults_to_functional(stack: Stack) -> None:
    stack.ingest(obs("Alice", "favorite color", "Blue"))
    (out,) = stack.ingest(obs("Alice", "favorite_color", "Green", day=1))
    assert out.decision == "SUPERSEDED"


def test_self_referential_triple_rejected(stack: Stack) -> None:
    with pytest.raises(ValueError, match="Self-referential"):
        stack.ingest(obs("Alice", "knows", "alice"))
    assert len(stack.ledger) == 0


def test_ledger_write_failure_rolls_back_graph(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai"))
    before = {e.id: e.model_dump() for e in stack.graph.all_edges()}

    def boom(_drafts):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(stack.ledger, "append_batch", boom)
    with pytest.raises(OSError):
        stack.ingest(obs("Alice", "lives_in", "London", day=10))

    after = {e.id: e.model_dump() for e in stack.graph.all_edges()}
    assert after == before
    assert stack.graph.get_entity("london") is None
    assert len(stack.ledger) == 1


def test_hash_chain_verifies_and_links(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai"), obs("Alice", "lives_in", "London", day=1))
    records = stack.ledger.records()
    assert [r.sequence for r in records] == [0, 1, 2]
    assert records[0].prev_hash == "0" * 64
    assert all(b.prev_hash == a.record_hash for a, b in zip(records, records[1:]))
    assert stack.ledger.verify().valid


def test_tampering_is_detected_on_load(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai"), obs("Alice", "lives_in", "London", day=1))
    lines = stack.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["deterministic_rationale"] = "nothing to see here"
    lines[1] = json.dumps(record)
    stack.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = AuditLedger(stack.path).load()
    assert not result.valid
    assert result.first_invalid_sequence == 1
    assert "tampered" in result.detail


def test_deleted_record_breaks_chain(stack: Stack) -> None:
    stack.ingest(*(obs("Alice", "lives_in", c, day=i) for i, c in enumerate(["A1", "B2", "C3"])))
    lines = stack.path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    stack.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not AuditLedger(stack.path).load().valid


def test_replay_rebuilds_identical_graph(stack: Stack) -> None:
    stack.ingest(
        obs("Alice", "lives_in", "Mumbai", day=0),
        obs("Alice", "lives_in", "London", day=10),
        obs("Alice", "lives_in", "London", day=11, source="sensor"),
        obs("Alice", "lives_in", "Pune", day=-30),
        obs("Alice", "works_at", "Acme", day=2),
    )
    graph2 = TemporalGraphEngine()
    ledger2 = AuditLedger(stack.path)
    assert ledger2.load().valid
    ReconciliationEngine(graph2, ledger2).replay()
    dump = lambda g: {e.id: e.model_dump() for e in g.all_edges()}  # noqa: E731
    assert dump(graph2) == dump(stack.graph)
    assert {n.id: n.entity_type for n in graph2.entities()} == {n.id: n.entity_type for n in stack.graph.entities()}


def test_archive_and_reset_preserves_old_ledger(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai"))
    archived = stack.ledger.archive_and_reset()
    assert archived is not None and archived.exists()
    assert len(stack.ledger) == 0
    assert stack.ledger.head_hash == "0" * 64


def test_provenance_trail_is_chronological(stack: Stack) -> None:
    stack.ingest(obs("Alice", "lives_in", "Mumbai"), obs("Alice", "lives_in", "London", day=10))
    trail = stack.reconciler.get_provenance_trail("alice")
    assert [e.record.mutation_type for e in trail.entries] == ["INSERT", "INVALIDATE", "INSERT"]
    assert all(e.narrative for e in trail.entries)
    mumbai_trail = stack.reconciler.get_provenance_trail("mumbai")
    assert len(mumbai_trail.entries) == 2
    with pytest.raises(KeyError):
        stack.reconciler.get_provenance_trail("nobody")


def test_resolve_short_id(stack: Stack) -> None:
    (out,) = stack.ingest(obs("Alice", "lives_in", "Mumbai"))
    record = out.mutations[0]
    assert stack.ledger.resolve_short_id(record.short_id) == record
    assert stack.ledger.resolve_short_id("deadbeef") is None


def test_replay_prefix_reconstructs_past_state(stack: Stack) -> None:
    from graph_engine import TemporalGraphEngine
    from reconciliation import ReconciliationEngine

    stack.ingest(obs("Alice", "lives_in", "Mumbai", day=0))
    first = len(stack.ledger) - 1
    stack.ingest(obs("Alice", "lives_in", "London", day=30))
    past = ReconciliationEngine(TemporalGraphEngine(), stack.ledger)
    assert past.replay(until_sequence=first) == first + 1
    assert [e.target_id for e in past.graph.active_edges_for("alice", "lives_in")] == ["mumbai"]
    assert [e.target_id for e in stack.graph.active_edges_for("alice", "lives_in")] == ["london"]
    assert stack.ledger.verify().valid
