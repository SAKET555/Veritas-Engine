"""Auditable entity linking: explicit and automatic aliases, ledger records, replay, rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from entity_linking import EntityLinker, normalize_surface
from graph_engine import TemporalGraphEngine
from pruning import SubgraphPruner
from reconciliation import AuditLedger, ReconciliationEngine
from tests.conftest import obs


class Linked:
    def __init__(self, path: Path, mode: str = "explicit") -> None:
        self.graph = TemporalGraphEngine()
        self.ledger = AuditLedger(path)
        self.linker = EntityLinker(mode)  # type: ignore[arg-type]
        self.reconciler = ReconciliationEngine(self.graph, self.ledger, linker=self.linker)
        self.pruner = SubgraphPruner(self.graph, self.ledger, linker=self.linker)


@pytest.mark.parametrize("raw,expected", [
    ("Dr. Alice Rao", "Alice Rao"), ("Prof Alice", "Alice"), ("Acme, Inc.", "Acme"), ("Acme Ltd", "Acme"),
    ("Globex Corporation", "Globex"), ("Initech GmbH", "Initech"), ("Inc", "Inc"), ("Alice", "Alice"),
])
def test_normalize_surface(raw: str, expected: str) -> None:
    assert normalize_surface(raw) == expected


def test_explicit_alias_is_ledgered_and_applied(tmp_path: Path) -> None:
    s = Linked(tmp_path / "l.jsonl")
    s.reconciler.ingest(obs("Robert Smith", "lives_in", "Oslo", day=0))
    record = s.reconciler.declare_alias("Bob", "Robert Smith")
    assert record.mutation_type == "ALIAS" and record.entity_ids == ["bob", "robert-smith"]
    out = s.reconciler.ingest(obs("Bob", "lives_in", "Bergen", day=10))
    assert out.decision == "SUPERSEDED" and out.triple.subject == "Robert Smith"
    assert s.graph.get_entity("bob") is None
    assert s.pruner.resolve_seeds("Where does Bob live?") == ["robert-smith"]
    assert s.pruner.resolve_seeds("q", explicit=["bob"]) == ["robert-smith"]
    assert s.ledger.verify().valid
    trail = s.reconciler.get_provenance_trail("robert-smith")
    assert any("ALIAS (explicit): 'Bob' → 'Robert Smith'" in e.narrative for e in trail.entries)


def test_alias_validation(tmp_path: Path) -> None:
    s = Linked(tmp_path / "l.jsonl")
    s.reconciler.declare_alias("Bob", "Robert")
    with pytest.raises(ValueError, match="same entity"):
        s.reconciler.declare_alias("robert", "Robert")
    with pytest.raises(ValueError, match="already an alias"):
        s.reconciler.declare_alias("Bob", "Roberta")
    with pytest.raises(ValueError, match="itself a canonical"):
        s.reconciler.declare_alias("Robert", "Rob")
    # Aliasing to an alias points at the final canonical entity instead of chaining.
    s.reconciler.declare_alias("Bobby", "Bob")
    assert {a.alias: a.canonical for a in s.linker.aliases()}["Bobby"] == "Robert"
    assert len(s.ledger) == 2


def test_auto_linking_records_the_merge(tmp_path: Path) -> None:
    s = Linked(tmp_path / "l.jsonl", mode="auto")
    s.reconciler.ingest(obs("Acme", "headquartered_in", "Berlin", day=0, source="system_of_record"))
    out = s.reconciler.ingest(obs("Acme Inc.", "headquartered_in", "Munich", day=30, source="system_of_record"))
    assert out.decision == "SUPERSEDED"
    assert [m.mutation_type for m in out.mutations] == ["ALIAS", "INVALIDATE", "INSERT"]
    assert "Automatic entity linking" in out.mutations[0].deterministic_rationale
    assert s.graph.get_entity("acme-inc") is None


def test_auto_linking_skips_ambiguous_and_unknown(tmp_path: Path) -> None:
    s = Linked(tmp_path / "l.jsonl")  # two legal entities recorded before auto linking was enabled
    s.reconciler.ingest(obs("Acme Ltd", "headquartered_in", "Berlin", day=0))
    s.reconciler.ingest(obs("Acme GmbH", "headquartered_in", "Vienna", day=0))
    s.linker.mode = "auto"
    out = s.reconciler.ingest(obs("Acme", "headquartered_in", "Paris", day=1))
    assert out.decision == "INSERTED" and not any(m.mutation_type == "ALIAS" for m in out.mutations)
    fresh = s.reconciler.ingest(obs("Dr. Nobody", "lives_in", "Oslo", day=1))
    assert fresh.triple.subject == "Dr. Nobody"


def test_explicit_mode_never_auto_links(tmp_path: Path) -> None:
    s = Linked(tmp_path / "l.jsonl")
    s.reconciler.ingest(obs("Acme", "headquartered_in", "Berlin", day=0))
    out = s.reconciler.ingest(obs("Acme Inc.", "headquartered_in", "Munich", day=30))
    assert out.decision == "INSERTED" and s.graph.get_entity("acme-inc") is not None


def test_replay_restores_aliases(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    s = Linked(path, mode="auto")
    s.reconciler.ingest(obs("Acme", "headquartered_in", "Berlin", day=0))
    s.reconciler.ingest(obs("Acme Inc.", "headquartered_in", "Munich", day=30))
    s.reconciler.declare_alias("ACM", "Acme")
    again = Linked(path, mode="auto")
    assert again.ledger.load().valid
    again.reconciler.replay()
    assert {a.alias for a in again.linker.aliases()} == {"Acme Inc.", "ACM"}
    assert again.reconciler.ingest(obs("ACM", "headquartered_in", "Paris", day=60)).triple.subject == "Acme"


def test_failed_ledger_write_rolls_back_auto_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = Linked(tmp_path / "l.jsonl", mode="auto")
    s.reconciler.ingest(obs("Acme", "headquartered_in", "Berlin", day=0))

    def boom(drafts):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(s.ledger, "append_batch", boom)
    with pytest.raises(OSError):
        s.reconciler.ingest(obs("Acme Inc.", "headquartered_in", "Munich", day=30))
    assert s.linker.aliases() == []
    assert [e.target_id for e in s.graph.active_edges_for("acme", "headquartered_in")] == ["berlin"]
