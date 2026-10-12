"""Audit-trailed, deterministic state reconciliation.

Every state transition is decided by explicit rules (no model calls), executed
atomically against the graph, and committed to an append-only JSONL ledger in
which each record carries the SHA-256 of its predecessor. If the ledger write
fails the in-memory graph is rolled back, so graph state and ledger can never
diverge. Because each record stores a full post-state snapshot, the ledger
alone is sufficient to rebuild the graph (``replay``).

Decision rules for an incoming triple ``(s, p, o_new, t, c)``
-------------------------------------------------------------
R1  Identical active fact exists            -> RECONCILE (noisy-OR reinforcement)
R2  ``p`` is multi-valued                   -> INSERT
R3  ``p`` functional, a recorded fact for (s, p) starts after ``t``
                                            -> retroactive history handling
R4  ``p`` functional, active fact ``o_old`` with ``t_start <= t``:
      tie on valid time        : supersede iff eff(new) >  eff(old)
      newer valid time         : supersede iff eff(new) >= eff(old) * (1 - tolerance)
      superseded               -> INVALIDATE old (t_end = t) + INSERT new ([t, inf))
      otherwise                -> RECONCILE (incumbent retained, candidate rejected)
R5  otherwise                               -> INSERT

``eff(x) = confidence(x) * authority(source(x))``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from graph_engine import TemporalGraphEngine
from models import (
    ChainVerification,
    EntityNode,
    MutationAuditRecord,
    MutationType,
    ProvenanceEntry,
    ProvenanceTrail,
    ReconciliationDecision,
    ReconciliationOutcome,
    TemporalEdge,
    TripleObservation,
    entity_id_for,
    new_uuid,
    utc_now,
)
from entity_linking import AliasLink, EntityLinker
from ontology import PredicateSpec, get_spec

GENESIS_HASH = "0" * 64

SOURCE_AUTHORITY: dict[str, float] = {
    "system_of_record": 1.0,
    "sensor": 0.95,
    "user_statement": 0.9,
    "agent_observation": 0.85,
    "web": 0.7,
    "agent_inference": 0.6,
    "hearsay": 0.4,
}
DEFAULT_AUTHORITY = 0.75


def authority_of(source: str) -> float:
    return SOURCE_AUTHORITY.get(source, DEFAULT_AUTHORITY)


def _iso(ts: Optional[datetime]) -> str:
    return "∞" if ts is None else ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Append-only, hash-chained ledger
# --------------------------------------------------------------------------- #


@dataclass
class _Draft:
    transaction_id: str
    timestamp: datetime
    mutation_type: MutationType
    affected_edge_id: str
    entity_ids: list[str]
    rationale: str
    prior: Optional[dict[str, Any]]
    new: dict[str, Any]
    mutation_id: str = field(default_factory=new_uuid)


class AuditLedger:
    """Sequential JSONL ledger with a SHA-256 hash chain."""

    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        # fsync after every transaction makes each committed mutation durable across power loss.
        # Turning it off trades that guarantee for throughput (the OS still flushes eventually).
        self.fsync = fsync
        self._records: list[MutationAuditRecord] = []
        self._by_edge: dict[str, list[MutationAuditRecord]] = {}
        self._by_short: dict[str, MutationAuditRecord] = {}
        self._lock = threading.RLock()

    # -- hashing ----------------------------------------------------------- #

    @staticmethod
    def compute_hash(record: MutationAuditRecord) -> str:
        body = record.model_dump(mode="json", exclude={"record_hash"})
        return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()

    @property
    def head_hash(self) -> str:
        with self._lock:
            return self._records[-1].record_hash if self._records else GENESIS_HASH

    # -- persistence ------------------------------------------------------- #

    def load(self) -> ChainVerification:
        """Load the ledger from disk and verify the chain."""
        with self._lock:
            self._reset_indexes()
            if not self.path.exists():
                return self.verify()
            with self.path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = MutationAuditRecord.model_validate_json(line)
                    except ValueError as exc:
                        raise ValueError(f"Corrupt ledger line {line_no} in {self.path}: {exc}") from exc
                    self._index(record)
            return self.verify()

    def append_batch(self, drafts: list[_Draft]) -> list[MutationAuditRecord]:
        """Seal drafts into chained records and durably append them as one unit."""
        with self._lock:
            prev = self.head_hash
            seq = len(self._records)
            sealed: list[MutationAuditRecord] = []
            for draft in drafts:
                unsigned = MutationAuditRecord(
                    mutation_id=draft.mutation_id,
                    sequence=seq,
                    transaction_id=draft.transaction_id,
                    timestamp=draft.timestamp,
                    mutation_type=draft.mutation_type,
                    affected_edge_id=draft.affected_edge_id,
                    entity_ids=draft.entity_ids,
                    deterministic_rationale=draft.rationale,
                    prior_state_snapshot=draft.prior,
                    new_state_snapshot=draft.new,
                    prev_hash=prev,
                )
                record = unsigned.model_copy(update={"record_hash": self.compute_hash(unsigned)})
                sealed.append(record)
                prev = record.record_hash
                seq += 1

            self.path.parent.mkdir(parents=True, exist_ok=True)
            size_before = self.path.stat().st_size if self.path.exists() else 0
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    for record in sealed:
                        fh.write(record.model_dump_json() + "\n")
                    fh.flush()
                    if self.fsync:
                        os.fsync(fh.fileno())
            except OSError:
                if self.path.exists():
                    with self.path.open("r+b") as fh:
                        fh.truncate(size_before)
                raise
            for record in sealed:
                self._index(record)
            return sealed

    def archive_and_reset(self) -> Optional[Path]:
        """Move the current ledger aside (never deleted) and start a fresh chain."""
        with self._lock:
            archived: Optional[Path] = None
            if self.path.exists() and self.path.stat().st_size > 0:
                stamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
                archived = self.path.with_name(f"{self.path.stem}.{stamp}.archived.jsonl")
                shutil.move(str(self.path), archived)
            self._reset_indexes()
            return archived

    # -- queries ----------------------------------------------------------- #

    def records(self) -> list[MutationAuditRecord]:
        with self._lock:
            return list(self._records)

    def for_edge(self, edge_id: str) -> list[MutationAuditRecord]:
        with self._lock:
            return list(self._by_edge.get(edge_id, []))

    def latest_for_edge(self, edge_id: str) -> Optional[MutationAuditRecord]:
        with self._lock:
            records = self._by_edge.get(edge_id)
            return records[-1] if records else None

    def resolve_short_id(self, short_id: str) -> Optional[MutationAuditRecord]:
        with self._lock:
            return self._by_short.get(short_id.lower())

    def for_entity(self, entity_id: str) -> list[MutationAuditRecord]:
        with self._lock:
            return [r for r in self._records if entity_id in r.entity_ids]

    def verify(self) -> ChainVerification:
        with self._lock:
            prev = GENESIS_HASH
            for idx, record in enumerate(self._records):
                if record.sequence != idx:
                    return ChainVerification(
                        valid=False, records_checked=idx, head_hash=prev,
                        first_invalid_sequence=idx,
                        detail=f"Sequence gap: expected {idx}, found {record.sequence}",
                    )
                if record.prev_hash != prev:
                    return ChainVerification(
                        valid=False, records_checked=idx, head_hash=prev,
                        first_invalid_sequence=idx,
                        detail=f"Broken link at #{idx}: prev_hash does not match predecessor",
                    )
                if self.compute_hash(record) != record.record_hash:
                    return ChainVerification(
                        valid=False, records_checked=idx, head_hash=prev,
                        first_invalid_sequence=idx,
                        detail=f"Record #{idx} content does not match its hash (tampered)",
                    )
                prev = record.record_hash
            return ChainVerification(
                valid=True, records_checked=len(self._records), head_hash=prev,
                detail=f"{len(self._records)} records verified; chain intact",
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # -- internals --------------------------------------------------------- #

    def _reset_indexes(self) -> None:
        self._records = []
        self._by_edge = {}
        self._by_short = {}

    def _index(self, record: MutationAuditRecord) -> None:
        self._records.append(record)
        self._by_edge.setdefault(record.affected_edge_id, []).append(record)
        self._by_short[record.short_id] = record


# --------------------------------------------------------------------------- #
# Reconciliation engine
# --------------------------------------------------------------------------- #


@dataclass
class _Transaction:
    """Pending drafts plus compensating actions for rollback."""

    transaction_id: str
    drafts: list[_Draft] = field(default_factory=list)
    undo: list[Callable[[], None]] = field(default_factory=list)

    def rollback(self) -> None:
        for action in reversed(self.undo):
            action()


class ReconciliationEngine:
    def __init__(
        self,
        graph: TemporalGraphEngine,
        ledger: AuditLedger,
        *,
        supersession_tolerance: float = 0.35,
        linker: Optional[EntityLinker] = None,
    ) -> None:
        if not 0.0 <= supersession_tolerance < 1.0:
            raise ValueError("supersession_tolerance must be in [0, 1)")
        self.graph = graph
        self.ledger = ledger
        self.tolerance = supersession_tolerance
        self.linker = linker

    # -- public API -------------------------------------------------------- #

    def ingest(self, obs: TripleObservation) -> ReconciliationOutcome:
        spec = get_spec(obs.predicate)
        with self.graph.lock:
            txn = _Transaction(transaction_id=new_uuid())
            try:
                obs = self._link_entities(txn, obs)
                subject_id = entity_id_for(obs.subject)
                object_id = entity_id_for(obs.object)
                if subject_id == object_id:
                    raise ValueError(f"Self-referential triple rejected: {obs.subject!r} {spec.name} {obs.object!r}")
                decision, active_id, summary = self._decide_and_apply(txn, obs, spec, subject_id, object_id)
                records = self.ledger.append_batch(txn.drafts)
            except Exception:
                txn.rollback()
                raise
            if self.linker is not None:
                for eid in (subject_id, object_id):
                    entity = self.graph.get_entity(eid)
                    if entity is not None:
                        self.linker.note_entity(eid, entity.name)
        return ReconciliationOutcome(
            triple=obs,
            decision=decision,
            transaction_id=txn.transaction_id,
            active_edge_id=active_id,
            mutations=records,
            summary=summary,
        )

    def declare_alias(self, alias: str, canonical: str) -> MutationAuditRecord:
        """Commit an operator-declared alias as an ALIAS ledger record, then apply it."""
        if self.linker is None:
            raise ValueError("Entity linking is not enabled for this engine")
        with self.graph.lock:
            link = self.linker.declare(alias, canonical)
            txn = _Transaction(transaction_id=new_uuid())
            self._alias_draft(txn, link)
            (record,) = self.ledger.append_batch(txn.drafts)
            self.linker.restore(link.snapshot())
            return record

    def _link_entities(self, txn: _Transaction, obs: TripleObservation) -> TripleObservation:
        """Resolve subject and object through the linker; record any new automatic link."""
        if self.linker is None:
            return obs

        def lookup(eid: str) -> Optional[str]:
            entity = self.graph.get_entity(eid)
            return entity.name if entity else None

        updates: dict[str, str] = {}
        for field_name in ("subject", "object"):
            surface = getattr(obs, field_name)
            canonical, link = self.linker.resolve(surface, lookup)
            if link is not None:
                txn.undo.append(lambda alias_id=link.alias_id: self.linker.remove(alias_id))  # type: ignore[union-attr]
                self._alias_draft(txn, link)
            if canonical != surface:
                updates[field_name] = canonical
        return obs.model_copy(update=updates) if updates else obs

    def _alias_draft(self, txn: _Transaction, link: AliasLink) -> None:
        txn.drafts.append(
            _Draft(
                transaction_id=txn.transaction_id,
                timestamp=utc_now(),
                mutation_type="ALIAS",
                affected_edge_id=f"alias:{link.alias_id}",
                entity_ids=[link.alias_id, link.canonical_id],
                rationale=link.rationale,
                prior=None,
                new=link.snapshot(),
            )
        )

    def get_provenance_trail(self, entity_id: str) -> ProvenanceTrail:
        entity = self.graph.get_entity(entity_id)
        if entity is None:
            raise KeyError(entity_id)
        entries: list[ProvenanceEntry] = []
        for record in self.ledger.for_entity(entity_id):
            snap = record.new_state_snapshot
            if record.mutation_type == "ALIAS":
                a = snap["alias"]
                entries.append(ProvenanceEntry(record=record, narrative=(
                    f"#{record.sequence} [{record.short_id}] {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"ALIAS ({a['mode']}): '{a['alias']}' → '{a['canonical']}' — {record.deterministic_rationale}"
                )))
                continue
            edge = snap["edge"]
            src = snap["source_entity"]["name"]
            tgt = snap["target_entity"]["name"]
            narrative = (
                f"#{record.sequence} [{record.short_id}] {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{record.mutation_type}: ({src}, {edge['relation']}, {tgt}) "
                f"valid [{edge['t_start']} → {edge['t_end'] or '∞'}) — {record.deterministic_rationale}"
            )
            entries.append(ProvenanceEntry(record=record, narrative=narrative))
        return ProvenanceTrail(entity=entity, entries=entries)

    def replay(self) -> int:
        """Rebuild the graph purely from ledger snapshots. Returns records applied."""
        with self.graph.lock:
            self.graph.clear()
            applied = 0
            if self.linker is not None:
                self.linker.clear()
            for record in self.ledger.records():
                snap = record.new_state_snapshot
                if record.mutation_type == "ALIAS":
                    if self.linker is not None:
                        self.linker.restore(snap)
                    applied += 1
                    continue
                # Entities only ever get refined (specific type, earlier first_seen),
                # so the most recent snapshot is always the most complete one.
                for key in ("source_entity", "target_entity"):
                    self.graph.restore_entity(EntityNode.model_validate(snap[key]))
                self.graph.restore_edge(TemporalEdge.model_validate(snap["edge"]))
                if self.linker is not None:
                    for key in ("source_entity", "target_entity"):
                        self.linker.note_entity(snap[key]["id"], snap[key]["name"])
                applied += 1
            return applied

    # -- decision procedure ------------------------------------------------ #

    def _decide_and_apply(
        self,
        txn: _Transaction,
        obs: TripleObservation,
        spec: PredicateSpec,
        subject_id: str,
        object_id: str,
    ) -> tuple[ReconciliationDecision, Optional[str], str]:
        ts = obs.timestamp
        history = self.graph.edges_for(subject_id, spec.name)
        active = [e for e in history if e.is_active]
        triple_txt = f"({obs.subject}, {spec.name}, {obs.object})"
        eff_new = obs.confidence * authority_of(obs.source)

        # R1: corroboration of an identical active fact
        same = next((e for e in active if e.target_id == object_id), None)
        if same is not None:
            return self._reinforce(txn, obs, same, triple_txt)

        if not spec.functional:
            # R2: set-valued predicate, no conflict is possible
            edge = self._insert(
                txn, obs, spec, subject_id, object_id, t_end=None,
                rationale=(
                    f"Predicate '{spec.name}' is multi-valued (non-functional); {triple_txt} "
                    f"coexists with {len(active)} other active object(s). Opened interval "
                    f"[{_iso(ts)}, ∞) with confidence {obs.confidence:.2f} from source '{obs.source}'."
                ),
            )
            return "INSERTED", edge.id, f"Inserted {triple_txt} (multi-valued predicate)."

        # R3: observation predates recorded history for this (s, p)
        later = [e for e in history if e.t_start > ts]
        if later:
            return self._retroactive(txn, obs, spec, subject_id, object_id, history, later, triple_txt)

        prior_note = self._prior_expectation_note(obs, active)

        if not active:
            # R5: first value for a functional predicate
            edge = self._insert(
                txn, obs, spec, subject_id, object_id, t_end=None,
                rationale=(
                    f"No active value for functional predicate '{spec.name}' on '{obs.subject}'. "
                    f"Opened interval [{_iso(ts)}, ∞) for {triple_txt} with confidence "
                    f"{obs.confidence:.2f} (source '{obs.source}', effective {eff_new:.3f}).{prior_note}"
                ),
            )
            return "INSERTED", edge.id, f"Inserted {triple_txt}."

        # R4: conflict with incumbent(s)
        incumbent = max(active, key=lambda e: (e.t_start, e.recorded_at))
        eff_old = incumbent.confidence_score * authority_of(incumbent.source)
        incumbent_name = self._name(incumbent.target_id)
        tie = ts == incumbent.t_start
        if tie:
            supersede = eff_new > eff_old
            rule = f"equal valid time, needing effective confidence {eff_new:.3f} > incumbent {eff_old:.3f}"
        else:
            bar = eff_old * (1.0 - self.tolerance)
            supersede = eff_new >= bar
            rule = (
                f"newer valid time than {_iso(incumbent.t_start)}, needing effective confidence "
                f"{eff_new:.3f} ≥ incumbent {eff_old:.3f} × (1 − {self.tolerance:.2f}) = {bar:.3f}"
            )

        if not supersede:
            self._record_noop(
                txn, incumbent,
                rationale=(
                    f"Rejected candidate {triple_txt}; incumbent ({obs.subject}, {spec.name}, "
                    f"{incumbent_name}) retained: {rule} — failed. Candidate source "
                    f"'{obs.source}' (authority {authority_of(obs.source):.2f}) × confidence "
                    f"{obs.confidence:.2f}. Provenance {obs.provenance_id} logged; no state change.{prior_note}"
                ),
            )
            return (
                "RETAINED_INCUMBENT", incumbent.id,
                f"Kept ({obs.subject}, {spec.name}, {incumbent_name}); candidate {triple_txt} lacked authority.",
            )

        new_edge_id = new_uuid()
        for old in sorted(active, key=lambda e: e.t_start):
            old_name = self._name(old.target_id)
            self._invalidate(
                txn, old, end_time=ts, invalidated_by=new_edge_id,
                rationale=(
                    f"Superseded by {triple_txt} observed for {_iso(ts)} with {rule} — passed. "
                    f"Functional predicate '{spec.name}' admits one active object per subject, so "
                    f"({obs.subject}, {spec.name}, {old_name}) was closed at {_iso(ts)} and kept as history.{prior_note}"
                ),
            )
        edge = self._insert(
            txn, obs, spec, subject_id, object_id, t_end=None, edge_id=new_edge_id,
            rationale=(
                f"Opened interval [{_iso(ts)}, ∞) for {triple_txt} superseding "
                f"({obs.subject}, {spec.name}, {incumbent_name}) [edge {incumbent.id[:8]}] in transaction "
                f"{txn.transaction_id[:8]}; confidence {obs.confidence:.2f}, source '{obs.source}'."
            ),
        )
        return (
            "SUPERSEDED", edge.id,
            f"{obs.subject}: {spec.name} {incumbent_name} → {obs.object} (old fact closed at {_iso(ts)}).",
        )

    def _retroactive(
        self,
        txn: _Transaction,
        obs: TripleObservation,
        spec: PredicateSpec,
        subject_id: str,
        object_id: str,
        history: list[TemporalEdge],
        later: list[TemporalEdge],
        triple_txt: str,
    ) -> tuple[ReconciliationDecision, Optional[str], str]:
        ts = obs.timestamp
        covering = next((e for e in history if e.valid_at(ts)), None)
        if covering is not None:
            if covering.target_id == object_id:
                return self._reinforce(txn, obs, covering, triple_txt)
            cover_name = self._name(covering.target_id)
            self._record_noop(
                txn, covering,
                rationale=(
                    f"Late-arriving observation {triple_txt} @ {_iso(ts)} contradicts recorded history: "
                    f"({obs.subject}, {spec.name}, {cover_name}) was valid over "
                    f"[{_iso(covering.t_start)}, {_iso(covering.t_end)}). Recorded history wins over "
                    f"out-of-order observations; candidate logged, no state change."
                ),
            )
            return (
                "RETAINED_INCUMBENT", covering.id,
                f"Out-of-order observation conflicts with history; kept {cover_name}.",
            )
        end = min(e.t_start for e in later)
        edge = self._insert(
            txn, obs, spec, subject_id, object_id, t_end=end,
            rationale=(
                f"Late-arriving observation: {triple_txt} @ {_iso(ts)} precedes the earliest recorded "
                f"'{spec.name}' fact for '{obs.subject}' (starting {_iso(end)}). Inserted as closed "
                f"historical interval [{_iso(ts)}, {_iso(end)}); current state unchanged."
            ),
        )
        return "RETROACTIVE_INSERT", edge.id, f"Backfilled history {triple_txt} until {_iso(end)}."

    # -- primitive operations (each emits one draft + one undo) ------------ #

    def _snapshot(self, edge: TemporalEdge) -> dict[str, Any]:
        src = self.graph.get_entity(edge.source_id)
        tgt = self.graph.get_entity(edge.target_id)
        assert src is not None and tgt is not None
        return {
            "edge": edge.model_dump(mode="json"),
            "source_entity": src.model_dump(mode="json"),
            "target_entity": tgt.model_dump(mode="json"),
        }

    def _name(self, entity_id: str) -> str:
        entity = self.graph.get_entity(entity_id)
        return entity.name if entity else entity_id

    def _insert(
        self,
        txn: _Transaction,
        obs: TripleObservation,
        spec: PredicateSpec,
        subject_id: str,
        object_id: str,
        *,
        t_end: Optional[datetime],
        rationale: str,
        edge_id: Optional[str] = None,
    ) -> TemporalEdge:
        created_entities: list[str] = []
        for eid, name, etype in (
            (subject_id, obs.subject, obs.subject_type or spec.subject_type),
            (object_id, obs.object, obs.object_type or spec.object_type),
        ):
            if self.graph.get_entity(eid) is None:
                created_entities.append(eid)
            self.graph.upsert_entity(name, etype, first_seen=obs.timestamp)

        edge = self.graph.add_temporal_edge(
            subject_id, spec.name, object_id,
            confidence=obs.confidence,
            provenance_id=obs.provenance_id,
            timestamp=obs.timestamp,
            t_end=t_end,
            source_label=obs.source,
            edge_id=edge_id,
        )

        def undo() -> None:
            self.graph.remove_edge(edge.id)
            for eid in created_entities:
                if self.graph.active_degree(eid) == 0 and not any(
                    e.source_id == eid or e.target_id == eid for e in self.graph.all_edges()
                ):
                    self.graph._g.remove_node(eid)  # noqa: SLF001 - transactional rollback

        txn.undo.append(undo)
        txn.drafts.append(
            _Draft(
                transaction_id=txn.transaction_id,
                timestamp=utc_now(),
                mutation_type="INSERT",
                affected_edge_id=edge.id,
                entity_ids=[subject_id, object_id],
                rationale=rationale,
                prior=None,
                new=self._snapshot(edge),
            )
        )
        return edge

    def _invalidate(
        self,
        txn: _Transaction,
        edge: TemporalEdge,
        *,
        end_time: datetime,
        invalidated_by: Optional[str],
        rationale: str,
    ) -> TemporalEdge:
        prior_snapshot = self._snapshot(edge)
        prior, updated = self.graph.invalidate_edge(
            edge.id, end_time, rationale, invalidated_by=invalidated_by
        )
        txn.undo.append(lambda: self.graph.restore_edge(prior))
        txn.drafts.append(
            _Draft(
                transaction_id=txn.transaction_id,
                timestamp=utc_now(),
                mutation_type="INVALIDATE",
                affected_edge_id=edge.id,
                entity_ids=[edge.source_id, edge.target_id],
                rationale=rationale,
                prior=prior_snapshot,
                new=self._snapshot(updated),
            )
        )
        return updated

    def _reinforce(
        self, txn: _Transaction, obs: TripleObservation, edge: TemporalEdge, triple_txt: str
    ) -> tuple[ReconciliationDecision, Optional[str], str]:
        prior_snapshot = self._snapshot(edge)
        combined = round(1.0 - (1.0 - edge.confidence_score) * (1.0 - obs.confidence), 4)
        combined = min(combined, 0.999)
        prior, updated = self.graph.update_confidence(edge.id, combined, obs.provenance_id)
        txn.undo.append(lambda: self.graph.restore_edge(prior))
        txn.drafts.append(
            _Draft(
                transaction_id=txn.transaction_id,
                timestamp=utc_now(),
                mutation_type="RECONCILE",
                affected_edge_id=edge.id,
                entity_ids=[edge.source_id, edge.target_id],
                rationale=(
                    f"Corroborating observation of existing fact {triple_txt} from source '{obs.source}'. "
                    f"Confidence fused by noisy-OR: 1 − (1 − {edge.confidence_score:.3f})(1 − "
                    f"{obs.confidence:.3f}) = {combined:.3f}. Valid-time interval unchanged "
                    f"[{_iso(edge.t_start)}, {_iso(edge.t_end)})."
                ),
                prior=prior_snapshot,
                new=self._snapshot(updated),
            )
        )
        return "REINFORCED", edge.id, f"Reinforced {triple_txt}: confidence {edge.confidence_score:.3f} → {combined:.3f}."

    def _record_noop(self, txn: _Transaction, edge: TemporalEdge, *, rationale: str) -> None:
        snapshot = self._snapshot(edge)
        txn.drafts.append(
            _Draft(
                transaction_id=txn.transaction_id,
                timestamp=utc_now(),
                mutation_type="RECONCILE",
                affected_edge_id=edge.id,
                entity_ids=[edge.source_id, edge.target_id],
                rationale=rationale,
                prior=snapshot,
                new=snapshot,
            )
        )

    def _prior_expectation_note(self, obs: TripleObservation, active: list[TemporalEdge]) -> str:
        if not obs.expected_prior_object:
            return ""
        expected_id = entity_id_for(obs.expected_prior_object)
        if not active:
            return (
                f" Observation asserted prior value '{obs.expected_prior_object}', "
                f"but no active prior value was on record."
            )
        if any(e.target_id == expected_id for e in active):
            return f" Observation's stated prior value '{obs.expected_prior_object}' matches the ledger."
        names = ", ".join(self._name(e.target_id) for e in active)
        return (
            f" Discrepancy: observation asserted prior value '{obs.expected_prior_object}' "
            f"but the ledger held '{names}'."
        )
