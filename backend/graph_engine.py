"""Bi-temporal graph engine built on ``networkx.MultiDiGraph``.

The bi-temporal fact model (valid time + system time, superseded facts closed
rather than deleted) follows Graphiti / Zep — Rasmussen et al., 2025,
"Zep: A Temporal Knowledge Graph Architecture for Agent Memory",
arXiv:2501.13956, https://github.com/getzep/graphiti — and the temporal-database
literature (Snodgrass, 1999). This is an independent implementation.

Design notes
------------
* Every fact is an edge keyed by its UUID, so parallel facts between the same
  pair of entities (e.g. a historical and a current ``works_at``) coexist.
* Facts are *never* deleted. Invalidation closes the valid-time interval and
  stamps the system-time ``superseded_at``; history stays queryable.
* An ``_active`` index makes ``get_active_subgraph`` O(|active edges|) rather
  than a scan of the whole history, which dominates as trajectories grow.
* A re-entrant lock guards all mutations; the reconciliation engine takes the
  same lock to make multi-step transitions atomic.
"""

from __future__ import annotations

import heapq
import threading
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable, Optional

import networkx as nx

from models import EntityNode, TemporalEdge, ensure_utc, entity_id_for, utc_now


class GraphEngineError(RuntimeError):
    """Raised on illegal temporal operations (unknown edge, double close, ...)."""


class _MaxMultiset:
    """Multiset of datetimes with O(log n) insert/remove and amortized O(1) max (lazy-deletion heap)."""

    def __init__(self) -> None:
        self._heap: list[float] = []
        self._count: Counter[float] = Counter()
        self._values: dict[float, datetime] = {}

    def add(self, value: datetime) -> None:
        key = value.timestamp()
        self._values[key] = value
        self._count[key] += 1
        heapq.heappush(self._heap, -key)

    def discard(self, value: datetime) -> None:
        key = value.timestamp()
        if self._count[key] > 0:
            self._count[key] -= 1

    def max(self) -> Optional[datetime]:
        while self._heap and self._count[-self._heap[0]] <= 0:
            heapq.heappop(self._heap)
        return self._values[-self._heap[0]] if self._heap else None

    def clear(self) -> None:
        self._heap.clear()
        self._count.clear()
        self._values.clear()


def _last_point(edge: TemporalEdge) -> datetime:
    return edge.t_end if edge.t_end is not None else edge.t_start


class TemporalGraphEngine:
    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._edge_index: dict[str, tuple[str, str]] = {}
        self._active: set[str] = set()
        self._by_subject_relation: dict[tuple[str, str], set[str]] = defaultdict(set)
        # Query-planning indexes: insertion order, relation vocabulary and the latest valid-time point.
        self._order: dict[str, int] = {}
        self._next_order = 0
        self._relation_counts: Counter[str] = Counter()
        self._time_points = _MaxMultiset()
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #

    def upsert_entity(
        self,
        name: str,
        entity_type: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        first_seen: Optional[datetime] = None,
    ) -> EntityNode:
        """Create the entity if absent; otherwise refine its type/attributes.

        A generic ``Entity`` type is upgraded when a more specific type is
        learned, but a specific type is never overwritten by a generic one.
        ``first_seen`` only ever moves earlier (late-arriving history).
        """
        node_id = entity_id_for(name)
        with self.lock:
            if node_id in self._g:
                node: EntityNode = self._g.nodes[node_id]["data"]
                updates: dict[str, Any] = {}
                if entity_type and entity_type != "Entity" and node.entity_type == "Entity":
                    updates["entity_type"] = entity_type
                if attributes:
                    updates["attributes"] = {**node.attributes, **attributes}
                if first_seen is not None and ensure_utc(first_seen) < node.first_seen:
                    updates["first_seen"] = ensure_utc(first_seen)
                if updates:
                    node = EntityNode.model_validate({**node.model_dump(), **updates})
                    self._g.nodes[node_id]["data"] = node
                return node
            node = EntityNode(
                id=node_id,
                name=name.strip(),
                entity_type=entity_type or "Entity",
                attributes=dict(attributes or {}),
                first_seen=ensure_utc(first_seen) if first_seen else utc_now(),
            )
            self._g.add_node(node_id, data=node)
            return node

    def restore_entity(self, node: EntityNode) -> None:
        """Insert an entity verbatim (used by ledger replay)."""
        with self.lock:
            self._g.add_node(node.id, data=node)

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        with self.lock:
            if entity_id not in self._g:
                return None
            return self._g.nodes[entity_id]["data"]

    def entities(self) -> list[EntityNode]:
        with self.lock:
            return [data["data"] for _, data in self._g.nodes(data=True)]

    # ------------------------------------------------------------------ #
    # Edges
    # ------------------------------------------------------------------ #

    def add_temporal_edge(
        self,
        source: str,
        relation: str,
        target: str,
        confidence: float,
        provenance_id: str,
        timestamp: Optional[datetime] = None,
        *,
        t_end: Optional[datetime] = None,
        source_label: str = "agent_observation",
        edge_id: Optional[str] = None,
    ) -> TemporalEdge:
        """Insert a fact with valid-time interval ``[timestamp, t_end)``.

        ``source``/``target`` are entity ids; unknown ids are created as
        generic entities so the engine can be used standalone.
        """
        t_start = ensure_utc(timestamp) if timestamp else utc_now()
        with self.lock:
            for node_id in (source, target):
                if node_id not in self._g:
                    self.upsert_entity(node_id, first_seen=t_start)
            payload: dict[str, Any] = {
                "source_id": source,
                "relation": relation,
                "target_id": target,
                "t_start": t_start,
                "t_end": ensure_utc(t_end) if t_end else None,
                "confidence_score": confidence,
                "provenance_id": provenance_id,
                "source": source_label,
            }
            if edge_id:
                payload["id"] = edge_id
            edge = TemporalEdge.model_validate(payload)
            if t_end is not None:
                edge = TemporalEdge.model_validate(
                    {**edge.model_dump(), "superseded_at": edge.recorded_at}
                )
            if edge.id in self._edge_index:
                raise GraphEngineError(f"Edge id collision: {edge.id}")
            self._store(edge)
            return edge

    def _store(self, edge: TemporalEdge) -> None:
        """Write (or overwrite) an edge and keep all indexes consistent."""
        if edge.id in self._edge_index:
            old_u, old_v = self._edge_index[edge.id]
            if (old_u, old_v) != (edge.source_id, edge.target_id):
                raise GraphEngineError(f"Edge {edge.id} cannot change endpoints")
            previous: TemporalEdge = self._g.edges[old_u, old_v, edge.id]["data"]
            self._time_points.discard(_last_point(previous))
            self._g.edges[old_u, old_v, edge.id]["data"] = edge
        else:
            self._g.add_edge(edge.source_id, edge.target_id, key=edge.id, data=edge)
            self._edge_index[edge.id] = (edge.source_id, edge.target_id)
            self._by_subject_relation[(edge.source_id, edge.relation)].add(edge.id)
            self._relation_counts[edge.relation] += 1
            if edge.id not in self._order:
                self._order[edge.id] = self._next_order
                self._next_order += 1
        self._time_points.add(_last_point(edge))
        if edge.is_active:
            self._active.add(edge.id)
        else:
            self._active.discard(edge.id)

    def restore_edge(self, edge: TemporalEdge) -> None:
        """Upsert an edge verbatim (ledger replay and transactional rollback)."""
        with self.lock:
            for node_id in (edge.source_id, edge.target_id):
                if node_id not in self._g:
                    self.upsert_entity(node_id, first_seen=edge.t_start)
            self._store(edge)

    def remove_edge(self, edge_id: str) -> None:
        """Physically remove an edge. Only used to roll back an uncommitted insert."""
        with self.lock:
            if edge_id not in self._edge_index:
                return
            u, v = self._edge_index.pop(edge_id)
            edge: TemporalEdge = self._g.edges[u, v, edge_id]["data"]
            self._g.remove_edge(u, v, key=edge_id)
            self._active.discard(edge_id)
            self._by_subject_relation[(edge.source_id, edge.relation)].discard(edge_id)
            self._relation_counts[edge.relation] -= 1
            if self._relation_counts[edge.relation] <= 0:
                del self._relation_counts[edge.relation]
            self._time_points.discard(_last_point(edge))
            self._order.pop(edge_id, None)

    def get_edge(self, edge_id: str) -> TemporalEdge:
        with self.lock:
            if edge_id not in self._edge_index:
                raise GraphEngineError(f"Unknown edge {edge_id}")
            u, v = self._edge_index[edge_id]
            return self._g.edges[u, v, edge_id]["data"]

    def has_edge(self, edge_id: str) -> bool:
        with self.lock:
            return edge_id in self._edge_index

    def all_edges(self) -> list[TemporalEdge]:
        with self.lock:
            return [data["data"] for _, _, data in self._g.edges(data=True)]

    def incident_edges(self, node_id: str) -> list[TemporalEdge]:
        """Every fact (any validity) with ``node_id`` as subject or object."""
        with self.lock:
            if node_id not in self._g:
                return []
            out = [d["data"] for _, _, d in self._g.out_edges(node_id, data=True)]
            out.extend(d["data"] for _, _, d in self._g.in_edges(node_id, data=True))
            return out

    def edge_order(self, edge_id: str) -> int:
        """Insertion ordinal of an edge (stable tie-breaker for query planning)."""
        return self._order.get(edge_id, -1)

    def relations(self) -> set[str]:
        """Every relation that labels at least one recorded fact."""
        with self.lock:
            return set(self._relation_counts)

    def latest_time_point(self) -> Optional[datetime]:
        """The latest t_start or t_end of any fact: the default reference time for recency."""
        with self.lock:
            return self._time_points.max()

    def edges_for(self, source_id: str, relation: str) -> list[TemporalEdge]:
        """All facts (any validity) for a (subject, predicate) pair, oldest first."""
        with self.lock:
            ids = self._by_subject_relation.get((source_id, relation), set())
            return sorted((self.get_edge(i) for i in ids), key=lambda e: (e.t_start, e.recorded_at))

    def active_edges_for(self, source_id: str, relation: str) -> list[TemporalEdge]:
        return [e for e in self.edges_for(source_id, relation) if e.is_active]

    def invalidate_edge(
        self,
        edge_id: str,
        end_time: datetime,
        rationale: str,
        *,
        invalidated_by: Optional[str] = None,
    ) -> tuple[TemporalEdge, TemporalEdge]:
        """Close an open valid-time interval. Returns ``(prior, updated)``.

        The edge and both endpoint nodes remain in the graph so point-in-time
        queries before ``end_time`` still see the fact.
        """
        end_time = ensure_utc(end_time)
        with self.lock:
            prior = self.get_edge(edge_id)
            if not prior.is_active:
                raise GraphEngineError(
                    f"Edge {edge_id} already closed at {prior.t_end.isoformat()}"  # type: ignore[union-attr]
                )
            if end_time < prior.t_start:
                raise GraphEngineError(
                    f"Cannot close edge {edge_id} at {end_time.isoformat()}: "
                    f"before its t_start {prior.t_start.isoformat()}"
                )
            updated = TemporalEdge.model_validate(
                {
                    **prior.model_dump(),
                    "t_end": end_time,
                    "superseded_at": utc_now(),
                    "invalidated_by": invalidated_by,
                    "invalidation_rationale": rationale,
                }
            )
            self._store(updated)
            return prior, updated

    def update_confidence(
        self, edge_id: str, confidence: float, corroborating_provenance: Optional[str] = None
    ) -> tuple[TemporalEdge, TemporalEdge]:
        with self.lock:
            prior = self.get_edge(edge_id)
            corroboration = list(prior.corroborating_provenance)
            if corroborating_provenance and corroborating_provenance not in corroboration:
                corroboration.append(corroborating_provenance)
            updated = TemporalEdge.model_validate(
                {
                    **prior.model_dump(),
                    "confidence_score": confidence,
                    "corroborating_provenance": corroboration,
                }
            )
            self._store(updated)
            return prior, updated

    # ------------------------------------------------------------------ #
    # Temporal queries
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_visible(edge: TemporalEdge, ts: datetime, as_known_at: Optional[datetime]) -> bool:
        """Bi-temporal visibility: valid at ``ts`` *as the system knew it* at ``as_known_at``."""
        if as_known_at is None:
            return edge.valid_at(ts)
        if edge.recorded_at > as_known_at:
            return False
        t_end = edge.t_end
        if edge.superseded_at is not None and edge.superseded_at > as_known_at:
            t_end = None  # the invalidation had not yet been learned
        return edge.t_start <= ts and (t_end is None or ts < t_end)

    def query_point_in_time(
        self,
        target_timestamp: datetime,
        *,
        as_known_at: Optional[datetime] = None,
    ) -> nx.MultiDiGraph:
        """Subgraph of facts with ``t_start <= target_timestamp < (t_end or +inf)``.

        Only nodes incident to at least one visible edge are included.
        """
        ts = ensure_utc(target_timestamp)
        known = ensure_utc(as_known_at) if as_known_at else None
        with self.lock:
            edges = [e for e in self.all_edges() if self.is_visible(e, ts, known)]
            return self._materialize(edges)

    def get_active_subgraph(self) -> nx.MultiDiGraph:
        """Fast path: only currently-open facts (``t_end is None``)."""
        with self.lock:
            return self._materialize(self.get_edge(i) for i in self._active)

    def history_subgraph(self, up_to: Optional[datetime] = None) -> nx.MultiDiGraph:
        """All facts that had started by ``up_to`` (active and invalidated)."""
        with self.lock:
            if up_to is None:
                return self._materialize(self.all_edges())
            cutoff = ensure_utc(up_to)
            return self._materialize(e for e in self.all_edges() if e.t_start <= cutoff)

    def _materialize(self, edges: Iterable[TemporalEdge]) -> nx.MultiDiGraph:
        sub = nx.MultiDiGraph()
        for edge in edges:
            for node_id in (edge.source_id, edge.target_id):
                if node_id not in sub:
                    sub.add_node(node_id, data=self._g.nodes[node_id]["data"])
            sub.add_edge(edge.source_id, edge.target_id, key=edge.id, data=edge)
        return sub

    def timeline(self) -> list[datetime]:
        """Sorted distinct valid-time transition points (T0 ... Tn)."""
        with self.lock:
            points: set[datetime] = set()
            for edge in self.all_edges():
                points.add(edge.t_start)
                if edge.t_end is not None:
                    points.add(edge.t_end)
            return sorted(points)

    def active_degree(self, node_id: str) -> int:
        with self.lock:
            if node_id not in self._g:
                return 0
            count = 0
            for _, _, data in self._g.in_edges(node_id, data=True):
                count += data["data"].is_active
            for _, _, data in self._g.out_edges(node_id, data=True):
                count += data["data"].is_active
            return count

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "entities": self._g.number_of_nodes(),
                "edges_total": len(self._edge_index),
                "edges_active": len(self._active),
                "edges_historical": len(self._edge_index) - len(self._active),
            }

    def clear(self) -> None:
        with self.lock:
            self._g.clear()
            self._edge_index.clear()
            self._active.clear()
            self._by_subject_relation.clear()
            self._order.clear()
            self._next_order = 0
            self._relation_counts.clear()
            self._time_points.clear()
