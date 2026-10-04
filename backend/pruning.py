"""Confidence-driven subgraph pruning and compact context serialization.

Pipeline
--------
1. Seed resolution  – entities mentioned in the question (or supplied explicitly).
2. Ego expansion    – radius-k traversal from the seeds. Each node carries a
   *path weight*: seeds = 1, a forward hop multiplies by ``hop_decay`` and a
   reverse hop (walking an edge against its direction) additionally by
   ``reverse_edge_penalty``. This keeps multi-hop chains the question follows
   (Alice → Acme → Berlin) while damping the fan-in explosion through hub
   nodes (every other person who also lives in London).
3. Composite score  – Score(e) = (α·Confidence + β·Recency + γ·Relevance) / (α+β+γ)
      Recency(e)   = 2^(−age / half_life), age measured from the fact's last
                     moment of validity to the reference time.
      Relevance(e) = structural(e) · (0.3 + 0.7 · lexical(e))
                     structural = ego path weight, lexical = question names the
                     relation (ontology aliases) or the target entity
4. Filtering        – inactive facts dropped unless historical intent is
   flagged; facts below τ dropped; top ``max_edges`` retained; isolated
   nodes discarded.
5. Serialization    – one line per fact, mutation-ID citations inline.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import networkx as nx

from graph_engine import TemporalGraphEngine
from models import PruningParams, ScoredEdge, TemporalEdge, ensure_utc
from ontology import predicate_tokens
from reconciliation import AuditLedger

_WORD = re.compile(r"[a-z0-9]+")
_HISTORY_CUES = frozenset({
    "previous", "previously", "before", "used", "formerly", "former", "history", "historical",
    "earlier", "originally", "changed", "change", "changes", "why", "when", "past", "prior",
    "old", "was", "were", "did", "provenance", "audit", "trail", "timeline", "moved", "left",
})


def tokenize(text: str) -> set[str]:
    """Lower-cased word tokens with light suffix stemming."""
    out: set[str] = set()
    for word in _WORD.findall(text.lower()):
        out.add(word)
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                out.add(word[: -len(suffix)])
                break
    return out


def fmt_ts(ts: Optional[datetime]) -> str:
    if ts is None:
        return "now"
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%dT%H:%MZ")


@dataclass
class PrunedContext:
    text: str
    output_format: str
    seeds: list[str]
    include_history: bool
    kept_edges: list[TemporalEdge]
    scored: list[ScoredEdge]
    citations: dict[str, str]  # short mutation id -> edge id
    reference_time: Optional[datetime]


class SubgraphPruner:
    def __init__(self, graph: TemporalGraphEngine, ledger: AuditLedger) -> None:
        self.graph = graph
        self.ledger = ledger

    # ------------------------------------------------------------------ #
    # Seeds & intent
    # ------------------------------------------------------------------ #

    def resolve_seeds(self, question: str, explicit: Optional[Iterable[str]] = None) -> list[str]:
        if explicit:
            resolved = []
            for raw in explicit:
                key = raw.strip().lower()
                for entity in self.graph.entities():
                    if entity.id == key or entity.name.lower() == key:
                        resolved.append(entity.id)
                        break
            return resolved

        lowered = question.lower()
        found: list[str] = []
        claimed: list[tuple[int, int]] = []
        for entity in sorted(self.graph.entities(), key=lambda e: len(e.name), reverse=True):
            pattern = r"(?<![a-z0-9])" + re.escape(entity.name.lower()) + r"(?:'s)?(?![a-z0-9])"
            for match in re.finditer(pattern, lowered):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in claimed):
                    continue
                claimed.append(span)
                if entity.id not in found:
                    found.append(entity.id)
        return found

    @staticmethod
    def detect_historical_intent(question: str) -> bool:
        return bool(tokenize(question) & _HISTORY_CUES)

    # ------------------------------------------------------------------ #
    # Pruning
    # ------------------------------------------------------------------ #

    def prune(
        self,
        question: str,
        params: PruningParams,
        *,
        seeds: Optional[list[str]] = None,
        as_of: Optional[datetime] = None,
    ) -> PrunedContext:
        as_of = ensure_utc(as_of) if as_of else None
        include_history = (
            params.include_history
            if params.include_history is not None
            else self.detect_historical_intent(question)
        )
        seed_ids = seeds if seeds is not None else self.resolve_seeds(question)

        universe = self.graph.history_subgraph(as_of)
        all_edges: list[TemporalEdge] = [d["data"] for _, _, d in universe.edges(data=True)]

        def current(edge: TemporalEdge) -> bool:
            return edge.valid_at(as_of) if as_of else edge.is_active

        traversal = nx.MultiDiGraph()
        traversal.add_nodes_from(universe.nodes)
        for edge in all_edges:
            if include_history or current(edge):
                traversal.add_edge(edge.source_id, edge.target_id, key=edge.id)

        weights = self._path_weights(traversal, seed_ids, params)
        reference = as_of or max((self._last_valid(e, None) for e in all_edges), default=None)
        q_tokens = tokenize(question)
        # A question that names no relation at all ("Tell me about Alice") carries no
        # relational signal, so every relation is treated as lexically relevant.
        lexical_neutral = not any(predicate_tokens(e.relation) & q_tokens for e in all_edges)

        weight_sum = params.alpha + params.beta + params.gamma
        if weight_sum <= 0:
            raise ValueError("alpha + beta + gamma must be positive")

        scored: list[tuple[ScoredEdge, TemporalEdge]] = []
        for edge in all_edges:
            src_w = weights.get(edge.source_id)
            tgt_w = weights.get(edge.target_id)
            in_ego = src_w is not None and tgt_w is not None
            is_current = current(edge)

            structural = 0.0
            if in_ego:
                structural = max(src_w or 0.0, (tgt_w or 0.0) * params.reverse_edge_penalty)
            lexical = 1.0 if lexical_neutral or predicate_tokens(edge.relation) & q_tokens else 0.0
            target_entity = self.graph.get_entity(edge.target_id)
            if lexical == 0.0 and target_entity and tokenize(target_entity.name) <= q_tokens:
                lexical = 0.5
            relevance = structural * (0.3 + 0.7 * lexical)
            recency = self._recency(edge, reference, params.recency_half_life_days)
            score = (
                params.alpha * edge.confidence_score
                + params.beta * recency
                + params.gamma * relevance
            ) / weight_sum

            if not is_current and not include_history:
                kept, reason = False, "inactive (t_end set) and no historical intent"
            elif not in_ego:
                kept, reason = False, f"outside {params.k_hops}-hop ego graph"
            elif score < params.tau:
                kept, reason = False, f"score {score:.3f} < τ {params.tau:.2f}"
            else:
                kept, reason = True, f"score {score:.3f} ≥ τ {params.tau:.2f}"

            scored.append((
                ScoredEdge(
                    edge_id=edge.id,
                    source=self._name(edge.source_id),
                    relation=edge.relation,
                    target=self._name(edge.target_id),
                    active=is_current,
                    confidence=round(edge.confidence_score, 4),
                    recency=round(recency, 4),
                    relevance=round(relevance, 4),
                    score=round(score, 4),
                    kept=kept,
                    reason=reason,
                ),
                edge,
            ))

        scored.sort(key=lambda pair: (pair[0].kept, pair[0].score), reverse=True)
        kept_edges: list[TemporalEdge] = []
        final_scored: list[ScoredEdge] = []
        for item, edge in scored:
            if item.kept and len(kept_edges) >= params.max_edges:
                item = item.model_copy(update={"kept": False, "reason": f"over max_edges={params.max_edges}"})
            if item.kept:
                kept_edges.append(edge)
            final_scored.append(item)

        # Stable, readable order: current facts first, then by subject and time.
        kept_edges.sort(key=lambda e: (not current(e), self._name(e.source_id), e.relation, e.t_start))
        text, citations = self.serialize(kept_edges, params.output_format, as_of, current)
        return PrunedContext(
            text=text,
            output_format=params.output_format,
            seeds=seed_ids,
            include_history=include_history,
            kept_edges=kept_edges,
            scored=final_scored,
            citations=citations,
            reference_time=reference,
        )

    def _path_weights(
        self, graph: nx.MultiDiGraph, seeds: list[str], params: PruningParams
    ) -> dict[str, float]:
        if not seeds:
            # No anchor: every node is equally (and only moderately) relevant.
            return {node: 0.5 for node in graph.nodes}
        weights = {s: 1.0 for s in seeds if s in graph}
        frontier = set(weights)
        for _ in range(params.k_hops):
            updated: set[str] = set()
            for u in frontier:
                base = weights[u]
                for _, v in graph.out_edges(u):
                    cand = base * params.hop_decay
                    if cand > weights.get(v, 0.0):
                        weights[v] = cand
                        updated.add(v)
                for v, _ in graph.in_edges(u):
                    cand = base * params.hop_decay * params.reverse_edge_penalty
                    if cand > weights.get(v, 0.0):
                        weights[v] = cand
                        updated.add(v)
            if not updated:
                break
            frontier = updated
        return weights

    @staticmethod
    def _last_valid(edge: TemporalEdge, reference: Optional[datetime]) -> datetime:
        if edge.t_end is not None:
            return edge.t_end
        return reference or edge.t_start

    def _recency(self, edge: TemporalEdge, reference: Optional[datetime], half_life_days: float) -> float:
        if reference is None:
            return 1.0
        anchor = edge.t_end if edge.t_end is not None and edge.t_end <= reference else edge.t_start
        age_days = max(0.0, (reference - anchor).total_seconds() / 86400.0)
        return math.pow(2.0, -age_days / half_life_days)

    def _name(self, entity_id: str) -> str:
        entity = self.graph.get_entity(entity_id)
        return entity.name if entity else entity_id

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def serialize(
        self,
        edges: list[TemporalEdge],
        output_format: str,
        as_of: Optional[datetime],
        is_current=None,  # type: ignore[no-untyped-def]
    ) -> tuple[str, dict[str, str]]:
        """Compact serialization. Returns (text, {short_mutation_id: edge_id})."""
        current = is_current or (lambda e: e.is_active)
        citations: dict[str, str] = {}
        rows: list[tuple[str, str, str, float, str, str, bool, str]] = []
        for edge in edges:
            record = self.ledger.latest_for_edge(edge.id)
            cite = record.short_id if record else edge.id[:8]
            citations[cite] = edge.id
            rows.append((
                self._name(edge.source_id), edge.relation, self._name(edge.target_id),
                edge.confidence_score, fmt_ts(edge.t_start),
                "now" if current(edge) else fmt_ts(edge.t_end),
                current(edge), cite,
            ))

        stamp = fmt_ts(as_of) if as_of else "now"
        if output_format == "json":
            payload = {
                "as_of": stamp,
                "cols": ["s", "p", "o", "conf", "from", "to", "cur", "cite"],
                "facts": [[s, p, o, round(c, 2), f, t, int(cur), m] for s, p, o, c, f, t, cur, m in rows],
            }
            return json.dumps(payload, separators=(",", ":"), ensure_ascii=False), citations

        lines = [f"## KG facts as_of {stamp} ({len(rows)})", "fmt: subject -relation-> object | conf | valid from..to | cite"]
        for s, p, o, c, f, t, cur, m in rows:
            flag = "" if cur else " [INVALIDATED]"
            lines.append(f"- {s} -{p}-> {o} | {c:.2f} | {f}..{t}{flag} | M:{m}")
        if not rows:
            lines.append("- (no facts passed pruning)")
        return "\n".join(lines), citations

    def serialize_compact(
        self,
        edges: list[TemporalEdge],
        as_of: Optional[datetime],
        is_current=None,  # type: ignore[no-untyped-def]
    ) -> tuple[str, dict[str, str], dict[str, str]]:
        """Token-lean serialization for small models.

        * Citations are prompt-local tags ``[1]``, ``[2]`` mapped server-side to
          ledger mutation IDs, instead of 8-hex IDs the model must copy.
        * No confidence column (everything shown already passed the τ filter).
        * Day-granularity dates.
        * Each past fact states what *replaced* it and is ordered newest-first
          right after the current value, so "what came before X" is a lookup
          rather than date arithmetic.

        Returns ``(text, {tag: short_mutation_id}, {short_mutation_id: edge_id})``.
        """
        current = is_current or (lambda e: e.is_active)
        live = sorted((e for e in edges if current(e)), key=lambda e: (self._name(e.source_id), e.relation))
        past = [e for e in edges if not current(e)]

        ordered: list[TemporalEdge] = []
        for edge in live:
            ordered.append(edge)
            same_slot = [p for p in past if (p.source_id, p.relation) == (edge.source_id, edge.relation)]
            ordered.extend(sorted(same_slot, key=lambda p: p.t_start, reverse=True))
        placed = {e.id for e in ordered}
        ordered.extend(sorted((p for p in past if p.id not in placed), key=lambda p: (self._name(p.source_id), p.relation, -p.t_start.timestamp())))

        day = lambda ts: ts.strftime("%Y-%m-%d")  # noqa: E731
        aliases: dict[str, str] = {}
        citations: dict[str, str] = {}
        stamp = day(as_of) if as_of else "now"
        lines = [f'Facts as of {stamp} ("past" = no longer true):']
        for n, edge in enumerate(ordered, start=1):
            record = self.ledger.latest_for_edge(edge.id)
            short = record.short_id if record else edge.id[:8]
            aliases[str(n)] = short
            citations[short] = edge.id
            head = f"[{n}] {self._name(edge.source_id)} -{edge.relation}-> {self._name(edge.target_id)}"
            if current(edge):
                lines.append(f"{head} (since {day(edge.t_start)})")
                continue
            successor = self._successor(edge)
            replaced = f", replaced by {self._name(successor.target_id)}" if successor else ""
            lines.append(f"{head} (past: {day(edge.t_start)} to {day(edge.t_end)}{replaced})")  # type: ignore[arg-type]
        if not ordered:
            lines.append("(none)")
        return "\n".join(lines), aliases, citations

    def _successor(self, edge: TemporalEdge) -> Optional[TemporalEdge]:
        """The fact that took over this (subject, relation) slot when ``edge`` ended."""
        if edge.t_end is None:
            return None
        later = [e for e in self.graph.edges_for(edge.source_id, edge.relation) if e.id != edge.id and e.t_start >= edge.t_end]
        return min(later, key=lambda e: e.t_start) if later else None

    def serialize_full_graph(self, output_format: str = "markdown") -> str:
        """Unpruned baseline: every fact ever recorded, same serializer."""
        edges = sorted(self.graph.all_edges(), key=lambda e: (self._name(e.source_id), e.relation, e.t_start))
        text, _ = self.serialize(edges, output_format, None)
        return text

    def serialize_audit_trail(
        self,
        edges: list[TemporalEdge],
        include_history: bool,
        question: str = "",
        max_records: int = 12,
        max_rationale_chars: int = 240,
    ) -> tuple[str, list[str]]:
        """Mutation records justifying the facts the question is *about*.

        Every kept fact is already cited inline; the trail adds the rationale
        only for facts whose relation the question names (falling back to all
        kept facts for relation-agnostic questions). In historical mode every
        record of those facts is included, so invalidations are explained.
        Returns (text, short ids).
        """
        q_tokens = tokenize(question)
        focus = [e for e in edges if predicate_tokens(e.relation) & q_tokens] or edges
        selected = []
        for edge in focus:
            records = self.ledger.for_edge(edge.id)
            if not records:
                continue
            selected.extend(records if include_history else records[-1:])
        selected.sort(key=lambda r: r.sequence)
        selected = selected[-max_records:]
        lines = ["## Mutation audit trail (verified, hash-chained)"]
        short_ids: list[str] = []
        for record in selected:
            edge = record.new_state_snapshot["edge"]
            src = record.new_state_snapshot["source_entity"]["name"]
            tgt = record.new_state_snapshot["target_entity"]["name"]
            rationale = record.deterministic_rationale
            if len(rationale) > max_rationale_chars:
                rationale = rationale[: max_rationale_chars - 1].rsplit(" ", 1)[0] + "…"
            lines.append(
                f"- M:{record.short_id} #{record.sequence} {record.mutation_type} "
                f"{src} -{edge['relation']}-> {tgt}: {rationale}"
            )
            short_ids.append(record.short_id)
        if not selected:
            lines.append("- (no mutations recorded for retained facts)")
        return "\n".join(lines), short_ids
