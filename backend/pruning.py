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
from functools import lru_cache
from datetime import datetime, timezone
from typing import Iterable, Optional


from graph_engine import TemporalGraphEngine
from models import PruningParams, ScoredEdge, TemporalEdge, ensure_utc
from ontology import predicate_tokens
from entity_linking import EntityLinker
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


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?Z?)?\b")
_MDY_DATE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)
_DMY_DATE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)


def extract_as_of(question: str) -> Optional[datetime]:
    """The valid-time instant a question asks about, if it names a calendar date.

    Recognizes ISO dates (``2026-03-15``, optionally with ``THH:MM``),
    ``March 15, 2026`` / ``Mar 15 2026`` and ``15 March 2026``. A bare date
    means midday UTC of that day, so "on 2026-03-01" sees a change recorded at
    09:00 that day. Returns ``None`` when no unambiguous date is present.
    """
    found: list[datetime] = []
    try:
        for m in _ISO_DATE.finditer(question):
            hour, minute, second = int(m.group(4) or 12), int(m.group(5) or 0), int(m.group(6) or 0)
            found.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), hour, minute, second,
                                  tzinfo=timezone.utc))
        for m in _MDY_DATE.finditer(question):
            found.append(datetime(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)), 12,
                                  tzinfo=timezone.utc))
        for m in _DMY_DATE.finditer(question):
            found.append(datetime(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)), 12,
                                  tzinfo=timezone.utc))
    except ValueError:  # e.g. February 30
        return None
    unique = set(found)
    return found[0] if len(unique) == 1 else None


@lru_cache(maxsize=65536)
def _name_tokens(name: str) -> frozenset[str]:
    return frozenset(tokenize(name))


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
    # Facts that provably cannot reach τ (outside the ego graph, or too weakly connected to it)
    # are dropped without a score record, so a query costs O(ego) instead of O(|history|).
    unscored: int = 0


class SubgraphPruner:
    def __init__(self, graph: TemporalGraphEngine, ledger: AuditLedger,
                 linker: Optional[EntityLinker] = None) -> None:
        self.graph = graph
        self.ledger = ledger
        self.linker = linker

    def _alias_surfaces(self) -> list[tuple[str, str]]:
        """(alias name, canonical entity id) for every alias whose canonical entity exists."""
        if self.linker is None:
            return []
        return [(a.alias, a.canonical_id) for a in self.linker.aliases() if self.graph.get_entity(a.canonical_id)]

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
                else:
                    resolved.extend(cid for alias, cid in self._alias_surfaces()
                                    if alias.lower() == key and cid not in resolved)
            return resolved

        lowered = question.lower()
        found: list[str] = []
        claimed: list[tuple[int, int]] = []
        surfaces = [(e.name, e.id) for e in self.graph.entities()] + self._alias_surfaces()
        for name, entity_id in sorted(surfaces, key=lambda pair: len(pair[0]), reverse=True):
            if name.lower() not in lowered:  # cheap filter before compiling a pattern
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])"
            for match in re.finditer(pattern, lowered):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in claimed):
                    continue
                claimed.append(span)
                if entity_id not in found:
                    found.append(entity_id)
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

        def visible(edge: TemporalEdge) -> bool:
            # Every fact that had started by as_of (the universe of graph.history_subgraph(as_of)).
            return as_of is None or edge.t_start <= as_of

        def current(edge: TemporalEdge) -> bool:
            return edge.valid_at(as_of) if as_of else edge.is_active

        def traversable(edge: TemporalEdge) -> bool:
            return visible(edge) and (include_history or current(edge))

        weight_sum = params.alpha + params.beta + params.gamma
        if weight_sum <= 0:
            raise ValueError("alpha + beta + gamma must be positive")
        # Upper bound on what any fact can score is (α + β + γ·structural)/Σ, since confidence and
        # recency are ≤ 1 and relevance ≤ structural. A fact whose structural weight is below w_min
        # can never reach τ, so it is dropped without a score record (counted in ``unscored``).
        w_min = (params.tau * weight_sum - params.alpha - params.beta) / params.gamma if params.gamma > 0 else 0.0

        if seed_ids and params.candidate_mode == "ego":
            weights, damp, candidates = self._ego(seed_ids, params, visible, traversable, w_min)
            relations = self.graph.relations()
            reference = as_of or self.graph.latest_time_point()
            if as_of is None:
                total = self.graph.stats()["edges_total"]
            else:
                total = sum(1 for e in self.graph.all_edges() if visible(e))
        elif seed_ids:
            # Default ("all"): score every fact in the universe, exactly as before; the ego walk
            # only supplies path weights (with no τ cut-off, so every reachable node is expanded).
            candidates = [e for e in self.graph.all_edges() if visible(e)]
            weights, damp, _ = self._ego(seed_ids, params, visible, traversable, float("-inf"), collect=False)
            relations = {e.relation for e in candidates}
            reference = as_of or max((self._last_valid(e, None) for e in candidates), default=None)
            total = len(candidates)
        else:
            # No anchor: every node is equally (and only moderately) relevant.
            candidates = [e for e in self.graph.all_edges() if visible(e)]
            weights = {n: 0.5 for e in candidates for n in (e.source_id, e.target_id)}
            damp = {}
            relations = {e.relation for e in candidates}
            reference = as_of or max((self._last_valid(e, None) for e in candidates), default=None)
            total = len(candidates)
        q_tokens = tokenize(question)
        # A question that names no relation at all ("Tell me about Alice") carries no
        # relational signal, so every relation is treated as lexically relevant.
        lexical_neutral = not any(predicate_tokens(r) & q_tokens for r in relations)

        scored: list[tuple[ScoredEdge, TemporalEdge]] = []
        for edge in candidates:
            src_w = weights.get(edge.source_id)
            tgt_w = weights.get(edge.target_id)
            in_ego = src_w is not None and tgt_w is not None
            is_current = current(edge)

            structural = 0.0
            if in_ego:
                reverse = (tgt_w or 0.0) * params.reverse_edge_penalty / damp.get(edge.target_id, 1.0)
                structural = max(src_w or 0.0, reverse)
            lexical = 1.0 if lexical_neutral or predicate_tokens(edge.relation) & q_tokens else 0.0
            target_entity = self.graph.get_entity(edge.target_id)
            if lexical == 0.0 and target_entity and _name_tokens(target_entity.name) <= q_tokens:
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
        unscored = total - len(candidates)

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
            unscored=unscored,
        )

    def _ego(
        self,
        seeds: list[str],
        params: PruningParams,
        visible,  # type: ignore[no-untyped-def]
        traversable,  # type: ignore[no-untyped-def]
        w_min: float,
        *,
        collect: bool = True,
    ) -> tuple[dict[str, float], dict[str, float], list[TemporalEdge]]:
        """Radius-k weighted expansion over the engine's adjacency, touching only the ego neighbourhood.

        Returns ``(path weights, hub damping per node, candidate facts)``. Path weights follow the
        module docstring; in addition, walking an edge *backwards* out of a non-seed node with
        fan-in ``d > 1`` is divided by ``d ** hub_damping``, so a city or employer shared by
        thousands of people does not pull all of them into every question about one person.
        Nodes whose weight is below ``w_min`` are not expanded: nothing past them can reach τ.
        """
        seed_set = set(seeds)
        incident: dict[str, list[TemporalEdge]] = {}

        def edges_of(node: str) -> list[TemporalEdge]:
            if node not in incident:
                incident[node] = [e for e in self.graph.incident_edges(node) if visible(e)]
            return incident[node]

        damp: dict[str, float] = {}

        def damping(node: str) -> float:
            if node not in damp:
                fan_in = sum(1 for e in edges_of(node) if e.target_id == node and traversable(e))
                damp[node] = fan_in ** params.hub_damping if fan_in > 1 and node not in seed_set else 1.0
            return damp[node]

        weights = {s: 1.0 for s in seeds if edges_of(s)}
        frontier = set(weights)
        for _ in range(params.k_hops):
            updated: set[str] = set()
            for u in frontier:
                base = weights[u]
                if base < w_min:
                    continue
                for e in edges_of(u):
                    if not traversable(e):
                        continue
                    if e.source_id == u:
                        v, cand = e.target_id, base * params.hop_decay
                    else:
                        v, cand = e.source_id, base * params.hop_decay * params.reverse_edge_penalty / damping(u)
                    if cand > weights.get(v, 0.0):
                        weights[v] = cand
                        updated.add(v)
            if not updated:
                break
            frontier = updated

        if not collect:
            for node in weights:
                damping(node)
            return weights, damp, []
        candidates: dict[str, TemporalEdge] = {}
        for node, weight in weights.items():
            if weight < w_min and node not in seed_set:
                # A fact can only pass τ through a strong endpoint, and is collected from that one.
                continue
            for e in edges_of(node):
                if e.id in candidates:
                    continue
                src_w = weights.get(e.source_id, 0.0)
                tgt_w = weights.get(e.target_id, 0.0) * params.reverse_edge_penalty / damping(e.target_id)
                if max(src_w, tgt_w) >= w_min or e.source_id in seed_set or e.target_id in seed_set:
                    candidates[e.id] = e
        ordered = sorted(candidates.values(), key=lambda e: self.graph.edge_order(e.id))
        return weights, damp, ordered

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
        # Keep the header neutral: an earlier header that defined "past" made a 3B model read
        # *every* fact as no longer true and answer UNKNOWN (measured: 5/11 → 12/15 when fixed).
        lines = [f"Facts (as of {day(as_of)}):" if as_of else "Facts:"]
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
            replaced = f"; replaced by {self._name(successor.target_id)}" if successor else ""
            lines.append(f"{head} (ENDED {day(edge.t_end)}, was true from {day(edge.t_start)}{replaced})")  # type: ignore[arg-type]
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
