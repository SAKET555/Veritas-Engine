"""Baseline memory systems that VERITAS is compared against.

Every baseline ingests the *same* observation stream (the same extracted
triples, in the same arrival order) and exposes the same retrieval interface,
so evaluations differ only in how memory is represented and selected:

DenseVectorRAG   append-only chunks, top-k by dense cosine similarity
                 (the standard vector-RAG memory)
HybridRAG        BM25 (Okapi, Robertson & Zaragoza 2009) fused with dense
                 retrieval by reciprocal rank fusion (Cormack et al. 2009) —
                 the "BM25 + dense" hybrid most production RAG stacks use
StaticGraphRAG   a knowledge graph of the same triples with no temporal
                 model: facts are deduplicated (s, p, o) edges, retrieval is a
                 k-hop ego graph around the entities the question names
                 (static Graph RAG in the sense of Edge et al. 2024 / Peng et
                 al. 2024 surveys, minus the LLM summarization step)
OverwriteMemory  mutable agent memory *without* audit-trailed reconciliation:
                 a new value for a single-valued slot overwrites the old one
                 in arrival order (the ADD/UPDATE/DELETE pattern of popular
                 agent-memory layers). No history, no source authority, no
                 ledger.

All four return a :class:`Retrieval`: the prompt context they would inject
plus the structured facts behind it, so a deterministic reader can score every
method with exactly the same rule (see ``temporal_bench.py``).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

import networkx as nx

from models import TripleObservation, entity_id_for
from ontology import get_spec

_TOK = re.compile(r"[a-z0-9]+")


def _stamp(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Shared value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Fact:
    """One structured fact as a memory method presents it to the reader.

    ``t_start``/``t_end`` carry whatever temporal knowledge the method has:
    VERITAS supplies closed validity intervals, timestamped baselines supply
    only the observation time (``t_end`` is always ``None`` because they never
    learn that a fact stopped being true), and the static graph supplies none.
    ``rank`` is the fact's position in the context (0 = first).
    """

    subject: str
    predicate: str
    object: str
    t_start: Optional[datetime] = None
    t_end: Optional[datetime] = None
    source: str = ""
    ref: str = ""
    rank: int = 0
    presented_as_current: bool = True


@dataclass
class Retrieval:
    context: str
    facts: list[Fact] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class ObservationLike(Protocol):
    timestamp: datetime
    text: str
    source: str


# --------------------------------------------------------------------------- #
# Dense encoders
# --------------------------------------------------------------------------- #


class HashedDenseEncoder:
    """Deterministic dense encoder (signed feature hashing of word + char-trigram features).

    Projects text into a fixed-width, L2-normalized dense vector. It captures
    exactly the property that matters here — lexical/semantic similarity with no
    notion of time — without requiring model downloads.
    """

    name = "hashed-dense-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = _TOK.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"#{w}#"
            feats.extend(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
        return feats

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (2.0 if feat.startswith("w:") else 1.0)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEncoder:
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self._model = SentenceTransformer(model)
        self.name = f"st/{model}"

    def encode(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]


# --------------------------------------------------------------------------- #
# Chunk-based retrieval (dense, sparse, hybrid)
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    doc_id: str
    text: str
    triple: TripleObservation
    vector: list[float]


def _chunk_fact(chunk: Chunk, rank: int) -> Fact:
    t = chunk.triple
    return Fact(t.subject, get_spec(t.predicate).name, t.object, t.timestamp, None, t.source, chunk.doc_id, rank)


class DenseVectorRAG:
    """Append-only chunk store with top-k cosine retrieval."""

    label = "vector RAG"

    def __init__(self, encoder, top_k: int = 6) -> None:  # type: ignore[no-untyped-def]
        self.encoder = encoder
        self.top_k = top_k
        self.chunks: list[Chunk] = []

    def ingest(self, obs: ObservationLike, triples: list[TripleObservation]) -> None:
        for triple in triples:
            text = f"[{obs.timestamp:%Y-%m-%d}] {obs.text}."
            self.chunks.append(Chunk(f"doc{len(self.chunks):04d}", text, triple, self.encoder.encode(text)))

    def retrieve(self, question: str) -> list[tuple[float, Chunk]]:
        q = self.encoder.encode(question)
        scored = [(sum(a * b for a, b in zip(q, c.vector)), c) for c in self.chunks]
        scored.sort(key=lambda x: -x[0])
        return scored[: self.top_k]

    @staticmethod
    def context(hits: list[tuple[float, Chunk]]) -> str:
        return "\n".join(f"- ({c.doc_id}, sim={s:.3f}) {c.text}" for s, c in hits)

    def search(self, question: str, as_of: Optional[datetime] = None) -> Retrieval:
        hits = self.retrieve(question)
        context = f"## Retrieved notes (top {len(hits)} by similarity)\n" + "\n".join(f"- {c.text}" for _, c in hits)
        return Retrieval(context, [_chunk_fact(c, i) for i, (_, c) in enumerate(hits)], {"retrieved": len(hits)})


def _bm25_tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if len(t) > 1]


class BM25Index:
    """Okapi BM25 with incremental indexing (k1 = 1.5, b = 0.75)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.df: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)

    def add(self, text: str) -> int:
        tf = Counter(_bm25_tokens(text))
        idx = len(self.docs)
        self.docs.append(tf)
        self.lengths.append(sum(tf.values()))
        for term in tf:
            self.df[term] += 1
            self.postings[term].append(idx)
        return idx

    def scores(self, query: str) -> dict[int, float]:
        n = len(self.docs)
        if n == 0:
            return {}
        avgdl = sum(self.lengths) / n or 1.0
        out: dict[int, float] = defaultdict(float)
        for term in set(_bm25_tokens(query)):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for idx in self.postings[term]:
                tf = self.docs[idx][term]
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.lengths[idx] / avgdl)
                out[idx] += idf * tf * (self.k1 + 1.0) / denom
        return out


class HybridRAG:
    """BM25 + dense retrieval fused with reciprocal rank fusion: score = Σ 1 / (k + rank)."""

    label = "hybrid BM25 + dense RAG"

    def __init__(self, encoder, top_k: int = 6, rrf_k: int = 60, candidates: int = 50) -> None:  # type: ignore[no-untyped-def]
        self.dense = DenseVectorRAG(encoder, top_k=candidates)
        self.bm25 = BM25Index()
        self.top_k = top_k
        self.rrf_k = rrf_k

    @property
    def chunks(self) -> list[Chunk]:
        return self.dense.chunks

    def ingest(self, obs: ObservationLike, triples: list[TripleObservation]) -> None:
        before = len(self.dense.chunks)
        self.dense.ingest(obs, triples)
        for chunk in self.dense.chunks[before:]:
            self.bm25.add(chunk.text)

    def retrieve(self, question: str) -> list[tuple[float, Chunk]]:
        dense_hits = self.dense.retrieve(question)
        position = {id(c): i for i, c in enumerate(self.dense.chunks)}
        fused: dict[int, float] = defaultdict(float)
        for rank, (_, chunk) in enumerate(dense_hits):
            fused[position[id(chunk)]] += 1.0 / (self.rrf_k + rank + 1)
        sparse = sorted(self.bm25.scores(question).items(), key=lambda kv: -kv[1])[: self.dense.top_k]
        for rank, (idx, _) in enumerate(sparse):
            fused[idx] += 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_k]
        return [(score, self.dense.chunks[idx]) for idx, score in ranked]

    def search(self, question: str, as_of: Optional[datetime] = None) -> Retrieval:
        hits = self.retrieve(question)
        context = (f"## Retrieved notes (top {len(hits)} by BM25 + dense rank fusion)\n"
                   + "\n".join(f"- {c.text}" for _, c in hits))
        return Retrieval(context, [_chunk_fact(c, i) for i, (_, c) in enumerate(hits)], {"retrieved": len(hits)})


# --------------------------------------------------------------------------- #
# Entity-anchored graph baselines
# --------------------------------------------------------------------------- #


class _EntityIndex:
    """Longest-match entity mention detection, the same strategy VERITAS's seed resolver uses."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}  # entity id -> surface name

    def add(self, name: str) -> str:
        eid = entity_id_for(name)
        self.names.setdefault(eid, name)
        return eid

    def seeds(self, question: str) -> list[str]:
        lowered = question.lower()
        found: list[str] = []
        claimed: list[tuple[int, int]] = []
        for eid, name in sorted(self.names.items(), key=lambda kv: len(kv[1]), reverse=True):
            pattern = r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])"
            for match in re.finditer(pattern, lowered):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in claimed):
                    continue
                claimed.append(span)
                if eid not in found:
                    found.append(eid)
        return found


def _ego(graph: nx.MultiDiGraph, seeds: list[str], k_hops: int) -> dict[str, int]:
    """Hop distance from the nearest seed, following edges in either direction."""
    dist = {s: 0 for s in seeds if s in graph}
    frontier = list(dist)
    for hop in range(1, k_hops + 1):
        nxt: list[str] = []
        for u in frontier:
            for v in list(graph.successors(u)) + list(graph.predecessors(u)):
                if v not in dist:
                    dist[v] = hop
                    nxt.append(v)
        frontier = nxt
    return dist


class StaticGraphRAG:
    """Knowledge-graph memory with no time dimension.

    Every distinct (subject, predicate, object) triple becomes one edge, kept
    forever. A later contradiction simply adds a second edge, so the graph holds
    every value a functional slot ever had, with nothing marking which is
    current. Retrieval takes the k-hop ego graph around the entities the
    question names and orders edges by hop distance, then by insertion order.
    """

    label = "static graph RAG"

    def __init__(self, k_hops: int = 2, max_edges: int = 40) -> None:
        self.graph = nx.MultiDiGraph()
        self.entities = _EntityIndex()
        self.k_hops = k_hops
        self.max_edges = max_edges
        self._seen: set[tuple[str, str, str]] = set()
        self._order = 0

    def ingest(self, obs: ObservationLike, triples: list[TripleObservation]) -> None:
        for t in triples:
            s, o = self.entities.add(t.subject), self.entities.add(t.object)
            p = get_spec(t.predicate).name
            if (s, p, o) in self._seen or s == o:
                continue
            self._seen.add((s, p, o))
            self.graph.add_edge(s, o, key=f"{p}:{self._order}", relation=p, order=self._order)
            self._order += 1

    def search(self, question: str, as_of: Optional[datetime] = None) -> Retrieval:
        seeds = self.entities.seeds(question)
        dist = _ego(self.graph, seeds, self.k_hops)
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in dist and v in dist:
                edges.append((min(dist[u], dist[v]), data["order"], u, data["relation"], v))
        edges.sort()
        edges = edges[: self.max_edges]
        name = self.entities.names
        lines = [f"## Knowledge graph facts ({len(edges)})", "fmt: subject -relation-> object"]
        lines += [f"- {name[u]} -{p}-> {name[v]}" for _, _, u, p, v in edges] or ["- (no facts found)"]
        facts = [Fact(name[u], p, name[v], ref=f"e{order}", rank=i) for i, (_, order, u, p, v) in enumerate(edges)]
        return Retrieval("\n".join(lines), facts, {"seeds": seeds, "edges": len(edges)})


class OverwriteMemory:
    """Mutable slot memory without audit-trailed reconciliation.

    Functional slots keep exactly one value: whatever arrived *last* (arrival
    order, regardless of valid time or source authority). Multi-valued slots
    accumulate. Overwritten values are discarded — there is no history and no
    ledger, so nothing can be cited, audited or rolled back.
    """

    label = "overwrite memory (no audit trail)"

    def __init__(self, k_hops: int = 2, max_edges: int = 40) -> None:
        self.slots: dict[tuple[str, str], dict[str, tuple[str, datetime, str]]] = {}
        self.entities = _EntityIndex()
        self.k_hops = k_hops
        self.max_edges = max_edges
        self.overwrites = 0

    def ingest(self, obs: ObservationLike, triples: list[TripleObservation]) -> None:
        for t in triples:
            s, o = self.entities.add(t.subject), self.entities.add(t.object)
            if s == o:
                continue
            spec = get_spec(t.predicate)
            values = self.slots.setdefault((s, spec.name), {})
            if spec.functional:
                if values and o not in values:
                    self.overwrites += 1
                values.clear()
            values[o] = (t.object, t.timestamp, t.source)

    def _graph(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for (s, p), values in self.slots.items():
            for o in values:
                g.add_edge(s, o, relation=p)
        return g

    def search(self, question: str, as_of: Optional[datetime] = None) -> Retrieval:
        seeds = self.entities.seeds(question)
        dist = _ego(self._graph(), seeds, self.k_hops)
        name = self.entities.names
        rows = []
        for (s, p), values in self.slots.items():
            for o, (_, ts, source) in values.items():
                if s in dist and o in dist:
                    rows.append((min(dist[s], dist[o]), name[s], p, name[o], ts, source))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        rows = rows[: self.max_edges]
        lines = [f"## Agent memory (current values, {len(rows)})"]
        lines += [f"- {s} {p.replace('_', ' ')}: {o} (updated {_stamp(ts)})" for _, s, p, o, ts, _ in rows] or ["- (empty)"]
        facts = [Fact(s, p, o, ts, None, src, rank=i) for i, (_, s, p, o, ts, src) in enumerate(rows)]
        return Retrieval("\n".join(lines), facts, {"seeds": seeds, "facts": len(rows)})


class FullHistoryMemory:
    """Append-only transcript memory: the whole observation log goes into every prompt."""

    label = "full history"

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.facts: list[Fact] = []

    def ingest(self, obs: ObservationLike, triples: list[TripleObservation]) -> None:
        self.lines.append(f"[{obs.timestamp:%Y-%m-%d %H:%M}] {obs.text}.")
        for t in triples:
            self.facts.append(Fact(t.subject, get_spec(t.predicate).name, t.object, t.timestamp, None, t.source,
                                   f"obs{len(self.lines) - 1}", len(self.facts)))

    def search(self, question: str, as_of: Optional[datetime] = None) -> Retrieval:
        context = f"## Observation log ({len(self.lines)} entries, oldest first)\n" + "\n".join(self.lines)
        return Retrieval(context, list(self.facts), {"entries": len(self.lines)})
