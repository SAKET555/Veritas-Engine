"""Baseline memories: each must behave like the system it stands in for, failure modes included."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from baselines import (
    BM25Index,
    FullHistoryMemory,
    HashedDenseEncoder,
    HybridRAG,
    OverwriteMemory,
    StaticGraphRAG,
)
from extraction import extract_triples
from scenarios import ScriptedObservation

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def feed(memory, *rows: tuple[int, str, str]) -> None:  # type: ignore[no-untyped-def]
    for day, text, source in rows:
        obs = ScriptedObservation("T", T0 + timedelta(days=day), text, 0.9, source)
        memory.ingest(obs, extract_triples(text, source=source, timestamp=obs.timestamp))


def test_bm25_prefers_rare_matching_terms() -> None:
    idx = BM25Index()
    for text in ("Alice lives in London", "Bob lives in London", "Carol works at Zephyr"):
        idx.add(text)
    scores = idx.scores("Where does Carol work?")
    assert max(scores, key=scores.get) == 2
    assert idx.scores("") == {}


def test_hybrid_rag_fuses_and_keeps_stale_chunks() -> None:
    rag = HybridRAG(HashedDenseEncoder(), top_k=3)
    feed(rag, (0, "Alice lives in Mumbai", "user_statement"), (30, "Alice moved to London", "user_statement"),
         (31, "Bob works at Acme", "user_statement"))
    found = rag.search("Where does Alice live?")
    objects = {f.object for f in found.facts if f.subject == "Alice"}
    assert objects == {"Mumbai", "London"}  # no invalidation: both values come back as current
    assert all(f.t_end is None for f in found.facts)
    assert found.context.startswith("## Retrieved notes")


def test_static_graph_has_no_time_and_keeps_contradictions() -> None:
    g = StaticGraphRAG()
    feed(g, (0, "Alice lives in Mumbai", "user_statement"), (30, "Alice moved to London", "user_statement"),
         (31, "Alice lives in London", "sensor"))
    found = g.search("Where does Alice live?")
    lives = [f for f in found.facts if f.predicate == "lives_in"]
    assert [f.object for f in lives] == ["Mumbai", "London"]  # deduplicated, insertion order, no timestamps
    assert all(f.t_start is None for f in lives)
    assert g.search("Who is Zed?").facts == []


def test_overwrite_memory_last_arrival_wins_even_when_wrong() -> None:
    m = OverwriteMemory()
    feed(m, (0, "Alice lives in Mumbai", "user_statement"), (30, "Alice moved to London", "sensor"),
         (35, "Alice lives in Paris", "hearsay"))
    assert [f.object for f in m.search("Where does Alice live?").facts] == ["Paris"]  # the rumor won
    # A late-arriving record about the past overwrites the present too.
    feed(m, (-200, "Alice lives in Pune", "system_of_record"))
    assert [f.object for f in m.search("Where does Alice live?").facts] == ["Pune"]
    assert m.overwrites == 3


def test_overwrite_memory_accumulates_multi_valued_predicates() -> None:
    m = OverwriteMemory()
    feed(m, (0, "Alice knows Bob", "user_statement"), (1, "Alice knows Carol", "user_statement"))
    assert {f.object for f in m.search("Who does Alice know?").facts} == {"Bob", "Carol"}


def test_full_history_returns_everything() -> None:
    m = FullHistoryMemory()
    feed(m, (0, "Alice lives in Mumbai", "user_statement"), (30, "Alice moved to London", "user_statement"))
    found = m.search("anything")
    assert len(found.facts) == 2 and "Observation log (2 entries" in found.context
