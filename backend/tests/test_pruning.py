from __future__ import annotations

import json

import pytest

from models import PruningParams
from tests.conftest import Stack, at, obs


@pytest.fixture
def world(stack: Stack) -> Stack:
    stack.ingest(
        obs("Alice", "lives_in", "Mumbai", day=0),
        obs("Alice", "works_at", "Infosys", day=0),
        obs("Acme", "headquartered_in", "Berlin", day=0, source="system_of_record"),
        obs("Alice", "knows", "Bob", day=0),
        obs("Alice", "lives_in", "London", day=50),
        obs("Alice", "works_at", "Acme", day=50),
    )
    # Distractors that share the London hub with Alice.
    for i, name in enumerate(["Dana", "Elif", "Farah", "Goran", "Hiro"]):
        stack.ingest(obs(name, "lives_in", "London", day=i), obs(name, "works_at", "Globex", day=i))
    return stack


def kept_triples(ctx) -> set[tuple[str, str, str]]:  # type: ignore[no-untyped-def]
    return {(e.source_id, e.relation, e.target_id) for e in ctx.kept_edges}


def test_seed_resolution_handles_possessives_and_longest_match(world: Stack) -> None:
    world.ingest(obs("Alice Smith", "lives_in", "Oslo"))
    assert world.pruner.resolve_seeds("What is Alice's job?") == ["alice"]
    assert world.pruner.resolve_seeds("Where does Alice Smith live?") == ["alice-smith"]
    assert world.pruner.resolve_seeds("q", explicit=["Acme"]) == ["acme"]


def test_historical_intent_detection(world: Stack) -> None:
    assert world.pruner.detect_historical_intent("Where did Alice live before?")
    assert not world.pruner.detect_historical_intent("Where does Alice live?")


def test_current_query_drops_stale_and_hub_distractors(world: Stack) -> None:
    ctx = world.pruner.prune("Where does Alice live?", PruningParams())
    assert ("alice", "lives_in", "london") in kept_triples(ctx)
    assert ("alice", "lives_in", "mumbai") not in kept_triples(ctx)
    assert not any(s in {"dana", "elif", "farah", "goran", "hiro"} for s, _, _ in kept_triples(ctx))
    assert not ctx.include_history
    stale = next(s for s in ctx.scored if s.target == "Mumbai")
    assert not stale.kept and "inactive" in stale.reason


def test_historical_query_keeps_invalidated_facts(world: Stack) -> None:
    ctx = world.pruner.prune("Where did Alice live before London?", PruningParams())
    assert ctx.include_history
    assert {("alice", "lives_in", "mumbai"), ("alice", "lives_in", "london")} <= kept_triples(ctx)
    assert "[INVALIDATED]" in ctx.text


def test_explicit_history_flag_overrides_detection(world: Stack) -> None:
    ctx = world.pruner.prune("Where did Alice live before?", PruningParams(include_history=False))
    assert ("alice", "lives_in", "mumbai") not in kept_triples(ctx)


def test_multi_hop_chain_is_retained(world: Stack) -> None:
    ctx = world.pruner.prune("Which city is Alice's employer headquartered in?", PruningParams())
    assert {("alice", "works_at", "acme"), ("acme", "headquartered_in", "berlin")} <= kept_triples(ctx)


def test_k_zero_limits_to_seed_only_edges(world: Stack) -> None:
    ctx = world.pruner.prune("Which city is Alice's employer headquartered in?", PruningParams(k_hops=0))
    assert ctx.kept_edges == []


def test_threshold_and_max_edges(world: Stack) -> None:
    assert world.pruner.prune("Tell me about Alice", PruningParams(tau=1.0)).kept_edges == []
    ctx = world.pruner.prune("Tell me about Alice", PruningParams(tau=0.0, max_edges=2))
    assert len(ctx.kept_edges) == 2
    assert any("max_edges" in s.reason for s in ctx.scored)


def test_relation_agnostic_question_keeps_seed_facts(world: Stack) -> None:
    ctx = world.pruner.prune("Tell me about Alice", PruningParams())
    assert {("alice", "lives_in", "london"), ("alice", "works_at", "acme"), ("alice", "knows", "bob")} <= kept_triples(ctx)


def test_point_in_time_pruning(world: Stack) -> None:
    ctx = world.pruner.prune("Where does Alice live?", PruningParams(), as_of=at(10))
    assert ("alice", "lives_in", "mumbai") in kept_triples(ctx)
    assert ("alice", "lives_in", "london") not in kept_triples(ctx)
    assert "now" in ctx.text and "[INVALIDATED]" not in ctx.text


def test_scores_are_normalized_and_weight_sum_validated(world: Stack) -> None:
    ctx = world.pruner.prune("Where does Alice live?", PruningParams())
    assert all(0.0 <= s.score <= 1.0 for s in ctx.scored)
    with pytest.raises(ValueError):
        world.pruner.prune("x", PruningParams(alpha=0, beta=0, gamma=0))


def test_serialization_formats_and_citations(world: Stack) -> None:
    md = world.pruner.prune("Where does Alice live?", PruningParams())
    assert md.text.startswith("## KG facts")
    for short_id, edge_id in md.citations.items():
        assert f"M:{short_id}" in md.text
        assert world.ledger.latest_for_edge(edge_id).short_id == short_id

    js = world.pruner.prune("Where does Alice live?", PruningParams(output_format="json"))
    payload = json.loads(js.text)
    assert payload["cols"][0] == "s"
    assert ["Alice", "lives_in", "London"] == payload["facts"][0][:3]


def test_pruned_context_is_much_smaller_than_full_graph(world: Stack) -> None:
    from tokens import count_tokens

    ctx = world.pruner.prune("Where does Alice live?", PruningParams())
    assert count_tokens(ctx.text) < 0.4 * count_tokens(world.pruner.serialize_full_graph())


def test_audit_trail_focuses_on_asked_relation(world: Stack) -> None:
    ctx = world.pruner.prune("Where did Alice live before London?", PruningParams())
    text, ids = world.pruner.serialize_audit_trail(ctx.kept_edges, True, "Where did Alice live before London?")
    assert "INVALIDATE" in text
    assert "works_at" not in text
    assert all(f"M:{i}" in text for i in ids)
