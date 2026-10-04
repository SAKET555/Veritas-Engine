from __future__ import annotations

import pytest

from agent import (
    LLMClient,
    LLMCompletion,
    LLMError,
    MockLLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    VerifiableAgent,
    build_llm_client_from_env,
)
from models import QueryRequest
from tests.conftest import Stack, obs


class ScriptedClient(LLMClient):
    """Returns a fixed answer, to test citation verification independently of the reader."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.last_user_prompt = ""

    @property
    def model_name(self) -> str:
        return "scripted"

    def complete(self, system: str, user: str) -> LLMCompletion:
        self.last_user_prompt = user
        return LLMCompletion(self.text, self.model_name)


@pytest.fixture
def world(stack: Stack) -> Stack:
    stack.ingest(
        obs("Alice", "lives_in", "Mumbai", day=0),
        obs("Alice", "lives_in", "London", day=50),
        obs("Alice", "works_at", "Acme", day=10),
        obs("Acme", "headquartered_in", "Munich", day=0, source="system_of_record"),
    )
    return stack


def agent_with(stack: Stack, client: LLMClient) -> VerifiableAgent:
    return VerifiableAgent(stack.pruner, stack.ledger, client)


def test_mock_answers_current_fact_with_verified_citation(world: Stack) -> None:
    resp = agent_with(world, MockLLMClient()).answer(QueryRequest(question="Where does Alice live?"))
    assert "London" in resp.answer and "Mumbai" not in resp.answer
    assert resp.grounded
    assert resp.citations and all(c.verified for c in resp.citations)
    assert resp.metrics.pruned_context_tokens < resp.metrics.full_graph_tokens
    assert 0 < resp.metrics.reduction_ratio < 1


def test_mock_historical_answer_cites_invalidation(world: Stack) -> None:
    resp = agent_with(world, MockLLMClient()).answer(
        QueryRequest(question="Where did Alice live before London and why did it change?")
    )
    assert "Previously, Alice lived in Mumbai" in resp.answer
    assert any(c.mutation_type == "INVALIDATE" and c.verified for c in resp.citations)


def test_mock_follows_multi_hop_chain(world: Stack) -> None:
    resp = agent_with(world, MockLLMClient()).answer(
        QueryRequest(question="Which city is Alice's employer headquartered in?")
    )
    assert "Munich" in resp.answer


def test_mock_abstains_without_context(stack: Stack) -> None:
    standard = agent_with(stack, MockLLMClient()).answer(
        QueryRequest(question="Where does Zed live?", prompt_style="standard")
    )
    assert standard.answer.startswith("INSUFFICIENT_CONTEXT")
    assert standard.grounded  # abstention is grounded behaviour
    compact = agent_with(stack, MockLLMClient()).answer(QueryRequest(question="Where does Zed live?", prompt_style="compact"))
    assert compact.answer.startswith("ANSWER: UNKNOWN")
    assert compact.grounded


def test_standard_prompt_contains_context_audit_and_question(world: Stack) -> None:
    client = ScriptedClient("INSUFFICIENT_CONTEXT: test")
    agent_with(world, client).answer(QueryRequest(question="Where does Alice live?", prompt_style="standard"))
    prompt = client.last_user_prompt
    assert "## KG facts" in prompt
    assert "## Mutation audit trail" in prompt
    assert prompt.rstrip().endswith("Question: Where does Alice live?")


def test_compact_prompt_is_smaller_and_omits_rationale(world: Stack) -> None:
    client = ScriptedClient("ANSWER: UNKNOWN")
    agent = agent_with(world, client)
    agent.answer(QueryRequest(question="Where did Alice live before London?", prompt_style="standard"))
    standard = client.last_user_prompt
    resp = agent.answer(QueryRequest(question="Where did Alice live before London?", prompt_style="compact"))
    compact = client.last_user_prompt
    assert resp.prompt_style == "compact"
    assert len(compact) < len(standard) / 2
    assert "Mutation audit trail" not in compact and "M:" not in compact
    assert "[1] Alice -lives_in-> London (since" in compact
    assert "Alice -lives_in-> Mumbai (ENDED" in compact and "replaced by London" in compact
    assert compact.startswith("Facts:\n")  # neutral header (see pruning.serialize_compact)


def test_compact_tags_resolve_to_ledger_records(world: Stack) -> None:
    agent = agent_with(world, ScriptedClient("ANSWER: London\nEXPLANATION: see [1] and [1, 9]."))
    resp = agent.answer(QueryRequest(question="Where does Alice live?", prompt_style="compact"))
    good = next(c for c in resp.citations if c.alias == "1")
    assert good.verified and good.mutation_id and world.ledger.resolve_short_id(good.short_id)
    bogus = next(c for c in resp.citations if c.alias == "9")
    assert not bogus.verified
    assert not resp.grounded  # one unverifiable tag makes the answer ungrounded


def test_fabricated_citation_is_flagged(world: Stack) -> None:
    resp = agent_with(world, ScriptedClient("Alice lives in Paris [M:deadbeef].")).answer(
        QueryRequest(question="Where does Alice live?")
    )
    assert not resp.grounded
    assert resp.citations[0].short_id == "deadbeef" and not resp.citations[0].verified


def test_real_but_out_of_context_citation_is_flagged(world: Stack) -> None:
    # A genuine ledger record that was NOT supplied in this prompt must not count as grounding.
    acme_record = world.ledger.for_entity("munich")[0]
    resp = agent_with(world, ScriptedClient(f"Alice lives in London [M:{acme_record.short_id}].")).answer(
        QueryRequest(question="Where does Alice live?")
    )
    assert resp.citations[0].mutation_id == acme_record.mutation_id
    assert not resp.citations[0].verified
    assert not resp.grounded


def test_uncited_answer_is_ungrounded(world: Stack) -> None:
    resp = agent_with(world, ScriptedClient("Alice lives in London.")).answer(
        QueryRequest(question="Where does Alice live?")
    )
    assert not resp.grounded and resp.citations == []


def test_client_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(build_llm_client_from_env(), MockLLMClient)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = build_llm_client_from_env()
    assert isinstance(client, OllamaClient) and client.base_url.endswith("11434") and client.model == "llama3.2"
    monkeypatch.setenv("LLM_PROVIDER", "llamacpp")
    client = build_llm_client_from_env()
    assert isinstance(client, OpenAICompatibleClient) and client.base_url.endswith("8080/v1")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        build_llm_client_from_env()
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(LLMError):
        build_llm_client_from_env()


def test_http_client_wraps_transport_errors() -> None:
    client = OpenAICompatibleClient("http://127.0.0.1:9", "m", timeout_s=0.5)
    with pytest.raises(LLMError):
        client.complete("s", "u")
