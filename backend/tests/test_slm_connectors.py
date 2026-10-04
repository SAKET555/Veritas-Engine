"""Connector tests against in-process fake servers (httpx.MockTransport) — no Ollama needed."""

from __future__ import annotations

import json

import httpx
import pytest

from agent import LLMError, OllamaClient, OpenAICompatibleClient, VerifiableAgent
from models import QueryRequest
from tests.conftest import Stack, obs


def ollama_transport(captured: list, *, status: int = 200, payload: dict | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        captured.append((request.method, request.url.path, body))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest", "model": "llama3.2:latest"}]})
        if status != 200:
            return httpx.Response(status, json=payload or {"error": "boom"})
        data = payload or {
            "message": {"role": "assistant", "content": "  ANSWER: Munich  "},
            "response": "ANSWER: Munich",
            "prompt_eval_count": 412,
            "eval_count": 20,
            "total_duration": 1_500_000_000,
            "load_duration": 10_000_000,
            "prompt_eval_duration": 400_000_000,
            "eval_duration": 1_000_000_000,
        }
        return httpx.Response(200, json=data)

    return httpx.MockTransport(handler)


def test_ollama_chat_request_shape_and_metrics() -> None:
    captured: list = []
    client = OllamaClient("llama3.2", num_ctx=3072, num_predict=99, num_gpu=0, transport=ollama_transport(captured))
    out = client.complete("SYS", "USER")

    method, path, body = captured[-1]
    assert (method, path) == ("POST", "/api/chat")
    assert body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 3072, "num_predict": 99, "num_gpu": 0}

    assert out.text == "ANSWER: Munich"
    assert out.model == "ollama/llama3.2"
    assert (out.prompt_tokens, out.completion_tokens) == (412, 20)
    assert out.timings_ms["total"] == 1500.0
    assert out.timings_ms["prompt_eval"] == 400.0
    assert out.timings_ms["generation"] == 1000.0
    assert out.timings_ms["tokens_per_s"] == 20.0
    assert "context_saturated" not in out.timings_ms


def test_ollama_generate_endpoint_and_gpu_default() -> None:
    captured: list = []
    client = OllamaClient("llama3.2", endpoint="generate", transport=ollama_transport(captured))
    assert client.complete("SYS", "USER").text == "ANSWER: Munich"
    _, path, body = captured[-1]
    assert path == "/api/generate"
    assert (body["system"], body["prompt"]) == ("SYS", "USER")
    assert "num_gpu" not in body["options"]  # let Ollama decide unless told otherwise


def test_ollama_flags_context_saturation() -> None:
    payload = {"message": {"content": "x"}, "prompt_eval_count": 2048, "eval_count": 1}
    client = OllamaClient("llama3.2", num_ctx=2048, transport=ollama_transport([], payload=payload))
    assert client.complete("s", "u").timings_ms["context_saturated"] == 1.0


def test_ollama_missing_model_gives_pull_hint() -> None:
    client = OllamaClient("qwen2.5:3b", transport=ollama_transport([], status=404))
    with pytest.raises(LLMError, match="ollama pull qwen2.5:3b"):
        client.complete("s", "u")
    with pytest.raises(LLMError, match="not pulled"):
        client.ensure_model()


def test_ollama_server_error_and_unreachable() -> None:
    client = OllamaClient("llama3.2", transport=ollama_transport([], status=500, payload={"error": "token repeat limit"}))
    with pytest.raises(LLMError, match="500"):
        client.complete("s", "u")

    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="ollama serve"):
        OllamaClient("llama3.2", transport=httpx.MockTransport(refuse)).complete("s", "u")


def test_ollama_warm_up_uses_same_context_size() -> None:
    captured: list = []
    client = OllamaClient("llama3.2", num_ctx=4096, transport=ollama_transport(captured))
    client.ensure_model()
    client.warm_up()
    _, path, body = captured[-1]
    assert path == "/api/generate"
    assert body["options"]["num_ctx"] == 4096  # a different num_ctx would force a reload on the first real call


def test_openai_compatible_parses_usage_and_llamacpp_timings() -> None:
    captured: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "choices": [{"message": {"content": "ANSWER: Munich"}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 12},
            "timings": {"prompt_ms": 120.5, "predicted_ms": 300.25, "predicted_per_second": 40.0},
        })

    client = OpenAICompatibleClient("http://localhost:8080/v1", "local", max_tokens=64,
                                    transport=httpx.MockTransport(handler))
    out = client.complete("s", "u")
    assert captured[-1]["max_tokens"] == 64 and captured[-1]["temperature"] == 0
    assert (out.prompt_tokens, out.completion_tokens) == (300, 12)
    assert out.timings_ms["prompt_eval"] == 120.5 and out.timings_ms["tokens_per_s"] == 40.0


def test_agent_surfaces_model_metrics_and_tolerates_citation_drift(stack: Stack) -> None:
    stack.ingest(obs("Acme", "headquartered_in", "Berlin", day=0), obs("Acme", "headquartered_in", "Munich", day=10))
    agent = VerifiableAgent(stack.pruner, stack.ledger, OllamaClient("llama3.2", transport=ollama_transport([])))
    ctx, _, audit_ids, _ = agent.build_prompt(QueryRequest(question="Where is Acme headquartered?"))
    cite = next(iter(ctx.citations))

    payload = {"message": {"content": f"ANSWER: Munich\nEXPLANATION: per M: {cite.upper()}"},
               "prompt_eval_count": 333, "eval_count": 9}
    agent.client = OllamaClient("llama3.2", transport=ollama_transport([], payload=payload))
    resp = agent.answer(QueryRequest(question="Where is Acme headquartered?"))
    assert resp.metrics.model_prompt_tokens == 333
    assert resp.metrics.model_completion_tokens == 9
    assert resp.grounded and resp.citations[0].short_id == cite and resp.citations[0].verified
