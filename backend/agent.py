"""Frozen-LLM agent harness.

The foundation model is never fine-tuned or edited: all knowledge-state change
lives in the temporal graph. At query time the harness

1. prunes the graph to a minimal, query-relevant, epistemically consistent
   subgraph,
2. attaches the hash-chained mutation records that justify each fact,
3. instructs the model to answer *only* from that context and to cite
   mutation IDs, and
4. verifies every citation in the answer against the ledger.

Clients
-------
``MockLLMClient``            – deterministic grounded extractive reader (offline, no keys)
``OllamaClient``             – native Ollama /api/chat or /api/generate for local SLMs (reports
                               model-tokenizer prompt counts and server-side timings)
``OpenAICompatibleClient``   – any ``/v1/chat/completions`` server: llama.cpp, OpenAI, LiteLLM
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import httpx

from models import Citation, QueryRequest, QueryResponse, TokenMetrics
from ontology import predicate_tokens
from pruning import SubgraphPruner, extract_as_of, tokenize
from reconciliation import AuditLedger
from tokens import count_tokens, tokenizer_name

SYSTEM_PROMPT = """You are a reasoning agent with a FROZEN knowledge base. Your only source of truth is the
verified temporal knowledge-graph context and mutation audit trail supplied in the user message.

Rules:
1. Answer strictly from the supplied facts. Do not use prior knowledge, do not guess.
2. Facts marked [INVALIDATED] are historical: they were true only inside their validity window.
   Never present them as current.
3. Every factual claim MUST end with the citation of the supporting mutation in the exact form [M:xxxxxxxx]
   using the 8-character IDs shown in the context.
4. If the context does not contain the answer, reply exactly: INSUFFICIENT_CONTEXT: <what is missing>.
5. Be concise: at most 4 sentences."""

# Wording measured on llama3.2-3B: no "[n]" placeholder (it gets copied literally) and the
# abstain clause last, so it reads as a fallback rather than an invitation.
SYSTEM_PROMPT_COMPACT = """Answer the question using only the numbered facts below. A fact marked ENDED is no longer true.
Reply in exactly this format:
ANSWER: <short answer>
EXPLANATION: <one sentence citing fact numbers like [1]>
If no fact is about the question, write ANSWER: UNKNOWN."""

# Prompt-local tags in compact style: "[2]", "[1, 3]", "[1][3]".
_ALIAS_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_COMPACT_FACT_RE = re.compile(
    r"^\[(?P<n>\d+)\] (?P<s>.+?) -(?P<p>[a-z0-9_]+)-> (?P<o>.+?) "
    r"\((?:since (?P<f>[\d-]+)|ENDED (?P<pt>[\d-]+), was true from (?P<pf>[\d-]+)(?:; replaced by (?P<r>.+?))?)\)$"
)
# Tolerant of small-model formatting drift: "[M:ab12cd34]", "M: AB12CD34", "(M:ab12cd34)".
_CITATION_RE = re.compile(r"M:\s*([0-9a-fA-F]{8})\b")
_FACT_RE = re.compile(
    r"^- (?P<s>.+?) -(?P<p>[a-z0-9_]+)-> (?P<o>.+?) \| (?P<c>[\d.]+) \| "
    r"(?P<f>[\dTZ:-]+)\.\.(?P<t>[\w:-]+)(?P<inv> \[INVALIDATED\])? \| M:(?P<m>[0-9a-f]{8})$"
)
_AUDIT_RE = re.compile(
    r"^- M:(?P<m>[0-9a-f]{8}) #(?P<seq>\d+) (?P<type>[A-Z]+) (?P<s>.+?) -(?P<p>[a-z0-9_]+)-> (?P<o>.+?): (?P<r>.*)$"
)
_QUESTION_RE = re.compile(r"^Question:\s*(?P<q>.+)$", re.MULTILINE)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "does", "do", "did", "what", "who", "which", "of", "in", "at",
    "to", "for", "and", "or", "now", "currently", "current", "s", "it", "its", "this", "that", "how",
})
_HISTORY_CUES = frozenset({
    "previous", "previously", "before", "used", "formerly", "former", "history", "earlier",
    "originally", "changed", "change", "why", "past", "prior", "provenance", "audit", "moved", "left",
    "when", "was", "were",
})
_VERBS: dict[str, tuple[str, str]] = {
    "lives_in": ("lives in", "lived in"),
    "works_at": ("works at", "worked at"),
    "has_role": ("holds the role", "held the role"),
    "reports_to": ("reports to", "reported to"),
    "married_to": ("is married to", "was married to"),
    "prefers": ("prefers", "preferred"),
    "headquartered_in": ("is headquartered in", "was headquartered in"),
    "located_in": ("is located in", "was located in"),
    "has_status": ("has status", "had status"),
    "owns": ("owns", "owned"),
    "knows": ("knows", "knew"),
    "member_of": ("is a member of", "was a member of"),
}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMCompletion:
    text: str
    model: str
    prompt_tokens: Optional[int] = None  # under the model's own tokenizer, when the server reports it
    completion_tokens: Optional[int] = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class LLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMCompletion: ...


# --------------------------------------------------------------------------- #
# Deterministic offline reader
# --------------------------------------------------------------------------- #


@dataclass
class _Fact:
    s: str
    p: str
    o: str
    conf: float
    t_from: str
    t_to: str
    current: bool
    cite: str
    tag: str = ""  # how to cite this fact in the answer; defaults to [M:cite]
    replaced_by: str = ""

    @property
    def citation(self) -> str:
        return self.tag or f"[M:{self.cite}]"


class MockLLMClient(LLMClient):
    """Deterministic extractive reader that only sees the prompt text.

    It parses the serialized facts exactly as an LLM would read them, ranks
    them lexically against the question, follows one hop of chaining for
    multi-hop questions, and emits cited sentences. It never consults the
    graph directly, so it is a faithful stand-in for a frozen model.
    """

    @property
    def model_name(self) -> str:
        return "mock-grounded-reader-v1"

    def complete(self, system: str, user: str) -> LLMCompletion:
        q_match = _QUESTION_RE.search(user)
        question = q_match.group("q").strip() if q_match else ""
        facts = self._parse_facts(user)
        audit = self._parse_audit(user)
        text = self._answer(question, facts, audit)
        if "ANSWER:" in system:  # honour the structured reply format when the prompt asks for it
            text = self._as_structured(text, facts, question)
        return LLMCompletion(text=text, model=self.model_name)

    def _as_structured(self, text: str, facts: list[_Fact], question: str) -> str:
        if text.startswith("INSUFFICIENT_CONTEXT"):
            return f"ANSWER: UNKNOWN\nEXPLANATION: {text.split(': ', 1)[-1]}"
        q = tokenize(question) - _STOPWORDS
        if tokenize(question) & {"before", "previously", "prior", "earlier", "formerly"}:
            past = [f for f in facts if not f.current and f.replaced_by and tokenize(f.replaced_by) & q]
            if past:
                return f"ANSWER: {past[0].o}\nEXPLANATION: {text}"
        # The answer is the cited current fact whose relation the question names; on ties the
        # last one in the reply wins (the end of a multi-hop chain).
        mentioned = [f for f in facts if f.current and f.citation in text]
        if not mentioned:
            return f"ANSWER: {text.split(' (')[0]}\nEXPLANATION: {text}"
        best = max(mentioned, key=lambda f: (bool(predicate_tokens(f.p) & q), text.index(f.citation)))
        return f"ANSWER: {best.o}\nEXPLANATION: {text}"

    @staticmethod
    def _parse_facts(prompt: str) -> list[_Fact]:
        facts: list[_Fact] = []
        for line in prompt.splitlines():
            line = line.strip()
            c = _COMPACT_FACT_RE.match(line)
            if c:
                is_current = c["f"] is not None
                facts.append(_Fact(
                    s=c["s"], p=c["p"], o=c["o"], conf=1.0,
                    t_from=c["f"] if is_current else c["pf"], t_to="now" if is_current else c["pt"],
                    current=is_current, cite=c["n"], tag=f"[{c['n']}]", replaced_by=c["r"] or "",
                ))
                continue
            m = _FACT_RE.match(line)
            if m:
                facts.append(_Fact(
                    s=m["s"], p=m["p"], o=m["o"], conf=float(m["c"]), t_from=m["f"], t_to=m["t"],
                    current=m["inv"] is None, cite=m["m"],
                ))
            elif line.startswith('{"as_of"'):
                payload = json.loads(line)
                for s, p, o, c, f, t, cur, cite in payload["facts"]:
                    facts.append(_Fact(s=s, p=p, o=o, conf=float(c), t_from=f, t_to=t, current=bool(cur), cite=cite))
        return facts

    @staticmethod
    def _parse_audit(prompt: str) -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        for line in prompt.splitlines():
            m = _AUDIT_RE.match(line.strip())
            if m:
                out[m["m"]] = (m["type"], m["r"])
        return out

    @staticmethod
    def _score(fact: _Fact, q: set[str]) -> float:
        score = 2.0 * len(tokenize(fact.s) & q)
        if predicate_tokens(fact.p) & q:
            score += 1.5
        score += 1.0 * len(tokenize(fact.o) & q)
        return score

    def _answer(self, question: str, facts: list[_Fact], audit: dict[str, tuple[str, str]]) -> str:
        q = tokenize(question) - _STOPWORDS
        if not facts or not q:
            return "INSUFFICIENT_CONTEXT: no verified facts were supplied for this question."
        historical = bool(tokenize(question) & _HISTORY_CUES)

        current = [f for f in facts if f.current]
        # When the question names a relation, only facts of that relation can be primary answers.
        if any(predicate_tokens(f.p) & q for f in current):
            current_primary = [f for f in current if predicate_tokens(f.p) & q]
        else:
            current_primary = current
        ranked = sorted(((self._score(f, q), f) for f in current_primary), key=lambda x: -x[0])
        best = ranked[0][0] if ranked else 0.0
        primary = [f for s, f in ranked if s >= max(1.5, 0.6 * best)][:3]

        chained: list[_Fact] = []
        for fact in primary:
            for nxt in current:
                if nxt.s == fact.o and nxt not in primary and nxt not in chained and self._score(nxt, q) > 0:
                    chained.append(nxt)

        if not primary:
            return (
                "INSUFFICIENT_CONTEXT: the verified knowledge graph contains no current fact "
                "matching this question."
            )

        sentences = []
        for fact in primary + chained:
            verb = _VERBS.get(fact.p, (fact.p.replace("_", " "), fact.p.replace("_", " ")))[0]
            conf = f", confidence {fact.conf:.2f}" if fact.conf < 1.0 else ""
            sentences.append(f"{fact.s} {verb} {fact.o} (valid since {fact.t_from}{conf}) {fact.citation}.")

        if historical:
            keys = {(f.s, f.p) for f in primary}
            past = sorted(
                (f for f in facts if not f.current and (f.s, f.p) in keys),
                key=lambda f: f.t_from,
            )
            for fact in past:
                verb = _VERBS.get(fact.p, (fact.p.replace("_", " "), fact.p.replace("_", " ")))[1]
                replaced = f", replaced by {fact.replaced_by}" if fact.replaced_by else ""
                sentences.append(
                    f"Previously, {fact.s} {verb} {fact.o} from {fact.t_from} until {fact.t_to}{replaced} {fact.citation}."
                )
                if fact.cite in audit and audit[fact.cite][0] == "INVALIDATE":
                    reason = audit[fact.cite][1].split(". ")[0].rstrip(".…")
                    sentences.append(f"Why it changed: {reason} [M:{fact.cite}].")
        return " ".join(sentences)


# --------------------------------------------------------------------------- #
# Local SLM connectors
# --------------------------------------------------------------------------- #

_NS_PER_MS = 1_000_000.0


class OllamaClient(LLMClient):
    """Native Ollama client for local small language models (Llama-3.2-3B, Qwen-2.5-3B, Phi-3.5, ...).

    Uses the native API rather than the OpenAI shim because it reports what an
    empirical comparison needs: ``prompt_eval_count`` (prompt tokens under the
    model's own tokenizer) and server-side load / prompt-eval / generation
    durations. ``num_ctx`` is always set explicitly: Ollama otherwise truncates
    long prompts silently to its default window, which would corrupt any
    comparison against a long append-only history.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        *,
        endpoint: str = "chat",
        num_ctx: int = 8192,
        num_predict: int = 256,
        temperature: float = 0.0,
        seed: int = 42,
        keep_alive: str = "15m",
        timeout_s: float = 300.0,
        num_gpu: Optional[int] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if endpoint not in {"chat", "generate"}:
            raise ValueError("endpoint must be 'chat' or 'generate'")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.seed = seed
        self.keep_alive = keep_alive
        self.num_gpu = num_gpu  # layers offloaded to GPU; 0 forces CPU-only inference
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout_s, transport=transport)

    @property
    def model_name(self) -> str:
        return f"ollama/{self.model}"

    def _options(self) -> dict:
        options = {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        return options

    def _post(self, path: str, body: dict) -> dict:
        try:
            resp = self._http.post(path, json=body)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url} ({exc}). Is it running? Start it with `ollama serve`."
            ) from exc
        if resp.status_code == 404:
            raise LLMError(f"Ollama model '{self.model}' not found. Pull it with: ollama pull {self.model}")
        if resp.status_code >= 400:
            raise LLMError(f"Ollama {path} returned {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise LLMError(f"Ollama {path} returned non-JSON output") from exc

    def ensure_model(self) -> None:
        """Fail fast with an actionable message if the server or model is missing."""
        try:
            tags = self._http.get("/api/tags").json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        models = tags.get("models", [])
        names = {m.get("name") for m in models} | {m.get("model") for m in models}
        if not names & {self.model, f"{self.model}:latest"}:
            available = ", ".join(sorted(n for n in names if n)) or "none"
            raise LLMError(f"Model '{self.model}' is not pulled (available: {available}). Run: ollama pull {self.model}")

    def warm_up(self) -> float:
        """Load the weights so measured latency excludes model loading. Returns load time in ms.

        Must use the same ``num_ctx`` as real calls: Ollama reloads the model
        whenever the context size changes, which would otherwise land a multi-
        second reload inside the first measured request.
        """
        started = time.perf_counter()
        self._post("/api/generate", {
            "model": self.model,
            "prompt": "Reply with OK.",
            "stream": False,
            "options": {**self._options(), "num_predict": 2},
            "keep_alive": self.keep_alive,
        })
        return (time.perf_counter() - started) * 1000.0

    def complete(self, system: str, user: str) -> LLMCompletion:
        if self.endpoint == "chat":
            data = self._post("/api/chat", {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": self._options(),
                "keep_alive": self.keep_alive,
            })
            text = (data.get("message") or {}).get("content", "")
        else:
            data = self._post("/api/generate", {
                "model": self.model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": self._options(),
                "keep_alive": self.keep_alive,
            })
            text = data.get("response", "")

        timings = {
            key: round(data[src] / _NS_PER_MS, 2)
            for key, src in (
                ("total", "total_duration"),
                ("load", "load_duration"),
                ("prompt_eval", "prompt_eval_duration"),
                ("generation", "eval_duration"),
            )
            if isinstance(data.get(src), (int, float))
        }
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        if completion_tokens and timings.get("generation"):
            timings["tokens_per_s"] = round(completion_tokens / (timings["generation"] / 1000.0), 2)
        if prompt_tokens is not None and prompt_tokens >= self.num_ctx - 1:
            timings["context_saturated"] = 1.0
        return LLMCompletion(
            text=text.strip(),
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timings_ms=timings,
        )


class OpenAICompatibleClient(LLMClient):
    """Any ``/v1/chat/completions`` server: llama.cpp ``llama-server``, OpenAI, LiteLLM, vLLM."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_s: float = 300.0,
        *,
        max_tokens: int = 256,
        seed: int = 42,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.seed = seed
        self._http = httpx.Client(timeout=timeout_s, transport=transport)

    @property
    def model_name(self) -> str:
        return self.model

    def complete(self, system: str, user: str) -> LLMCompletion:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        started = time.perf_counter()
        try:
            resp = self._http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM call to {self.base_url} failed: {exc}") from exc
        usage = data.get("usage") or {}
        timings = {"total": round((time.perf_counter() - started) * 1000.0, 2)}
        server = data.get("timings") or {}  # llama.cpp's llama-server reports its own timings
        if "prompt_ms" in server:
            timings["prompt_eval"] = round(server["prompt_ms"], 2)
        if "predicted_ms" in server:
            timings["generation"] = round(server["predicted_ms"], 2)
        if "predicted_per_second" in server:
            timings["tokens_per_s"] = round(server["predicted_per_second"], 2)
        return LLMCompletion(
            text=text,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            timings_ms=timings,
        )


def build_llm_client_from_env() -> LLMClient:
    """LLM_PROVIDER = mock | ollama | llamacpp | openai | litellm (default: mock)."""
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "ollama":
        return OllamaClient(
            model=os.getenv("LLM_MODEL", "llama3.2"),
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
            num_ctx=int(os.getenv("LLM_NUM_CTX", "8192")),
        )
    if provider == "llamacpp":
        return OpenAICompatibleClient(
            os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"), os.getenv("LLM_MODEL", "local-gguf")
        )
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAICompatibleClient(
            os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), os.getenv("LLM_MODEL", "gpt-4o-mini"), key
        )
    if provider == "litellm":
        return OpenAICompatibleClient(
            os.getenv("LLM_BASE_URL", "http://localhost:4000"),
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
            os.getenv("LITELLM_API_KEY"),
        )
    raise LLMError(f"Unknown LLM_PROVIDER {provider!r}")


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


@dataclass
class PromptPlan:
    """Everything sent to the model for one question, plus what is needed to verify the reply."""

    style: str
    system: str
    user: str
    context: str  # the fact block
    audit: str  # rationale trail ("" in compact style: it stays in the ledger)
    allowed: set[str]  # short mutation ids the model may cite
    aliases: dict[str, str]  # prompt tag -> short mutation id (compact style)
    ctx: object  # PrunedContext


class VerifiableAgent:
    def __init__(self, pruner: SubgraphPruner, ledger: AuditLedger, client: LLMClient) -> None:
        self.pruner = pruner
        self.ledger = ledger
        self.client = client

    @staticmethod
    def resolve_as_of(req: QueryRequest) -> QueryRequest:
        """Fill ``as_of`` from a date named in the question ("on March 3, 2025") when none was given."""
        if req.as_of is not None or not req.detect_as_of:
            return req
        detected = extract_as_of(req.question)
        return req.model_copy(update={"as_of": detected}) if detected else req

    def plan(self, req: QueryRequest) -> PromptPlan:
        req = self.resolve_as_of(req)
        if req.prompt_style == "standard":
            ctx, audit_text, audit_ids, user = self.build_prompt(req)
            return PromptPlan("standard", SYSTEM_PROMPT, user, ctx.text, audit_text,
                              set(ctx.citations) | set(audit_ids), {}, ctx)
        seeds = self.pruner.resolve_seeds(req.question, req.seed_entities)
        ctx = self.pruner.prune(req.question, req.pruning, seeds=seeds, as_of=req.as_of)
        as_of = req.as_of
        current = (lambda e: e.valid_at(as_of)) if as_of else None
        text, aliases, citations = self.pruner.serialize_compact(ctx.kept_edges, as_of, current)
        user = f"{text}\n\nQuestion: {req.question}"
        return PromptPlan("compact", SYSTEM_PROMPT_COMPACT, user, text, "", set(citations), aliases, ctx)

    def build_prompt(self, req: QueryRequest):  # type: ignore[no-untyped-def]
        req = self.resolve_as_of(req)
        seeds = self.pruner.resolve_seeds(req.question, req.seed_entities)
        ctx = self.pruner.prune(req.question, req.pruning, seeds=seeds, as_of=req.as_of)
        audit_text, audit_ids = self.pruner.serialize_audit_trail(
            ctx.kept_edges, ctx.include_history, req.question
        )
        user = f"{ctx.text}\n\n{audit_text}\n\nQuestion: {req.question}"
        return ctx, audit_text, audit_ids, user

    def answer(self, req: QueryRequest) -> QueryResponse:
        req = self.resolve_as_of(req)
        plan = self.plan(req)
        ctx, audit_text, user_prompt = plan.ctx, plan.audit, plan.user

        started = time.perf_counter()
        completion = self.client.complete(plan.system, user_prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0

        citations = self.verify_citations(completion.text, plan.allowed, plan.aliases)
        head = completion.text[:120]
        abstained = "INSUFFICIENT_CONTEXT" in head or "ANSWER: UNKNOWN" in head.upper()
        grounded = abstained or (bool(citations) and all(c.verified for c in citations))

        full_graph = self.pruner.serialize_full_graph(req.pruning.output_format)
        full_tokens = count_tokens(full_graph)
        pruned_tokens = count_tokens(plan.context)
        audit_tokens = count_tokens(audit_text)
        metrics = TokenMetrics(
            tokenizer=tokenizer_name(),
            full_graph_tokens=full_tokens,
            pruned_context_tokens=pruned_tokens,
            audit_trail_tokens=audit_tokens,
            prompt_tokens_total=count_tokens(plan.system) + count_tokens(user_prompt),
            reduction_ratio=round(1.0 - pruned_tokens / full_tokens, 4) if full_tokens else 0.0,
            full_graph_edges=len(self.pruner.graph.all_edges()),
            pruned_edges=len(ctx.kept_edges),
            edges_unscored=ctx.unscored,
            model_prompt_tokens=completion.prompt_tokens,
            model_completion_tokens=completion.completion_tokens,
        )
        return QueryResponse(
            question=req.question,
            answer=completion.text,
            model=completion.model,
            prompt_style=plan.style,
            latency_ms=round(latency_ms, 2),
            grounded=grounded,
            citations=citations,
            seed_entities=ctx.seeds,
            include_history=ctx.include_history,
            as_of=req.as_of,
            pruned_context=plan.context,
            audit_context=audit_text,
            system_prompt=plan.system,
            scored_edges=ctx.scored,
            metrics=metrics,
            model_timings_ms=completion.timings_ms,
        )

    def verify_citations(
        self, text: str, allowed: set[str], aliases: Optional[dict[str, str]] = None
    ) -> list[Citation]:
        """Resolve every citation in ``text`` against the ledger.

        A citation is verified only if it resolves to a real ledger record *and*
        that record was actually supplied in this prompt. Compact-style ``[n]``
        tags are translated through the prompt's alias map first.
        """
        found: list[tuple[str, Optional[str]]] = []  # (short id or unknown tag, alias)
        for short in (m.lower() for m in _CITATION_RE.findall(text)):
            if all(short != s for s, _ in found):
                found.append((short, None))
        if aliases is not None:
            for group in _ALIAS_RE.findall(text):
                for tag in (t.strip() for t in group.split(",")):
                    short = aliases.get(tag, f"[{tag}]")
                    if all(short != s for s, _ in found):
                        found.append((short, tag))
        out: list[Citation] = []
        for short, alias in found:
            record = self.ledger.resolve_short_id(short)
            out.append(Citation(
                short_id=short,
                alias=alias,
                verified=record is not None and short in allowed,
                mutation_id=record.mutation_id if record else None,
                mutation_type=record.mutation_type if record else None,
                rationale=record.deterministic_rationale if record else None,
            ))
        return out
