"""Head-to-head A/B evaluation of memory strategies with a real local SLM.

    python ab_eval.py                                   # llama3.2 via Ollama, all methods
    python ab_eval.py --model qwen2.5:3b --repeats 3    # another SLM, 3 timed repetitions
    python ab_eval.py --scale 4                         # 4x larger accumulated history
    python ab_eval.py --provider llamacpp --base-url http://localhost:8080/v1

Every method sees the *same* chronological observation stream and is queried
by the *same* frozen SLM with the *same* answer format. Only the memory that is
turned into prompt context differs:

A1  append-only history  every observation ever made, timestamped, oldest first
                         (the naive "keep the whole transcript" agent memory)
A2  vector retrieval     top-k observations by dense cosine similarity, no state
                         invalidation (the standard RAG memory)
B   VERITAS (standard)   the pruned, active, verified subgraph plus the full
                         rationale audit trail and 8-hex mutation-ID citations,
                         built by the real reconciliation engine and ledger
B2  VERITAS (compact)    the same pruned subgraph in the token-lean format:
                         short instructions, [n] citation tags mapped to ledger
                         IDs server-side, "replaced by" links, no rationale text

Baselines are told that facts change over time and to report the current state,
so the comparison measures the memory representation, not prompt sandbagging.

Outputs: a side-by-side console report and ``reports/ab_<model>_<timestamp>/``
containing ``report.json`` (every raw row, including prompts), ``report.md``,
and ``veritas_ledger.jsonl`` (the verified ledger that method B answered from).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from agent import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_COMPACT,
    LLMClient,
    LLMCompletion,
    LLMError,
    MockLLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    VerifiableAgent,
)
from benchmark import DenseVectorRAG, HashedDenseEncoder
from extraction import extract_triples
from graph_engine import TemporalGraphEngine
from models import PruningParams, QueryRequest, TripleObservation
from pruning import SubgraphPruner
from reconciliation import AuditLedger, ReconciliationEngine
from scenarios import T0, T1, ScriptedObservation, trajectory
from tokens import count_tokens, tokenizer_name

# --------------------------------------------------------------------------- #
# Query set with ground truth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ABQuery:
    question: str
    answer: str
    wrong: tuple[str, ...]  # values whose presence in the answer means stale/conflicting state
    category: str  # changed | multi-hop | historical | stable


AB_QUERIES: tuple[ABQuery, ...] = (
    ABQuery("Where is Acme currently headquartered?", "Munich", ("Berlin",), "changed"),
    ABQuery("Where does Alice live now?", "London", ("Mumbai", "Paris", "Pune"), "changed"),
    ABQuery("Which company does Alice work at?", "Acme", ("Infosys",), "changed"),
    ABQuery("What is Alice's current role?", "Staff Engineer", ("Software Engineer",), "changed"),
    ABQuery("What does Alice prefer?", "Coffee", ("Tea",), "changed"),
    ABQuery("Which city is Alice's employer headquartered in?", "Munich", ("Berlin", "Bengaluru"), "multi-hop"),
    ABQuery("Where did Alice live immediately before London?", "Mumbai", ("Pune", "Paris"), "historical"),
    # Held out: added after the compact format was designed, to guard against tuning to the test set.
    ABQuery("Where was Acme headquartered before Munich?", "Berlin", ("Munich", "Bengaluru"), "historical"),
    ABQuery("What was Alice's role before Staff Engineer?", "Software Engineer", ("Staff Engineer",), "historical"),
    ABQuery("Where does Bob live?", "London", (), "stable"),
    ABQuery("Who does Alice report to?", "Bob", (), "stable"),
)

FORMAT_INSTRUCTIONS = """Reply in exactly this format and nothing else:
ANSWER: <only the answer value, as short as possible>
EXPLANATION: <one or two sentences>"""

HISTORY_SYSTEM = f"""Memory mode: accumulated observation log.
You are an assistant agent. Your memory is the chronological log of observations below, oldest first,
each with the time it was recorded. The world changes over time, so observations may conflict;
when they do, work out which value is true now. Answer only from the memory.
If the memory does not contain the answer, reply ANSWER: UNKNOWN.

{FORMAT_INSTRUCTIONS}"""

RAG_SYSTEM = f"""Memory mode: retrieved notes.
You are an assistant agent. The notes below were retrieved from your long-term memory by similarity
to the question; each has the time it was recorded. The world changes over time, so notes may conflict;
when they do, work out which value is true now. Answer only from the notes.
If the notes do not contain the answer, reply ANSWER: UNKNOWN.

{FORMAT_INSTRUCTIONS}"""

VERITAS_SYSTEM = f"""{SYSTEM_PROMPT}

{FORMAT_INSTRUCTIONS}
Put the [M:xxxxxxxx] citations at the end of the EXPLANATION line."""


# --------------------------------------------------------------------------- #
# Memory methods
# --------------------------------------------------------------------------- #


@dataclass
class PromptBundle:
    system: str
    user: str
    context: str
    meta: dict = field(default_factory=dict)


class MemoryMethod(ABC):
    key: str
    label: str

    @abstractmethod
    def observe(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None: ...

    @abstractmethod
    def build(self, question: str) -> PromptBundle: ...

    def post_process(self, bundle: PromptBundle, text: str) -> dict:
        return {}


def _stamp(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


class AppendOnlyHistory(MemoryMethod):
    key, label = "A1", "append-only history"

    def __init__(self) -> None:
        self.lines: list[str] = []

    def observe(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None:
        self.lines.append(f"[{_stamp(obs.timestamp)}] {obs.text}.")

    def build(self, question: str) -> PromptBundle:
        context = f"## Observation log ({len(self.lines)} entries, oldest first)\n" + "\n".join(self.lines)
        return PromptBundle(HISTORY_SYSTEM, f"{context}\n\nQuestion: {question}", context)


class VectorRetrieval(MemoryMethod):
    key, label = "A2", "vector RAG (no invalidation)"

    def __init__(self, top_k: int = 6) -> None:
        self.rag = DenseVectorRAG(HashedDenseEncoder(), top_k=top_k)

    def observe(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None:
        self.rag.ingest(obs, triples[:1])  # one chunk per observation, like a raw memory store

    def build(self, question: str) -> PromptBundle:
        hits = self.rag.retrieve(question)
        context = f"## Retrieved notes (top {len(hits)} by similarity)\n" + "\n".join(f"- {c.text}" for _, c in hits)
        return PromptBundle(RAG_SYSTEM, f"{context}\n\nQuestion: {question}", context, {"retrieved": len(hits)})


class VeritasMemory(MemoryMethod):
    key, label = "B", "VERITAS (standard prompt)"

    def __init__(self, ledger_path: Path, client: LLMClient) -> None:
        self.graph = TemporalGraphEngine()
        self.ledger = AuditLedger(ledger_path)
        self.reconciler = ReconciliationEngine(self.graph, self.ledger)
        self.pruner = SubgraphPruner(self.graph, self.ledger)
        self.agent = VerifiableAgent(self.pruner, self.ledger, client)
        self.decisions: dict[str, int] = {}

    def observe(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None:
        for triple in triples:
            outcome = self.reconciler.ingest(triple)
            self.decisions[outcome.decision] = self.decisions.get(outcome.decision, 0) + 1

    style = "standard"

    def build(self, question: str) -> PromptBundle:
        plan = self.agent.plan(QueryRequest(question=question, pruning=PruningParams(), prompt_style=self.style))
        system = VERITAS_SYSTEM if self.style == "standard" else plan.system
        context = f"{plan.context}\n\n{plan.audit}" if plan.audit else plan.context
        return PromptBundle(
            system, plan.user, context,
            {
                "allowed_citations": sorted(plan.allowed),
                "aliases": plan.aliases,
                "pruned_edges": len(plan.ctx.kept_edges),
                "include_history": plan.ctx.include_history,
            },
        )

    def post_process(self, bundle: PromptBundle, text: str) -> dict:
        aliases = bundle.meta["aliases"] if self.style == "compact" else None
        citations = self.agent.verify_citations(text, set(bundle.meta["allowed_citations"]), aliases)
        return {
            "citations": [{"id": c.short_id, "tag": c.alias, "verified": c.verified, "type": c.mutation_type}
                          for c in citations],
            "citations_verified": bool(citations) and all(c.verified for c in citations),
        }


class VeritasCompact(VeritasMemory):
    """Same graph, ledger and pruning as method B (shared, not re-ingested); only the prompt format differs."""

    key, label = "B2", "VERITAS (compact prompt)"
    style = "compact"

    def __init__(self, base: VeritasMemory) -> None:  # noqa: D107 - shares base's pipeline
        self.graph, self.ledger, self.reconciler = base.graph, base.ledger, base.reconciler
        self.pruner, self.agent, self.decisions = base.pruner, base.agent, base.decisions

    def observe(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None:
        """No-op: method B already ingested this observation into the shared graph."""


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

_ANSWER_RE = re.compile(r"^[\s*_#>-]*answer[\s*_]*:[\s*_]*(?P<a>.+?)[\s*_]*$", re.IGNORECASE | re.MULTILINE)


def extract_answer(text: str) -> tuple[str, bool]:
    """Return (answer value, format_ok). Falls back to the first non-empty line."""
    match = _ANSWER_RE.search(text)
    if match:
        return match.group("a").strip(), True
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first, False


def mentions(text: str, value: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(value) + r"(?![A-Za-z])", text, re.IGNORECASE) is not None


@dataclass
class Grade:
    answer: str
    format_ok: bool
    correct: bool
    stale_values: list[str]


def grade(query: ABQuery, text: str) -> Grade:
    answer, format_ok = extract_answer(text)
    stale = [v for v in query.wrong if mentions(answer, v)]
    return Grade(answer, format_ok, mentions(answer, query.answer) and not stale, stale)


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #

_CITIES = ["Paris", "Tokyo", "Toronto", "Sydney", "Madrid", "Seoul", "Chicago", "Dublin", "Lisbon", "Austin"]
_ORGS = ["Globex", "Initech", "Umbrella", "Hooli", "Soylent", "Vandelay"]


def scaled_trajectory(scale: int, seed: int = 11) -> list[ScriptedObservation]:
    """The demo trajectory plus ``(scale - 1) x 24`` synthetic agents with their own drift.

    Extra agents move and change jobs at T1, so the accumulated history grows
    with both new facts *and* superseded ones, as a long-horizon agent's would.
    """
    base = trajectory()
    if scale <= 1:
        return base
    rng = random.Random(seed)
    extra: list[ScriptedObservation] = []
    for i in range((scale - 1) * 24):
        name = f"Agent{i:03d}"
        ts0 = T0 + timedelta(hours=2, minutes=i)
        extra.append(ScriptedObservation("T0", ts0, f"{name} lives in {rng.choice(_CITIES)}"))
        extra.append(ScriptedObservation("T0", ts0, f"{name} works at {rng.choice(_ORGS)}"))
        if i % 2 == 0:
            ts1 = T1 - timedelta(days=rng.randint(1, 30), minutes=i)
            extra.append(ScriptedObservation("T1", ts1, f"{name} moved to {rng.choice(_CITIES)}"))
        if i % 3 == 0:
            ts1 = T1 - timedelta(days=rng.randint(1, 30), minutes=i)
            extra.append(ScriptedObservation("T1", ts1, f"{name} works at {rng.choice(_ORGS)}"))
    order = {"T0": 0, "T1": 1, "T2": 2}
    return sorted(base + extra, key=lambda o: (order[o.phase], o.timestamp))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    repeat: int
    query: str
    category: str
    expected: str
    method: str
    method_label: str
    raw_output: str
    answer: str
    format_ok: bool
    correct: bool
    stale_values: list[str]
    model_prompt_tokens: Optional[int]
    est_prompt_tokens: int
    context_tokens_est: int
    completion_tokens: Optional[int]
    wall_ms: float
    timings_ms: dict
    extra: dict
    system_prompt: str = ""
    user_prompt: str = ""


def _pctl(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def summarize(rows: list[Row], methods: list[MemoryMethod]) -> dict:
    out: dict = {}
    for m in methods:
        mine = [r for r in rows if r.method == m.key]
        if not mine:
            continue
        model_tokens = [r.model_prompt_tokens for r in mine if r.model_prompt_tokens is not None]
        prompt_tokens = model_tokens if len(model_tokens) == len(mine) else [r.est_prompt_tokens for r in mine]
        walls = [r.wall_ms for r in mine]
        gen = [r.timings_ms["generation"] for r in mine if "generation" in r.timings_ms]
        pev = [r.timings_ms["prompt_eval"] for r in mine if "prompt_eval" in r.timings_ms]
        tps = [r.timings_ms["tokens_per_s"] for r in mine if "tokens_per_s" in r.timings_ms]
        by_cat: dict[str, float] = {}
        for cat in sorted({r.category for r in mine}):
            cat_rows = [r for r in mine if r.category == cat]
            by_cat[cat] = sum(r.correct for r in cat_rows) / len(cat_rows)
        summary = {
            "label": m.label,
            "n": len(mine),
            "accuracy": sum(r.correct for r in mine) / len(mine),
            "accuracy_by_category": by_cat,
            "stale_answer_rate": sum(bool(r.stale_values) for r in mine) / len(mine),
            "format_compliance": sum(r.format_ok for r in mine) / len(mine),
            "prompt_tokens_source": "model" if prompt_tokens is model_tokens else "estimate",
            "prompt_tokens_mean": statistics.fmean(prompt_tokens),
            "context_tokens_est_mean": statistics.fmean(r.context_tokens_est for r in mine),
            "latency_ms_median": statistics.median(walls),
            "latency_ms_p95": _pctl(walls, 0.95),
            "latency_ms_mean": statistics.fmean(walls),
            "prompt_eval_ms_mean": statistics.fmean(pev) if pev else None,
            "generation_ms_mean": statistics.fmean(gen) if gen else None,
            "tokens_per_s_mean": statistics.fmean(tps) if tps else None,
            "context_saturated": sum(1 for r in mine if r.timings_ms.get("context_saturated")),
            "generation_errors": sum(1 for r in mine if r.timings_ms.get("generation_error")),
        }
        if m.key in ("B", "B2"):
            summary["citation_verified_rate"] = sum(bool(r.extra.get("citations_verified")) for r in mine) / len(mine)
        out[m.key] = summary
    return out


_GENERATION_FAILURE_MARKERS = ("repeat limit", "prediction aborted", "context length", "exceeds")
# The inference runner died (e.g. a CUDA fault); Ollama restarts it on the next request.
_RUNNER_CRASH_MARKERS = ("cuda error", "terminated", "runner process", "connection reset",
                         "server disconnected", "remoteprotocolerror", "cannot reach")


def _is_generation_failure(exc: LLMError) -> bool:
    """Server-side aborts of a single generation, as opposed to infrastructure errors."""
    text = str(exc).lower()
    return any(marker in text for marker in _GENERATION_FAILURE_MARKERS)


def _is_runner_crash(exc: LLMError) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RUNNER_CRASH_MARKERS)


def required_context(methods: Sequence["MemoryMethod"], queries: Sequence[ABQuery], max_tokens: int) -> int:
    """Smallest 1024-aligned window that fits every prompt (heuristic counts over-estimate ~20%).

    A tight window keeps the KV cache small so more layers fit on a small GPU.
    """
    longest = max(
        count_tokens(b.system) + count_tokens(b.user)
        for b in (m.build(q.question) for m in methods for q in queries)
    )
    needed = int(longest * 1.15) + max_tokens + 64
    return max(2048, -(-needed // 1024) * 1024)


class ABRunner:
    def __init__(self, client: LLMClient, methods: list[MemoryMethod], queries: Sequence[ABQuery],
                 repeats: int = 1, verbose: bool = False, *, bust_prompt_cache: bool = True,
                 crash_retries: int = 2) -> None:
        self.client = client
        self.methods = methods
        self.queries = queries
        self.repeats = repeats
        self.verbose = verbose
        self.bust_prompt_cache = bust_prompt_cache
        self.crash_retries = crash_retries

    def _call(self, bundle: PromptBundle) -> tuple[LLMCompletion, int]:
        """One generation with cold-cache measurement and bounded retries on runner crashes.

        A unique request id at the very start of the system prompt makes the
        token prefix differ from every earlier request, so the server's prompt
        cache cannot reuse KV entries: every call pays (and reports) its full
        prompt cost, which is what an apples-to-apples comparison needs.
        """
        system = f"[request {uuid.uuid4().hex[:8]}]\n{bundle.system}" if self.bust_prompt_cache else bundle.system
        retries = 0
        while True:
            try:
                return self.client.complete(system, bundle.user), retries
            except LLMError as exc:
                if _is_runner_crash(exc) and retries < self.crash_retries:
                    retries += 1
                    print(f"\n  inference runner crashed ({str(exc)[:90]}…); retry {retries}/{self.crash_retries}")
                    time.sleep(3.0)
                    continue
                raise

    def ingest(self, observations: list[ScriptedObservation],
               methods: Optional[Sequence[MemoryMethod]] = None) -> None:
        targets = list(methods if methods is not None else self.methods)
        for obs in observations:
            triples = extract_triples(obs.text, confidence=obs.confidence, source=obs.source, timestamp=obs.timestamp)
            for method in targets:
                method.observe(obs, triples)

    def warm_up(self) -> None:
        if isinstance(self.client, OllamaClient):
            self.client.ensure_model()
            ms = self.client.warm_up()
            print(f"  model loaded in {ms / 1000:.1f}s")
        # One discarded call per method so GPU kernels / buffers for each prompt size are initialised.
        for method in self.methods:
            bundle = method.build("Summarize your memory in one sentence.")
            try:
                self._call(bundle)
            except LLMError as exc:  # a degenerate warm-up generation is irrelevant to the measurement
                print(f"  warm-up call for {method.key} failed ({exc}); continuing")

    def run(
        self,
        on_row: Optional[Callable[[Row, int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        quiet: bool = False,
    ) -> list[Row]:
        """Execute every (repeat, query, method) call. ``on_row(row, done, total)`` streams progress;
        ``should_stop()`` is polled between calls for cooperative cancellation."""
        rows: list[Row] = []
        total = self.repeats * len(self.queries) * len(self.methods)
        done = 0
        for rep in range(self.repeats):
            for qi, query in enumerate(self.queries):
                # Rotate method order per query so no method systematically benefits from a
                # warm prompt cache or suffers from thermal throttling.
                shift = (qi + rep) % len(self.methods)
                for method in self.methods[shift:] + self.methods[:shift]:
                    bundle = method.build(query.question)
                    started = time.perf_counter()
                    retries = 0
                    try:
                        completion, retries = self._call(bundle)
                    except LLMError as exc:
                        if not _is_generation_failure(exc):
                            raise  # server down / model missing / repeated crashes: abort the run
                        # The model degenerated (e.g. a repetition loop) — a genuine failed answer.
                        completion = LLMCompletion(text=f"<generation failed: {exc}>", model=self.client.model_name,
                                                   timings_ms={"generation_error": 1.0})
                    wall_ms = (time.perf_counter() - started) * 1000.0
                    g = grade(query, completion.text)
                    extra = {**bundle.meta, **method.post_process(bundle, completion.text)}
                    extra.pop("allowed_citations", None)
                    extra.pop("aliases", None)
                    if retries:
                        extra["runner_crash_retries"] = retries
                    rows.append(Row(
                        repeat=rep, query=query.question, category=query.category, expected=query.answer,
                        method=method.key, method_label=method.label, raw_output=completion.text,
                        answer=g.answer, format_ok=g.format_ok, correct=g.correct, stale_values=g.stale_values,
                        model_prompt_tokens=completion.prompt_tokens,
                        est_prompt_tokens=count_tokens(bundle.system) + count_tokens(bundle.user),
                        context_tokens_est=count_tokens(bundle.context),
                        completion_tokens=completion.completion_tokens,
                        wall_ms=round(wall_ms, 2), timings_ms=completion.timings_ms, extra=extra,
                        system_prompt=bundle.system, user_prompt=bundle.user,
                    ))
                    done += 1
                    if on_row is not None:
                        on_row(rows[-1], done, total)
                    if should_stop is not None and should_stop():
                        return rows
                    if quiet:
                        continue
                    if self.verbose:
                        mark = "✓" if g.correct else "✗"
                        print(f"  [{done:3d}/{total}] {method.key:<2} {mark} {query.question[:48]:<48} "
                              f"→ {g.answer[:40]!r} ({wall_ms:,.0f} ms)")
                    else:
                        print(f"\r  running {done}/{total}", end="", flush=True)
        if not self.verbose and not quiet:
            print()
        return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt_tokens(r: Row) -> str:
    return f"{r.model_prompt_tokens:,}" if r.model_prompt_tokens is not None else f"~{r.est_prompt_tokens:,}"


def print_side_by_side(rows: list[Row], methods: list[MemoryMethod], queries: Sequence[ABQuery]) -> None:
    first = [r for r in rows if r.repeat == 0]
    print("\n" + "=" * 108)
    print("PER-QUERY RESULTS (repeat 0)")
    print("=" * 108)
    for q in queries:
        wrong = f"   wrong if: {', '.join(q.wrong)}" if q.wrong else ""
        print(f"[{q.category}] {q.question}\n  expected: {q.answer}{wrong}")
        for m in methods:
            r = next((x for x in first if x.query == q.question and x.method == m.key), None)
            if r is None:
                continue
            mark = "✓" if r.correct else ("✗ STALE" if r.stale_values else "✗")
            cite = ""
            if m.key in ("B", "B2"):
                cite = " cites✓" if r.extra.get("citations_verified") else " cites✗"
            answer = r.answer if len(r.answer) <= 34 else r.answer[:31] + "..."
            print(f"  {m.key:<2} {m.label:<32} {mark:<8} {answer!r:<38}{cite:<8} "
                  f"{_fmt_tokens(r):>7} tok {r.wall_ms:>8,.0f} ms")
        print()


def print_summary(summary: dict, methods: list[MemoryMethod]) -> None:
    keys = [m.key for m in methods if m.key in summary]
    width = 22

    def row(label: str, values: list[str]) -> None:
        print(f"{label:<34}" + "".join(f"{v:>{width}}" for v in values))

    print("=" * (34 + width * len(keys)))
    row("metric", [f"{k} {summary[k]['label'][:width - 4]}" for k in keys])
    print("-" * (34 + width * len(keys)))
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    row("accuracy (all)", [pct(summary[k]["accuracy"]) for k in keys])
    cats = sorted({c for k in keys for c in summary[k]["accuracy_by_category"]})
    for cat in cats:
        row(f"  accuracy · {cat}", [pct(summary[k]["accuracy_by_category"].get(cat, float("nan"))) for k in keys])
    row("stale / conflicting answer", [pct(summary[k]["stale_answer_rate"]) for k in keys])
    row("answer-format compliance", [pct(summary[k]["format_compliance"]) for k in keys])
    row("degenerate generations", [str(summary[k]["generation_errors"]) for k in keys])
    if "B" in summary or "B2" in summary:
        row("verified citations", [pct(summary[k]["citation_verified_rate"]) if "citation_verified_rate" in summary[k]
                                   else "n/a" for k in keys])
    row("prompt tokens (mean)", [f"{summary[k]['prompt_tokens_mean']:,.0f} ({summary[k]['prompt_tokens_source'][:5]})" for k in keys])
    row("memory-context tokens (est.)", [f"{summary[k]['context_tokens_est_mean']:,.0f}" for k in keys])
    row("latency median (ms)", [f"{summary[k]['latency_ms_median']:,.0f}" for k in keys])
    row("latency p95 (ms)", [f"{summary[k]['latency_ms_p95']:,.0f}" for k in keys])
    row("prompt-eval mean (ms)", [f"{summary[k]['prompt_eval_ms_mean']:,.0f}" if summary[k]["prompt_eval_ms_mean"] is not None else "n/a" for k in keys])
    row("generation mean (ms)", [f"{summary[k]['generation_ms_mean']:,.0f}" if summary[k]["generation_ms_mean"] is not None else "n/a" for k in keys])
    row("decode speed (tok/s)", [f"{summary[k]['tokens_per_s_mean']:.1f}" if summary[k]["tokens_per_s_mean"] else "n/a" for k in keys])
    print("-" * (34 + width * len(keys)))
    for base, challenger in (("A1", "B"), ("A1", "B2"), ("B", "B2")):
        if base in summary and challenger in summary:
            a, b = summary[base], summary[challenger]
            print(f"{challenger} vs {base}: {100 * (1 - b['prompt_tokens_mean'] / a['prompt_tokens_mean']):.1f}% fewer "
                  f"prompt tokens, {a['latency_ms_median'] / b['latency_ms_median']:.2f}x median latency speed-up, "
                  f"accuracy {100 * a['accuracy']:.0f}% → {100 * b['accuracy']:.0f}%")
    saturated = {k: summary[k]["context_saturated"] for k in keys if summary[k]["context_saturated"]}
    if saturated:
        print(f"WARNING: prompts hit num_ctx and were truncated by the server: {saturated} — raise --num-ctx")
    print("=" * (34 + width * len(keys)))


def write_markdown(path: Path, config: dict, summary: dict, rows: list[Row], methods: list[MemoryMethod],
                   queries: Sequence[ABQuery]) -> None:
    keys = [m.key for m in methods if m.key in summary]
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        f"# VERITAS A/B evaluation — {config['model']}",
        "",
        f"- Run: {config['started_at']} · repeats: {config['repeats']} · history scale: {config['scale']} "
        f"({config['observations']} observations) · num_ctx: {config.get('num_ctx')}",
        f"- Ledger for method B: `{config['ledger_path']}` — {config['ledger_verification']}",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(f"{k} · {summary[k]['label']}" for k in keys) + " |",
        "|---|" + "---|" * len(keys),
        "| Accuracy | " + " | ".join(pct(summary[k]["accuracy"]) for k in keys) + " |",
    ]
    for cat in sorted({c for k in keys for c in summary[k]["accuracy_by_category"]}):
        lines.append(f"| Accuracy · {cat} | " + " | ".join(
            pct(summary[k]["accuracy_by_category"].get(cat, float("nan"))) for k in keys) + " |")
    lines += [
        "| Stale / conflicting answers | " + " | ".join(pct(summary[k]["stale_answer_rate"]) for k in keys) + " |",
        "| Prompt tokens (mean) | " + " | ".join(f"{summary[k]['prompt_tokens_mean']:,.0f}" for k in keys) + " |",
        "| Latency median (ms) | " + " | ".join(f"{summary[k]['latency_ms_median']:,.0f}" for k in keys) + " |",
        "| Latency p95 (ms) | " + " | ".join(f"{summary[k]['latency_ms_p95']:,.0f}" for k in keys) + " |",
        "",
        "## Per-query answers (repeat 0)",
        "",
        "| Category | Question | Expected | " + " | ".join(keys) + " |",
        "|---|---|---|" + "---|" * len(keys),
    ]
    for q in queries:
        cells = []
        for k in keys:
            r = next((x for x in rows if x.repeat == 0 and x.query == q.question and x.method == k), None)
            if r:
                cells.append(f"{'✅' if r.correct else '❌'} {r.answer[:40].replace('|', '/')} · {_fmt_tokens(r)} tok · {r.wall_ms:,.0f} ms")
            else:
                cells.append("–")
        lines.append(f"| {q.category} | {q.question} | {q.answer} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_client(args: argparse.Namespace) -> LLMClient:
    if args.provider == "mock":
        return MockLLMClient()
    if args.provider == "llamacpp":
        return OpenAICompatibleClient(args.base_url or "http://localhost:8080/v1", args.model,
                                      max_tokens=args.max_tokens)
    return OllamaClient(args.model, args.base_url or "http://localhost:11434", endpoint=args.endpoint,
                        num_ctx=args.num_ctx or 8192, num_predict=args.max_tokens, num_gpu=args.num_gpu)


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["ollama", "llamacpp", "mock"], default="ollama")
    p.add_argument("--model", default="llama3.2", help="e.g. llama3.2, qwen2.5:3b, phi3.5")
    p.add_argument("--base-url", default=None)
    p.add_argument("--endpoint", choices=["chat", "generate"], default="chat", help="Ollama API endpoint")
    p.add_argument("--methods", default="A1,A2,B,B2", help="comma-separated subset of A1,A2,B,B2")
    p.add_argument("--repeats", type=int, default=1, help="timed repetitions of the full query set")
    p.add_argument("--scale", type=int, default=1, help="multiply the background history (1 = demo trajectory)")
    p.add_argument("--top-k", type=int, default=6, help="retrieval depth for A2")
    p.add_argument("--num-ctx", type=int, default=0, help="Ollama context window; 0 = smallest that fits every prompt")
    p.add_argument("--num-gpu", type=int, default=None, help="layers to offload to GPU (0 = CPU only)")
    p.add_argument("--allow-prompt-cache", action="store_true",
                   help="let the server reuse cached prompt prefixes (default: measure every call cold)")
    p.add_argument("--retries", type=int, default=2, help="retries after an inference-runner crash")
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "reports")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    started_at = datetime.now(timezone.utc)
    run_dir = args.out / f"ab_{re.sub(r'[^A-Za-z0-9.]+', '-', args.model)}_{started_at:%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args)
    veritas_base = VeritasMemory(run_dir / "veritas_ledger.jsonl", client)
    available: dict[str, MemoryMethod] = {
        "A1": AppendOnlyHistory(),
        "A2": VectorRetrieval(args.top_k),
        "B": veritas_base,
        "B2": VeritasCompact(veritas_base),
    }
    wanted = [k.strip().upper() for k in args.methods.split(",") if k.strip()]
    unknown = [k for k in wanted if k not in available]
    if unknown:
        p.error(f"unknown methods: {unknown}")
    methods = [available[k] for k in wanted]
    if "B2" in wanted and "B" not in wanted:
        # B2 reads the graph that B's ingestion builds; ingest through B without querying it.
        methods_for_ingest = [*methods, veritas_base]
    else:
        methods_for_ingest = methods

    observations = scaled_trajectory(args.scale)
    print("=" * 108)
    print(f"VERITAS A/B evaluation · model={client.model_name} · methods={','.join(wanted)} · "
          f"repeats={args.repeats} · observations={len(observations)}")
    print("=" * 108)
    runner = ABRunner(client, methods, AB_QUERIES, repeats=args.repeats, verbose=args.verbose,
                      bust_prompt_cache=not args.allow_prompt_cache, crash_retries=args.retries)
    runner.ingest(observations, methods_for_ingest)
    if isinstance(client, OllamaClient):
        if not args.num_ctx:
            client.num_ctx = required_context(methods, AB_QUERIES, args.max_tokens)
        print(f"  num_ctx={client.num_ctx} · num_gpu={'auto' if args.num_gpu is None else args.num_gpu} · "
              f"prompt cache {'allowed' if args.allow_prompt_cache else 'disabled (cold measurement)'}")

    veritas = veritas_base
    verification = veritas.ledger.verify()
    if "B" in wanted or "B2" in wanted:
        print(f"  VERITAS ledger: {verification.detail} · decisions {veritas.decisions}")

    try:
        if not args.no_warmup:
            print("  warming up…")
            runner.warm_up()
        rows = runner.run()
    except LLMError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    summary = summarize(rows, methods)
    print_side_by_side(rows, methods, AB_QUERIES)
    print_summary(summary, methods)

    config = {
        "model": client.model_name,
        "provider": args.provider,
        "endpoint": args.endpoint,
        "num_ctx": client.num_ctx if isinstance(client, OllamaClient) else None,
        "num_gpu": args.num_gpu,
        "prompt_cache": "allowed" if args.allow_prompt_cache else "disabled",
        "max_tokens": args.max_tokens,
        "methods": wanted,
        "repeats": args.repeats,
        "scale": args.scale,
        "observations": len(observations),
        "top_k": args.top_k,
        "tokenizer_estimate": tokenizer_name(),
        "started_at": started_at.isoformat(),
        "ledger_path": str(run_dir / "veritas_ledger.jsonl"),
        "ledger_verification": verification.detail,
        "ledger_head_hash": verification.head_hash,
        "reconciliation_decisions": veritas.decisions,
    }
    report = {"config": config, "summary": summary, "rows": [asdict(r) for r in rows]}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown(run_dir / "report.md", config, summary, rows, methods, AB_QUERIES)
    print(f"Reports written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
