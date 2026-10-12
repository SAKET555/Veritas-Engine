"""Offline temporal-memory benchmark: VERITAS vs. five baselines on synthetic multi-session data.

    python temporal_bench.py                            # 40 people, 6 sessions, 60 questions
    python temporal_bench.py --people 120 --sessions 8  # a larger world
    python temporal_bench.py --scaling 25,100,400,1000  # latency / token scaling across graph sizes
    python temporal_bench.py --json out.json --markdown out.md
    python temporal_bench.py --export-dataset ds.json   # the generated observations + questions

What it measures (proposal §5: epistemic accuracy, provenance verifiability,
token efficiency, latency overhead)
-------------------------------------------------------------------------
Every method ingests the same observation stream (``datasets.generate``) and is
asked the same questions. Each method's retrieval is read by the *same*
deterministic reader, which is deliberately generous to the baselines:

* current value    the newest fact for (subject, predicate) — newest by the
                   fact's validity time when the method knows it, otherwise
                   the last one listed. For VERITAS, the open interval.
* value on a date  the newest fact that had started by that date (VERITAS:
                   the interval containing it).
* previous value   the value listed immediately before the current one in
                   time order.
* multi-hop        current employer, then that employer's current HQ.

So a baseline is never penalized for a weak reader: it loses only when its
*memory* hands the reader the wrong facts (stale, rumored, missing). No LLM is
involved, so the run is deterministic and needs no GPU. Use ``ab_eval.py
--dataset synthetic`` to put the same data in front of a real model.

Ground truth comes from the simulated world in ``datasets.py``, not from
VERITAS's reconciliation rules.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from agent import MockLLMClient, VerifiableAgent
from baselines import (
    DenseVectorRAG,
    Fact,
    FullHistoryMemory,
    HashedDenseEncoder,
    HybridRAG,
    OverwriteMemory,
    Retrieval,
    StaticGraphRAG,
)
from datasets import CATEGORIES, Dataset, TemporalQuestion, generate
from extraction import extract_triples
from graph_engine import TemporalGraphEngine
from models import PruningParams, QueryRequest, TripleObservation
from ontology import get_spec
from pruning import SubgraphPruner, extract_as_of
from reconciliation import AuditLedger, ReconciliationEngine
from scenarios import ScriptedObservation
from tokens import count_tokens, tokenizer_name

# --------------------------------------------------------------------------- #
# VERITAS behind the common retrieval interface
# --------------------------------------------------------------------------- #


class VeritasRetriever:
    """The real VERITAS pipeline (reconciliation → hash-chained ledger → pruning) as a retriever."""

    label = "VERITAS"

    def __init__(self, ledger_path: Path, *, fsync: bool = True, pruning: Optional[PruningParams] = None) -> None:
        self.pruning = pruning or PruningParams()
        self.graph = TemporalGraphEngine()
        self.ledger = AuditLedger(ledger_path, fsync=fsync)
        self.reconciler = ReconciliationEngine(self.graph, self.ledger)
        self.pruner = SubgraphPruner(self.graph, self.ledger)
        self.agent = VerifiableAgent(self.pruner, self.ledger, MockLLMClient())
        self.decisions: dict[str, int] = {}

    def ingest(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None:
        for triple in triples:
            outcome = self.reconciler.ingest(triple)
            self.decisions[outcome.decision] = self.decisions.get(outcome.decision, 0) + 1

    def _name(self, entity_id: str) -> str:
        entity = self.graph.get_entity(entity_id)
        return entity.name if entity else entity_id

    def search(self, question: str, as_of: Optional[datetime] = None) -> Retrieval:
        req = QueryRequest(question=question, as_of=as_of, pruning=self.pruning, detect_as_of=True)
        plan = self.agent.plan(req)
        when = req.as_of or extract_as_of(question)
        facts = []
        for rank, edge in enumerate(plan.ctx.kept_edges):  # type: ignore[attr-defined]
            current = edge.valid_at(when) if when else edge.is_active
            record = self.ledger.latest_for_edge(edge.id)
            facts.append(Fact(self._name(edge.source_id), edge.relation, self._name(edge.target_id),
                              edge.t_start, edge.t_end, edge.source, record.short_id if record else "",
                              rank, current))
        context = f"{plan.context}\n\n{plan.audit}" if plan.audit else plan.context
        return Retrieval(context, facts, {"as_of": when.isoformat() if when else None})

    def compact_context(self, question: str) -> str:
        return self.agent.plan(QueryRequest(question=question, prompt_style="compact", pruning=self.pruning,
                                            detect_as_of=True)).context

    def verifies(self, fact: Fact) -> bool:
        return bool(fact.ref) and self.ledger.resolve_short_id(fact.ref) is not None


OPTIMIZED_PRUNING = PruningParams(candidate_mode="ego", hub_damping=0.5)


class VeritasView(VeritasRetriever):
    """Another pruning configuration over the *same* graph and ledger (ingests nothing itself)."""

    def __init__(self, base: VeritasRetriever, pruning: PruningParams) -> None:  # noqa: D107 - shares base
        self.pruning = pruning
        self.graph, self.ledger, self.reconciler = base.graph, base.ledger, base.reconciler
        self.pruner, self.agent, self.decisions = base.pruner, base.agent, base.decisions

    def ingest(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None:
        """No-op: the base retriever already reconciled this observation into the shared graph."""


# --------------------------------------------------------------------------- #
# The shared deterministic reader
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _candidates(facts: Sequence[Fact], subject: str, predicate: str) -> list[Fact]:
    key = _norm(subject)
    return [f for f in facts if _norm(f.subject) == key and f.predicate == predicate]


def _value_at(cands: list[Fact], at: Optional[datetime]) -> Optional[Fact]:
    if not cands:
        return None
    timed = [f for f in cands if f.t_start is not None]
    if not timed:  # a memory with no notion of time: the last one listed
        return max(cands, key=lambda f: f.rank)
    if at is None:
        pool = [f for f in timed if f.t_end is None and f.presented_as_current] or [f for f in timed if f.t_end is None]
    else:
        pool = [f for f in timed if f.t_start <= at and (f.t_end is None or at < f.t_end)]
    if not pool:
        return None
    return max(pool, key=lambda f: (f.t_start, f.rank))


def _previous(cands: list[Fact], before: str) -> Optional[Fact]:
    ordered = sorted(cands, key=lambda f: (f.t_start, f.rank)) if all(f.t_start for f in cands) else \
        sorted(cands, key=lambda f: f.rank)
    sequence: list[Fact] = []
    for f in ordered:
        if not sequence or _norm(sequence[-1].object) != _norm(f.object):
            sequence.append(f)
    for i in range(len(sequence) - 1, 0, -1):
        if _norm(sequence[i].object) == _norm(before):
            return sequence[i - 1]
    return None


def read(facts: Sequence[Fact], q: TemporalQuestion) -> tuple[Optional[Fact], list[Fact]]:
    """Answer ``q`` from ``facts``. Returns (answering fact, every fact the answer depends on)."""
    first = _candidates(facts, q.subject, q.path[0])
    if q.kind == "previous":
        f = _previous(first, q.before or "")
        return f, [f] if f else []
    hop = _value_at(first, q.as_of if q.kind == "as_of" else None)
    if hop is None or len(q.path) == 1:
        return hop, [hop] if hop else []
    second = _value_at(_candidates(facts, hop.object, q.path[1]), None)
    return second, [hop, second] if second else [hop]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

METHOD_LABELS = {
    "full_history": "Full history",
    "vector_rag": "Vector RAG",
    "hybrid_rag": "Hybrid BM25+dense",
    "static_graph_rag": "Static graph RAG",
    "overwrite_memory": "Overwrite memory",
    "veritas": "VERITAS",
    "veritas_optimized": "VERITAS optimized",
}


@dataclass
class Outcome:
    method: str
    question: str
    category: str
    expected: str
    answer: Optional[str]
    correct: bool
    stale: bool
    evidence: bool
    contradiction: bool
    verifiable: bool
    context_tokens: int
    latency_ms: float
    extra: dict = field(default_factory=dict)


def build_methods(workdir: Path, *, top_k: int = 6, fsync: bool = True) -> dict[str, object]:
    encoder = HashedDenseEncoder()
    veritas = VeritasRetriever(workdir / "veritas_ledger.jsonl", fsync=fsync)
    return {
        "full_history": FullHistoryMemory(),
        "vector_rag": DenseVectorRAG(encoder, top_k=top_k),
        "hybrid_rag": HybridRAG(encoder, top_k=top_k),
        "static_graph_rag": StaticGraphRAG(),
        "overwrite_memory": OverwriteMemory(),
        "veritas": veritas,
        "veritas_optimized": VeritasView(veritas, OPTIMIZED_PRUNING),
    }


def _contradiction(facts: Sequence[Fact]) -> bool:
    groups: dict[tuple[str, str], set[str]] = {}
    for f in facts:
        if f.presented_as_current and get_spec(f.predicate).functional:
            groups.setdefault((_norm(f.subject), f.predicate), set()).add(_norm(f.object))
    return any(len(v) > 1 for v in groups.values())


def _pctl(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def evaluate(dataset: Dataset, *, top_k: int = 6, fsync: bool = True, workdir: Optional[Path] = None,
             methods: Optional[Sequence[str]] = None) -> dict:
    """Ingest the dataset into every method, ask every question and aggregate the metrics."""
    workdir = Path(workdir or tempfile.mkdtemp(prefix="veritas-tbench-"))
    memories = build_methods(workdir, top_k=top_k, fsync=fsync)
    if methods:
        memories = {k: v for k, v in memories.items() if k in methods}

    extracted = [(obs, extract_triples(obs.text, confidence=obs.confidence, source=obs.source,
                                       timestamp=obs.timestamp)) for obs in dataset.observations]
    ingest_ms: dict[str, list[float]] = {k: [] for k in memories}
    for obs, triples in extracted:
        for key, memory in memories.items():
            started = time.perf_counter()
            memory.ingest(obs, triples)  # type: ignore[attr-defined]
            ingest_ms[key].append((time.perf_counter() - started) * 1000.0)

    veritas = memories.get("veritas")
    chain = veritas.ledger.verify() if isinstance(veritas, VeritasRetriever) else None
    outcomes: list[Outcome] = []
    for q in dataset.questions:
        for key, memory in memories.items():
            started = time.perf_counter()
            retrieval: Retrieval = memory.search(q.question)  # type: ignore[attr-defined]
            latency = (time.perf_counter() - started) * 1000.0
            fact, support = read(retrieval.facts, q)
            answer = fact.object if fact else None
            values = {_norm(f.object) for f in retrieval.facts}
            verifiable = (isinstance(memory, VeritasRetriever) and bool(chain and chain.valid)
                          and bool(support) and fact is not None and all(memory.verifies(f) for f in support))
            extra = {}
            if isinstance(memory, VeritasRetriever):
                extra["compact_context_tokens"] = count_tokens(memory.compact_context(q.question))
            outcomes.append(Outcome(
                method=key, question=q.question, category=q.category, expected=q.answer, answer=answer,
                correct=answer is not None and _norm(answer) == _norm(q.answer),
                stale=answer is not None and any(_norm(answer) == _norm(w) for w in q.wrong),
                evidence=_norm(q.answer) in values,
                contradiction=_contradiction(retrieval.facts),
                verifiable=verifiable,
                context_tokens=count_tokens(retrieval.context),
                latency_ms=latency,
                extra=extra,
            ))

    for key, memory in memories.items():
        if isinstance(memory, VeritasView) and "veritas" in ingest_ms:
            ingest_ms[key] = ingest_ms["veritas"]  # shares the base graph: same ingestion cost
    summary: dict[str, dict] = {}
    for key in memories:
        mine = [o for o in outcomes if o.method == key]
        n = len(mine) or 1
        by_cat = {}
        for cat in CATEGORIES:
            rows = [o for o in mine if o.category == cat]
            if rows:
                by_cat[cat] = sum(o.correct for o in rows) / len(rows)
        entry = {
            "label": METHOD_LABELS[key],
            "n": len(mine),
            "accuracy": sum(o.correct for o in mine) / n,
            "accuracy_by_category": by_cat,
            "stale_answer_rate": sum(o.stale for o in mine) / n,
            "abstain_rate": sum(o.answer is None for o in mine) / n,
            "evidence_recall": sum(o.evidence for o in mine) / n,
            "contradictory_context_rate": sum(o.contradiction for o in mine) / n,
            "verifiable_rate": sum(o.verifiable for o in mine) / n,
            "context_tokens_mean": statistics.fmean(o.context_tokens for o in mine) if mine else 0.0,
            "context_tokens_p95": _pctl([o.context_tokens for o in mine], 0.95),
            "query_ms_mean": statistics.fmean(o.latency_ms for o in mine) if mine else 0.0,
            "query_ms_p95": _pctl([o.latency_ms for o in mine], 0.95),
            "ingest_ms_per_observation_mean": statistics.fmean(ingest_ms[key]) if ingest_ms[key] else 0.0,
            "ingest_ms_per_observation_p95": _pctl(ingest_ms[key], 0.95),
        }
        if key.startswith("veritas"):
            entry["compact_context_tokens_mean"] = statistics.fmean(o.extra["compact_context_tokens"] for o in mine)
        summary[key] = entry

    graph_info: dict = {}
    if isinstance(veritas, VeritasRetriever):
        started = time.perf_counter()
        verification = veritas.ledger.verify()
        verify_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        full_graph_tokens = count_tokens(veritas.pruner.serialize_full_graph())
        graph_info = {
            **veritas.graph.stats(),
            "ledger_records": len(veritas.ledger),
            "ledger_valid": verification.valid,
            "ledger_verify_ms": round(verify_ms, 2),
            "full_graph_tokens": full_graph_tokens,
            "decisions": veritas.decisions,
        }
    return {
        "dataset": {**dataset.config, **dataset.stats()},
        "tokenizer": tokenizer_name(),
        "fsync": fsync,
        "summary": summary,
        "graph": graph_info,
        "outcomes": [o.__dict__ for o in outcomes],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def print_report(report: dict) -> None:
    summary = report["summary"]
    keys = list(summary)
    width = 18
    line = "=" * (30 + width * len(keys))
    ds = report["dataset"]
    print(line)
    print(f"VERITAS temporal benchmark · {ds['people']} people · {ds['orgs']} orgs · {ds['sessions']} sessions · "
          f"{ds['observations']} observations · {ds['questions']} questions · seed {ds['seed']}")
    print(f"sources {ds['by_source']} · tokenizer {report['tokenizer']} · ledger fsync {report['fsync']}")
    print(line)

    def row(label: str, cells: list[str]) -> None:
        print(f"{label:<30}" + "".join(f"{c:>{width}}" for c in cells))

    row("metric", [summary[k]["label"] for k in keys])
    print("-" * len(line))
    row("accuracy", [_pct(summary[k]["accuracy"]) for k in keys])
    for cat in CATEGORIES:
        if any(cat in summary[k]["accuracy_by_category"] for k in keys):
            row(f"  · {cat}", [_pct(summary[k]["accuracy_by_category"].get(cat, float("nan"))) for k in keys])
    row("stale / wrong value answered", [_pct(summary[k]["stale_answer_rate"]) for k in keys])
    row("no answer found", [_pct(summary[k]["abstain_rate"]) for k in keys])
    row("answer present in context", [_pct(summary[k]["evidence_recall"]) for k in keys])
    row("contradictory context", [_pct(summary[k]["contradictory_context_rate"]) for k in keys])
    row("verifiable (ledger-cited)", [_pct(summary[k]["verifiable_rate"]) for k in keys])
    row("context tokens (mean)", [f"{summary[k]['context_tokens_mean']:,.0f}" for k in keys])
    row("context tokens (p95)", [f"{summary[k]['context_tokens_p95']:,.0f}" for k in keys])
    row("query latency mean (ms)", [f"{summary[k]['query_ms_mean']:,.2f}" for k in keys])
    row("ingest per obs mean (ms)", [f"{summary[k]['ingest_ms_per_observation_mean']:,.3f}" for k in keys])
    row("ingest per obs p95 (ms)", [f"{summary[k]['ingest_ms_per_observation_p95']:,.3f}" for k in keys])
    print("-" * len(line))
    g = report.get("graph") or {}
    if g:
        v = summary["veritas"]
        print(f"VERITAS graph: {g['entities']} entities, {g['edges_total']} facts ({g['edges_active']} active), "
              f"{g['ledger_records']} ledger records — chain {'VALID' if g['ledger_valid'] else 'BROKEN'} "
              f"(verified in {g['ledger_verify_ms']:,.1f} ms)")
        print(f"VERITAS context: {v['context_tokens_mean']:,.0f} tokens standard / "
              f"{v['compact_context_tokens_mean']:,.0f} compact vs {g['full_graph_tokens']:,} for the unpruned graph "
              f"({100 * (1 - v['context_tokens_mean'] / max(1, g['full_graph_tokens'])):.1f}% reduction)")
        o = summary.get("veritas_optimized")
        if o:
            print(f"VERITAS optimized (candidate_mode=ego, hub_damping=0.5): {o['context_tokens_mean']:,.0f} tokens, "
                  f"{o['query_ms_mean']:,.2f} ms per query vs {v['query_ms_mean']:,.2f} ms with the default settings")
        print(f"Reconciliation decisions: {g['decisions']}")
    print(line)


def write_markdown(path: Path, report: dict) -> None:
    summary = report["summary"]
    keys = list(summary)
    ds = report["dataset"]
    lines = [
        "# VERITAS temporal benchmark",
        "",
        f"{ds['people']} people, {ds['orgs']} organizations, {ds['sessions']} sessions, {ds['observations']} "
        f"observations, {ds['questions']} questions (seed {ds['seed']}, drift {ds['drift']}, rumor rate "
        f"{ds['rumor_rate']}, late-arrival rate {ds['late_rate']}). Tokenizer: {report['tokenizer']}.",
        "",
        "| Metric | " + " | ".join(summary[k]["label"] for k in keys) + " |",
        "|---|" + "---|" * len(keys),
        "| Accuracy | " + " | ".join(_pct(summary[k]["accuracy"]) for k in keys) + " |",
    ]
    for cat in CATEGORIES:
        if any(cat in summary[k]["accuracy_by_category"] for k in keys):
            lines.append(f"| · {cat} | " + " | ".join(
                _pct(summary[k]["accuracy_by_category"].get(cat, float("nan"))) for k in keys) + " |")
    for label, key, fmt in (
        ("Stale / wrong value answered", "stale_answer_rate", _pct),
        ("Answer present in context", "evidence_recall", _pct),
        ("Contradictory context", "contradictory_context_rate", _pct),
        ("Verifiable (ledger-cited)", "verifiable_rate", _pct),
        ("Context tokens (mean)", "context_tokens_mean", lambda x: f"{x:,.0f}"),
        ("Query latency mean (ms)", "query_ms_mean", lambda x: f"{x:,.2f}"),
        ("Ingest per observation (ms)", "ingest_ms_per_observation_mean", lambda x: f"{x:,.3f}"),
    ):
        lines.append(f"| {label} | " + " | ".join(fmt(summary[k][key]) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


def scaling(sizes: Sequence[int], *, sessions: int, questions: int, seed: int, fsync: bool, top_k: int) -> dict:
    """Run the benchmark at several world sizes and tabulate latency and token growth."""
    points = []
    for people in sizes:
        dataset = generate(people=people, sessions=sessions, max_questions=questions, seed=seed)
        print(f"  · {people} people: {len(dataset.observations)} observations …", flush=True)
        report = evaluate(dataset, top_k=top_k, fsync=fsync)
        points.append({"people": people, "observations": len(dataset.observations), "graph": report["graph"],
                       "summary": report["summary"]})

    keys = list(points[0]["summary"])
    print("=" * 150)
    print("SCALING — VERITAS reconciliation and retrieval cost as the knowledge graph grows")
    print("=" * 150)
    print(f"{'people':>7}{'obs':>8}{'facts':>8}{'ledger':>8}{'ingest mean':>13}{'ingest p95':>12}{'verify':>10}"
          f"{'full graph':>12} │ {'query mean/p95 default':>24}{'optimized':>20} │ {'ctx tok def/opt':>16}")
    for p in points:
        v, o, g = p["summary"]["veritas"], p["summary"]["veritas_optimized"], p["graph"]
        print(f"{p['people']:>7}{p['observations']:>8}{g['edges_total']:>8}{g['ledger_records']:>8}"
              f"{v['ingest_ms_per_observation_mean']:>11.2f}ms{v['ingest_ms_per_observation_p95']:>10.2f}ms"
              f"{g['ledger_verify_ms']:>8.0f}ms{g['full_graph_tokens']:>12,} │ "
              f"{v['query_ms_mean']:>11.1f} / {v['query_ms_p95']:>6.1f} ms{o['query_ms_mean']:>9.1f} / {o['query_ms_p95']:>6.1f} ms │ "
              f"{v['context_tokens_mean']:>7.0f} / {o['context_tokens_mean']:<6.0f}")
    print("-" * 150)
    print("Context tokens per question (mean) by method:")
    print(f"{'people':>7}" + "".join(f"{METHOD_LABELS[k]:>19}" for k in keys))  # noqa: E501
    for p in points:
        print(f"{p['people']:>7}" + "".join(f"{p['summary'][k]['context_tokens_mean']:>19,.0f}" for k in keys))
    print("Accuracy by method:")
    for p in points:
        print(f"{p['people']:>7}" + "".join(f"{_pct(p['summary'][k]['accuracy']):>19}" for k in keys))
    print("=" * 150)
    return {"points": points}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--people", type=int, default=40)
    p.add_argument("--orgs", type=int, default=None)
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--drift", type=float, default=0.35, help="per-session probability a fact changes")
    p.add_argument("--rumor-rate", type=float, default=0.15)
    p.add_argument("--late-rate", type=float, default=0.15)
    p.add_argument("--questions", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--top-k", type=int, default=6, help="retrieval depth for vector and hybrid RAG")
    p.add_argument("--no-fsync", action="store_true", help="skip the per-transaction ledger fsync")
    p.add_argument("--scaling", default="", help="comma-separated world sizes (people), e.g. 25,100,400")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--markdown", type=Path, default=None)
    p.add_argument("--export-dataset", type=Path, default=None)
    args = p.parse_args(argv)

    if args.scaling:
        sizes = [int(x) for x in args.scaling.split(",") if x.strip()]
        result = scaling(sizes, sessions=args.sessions, questions=args.questions, seed=args.seed,
                         fsync=not args.no_fsync, top_k=args.top_k)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return 0 if all(pt["graph"]["ledger_valid"] for pt in result["points"]) else 1

    dataset = generate(people=args.people, orgs=args.orgs, sessions=args.sessions, drift=args.drift,
                       rumor_rate=args.rumor_rate, late_rate=args.late_rate, max_questions=args.questions,
                       seed=args.seed)
    if args.export_dataset:
        args.export_dataset.write_text(json.dumps(dataset.to_json(), indent=2), encoding="utf-8")
        print(f"Wrote {args.export_dataset}")
    report = evaluate(dataset, top_k=args.top_k, fsync=not args.no_fsync)
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.json}")
    if args.markdown:
        write_markdown(args.markdown, report)
        print(f"Wrote {args.markdown}")
    return 0 if report["graph"].get("ledger_valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
