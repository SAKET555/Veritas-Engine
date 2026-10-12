"""Comparative benchmark: Verifiable TKG vs. RAG and graph baselines on a non-stationary trajectory.

    python benchmark.py                     # human-readable report
    python benchmark.py --json report.json  # also write machine-readable results
    python benchmark.py --embedder st       # use sentence-transformers if installed

Protocol
--------
Step 1 (T0)  initial facts are observed.
Step 2 (T1)  contradictory observations arrive (relocation, job change, rumor,
             late-arriving history); T2 adds further drift.
Step 3 (T2)  the agent is challenged on the mutated facts and their provenance.

Both systems ingest the *same* observation stream.

* Hybrid RAG (BM25 + dense, reciprocal rank fusion), static Graph RAG (no
  time dimension) and overwrite memory (last arrival wins, no audit trail)
  ingest the same triples; see ``baselines.py``. Their reader is the same
  top-ranked-fact rule as vector RAG's.
* Vector RAG stores every observation as an append-only chunk, embeds it with a
  dense encoder and retrieves top-k by cosine similarity. Its reader takes the
  highest-ranked retrieved chunk that addresses the asked (subject, predicate).
* The TKG reconciles each observation into the bi-temporal graph, prunes a
  query subgraph and answers through the frozen-LLM harness (mock reader by
  default — swap via LLM_PROVIDER).

Metrics
-------
accuracy                answer names the correct current value
stale-in-context        retrieved context asserts a superseded value as current
contradiction rate      context holds ≥2 different current values for one functional (s, p)
epistemic consistency   correct AND no stale/contradictory facts presented as current
verifiable              every claim carries a citation that resolves in the hash-chained ledger
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from agent import MockLLMClient, VerifiableAgent, build_llm_client_from_env
from baselines import (  # noqa: F401 - re-exported for existing importers
    Chunk,
    DenseVectorRAG,
    HashedDenseEncoder,
    HybridRAG,
    OverwriteMemory,
    Retrieval,
    SentenceTransformerEncoder,
    StaticGraphRAG,
)
from extraction import extract_triples
from graph_engine import TemporalGraphEngine
from models import PruningParams, QueryRequest, TripleObservation
from ontology import get_spec
from pruning import SubgraphPruner
from reconciliation import AuditLedger, ReconciliationEngine
from scenarios import QUERIES, BenchmarkQuery, ScriptedObservation, trajectory
from tokens import count_tokens, tokenizer_name

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class QueryResult:
    question: str
    expected: str
    answer: str
    correct: bool
    stale_in_context: list[str]
    contradiction: bool
    consistent: bool
    verifiable: bool
    context_tokens: int
    extra: dict = field(default_factory=dict)


def _mentions(text: str, value: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(value) + r"(?![A-Za-z])", text, re.IGNORECASE) is not None


def evaluate_rag(rag: DenseVectorRAG, q: BenchmarkQuery) -> QueryResult:
    hits = rag.retrieve(q.question)
    ctx = rag.context(hits)
    relevant = [(s, c) for s, c in hits
                if c.triple.subject.lower() == q.subject.lower() and get_spec(c.triple.predicate).name == q.predicate]
    answer = relevant[0][1].triple.object if relevant else "(no relevant chunk retrieved)"
    values = {c.triple.object for _, c in relevant}
    stale = sorted(v for v in values if v in q.stale or (v in q.expected_also and not q.historical))
    functional_groups: dict[tuple[str, str], set[str]] = {}
    for _, c in hits:
        spec = get_spec(c.triple.predicate)
        if spec.functional:
            functional_groups.setdefault((c.triple.subject.lower(), spec.name), set()).add(c.triple.object)
    contradiction = any(len(v) > 1 for v in functional_groups.values())
    correct = answer.lower() == q.expected.lower()
    if q.historical:
        correct = correct and all(any(v.lower() == c.triple.object.lower() for _, c in relevant) for v in q.expected_also)
    return QueryResult(
        question=q.question, expected=q.expected, answer=answer, correct=correct,
        stale_in_context=stale, contradiction=contradiction,
        consistent=correct and not stale and not contradiction,
        verifiable=False, context_tokens=count_tokens(ctx),
        extra={"retrieved": [c.doc_id + ": " + c.text for _, c in hits]},
    )


def evaluate_baseline(memory, q: BenchmarkQuery) -> QueryResult:  # type: ignore[no-untyped-def]
    """Score any :mod:`baselines` memory with the same top-ranked-fact reader used for vector RAG."""
    retrieval: Retrieval = memory.search(q.question)
    relevant = [f for f in retrieval.facts if f.subject.lower() == q.subject.lower() and f.predicate == q.predicate]
    answer = relevant[0].object if relevant else "(no relevant fact retrieved)"
    values = {f.object for f in relevant}
    stale = sorted(v for v in values if v in q.stale or (v in q.expected_also and not q.historical))
    groups: dict[tuple[str, str], set[str]] = {}
    for f in retrieval.facts:
        if f.presented_as_current and get_spec(f.predicate).functional:
            groups.setdefault((f.subject.lower(), f.predicate), set()).add(f.object)
    contradiction = any(len(v) > 1 for v in groups.values())
    correct = answer.lower() == q.expected.lower()
    if q.historical:
        correct = correct and all(any(v.lower() == f.object.lower() for f in relevant) for v in q.expected_also)
    return QueryResult(
        question=q.question, expected=q.expected, answer=answer, correct=correct,
        stale_in_context=stale, contradiction=contradiction,
        consistent=correct and not stale and not contradiction,
        verifiable=False, context_tokens=count_tokens(retrieval.context),
        extra={"retrieved": retrieval.context.splitlines()[1:]},
    )


def evaluate_tkg(agent: VerifiableAgent, q: BenchmarkQuery) -> QueryResult:
    resp = agent.answer(QueryRequest(question=q.question, pruning=PruningParams()))
    current_lines = [ln for ln in resp.pruned_context.splitlines() if ln.startswith("- ") and "[INVALIDATED]" not in ln]
    # Stale = a superseded value asserted *as current* in the context or the answer.
    stale = sorted({v for v in q.stale if any(f"-{q.predicate}-> {v} |" in ln for ln in current_lines)})
    answer_current = re.split(r"\bPreviously\b", resp.answer)[0]
    stale_in_answer = [v for v in q.stale + (() if q.historical else q.expected_also) if _mentions(answer_current, v)]

    groups: dict[tuple[str, str], set[str]] = {}
    for ln in current_lines:
        m = re.match(r"^- (.+?) -([a-z0-9_]+)-> (.+?) \|", ln)
        if m and get_spec(m.group(2)).functional:
            groups.setdefault((m.group(1), m.group(2)), set()).add(m.group(3))
    contradiction = any(len(v) > 1 for v in groups.values())

    correct = _mentions(answer_current, q.expected)
    if q.historical:
        correct = correct and all(_mentions(resp.answer, v) for v in q.expected_also)
        correct = correct and any(c.mutation_type == "INVALIDATE" and c.verified for c in resp.citations)
    return QueryResult(
        question=q.question, expected=q.expected, answer=resp.answer, correct=correct,
        stale_in_context=sorted(set(stale) | set(stale_in_answer)), contradiction=contradiction,
        consistent=correct and not stale and not stale_in_answer and not contradiction,
        verifiable=resp.grounded and bool(resp.citations),
        context_tokens=resp.metrics.pruned_context_tokens + resp.metrics.audit_trail_tokens,
        extra={
            "pruned_context_tokens": resp.metrics.pruned_context_tokens,
            "audit_tokens": resp.metrics.audit_trail_tokens,
            "full_graph_tokens": resp.metrics.full_graph_tokens,
            "reduction_ratio": resp.metrics.reduction_ratio,
            "pruned_edges": resp.metrics.pruned_edges,
            "full_graph_edges": resp.metrics.full_graph_edges,
            "citations": [f"{c.short_id}:{c.mutation_type}:{'ok' if c.verified else 'UNVERIFIED'}" for c in resp.citations],
            "pruned_context": resp.pruned_context,
        },
    )


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):5.1f}%" if d else "  n/a"


def run(embedder: str, top_k: int, verbose: bool, all_baselines: bool = False) -> dict:
    observations = trajectory()
    encoder: Callable = SentenceTransformerEncoder() if embedder == "st" else HashedDenseEncoder()  # type: ignore[assignment]

    workdir = Path(tempfile.mkdtemp(prefix="veritas-bench-"))
    graph = TemporalGraphEngine()
    ledger = AuditLedger(workdir / "audit.jsonl")
    reconciler = ReconciliationEngine(graph, ledger)
    pruner = SubgraphPruner(graph, ledger)
    try:
        client = build_llm_client_from_env()
    except Exception:  # noqa: BLE001 - benchmark must run offline
        client = MockLLMClient()
    agent = VerifiableAgent(pruner, ledger, client)
    rag = DenseVectorRAG(encoder, top_k=top_k)
    extra_baselines = {
        "hybrid_rag": HybridRAG(encoder, top_k=top_k),
        "static_graph_rag": StaticGraphRAG(),
        "overwrite_memory": OverwriteMemory(),
    } if all_baselines else {}

    print("=" * 96)
    print("VERITAS benchmark — Verifiable Temporal KG vs. "
          + ("vector, hybrid, static-graph and overwrite memories" if all_baselines else "Dense Vector RAG"))
    print(f"encoder={encoder.name}  reader={client.model_name}  tokenizer={tokenizer_name()}  rag_top_k={top_k}")
    print("=" * 96)

    decisions: dict[str, int] = {}
    for phase in ("T0", "T1", "T2"):
        batch = [o for o in observations if o.phase == phase]
        for obs in batch:
            triples = extract_triples(obs.text, confidence=obs.confidence, source=obs.source, timestamp=obs.timestamp)
            rag.ingest(obs, triples)
            for memory in extra_baselines.values():
                memory.ingest(obs, triples)
            for triple in triples:
                outcome = reconciler.ingest(triple)
                decisions[outcome.decision] = decisions.get(outcome.decision, 0) + 1
                if verbose and triple.subject in {"Alice", "Acme"}:
                    print(f"  [{phase}] {outcome.decision:<19} {outcome.summary}")
        print(f"Step {phase}: ingested {len(batch):3d} observations | graph {graph.stats()}")

    chain = ledger.verify()
    print(f"Reconciliation decisions: {decisions}")
    print(f"Ledger: {chain.detail} (head {chain.head_hash[:16]}…)")
    print("-" * 96)

    rag_results = [evaluate_rag(rag, q) for q in QUERIES]
    extra_results = {key: [evaluate_baseline(m, q) for q in QUERIES] for key, m in extra_baselines.items()}
    tkg_results = [evaluate_tkg(agent, q) for q in QUERIES]

    for q, r, t in zip(QUERIES, rag_results, tkg_results):
        print(f"Q: {q.question}   (expected: {q.expected}{' + ' + ', '.join(q.expected_also) if q.expected_also else ''})")
        print(f"  RAG  answer={r.answer!r:<22} correct={r.correct!s:<5} stale_in_ctx={r.stale_in_context or '-'} "
              f"contradiction={r.contradiction!s:<5} tokens={r.context_tokens}")
        if verbose:
            for line in r.extra["retrieved"]:
                print(f"         {line}")
        print(f"  TKG  correct={t.correct!s:<5} stale={t.stale_in_context or '-'} contradiction={t.contradiction!s:<5} "
              f"tokens={t.context_tokens} (graph {t.extra['pruned_context_tokens']} vs full {t.extra['full_graph_tokens']}, "
              f"{t.extra['pruned_edges']}/{t.extra['full_graph_edges']} edges) cites={t.extra['citations']}")
        print(f"       answer: {t.answer}")
        if verbose:
            for line in t.extra["pruned_context"].splitlines():
                print(f"         {line}")
        print()

    n = len(QUERIES)

    def agg(results: list[QueryResult]) -> dict:
        return {
            "accuracy": sum(r.correct for r in results) / n,
            "stale_in_context_rate": sum(bool(r.stale_in_context) for r in results) / n,
            "contradiction_rate": sum(r.contradiction for r in results) / n,
            "epistemic_consistency": sum(r.consistent for r in results) / n,
            "verifiable_rate": sum(r.verifiable for r in results) / n,
            "mean_context_tokens": sum(r.context_tokens for r in results) / n,
        }

    rag_summary, tkg_summary = agg(rag_results), agg(tkg_results)
    extra_summaries = {key: agg(results) for key, results in extra_results.items()}
    full_tokens = tkg_results[0].extra["full_graph_tokens"]
    mean_pruned = sum(r.extra["pruned_context_tokens"] for r in tkg_results) / n

    if all_baselines:
        columns = [("Dense Vector RAG", rag_summary), ("Hybrid BM25+Dense", extra_summaries["hybrid_rag"]),
                   ("Static Graph RAG", extra_summaries["static_graph_rag"]),
                   ("Overwrite memory", extra_summaries["overwrite_memory"]), ("Verifiable TKG", tkg_summary)]
        col = 19
    else:
        columns = [("Dense Vector RAG", rag_summary), ("Verifiable TKG", tkg_summary)]
        col = 22
    width = 28 + col * len(columns)
    outer = width if all_baselines else 96
    print("=" * outer)
    print(f"{'metric':<28}" + "".join(f"{label:>{col}}" for label, _ in columns))
    print("-" * (width if all_baselines else 72))
    for key, label in (
        ("accuracy", "accuracy"),
        ("epistemic_consistency", "epistemic consistency"),
        ("stale_in_context_rate", "stale fact presented"),
        ("contradiction_rate", "contradictory context"),
        ("verifiable_rate", "verifiable (cited+chained)"),
    ):
        print(f"{label:<28}" + "".join(f"{_pct(round(summ[key] * n), n):>{col}}" for _, summ in columns))
    print(f"{'mean context tokens':<28}" + "".join(f"{summ['mean_context_tokens']:>{col}.1f}" for _, summ in columns))
    print("-" * (width if all_baselines else 72))
    print(f"TKG graph-context tokens: mean {mean_pruned:.1f} vs unpruned full graph {full_tokens} "
          f"→ {100.0 * (1 - mean_pruned / full_tokens):.1f}% reduction")
    print(f"Audit ledger integrity: {'VALID' if chain.valid else 'BROKEN'} — {chain.records_checked} records")
    print("=" * outer)

    return {
        "config": {"encoder": encoder.name, "reader": client.model_name, "tokenizer": tokenizer_name(), "rag_top_k": top_k},
        "decisions": decisions,
        "ledger": chain.model_dump(),
        "summary": {"vector_rag": rag_summary, **extra_summaries, "tkg": tkg_summary,
                    "full_graph_tokens": full_tokens, "mean_pruned_graph_tokens": mean_pruned},
        "per_query": {
            "vector_rag": [asdict(r) for r in rag_results],
            **{key: [asdict(r) for r in results] for key, results in extra_results.items()},
            "tkg": [asdict(r) for r in tkg_results],
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=["hashed", "st"], default="hashed")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", type=Path, default=None, help="write full results to this path")
    parser.add_argument("-v", "--verbose", action="store_true", help="print retrieved chunks and pruned contexts")
    parser.add_argument("--all-baselines", action="store_true",
                        help="also compare hybrid BM25+dense RAG, static graph RAG and overwrite memory")
    args = parser.parse_args(argv)

    report = run(args.embedder, args.top_k, args.verbose, args.all_baselines)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.json}")
    tkg = report["summary"]["tkg"]
    return 0 if tkg["epistemic_consistency"] == 1.0 and report["ledger"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
