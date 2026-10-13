"""Robustness experiments on the synthetic benchmark (no LLM; deterministic per seed).

    python experiments.py seeds        # every method over several seeds: mean, SD, 95% Wilson CI
    python experiments.py ablations    # remove one VERITAS mechanism at a time
    python experiments.py sweep        # threshold τ and score weights (α, β, γ)
    python experiments.py adversarial  # high-authority sources that lie
    python experiments.py all --json results.json

All experiments reuse ``temporal_bench.evaluate`` and its shared reader, so the
numbers are directly comparable with ``temporal_bench.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from baselines import Fact, Retrieval
from datasets import CATEGORIES, generate
from models import PruningParams, TripleObservation
from scenarios import ScriptedObservation
from temporal_bench import (
    METHOD_LABELS,
    OPTIMIZED_PRUNING,
    VeritasRetriever,
    VeritasView,
    build_methods,
    evaluate,
)
from tokens import count_tokens

DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _dataset(seed: int, **overrides):  # type: ignore[no-untyped-def]
    params = {"people": 40, "sessions": 6, "max_questions": 60}
    params.update(overrides)
    return generate(seed=seed, **params)


def _pooled(reports: list[dict], key: str) -> dict:
    rows = [o for r in reports for o in r["outcomes"] if o["method"] == key]
    correct = sum(o["correct"] for o in rows)
    per_seed = [r["summary"][key]["accuracy"] for r in reports]
    by_cat = {}
    for cat in CATEGORIES:
        cat_rows = [o for o in rows if o["category"] == cat]
        if cat_rows:
            by_cat[cat] = sum(o["correct"] for o in cat_rows) / len(cat_rows)
    lo, hi = wilson(correct, len(rows))
    return {
        "label": reports[0]["summary"][key]["label"],
        "n_answers": len(rows),
        "accuracy_pooled": correct / len(rows) if rows else float("nan"),
        "accuracy_mean": statistics.fmean(per_seed),
        "accuracy_sd": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
        "accuracy_ci95": [lo, hi],
        "accuracy_by_category": by_cat,
        "stale_rate": sum(o["stale"] for o in rows) / len(rows),
        "verifiable_rate": sum(o["verifiable"] for o in rows) / len(rows),
        "context_tokens_mean": statistics.fmean(o["context_tokens"] for o in rows),
    }


def _print_table(title: str, results: dict[str, dict], extra: Sequence[tuple[str, str]] = ()) -> None:
    print("=" * 112)
    print(title)
    print("=" * 112)
    print(f"{'configuration':<36}{'accuracy (pooled)':>18}{'mean ± SD':>16}{'95% CI':>18}{'stale':>9}{'tokens':>9}"
          + "".join(f"{label:>12}" for label, _ in extra))
    for key, r in results.items():
        lo, hi = r["accuracy_ci95"]
        print(f"{r['label'][:35]:<36}{100 * r['accuracy_pooled']:>17.1f}%"
              f"{100 * r['accuracy_mean']:>9.1f} ± {100 * r['accuracy_sd']:<4.1f}"
              f"{'[' + f'{100 * lo:.1f}, {100 * hi:.1f}' + ']':>18}{100 * r['stale_rate']:>8.1f}%"
              f"{r['context_tokens_mean']:>9.0f}" + "".join(f"{r.get(k, float('nan')):>12.3f}" for _, k in extra))


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #


def seeds_experiment(seeds: Sequence[int] = DEFAULT_SEEDS, **dataset_options) -> dict:  # type: ignore[no-untyped-def]
    reports = [evaluate(_dataset(s, **dataset_options), fsync=False) for s in seeds]
    keys = list(reports[0]["summary"])
    results = {k: _pooled(reports, k) for k in keys}
    _print_table(f"MULTI-SEED · seeds {list(seeds)} · {reports[0]['dataset']['questions']} questions per seed", results)
    return {"seeds": list(seeds), "results": results}


class ArrivalTimeVeritas(VeritasRetriever):
    """Ablation: ignore valid time; every observation is stamped with its arrival time."""

    def __init__(self, ledger_path: Path) -> None:
        super().__init__(ledger_path, fsync=False)
        self._clock = None

    def ingest(self, obs: ScriptedObservation, triples: list[TripleObservation]) -> None:
        self._clock = obs.timestamp if self._clock is None else max(self._clock, obs.timestamp)
        stamped = [t.model_copy(update={"timestamp": self._clock}) for t in triples]
        super().ingest(replace(obs, timestamp=self._clock), stamped)


class UnprunedVeritas(VeritasView):
    """Ablation: no pruning; the whole graph (with intervals) is the context."""

    def search(self, question: str, as_of=None) -> Retrieval:  # type: ignore[no-untyped-def]
        facts = []
        for rank, edge in enumerate(self.graph.all_edges()):
            record = self.ledger.latest_for_edge(edge.id)
            facts.append(Fact(self._name(edge.source_id), edge.relation, self._name(edge.target_id),
                              edge.t_start, edge.t_end, edge.source, record.short_id if record else "", rank,
                              edge.is_active))
        return Retrieval(self.pruner.serialize_full_graph(), facts)

    def compact_context(self, question: str) -> str:
        return self.pruner.serialize_full_graph()


def ablation_memories(workdir: Path) -> dict:
    full = VeritasRetriever(workdir / "full.jsonl", fsync=False)
    return {
        "veritas": full,
        "no_authority": VeritasRetriever(workdir / "no_auth.jsonl", fsync=False, authority=lambda _source: 1.0),
        "strict_tolerance": VeritasRetriever(workdir / "strict.jsonl", fsync=False, supersession_tolerance=0.0),
        "arrival_time": ArrivalTimeVeritas(workdir / "arrival.jsonl"),
        "no_pruning": UnprunedVeritas(full, PruningParams()),
        "optimized_pruning": VeritasView(full, OPTIMIZED_PRUNING),
    }


ABLATION_LABELS = {
    "veritas": "VERITAS (full)",
    "no_authority": "− source authority (all sources equal)",
    "strict_tolerance": "− supersession tolerance (δ = 0)",
    "arrival_time": "− valid time (arrival order only)",
    "no_pruning": "− pruning (whole graph in context)",
    "optimized_pruning": "+ ego-local scoring and hub damping",
}


def ablations_experiment(seeds: Sequence[int] = DEFAULT_SEEDS, **dataset_options) -> dict:  # type: ignore[no-untyped-def]
    reports = []
    for s in seeds:
        workdir = Path(tempfile.mkdtemp(prefix="veritas-ablation-"))
        reports.append(evaluate(_dataset(s, **dataset_options), memories=ablation_memories(workdir), fsync=False))
    results = {}
    for key in reports[0]["summary"]:
        r = _pooled(reports, key)
        r["label"] = ABLATION_LABELS[key]
        results[key] = r
    _print_table(f"ABLATIONS · seeds {list(seeds)}", results)
    print("Accuracy by category:")
    print(f"{'':<36}" + "".join(f"{c:>15}" for c in CATEGORIES))
    for r in results.values():
        print(f"{r['label'][:35]:<36}" + "".join(f"{100 * r['accuracy_by_category'].get(c, float('nan')):>14.1f}%"
                                                  for c in CATEGORIES))
    return {"seeds": list(seeds), "results": results}


WEIGHT_SETS = {
    "default (0.35, 0.15, 0.50)": (0.35, 0.15, 0.50),
    "confidence-heavy (0.60, 0.10, 0.30)": (0.60, 0.10, 0.30),
    "recency-heavy (0.20, 0.50, 0.30)": (0.20, 0.50, 0.30),
    "relevance-heavy (0.20, 0.10, 0.70)": (0.20, 0.10, 0.70),
    "equal (0.33, 0.33, 0.34)": (0.33, 0.33, 0.34),
}
TAUS = (0.4, 0.5, 0.6, 0.7, 0.8)


def sweep_experiment(seeds: Sequence[int] = DEFAULT_SEEDS, **dataset_options) -> dict:  # type: ignore[no-untyped-def]
    configs: dict[str, PruningParams] = {}
    for tau in TAUS:
        configs[f"τ = {tau:.1f}"] = PruningParams(tau=tau)
    for name, (a, b, g) in WEIGHT_SETS.items():
        configs[name] = PruningParams(alpha=a, beta=b, gamma=g)
    reports = []
    for s in seeds:
        workdir = Path(tempfile.mkdtemp(prefix="veritas-sweep-"))
        base = VeritasRetriever(workdir / "ledger.jsonl", fsync=False)
        memories = {"base": base}
        memories.update({name: VeritasView(base, params) for name, params in configs.items()})
        reports.append(evaluate(_dataset(s, **dataset_options), memories=memories, fsync=False))
    results = {}
    for key in configs:
        r = _pooled(reports, key)
        r["label"] = key
        rows = [o for rep in reports for o in rep["outcomes"] if o["method"] == key]
        r["evidence_recall"] = sum(o["evidence"] for o in rows) / len(rows)
        results[key] = r
    _print_table(f"PRUNING SWEEP · seeds {list(seeds)}", results, extra=(("recall", "evidence_recall"),))
    return {"seeds": list(seeds), "results": results}


def adversarial_experiment(seeds: Sequence[int] = DEFAULT_SEEDS, lie_rate: float = 0.15,
                           **dataset_options) -> dict:  # type: ignore[no-untyped-def]
    reports = [evaluate(_dataset(s, lie_rate=lie_rate, **dataset_options), fsync=False) for s in seeds]
    results = {k: _pooled(reports, k) for k in reports[0]["summary"]}
    _print_table(f"ADVERSARIAL · high-authority false claims (lie rate {lie_rate}) · seeds {list(seeds)}", results)
    print("Contested-category accuracy:", {results[k]["label"]: round(results[k]["accuracy_by_category"].get(
        "contested", float("nan")), 3) for k in results})
    return {"seeds": list(seeds), "lie_rate": lie_rate, "results": results}


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["seeds", "ablations", "sweep", "adversarial", "all"])
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--people", type=int, default=40)
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--questions", type=int, default=60)
    p.add_argument("--lie-rate", type=float, default=0.15)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    opts = {"people": args.people, "sessions": args.sessions, "max_questions": args.questions}
    out: dict = {"dataset": opts}
    if args.experiment in ("seeds", "all"):
        out["seeds"] = seeds_experiment(seeds, **opts)
    if args.experiment in ("ablations", "all"):
        out["ablations"] = ablations_experiment(seeds, **opts)
    if args.experiment in ("sweep", "all"):
        out["sweep"] = sweep_experiment(seeds, **opts)
    if args.experiment in ("adversarial", "all"):
        out["adversarial"] = adversarial_experiment(seeds, args.lie_rate, **opts)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
