"""A/B Lab service: run memory-method comparisons on user-supplied facts and questions.

Used by the dashboard's A/B Lab page through ``main.py``. A run is a background
job (one at a time, so a local SLM is never asked to serve two evaluations at
once) whose rows and running summary can be polled while it executes. Every
finished run is also written to ``reports/`` exactly like the CLI harness.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from ab_eval import (
    AB_QUERIES,
    ABQuery,
    ALL_METHODS,
    DEFAULT_METHODS,
    ABRunner,
    MemoryMethod,
    VeritasMemory,
    build_methods,
    required_context,
    scaled_trajectory,
    summarize,
    write_markdown,
)
from agent import LLMClient, LLMError, MockLLMClient, OllamaClient, OpenAICompatibleClient
from extraction import ExtractionError, extract_triples
from models import ensure_utc
from ontology import get_spec
from scenarios import ScriptedObservation

METHOD_KEYS = ALL_METHODS
Category = Literal["changed", "stable", "multi-hop", "historical", "contested", "point-in-time"]
_DEFAULT_START = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #


class ObservationIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    timestamp: Optional[datetime] = None
    source: str = "user_statement"
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    expected: str = Field(min_length=1, max_length=120)
    wrong: list[str] = Field(default_factory=list)
    category: Category = "changed"

    @field_validator("wrong")
    @classmethod
    def _clean(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]


class ABRunRequest(BaseModel):
    observations: list[ObservationIn] = Field(min_length=1, max_length=3000)
    queries: list[QueryIn] = Field(min_length=1, max_length=60)
    methods: list[str] = Field(default_factory=lambda: list(DEFAULT_METHODS))
    provider: Literal["ollama", "llamacpp", "mock"] = "ollama"
    model: str = "llama3.2"
    base_url: Optional[str] = None
    num_gpu: Optional[int] = Field(default=None, ge=0, le=200)
    repeats: int = Field(default=1, ge=1, le=5)
    top_k: int = Field(default=6, ge=1, le=30)
    max_tokens: int = Field(default=160, ge=16, le=1024)

    @field_validator("methods")
    @classmethod
    def _known_methods(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip().upper() for v in values]
        unknown = [v for v in cleaned if v not in METHOD_KEYS]
        if unknown or not cleaned:
            raise ValueError(f"methods must be a non-empty subset of {METHOD_KEYS}; got {values}")
        return list(dict.fromkeys(cleaned))


class SuggestRequest(BaseModel):
    observations: list[ObservationIn] = Field(min_length=1, max_length=3000)
    max_questions: int = Field(default=12, ge=1, le=40)


class ObservationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__(f"{len(errors)} observation(s) could not be parsed")
        self.errors = errors


class BusyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Input conversion
# --------------------------------------------------------------------------- #


def to_scripted(observations: list[ObservationIn]) -> list[ScriptedObservation]:
    """Validate every statement and assign timestamps.

    List order is arrival order. A missing timestamp is one hour after the
    previous observation, so a plain list of statements is a valid trajectory.
    """
    out: list[ScriptedObservation] = []
    errors: list[dict[str, Any]] = []
    clock = _DEFAULT_START - timedelta(hours=1)
    for idx, item in enumerate(observations):
        ts = ensure_utc(item.timestamp) if item.timestamp else clock + timedelta(hours=1)
        clock = ts
        try:
            extract_triples(item.text, confidence=item.confidence, source=item.source, timestamp=ts)
        except ExtractionError as exc:
            errors.append({"index": idx, "text": item.text, "error": str(exc)})
            continue
        out.append(ScriptedObservation("user", ts, item.text.strip().rstrip("."), item.confidence, item.source))
    if errors:
        raise ObservationError(errors)
    return out


def to_queries(queries: list[QueryIn]) -> list[ABQuery]:
    return [ABQuery(q.question.strip(), q.expected.strip(), tuple(q.wrong), q.category) for q in queries]


def preset() -> dict[str, Any]:
    """The demo trajectory and question set, in the lab's editable format."""
    return {
        "observations": [
            {"text": o.text, "timestamp": o.timestamp.isoformat(), "source": o.source, "confidence": o.confidence}
            for o in scaled_trajectory(1)
        ],
        "queries": [
            {"question": q.question, "expected": q.answer, "wrong": list(q.wrong), "category": q.category}
            for q in AB_QUERIES
        ],
    }


# --------------------------------------------------------------------------- #
# Automatic question generation
# --------------------------------------------------------------------------- #

_TEMPLATES: dict[str, tuple[str, str]] = {
    "lives_in": ("Where does {s} live now?", "Where did {s} live before {cur}?"),
    "works_at": ("Where does {s} work now?", "Where did {s} work before {cur}?"),
    "has_role": ("What is {s}'s current role?", "What was {s}'s role before {cur}?"),
    "reports_to": ("Who does {s} report to now?", "Who did {s} report to before {cur}?"),
    "married_to": ("Who is {s} married to now?", "Who was {s} married to before {cur}?"),
    "prefers": ("What does {s} prefer now?", "What did {s} prefer before {cur}?"),
    "headquartered_in": ("Where is {s} headquartered now?", "Where was {s} headquartered before {cur}?"),
    "located_in": ("Where is {s} located now?", "Where was {s} located before {cur}?"),
    "has_status": ("What is {s}'s current status?", "What was {s}'s status before {cur}?"),
}


def _templates(predicate: str) -> tuple[str, str]:
    if predicate in _TEMPLATES:
        return _TEMPLATES[predicate]
    words = predicate.replace("_", " ")
    return f"What is {{s}}'s current {words}?", f"What was {{s}}'s {words} before {{cur}}?"


def suggest_queries(observations: list[ObservationIn], max_questions: int = 12, workdir: Optional[Path] = None) -> dict:
    """Generate test questions with ground truth from the user's own facts.

    The facts are reconciled by the VERITAS engine; every single-valued
    (subject, relation) slot whose value *changed* yields a "current value"
    question (wrong answers = every superseded value) and a "what came before"
    question. A few never-changed slots are added as stable controls.
    Expected answers therefore follow VERITAS's reconciliation rules (latest
    valid time, source authority), so the UI shows them for the user to check.
    """
    import tempfile

    scripted = to_scripted(observations)
    base = Path(workdir or tempfile.mkdtemp(prefix="veritas-suggest-"))
    memory = VeritasMemory(base / "suggest_ledger.jsonl", MockLLMClient())
    for obs in scripted:
        memory.observe(obs, extract_triples(obs.text, confidence=obs.confidence, source=obs.source,
                                            timestamp=obs.timestamp))
    graph = memory.graph
    name = lambda eid: (graph.get_entity(eid).name if graph.get_entity(eid) else eid)  # noqa: E731

    slots: dict[tuple[str, str], list] = {}
    for edge in graph.all_edges():
        if get_spec(edge.relation).functional:
            slots.setdefault((edge.source_id, edge.relation), []).append(edge)

    changed: list[dict] = []
    historical: list[dict] = []
    stable: list[dict] = []
    for (sid, rel), edges in sorted(slots.items(), key=lambda kv: (name(kv[0][0]), kv[0][1])):
        edges.sort(key=lambda e: e.t_start)
        active = [e for e in edges if e.is_active]
        if len(active) != 1:
            continue
        current = active[0]
        past = [e for e in edges if not e.is_active]
        now_q, before_q = _templates(rel)
        s, cur = name(sid), name(current.target_id)
        past_names = list(dict.fromkeys(name(e.target_id) for e in past if e.target_id != current.target_id))
        if past_names:
            changed.append({"question": now_q.format(s=s, cur=cur), "expected": cur,
                            "wrong": past_names, "category": "changed"})
            before = [e for e in past if e.t_end is not None and e.t_end <= current.t_start]
            if before:
                prev = max(before, key=lambda e: e.t_end)
                prev_name = name(prev.target_id)
                if prev_name != cur:
                    others = [n for n in past_names if n != prev_name]
                    historical.append({"question": before_q.format(s=s, cur=cur), "expected": prev_name,
                                       "wrong": [cur, *others], "category": "historical"})
        else:
            stable.append({"question": now_q.format(s=s, cur=cur), "expected": cur, "wrong": [],
                           "category": "stable"})

    available = {"changed": len(changed), "historical": len(historical), "stable": len(stable)}
    # Interleave so a short budget still covers every kind of question.
    picked: list[dict] = []
    budget_stable = min(len(stable), max(1, max_questions // 5))
    pools = [changed, historical]
    while len(picked) < max_questions - budget_stable and any(pools):
        for pool in pools:
            if pool and len(picked) < max_questions - budget_stable:
                picked.append(pool.pop(0))
    picked.extend(stable[: max_questions - len(picked)])
    return {
        "queries": picked,
        "ground_truth_source": "VERITAS reconciliation of your observations (check before relying on it)",
        "available": available,
    }


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


def list_models(base_url: str = "http://localhost:11434") -> dict[str, Any]:
    try:
        tags = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.5).json()
        models = sorted(m["name"] for m in tags.get("models", []))
        return {"ollama_available": True, "models": models}
    except (httpx.HTTPError, ValueError, KeyError):
        return {"ollama_available": False, "models": []}


class ABJob:
    def __init__(self, request: ABRunRequest) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status: Literal["queued", "running", "done", "cancelled", "error"] = "queued"
        self.phase = "queued"
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.done = 0
        self.total = request.repeats * len(request.queries) * len(request.methods)
        self.rows: list[dict] = []
        self.summary: dict = {}
        self.config: dict = {}
        self.error: Optional[str] = None
        self.report_dir: Optional[str] = None
        self.cancel_requested = False
        self.lock = threading.Lock()

    def view(self, include_rows: bool = True) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "phase": self.phase,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "progress": {"done": self.done, "total": self.total},
                "config": self.config,
                "summary": self.summary,
                "rows": list(self.rows) if include_rows else [],
                "error": self.error,
                "report_dir": self.report_dir,
            }


class ABJobManager:
    def __init__(self, reports_dir: Path, history: int = 20) -> None:
        self.reports_dir = Path(reports_dir)
        self.history = history
        self.jobs: "OrderedDict[str, ABJob]" = OrderedDict()
        self._active: Optional[ABJob] = None
        self._lock = threading.Lock()

    def submit(self, request: ABRunRequest) -> ABJob:
        observations = to_scripted(request.observations)  # validate synchronously → 422 on bad input
        with self._lock:
            if self._active is not None and self._active.status in ("queued", "running"):
                raise BusyError(f"A/B run {self._active.id} is still running; cancel it or wait.")
            job = ABJob(request)
            self.jobs[job.id] = job
            while len(self.jobs) > self.history:
                self.jobs.popitem(last=False)
            self._active = job
        threading.Thread(target=self._run, args=(job, observations), daemon=True, name=f"ab-{job.id}").start()
        return job

    def get(self, job_id: str) -> Optional[ABJob]:
        return self.jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        return [job.view(include_rows=False) for job in reversed(self.jobs.values())]

    def cancel(self, job_id: str) -> Optional[ABJob]:
        job = self.jobs.get(job_id)
        if job is not None:
            job.cancel_requested = True
        return job

    # -- execution --------------------------------------------------------- #

    @staticmethod
    def _client(req: ABRunRequest) -> LLMClient:
        if req.provider == "mock":
            return MockLLMClient()
        if req.provider == "llamacpp":
            return OpenAICompatibleClient(req.base_url or "http://localhost:8080/v1", req.model,
                                          max_tokens=req.max_tokens)
        return OllamaClient(req.model, req.base_url or "http://localhost:11434", num_predict=req.max_tokens,
                            num_gpu=req.num_gpu)

    def _run(self, job: ABJob, observations: list[ScriptedObservation]) -> None:
        req = job.request
        with job.lock:
            job.status, job.phase, job.started_at = "running", "building memories", datetime.now(timezone.utc)
        run_dir = self.reports_dir / f"lab_{job.created_at:%Y%m%dT%H%M%SZ}_{job.id}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            client = self._client(req)
            available, base = build_methods(client, run_dir / "veritas_ledger.jsonl", req.top_k)
            methods = [available[k] for k in req.methods]
            ingest = methods if "B" in req.methods else [*methods, base]
            queries = to_queries(req.queries)
            runner = ABRunner(client, methods, queries, repeats=req.repeats)
            runner.ingest(observations, ingest)
            if isinstance(client, OllamaClient):
                client.num_ctx = required_context(methods, queries, req.max_tokens)
            verification = base.ledger.verify()
            with job.lock:
                job.config = {
                    "model": client.model_name,
                    "provider": req.provider,
                    "methods": req.methods,
                    "repeats": req.repeats,
                    "observations": len(observations),
                    "queries": len(queries),
                    "num_ctx": getattr(client, "num_ctx", None),
                    "num_gpu": req.num_gpu,
                    "ledger_verification": verification.detail,
                    "ledger_head_hash": verification.head_hash,
                    "reconciliation_decisions": dict(base.decisions),
                    "method_labels": {m.key: m.label for m in methods},
                }
                job.phase = "warming up model"
            if req.provider != "mock":
                runner.warm_up()
            with job.lock:
                job.phase = "running"

            collected = []

            def on_row(row, done: int, total: int) -> None:  # type: ignore[no-untyped-def]
                collected.append(row)
                view = asdict(row)
                view["prompt"] = view.pop("user_prompt")[:6000]
                view.pop("system_prompt", None)
                summary = summarize(collected, methods)
                with job.lock:
                    job.rows.append(view)
                    job.done, job.total = done, total
                    job.summary = summary

            rows = runner.run(on_row=on_row, should_stop=lambda: job.cancel_requested, quiet=True)
            summary = summarize(rows, methods) if rows else {}
            report = {"config": job.config, "summary": summary, "rows": [asdict(r) for r in rows]}
            (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            if rows:
                md_config = {**job.config, "started_at": job.started_at.isoformat() if job.started_at else "",
                             "scale": "custom", "ledger_path": str(run_dir / "veritas_ledger.jsonl")}
                write_markdown(run_dir / "report.md", md_config, summary, rows, methods, queries)
            with job.lock:
                job.summary = summary
                job.status = "cancelled" if job.cancel_requested else "done"
                job.phase = job.status
                job.report_dir = str(run_dir)
        except (LLMError, ValueError, OSError) as exc:
            with job.lock:
                job.status, job.phase, job.error = "error", "error", str(exc)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of a silent dead thread
            with job.lock:
                job.status, job.phase, job.error = "error", "error", f"{type(exc).__name__}: {exc}"
        finally:
            with job.lock:
                job.finished_at = datetime.now(timezone.utc)
