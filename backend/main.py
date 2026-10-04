"""FastAPI server exposing the verifiable temporal knowledge-state engine.

Run:  uvicorn main:app --reload --port 8000   (from the backend/ directory)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from agent import LLMError, VerifiableAgent, build_llm_client_from_env
from extraction import ExtractionError, extract_triples
from graph_engine import GraphEngineError, TemporalGraphEngine
from models import (
    AuditLedgerResponse,
    ChainVerification,
    GraphEdgeView,
    GraphNodeView,
    GraphResponse,
    ObservationRequest,
    ObservationResponse,
    ProvenanceTrail,
    QueryRequest,
    QueryResponse,
    TimelineResponse,
    TripleObservation,
    ensure_utc,
    new_uuid,
    utc_now,
)
from pruning import SubgraphPruner
from reconciliation import AuditLedger, ReconciliationEngine
from scenarios import trajectory
import ab_service

logger = logging.getLogger("veritas")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", Path(__file__).parent / "data" / "audit_log.jsonl"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", Path(__file__).parent / "reports"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]


class AppState:
    """Composition root: one graph, one ledger, one reconciler, one agent."""

    def __init__(self, ledger_path: Path) -> None:
        self.graph = TemporalGraphEngine()
        self.ledger = AuditLedger(ledger_path)
        self.reconciler = ReconciliationEngine(self.graph, self.ledger)
        self.pruner = SubgraphPruner(self.graph, self.ledger)
        self.agent = VerifiableAgent(self.pruner, self.ledger, build_llm_client_from_env())
        # Raw observation journal (what was *said*, in arrival order). The ledger records what the
        # system *concluded*; the A/B Lab needs the raw stream to build the append-only baselines.
        self.journal_path = ledger_path.with_name("observations.jsonl")
        self.journal: list[dict] = []
        self.ab_jobs = ab_service.ABJobManager(REPORTS_DIR)

    def boot(self) -> ChainVerification:
        verification = self.ledger.load()
        if not verification.valid:
            raise RuntimeError(f"Audit ledger failed verification: {verification.detail}")
        applied = self.reconciler.replay()
        logger.info("Replayed %d ledger records from %s", applied, self.ledger.path)
        if self.journal_path.exists():
            with self.journal_path.open(encoding="utf-8") as fh:
                self.journal = [json.loads(line) for line in fh if line.strip()]
        return verification

    def record(self, text: str, timestamp: datetime, source: str, confidence: float) -> None:
        entry = {"text": text, "timestamp": timestamp.isoformat(), "source": source, "confidence": confidence}
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self.journal.append(entry)

    def archive_journal(self) -> None:
        if self.journal_path.exists() and self.journal_path.stat().st_size:
            stamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
            self.journal_path.rename(self.journal_path.with_name(f"observations.{stamp}.archived.jsonl"))
        self.journal = []

    def ingest(self, triples: list[TripleObservation]) -> list:
        return [self.reconciler.ingest(t) for t in triples]


state: AppState


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global state
    state = AppState(AUDIT_LOG_PATH)
    state.boot()
    yield


app = FastAPI(
    title="VERITAS — Verifiable Knowledge State Editing",
    version="1.0.0",
    description="Bi-temporal knowledge graph with audit-trailed reconciliation for frozen-LLM agents.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_model": state.agent.client.model_name,
        "ledger_records": len(state.ledger),
        "graph": state.graph.stats(),
    }


@app.post("/api/observe", response_model=ObservationResponse)
def observe(req: ObservationRequest) -> ObservationResponse:
    provenance_id = new_uuid()
    ts = req.timestamp or utc_now()
    if req.text and not req.subject:
        try:
            triples = extract_triples(
                req.text, confidence=req.confidence, source=req.source, timestamp=ts, provenance_id=provenance_id
            )
        except ExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        triples = [
            TripleObservation(
                subject=req.subject or "", predicate=req.predicate or "", object=req.object or "",
                confidence=req.confidence, source=req.source, provenance_id=provenance_id,
                timestamp=ts, raw_text=req.text,
            )
        ]
    try:
        outcomes = state.ingest(triples)
    except (ValueError, GraphEngineError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    journal_text = req.text if req.text and not req.subject else f"{req.subject} | {req.predicate} | {req.object}"
    state.record(journal_text, ts, req.source, req.confidence)
    return ObservationResponse(provenance_id=provenance_id, extracted_triples=triples, outcomes=outcomes)


@app.get("/api/graph", response_model=GraphResponse)
def graph(
    timestamp: Optional[datetime] = Query(default=None, description="Valid-time instant; omit for current state"),
    as_known_at: Optional[datetime] = Query(default=None, description="System-time instant (bi-temporal)"),
    include_history: bool = Query(default=False, description="Also return invalidated / not-yet-valid edges"),
) -> GraphResponse:
    ts = ensure_utc(timestamp) if timestamp else None
    known = ensure_utc(as_known_at) if as_known_at else None
    g = state.graph

    if include_history:
        sub = g.history_subgraph(ts)
    elif ts is not None or known is not None:
        sub = g.query_point_in_time(ts or utc_now(), as_known_at=known)
    else:
        sub = g.get_active_subgraph()

    edges: list[GraphEdgeView] = []
    for _, _, data in sub.edges(data=True):
        edge = data["data"]
        valid = g.is_visible(edge, ts, known) if ts is not None else edge.is_active
        edges.append(GraphEdgeView(edge=edge, valid_at_query_time=valid))
    edges.sort(key=lambda v: (v.edge.t_start, v.edge.id))

    nodes = [
        GraphNodeView(
            id=nid, name=d["data"].name, entity_type=d["data"].entity_type, attributes=d["data"].attributes,
            first_seen=d["data"].first_seen, degree_active=g.active_degree(nid),
        )
        for nid, d in sub.nodes(data=True)
    ]
    nodes.sort(key=lambda n: n.id)
    return GraphResponse(
        timestamp=ts, as_known_at=known, include_history=include_history,
        nodes=nodes, edges=edges,
        stats={**g.stats(), "nodes_returned": len(nodes), "edges_returned": len(edges)},
    )


@app.get("/api/timeline", response_model=TimelineResponse)
def timeline() -> TimelineResponse:
    return TimelineResponse(timestamps=state.graph.timeline())


@app.get("/api/audit", response_model=AuditLedgerResponse)
def audit(
    entity_id: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=10000),
) -> AuditLedgerResponse:
    records = state.ledger.for_entity(entity_id) if entity_id else state.ledger.records()
    return AuditLedgerResponse(count=len(records), head_hash=state.ledger.head_hash, records=records[-limit:])


@app.get("/api/audit/verify", response_model=ChainVerification)
def verify_audit() -> ChainVerification:
    return state.ledger.verify()


@app.get("/api/provenance/{entity_id}", response_model=ProvenanceTrail)
def provenance(entity_id: str) -> ProvenanceTrail:
    try:
        return state.reconciler.get_provenance_trail(entity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown entity {entity_id!r}") from exc


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return state.agent.answer(req)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/reset")
def reset() -> dict:
    """Archive the current ledger (never deleted) and start an empty graph."""
    with state.graph.lock:
        archived = state.ledger.archive_and_reset()
        state.graph.clear()
        state.archive_journal()
    return {"archived_to": str(archived) if archived else None}


@app.post("/api/demo/seed")
def seed_demo() -> dict:
    """Reset, then replay the scripted T0 → T1 → T2 trajectory."""
    reset()
    decisions: dict[str, int] = {}
    for obs in trajectory():
        state.record(obs.text, obs.timestamp, obs.source, obs.confidence)
        prov = new_uuid()
        for triple in extract_triples(obs.text, confidence=obs.confidence, source=obs.source,
                                      timestamp=obs.timestamp, provenance_id=prov):
            outcome = state.reconciler.ingest(triple)
            decisions[outcome.decision] = decisions.get(outcome.decision, 0) + 1
    return {"ledger_records": len(state.ledger), "decisions": decisions, "graph": state.graph.stats()}


# --------------------------------------------------------------------------- #
# A/B Lab
# --------------------------------------------------------------------------- #


@app.get("/api/observations")
def observations() -> dict:
    """Raw observations made through this dashboard, in arrival order."""
    return {"count": len(state.journal), "observations": state.journal}


@app.get("/api/ab/models")
def ab_models() -> dict:
    return {**ab_service.list_models(os.getenv("OLLAMA_URL", "http://localhost:11434")), "default": "llama3.2"}


@app.get("/api/ab/preset")
def ab_preset() -> dict:
    return ab_service.preset()


@app.post("/api/ab/suggest-queries")
def ab_suggest(req: ab_service.SuggestRequest) -> dict:
    try:
        return ab_service.suggest_queries(req.observations, req.max_questions)
    except ab_service.ObservationError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "errors": exc.errors}) from exc


@app.post("/api/ab/runs")
def ab_start(req: ab_service.ABRunRequest) -> dict:
    try:
        job = state.ab_jobs.submit(req)
    except ab_service.ObservationError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "errors": exc.errors}) from exc
    except ab_service.BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.view()


@app.get("/api/ab/runs")
def ab_list() -> dict:
    return {"runs": state.ab_jobs.list()}


@app.get("/api/ab/runs/{job_id}")
def ab_get(job_id: str) -> dict:
    job = state.ab_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown A/B run {job_id!r}")
    return job.view()


@app.post("/api/ab/runs/{job_id}/cancel")
def ab_cancel(job_id: str) -> dict:
    job = state.ab_jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown A/B run {job_id!r}")
    return job.view(include_rows=False)
