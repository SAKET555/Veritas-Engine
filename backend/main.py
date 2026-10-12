"""FastAPI server exposing the verifiable temporal knowledge-state engine.

Run:  uvicorn main:app --reload --port 8000   (from the backend/ directory)

Cloud deployment options (all off by default, so a plain local run behaves as before):

* Multi-tenancy   send ``X-Tenant-ID: <id>``; each tenant gets its own graph, its own
                  hash-chained ledger and its own observation journal under
                  ``<data>/tenants/<id>/``. No header = the ``default`` tenant, whose ledger
                  is ``AUDIT_LOG_PATH`` exactly as in single-tenant mode.
* API keys        ``VERITAS_API_KEYS="key1,key2:acme"``: every request except the
                  health/readiness probes must send ``X-API-Key`` (or ``Authorization:
                  Bearer``). A key written ``key:tenant`` is bound to that tenant.
* Observability   ``GET /api/metrics`` (Prometheus text format), ``GET /api/ready``.
* Entity linking  ``ENTITY_LINKING=explicit`` (default: only operator-declared aliases)
                  or ``auto`` (also link "Acme Inc." to an existing "Acme"). Every link is
                  an ALIAS ledger record.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from agent import LLMError, VerifiableAgent, build_llm_client_from_env
from entity_linking import EntityLinker
from extraction import ExtractionError, extract_triples
from graph_engine import GraphEngineError, TemporalGraphEngine
from models import (
    AliasRequest,
    AuditLedgerResponse,
    ChainVerification,
    GraphEdgeView,
    GraphNodeView,
    GraphResponse,
    MutationAuditRecord,
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
from observability import Counter, GaugeCallback, Histogram, Registry
from pruning import SubgraphPruner
from reconciliation import AuditLedger, ReconciliationEngine
from scenarios import trajectory
import ab_service

logger = logging.getLogger("veritas")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", Path(__file__).parent / "data" / "audit_log.jsonl"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", Path(__file__).parent / "reports"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

DEFAULT_TENANT = "default"
TENANT_HEADER = "X-Tenant-ID"
_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_PUBLIC_PATHS = frozenset({"/api/health", "/api/ready", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

metrics = Registry()
HTTP_REQUESTS = metrics.register(Counter("veritas_http_requests_total", "HTTP requests by route and status."))
HTTP_LATENCY = metrics.register(Histogram("veritas_http_request_duration_seconds", "HTTP request latency."))
DECISIONS = metrics.register(Counter("veritas_reconciliation_decisions_total", "Reconciliation outcomes."))
QUERY_TOKENS = metrics.register(Counter("veritas_query_context_tokens_total",
                                        "Context tokens: pruned (sent) vs the unpruned graph (avoided)."))
QUERY_LATENCY = metrics.register(Histogram("veritas_query_llm_seconds", "Frozen-LLM generation latency per query."))
CITATIONS = metrics.register(Counter("veritas_citations_total", "Citations in answers by verification result."))


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class AppState:
    """Composition root for one tenant: graph, ledger, reconciler, pruner, agent and journal."""

    def __init__(self, ledger_path: Path, tenant: str = DEFAULT_TENANT,
                 ab_jobs: Optional[ab_service.ABJobManager] = None) -> None:
        self.tenant = tenant
        self.graph = TemporalGraphEngine()
        self.ledger = AuditLedger(ledger_path, fsync=_env_flag("LEDGER_FSYNC", True))
        mode = os.getenv("ENTITY_LINKING", "explicit").strip().lower()
        self.linker = EntityLinker("auto" if mode == "auto" else "explicit")
        self.reconciler = ReconciliationEngine(self.graph, self.ledger, linker=self.linker)
        self.pruner = SubgraphPruner(self.graph, self.ledger, linker=self.linker)
        self.agent = VerifiableAgent(self.pruner, self.ledger, build_llm_client_from_env())
        # Raw observation journal (what was *said*, in arrival order). The ledger records what the
        # system *concluded*; the A/B Lab needs the raw stream to build the append-only baselines.
        self.journal_path = ledger_path.with_name("observations.jsonl")
        self.journal: list[dict] = []
        self.ab_jobs = ab_jobs or ab_service.ABJobManager(REPORTS_DIR)
        self.verification: Optional[ChainVerification] = None

    def boot(self) -> ChainVerification:
        verification = self.ledger.load()
        if not verification.valid:
            raise RuntimeError(f"Audit ledger failed verification: {verification.detail}")
        applied = self.reconciler.replay()
        logger.info("[%s] replayed %d ledger records from %s", self.tenant, applied, self.ledger.path)
        if self.journal_path.exists():
            with self.journal_path.open(encoding="utf-8") as fh:
                self.journal = [json.loads(line) for line in fh if line.strip()]
        self.verification = verification
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
        outcomes = [self.reconciler.ingest(t) for t in triples]
        for outcome in outcomes:
            DECISIONS.inc(tenant=self.tenant, decision=outcome.decision)
        return outcomes


class TenantLimitError(RuntimeError):
    pass


class TenantRegistry:
    """Lazily boots one :class:`AppState` per tenant; all share the A/B job manager (one local SLM)."""

    def __init__(self, base_ledger_path: Path, reports_dir: Path, max_tenants: int) -> None:
        self.base = base_ledger_path
        self.max_tenants = max_tenants
        self.ab_jobs = ab_service.ABJobManager(reports_dir)
        self._states: dict[str, AppState] = {}
        self._lock = threading.Lock()

    def path_for(self, tenant: str) -> Path:
        if tenant == DEFAULT_TENANT:
            return self.base
        return self.base.parent / "tenants" / tenant / self.base.name

    def get(self, tenant: str) -> AppState:
        existing = self._states.get(tenant)
        if existing is not None:
            return existing
        with self._lock:
            if tenant in self._states:
                return self._states[tenant]
            if len(self._states) >= self.max_tenants:
                raise TenantLimitError(f"Tenant limit reached ({self.max_tenants}); raise MAX_TENANTS")
            st = AppState(self.path_for(tenant), tenant, self.ab_jobs)
            st.boot()
            self._states[tenant] = st
            return st

    def loaded(self) -> list[AppState]:
        return list(self._states.values())


def _parse_api_keys(raw: str) -> dict[str, Optional[str]]:
    keys: dict[str, Optional[str]] = {}
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        key, _, tenant = item.partition(":")
        tenant = tenant.strip().lower() or None
        if tenant is not None and not _TENANT_RE.match(tenant):
            raise ValueError(f"VERITAS_API_KEYS: invalid tenant id {tenant!r}")
        keys[key.strip()] = tenant
    return keys


registry: TenantRegistry
state: AppState  # the default tenant (kept for single-tenant callers)
API_KEYS: dict[str, Optional[str]] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global registry, state, API_KEYS
    API_KEYS = _parse_api_keys(os.getenv("VERITAS_API_KEYS", ""))
    registry = TenantRegistry(AUDIT_LOG_PATH, REPORTS_DIR, int(os.getenv("MAX_TENANTS", "100")))
    state = registry.get(DEFAULT_TENANT)
    if API_KEYS:
        logger.info("API key authentication enabled (%d keys)", len(API_KEYS))
    yield


app = FastAPI(
    title="VERITAS — Verifiable Knowledge State Editing",
    version="1.1.0",
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


@app.middleware("http")
async def access_control_and_metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    if request.method != "OPTIONS":
        denial = _authorize(request)
        if denial is not None:
            _observe_http(request, denial.status_code, started)
            return denial
    response = await call_next(request)
    _observe_http(request, response.status_code, started)
    return response


def _observe_http(request: Request, status: int, started: float) -> None:
    route = request.scope.get("route")
    template = getattr(route, "path", None) or "unmatched"
    HTTP_REQUESTS.inc(method=request.method, route=template, status=str(status))
    HTTP_LATENCY.observe(time.perf_counter() - started, method=request.method, route=template)


def _authorize(request: Request) -> Optional[JSONResponse]:
    """Resolve the tenant into ``request.state.tenant``; return an error response to deny."""
    requested = request.headers.get(TENANT_HEADER, "").strip().lower() or None
    if requested is not None and not _TENANT_RE.match(requested):
        return JSONResponse({"detail": f"Invalid {TENANT_HEADER}: use 1–63 of a-z, 0-9, '-', '_'"}, status_code=400)
    tenant = requested or DEFAULT_TENANT
    if API_KEYS and request.url.path not in _PUBLIC_PATHS:
        token = request.headers.get("X-API-Key", "").strip()
        auth = request.headers.get("Authorization", "")
        if not token and auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        if token not in API_KEYS:
            return JSONResponse({"detail": "Missing or invalid API key"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
        bound = API_KEYS[token]
        if bound is not None:
            if requested is not None and requested != bound:
                return JSONResponse({"detail": f"This API key is bound to tenant {bound!r}"}, status_code=403)
            tenant = bound
        request.state.key_bound = bound is not None
    request.state.tenant = tenant
    return None


def tenant_state(request: Request) -> AppState:
    tenant = getattr(request.state, "tenant", DEFAULT_TENANT)
    try:
        return registry.get(tenant)
    except TenantLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


Tenant = Depends(tenant_state)


def _tenant_gauges() -> GaugeCallback:
    def collect():  # type: ignore[no-untyped-def]
        for st in registry.loaded():
            stats = st.graph.stats()
            yield {"tenant": st.tenant, "kind": "ledger_records"}, len(st.ledger)
            yield {"tenant": st.tenant, "kind": "entities"}, stats["entities"]
            yield {"tenant": st.tenant, "kind": "edges_active"}, stats["edges_active"]
            yield {"tenant": st.tenant, "kind": "edges_historical"}, stats["edges_historical"]
            yield {"tenant": st.tenant, "kind": "aliases"}, len(st.linker.aliases())
            valid = st.verification.valid if st.verification else True
            yield {"tenant": st.tenant, "kind": "ledger_chain_valid"}, 1.0 if valid else 0.0
    return GaugeCallback("veritas_tenant_state", "Per-tenant graph and ledger size.", collect)


metrics.register(_tenant_gauges())


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health(request: Request) -> dict:
    st = tenant_state(request)
    return {
        "status": "ok",
        "tenant": st.tenant,
        "llm_model": st.agent.client.model_name,
        "ledger_records": len(st.ledger),
        "graph": st.graph.stats(),
    }


@app.get("/api/ready")
def ready() -> Response:
    """Readiness probe: the default tenant's ledger loaded and its hash chain verified at boot."""
    ok = state.verification is not None and state.verification.valid
    body = {"ready": ok, "ledger_records": len(state.ledger)}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/api/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/api/tenants")
def tenants(request: Request) -> dict:
    """Tenants loaded in this process (hidden from tenant-bound API keys)."""
    if getattr(request.state, "key_bound", False):
        raise HTTPException(status_code=403, detail="Tenant-bound keys cannot list tenants")
    return {"tenants": [{"id": st.tenant, "ledger_records": len(st.ledger), **st.graph.stats()}
                        for st in registry.loaded()]}


# --------------------------------------------------------------------------- #
# Knowledge state
# --------------------------------------------------------------------------- #


@app.post("/api/observe", response_model=ObservationResponse)
def observe(req: ObservationRequest, st: AppState = Tenant) -> ObservationResponse:
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
        outcomes = st.ingest(triples)
    except (ValueError, GraphEngineError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    journal_text = req.text if req.text and not req.subject else f"{req.subject} | {req.predicate} | {req.object}"
    st.record(journal_text, ts, req.source, req.confidence)
    return ObservationResponse(provenance_id=provenance_id, extracted_triples=triples, outcomes=outcomes)


@app.get("/api/graph", response_model=GraphResponse)
def graph(
    timestamp: Optional[datetime] = Query(default=None, description="Valid-time instant; omit for current state"),
    as_known_at: Optional[datetime] = Query(default=None, description="System-time instant (bi-temporal)"),
    include_history: bool = Query(default=False, description="Also return invalidated / not-yet-valid edges"),
    st: AppState = Tenant,
) -> GraphResponse:
    ts = ensure_utc(timestamp) if timestamp else None
    known = ensure_utc(as_known_at) if as_known_at else None
    g = st.graph

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
def timeline(st: AppState = Tenant) -> TimelineResponse:
    return TimelineResponse(timestamps=st.graph.timeline())


@app.get("/api/audit", response_model=AuditLedgerResponse)
def audit(
    entity_id: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=10000),
    st: AppState = Tenant,
) -> AuditLedgerResponse:
    records = st.ledger.for_entity(entity_id) if entity_id else st.ledger.records()
    return AuditLedgerResponse(count=len(records), head_hash=st.ledger.head_hash, records=records[-limit:])


@app.get("/api/audit/verify", response_model=ChainVerification)
def verify_audit(st: AppState = Tenant) -> ChainVerification:
    st.verification = st.ledger.verify()
    return st.verification


@app.get("/api/provenance/{entity_id}", response_model=ProvenanceTrail)
def provenance(entity_id: str, st: AppState = Tenant) -> ProvenanceTrail:
    try:
        return st.reconciler.get_provenance_trail(entity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown entity {entity_id!r}") from exc


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest, st: AppState = Tenant) -> QueryResponse:
    try:
        response = st.agent.answer(req)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    m = response.metrics
    QUERY_TOKENS.inc(m.pruned_context_tokens + m.audit_trail_tokens, tenant=st.tenant, kind="sent")
    QUERY_TOKENS.inc(max(0, m.full_graph_tokens - m.pruned_context_tokens), tenant=st.tenant, kind="avoided")
    QUERY_LATENCY.observe(response.latency_ms / 1000.0, tenant=st.tenant)
    for c in response.citations:
        CITATIONS.inc(tenant=st.tenant, verified=str(c.verified).lower())
    return response


@app.post("/api/entities/alias", response_model=MutationAuditRecord)
def declare_alias(req: AliasRequest, st: AppState = Tenant) -> MutationAuditRecord:
    """Declare that ``alias`` names the same entity as ``canonical`` (an audited ALIAS mutation)."""
    try:
        return st.reconciler.declare_alias(req.alias, req.canonical)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/entities/aliases")
def aliases(st: AppState = Tenant) -> dict:
    return {
        "mode": st.linker.mode,
        "aliases": [
            {"alias": a.alias, "alias_id": a.alias_id, "canonical": a.canonical, "canonical_id": a.canonical_id,
             "mode": a.mode}
            for a in st.linker.aliases()
        ],
    }


def _reset(st: AppState) -> dict:
    with st.graph.lock:
        archived = st.ledger.archive_and_reset()
        st.graph.clear()
        st.linker.clear()
        st.archive_journal()
        st.verification = st.ledger.verify()
    return {"archived_to": str(archived) if archived else None}


@app.post("/api/reset")
def reset(st: AppState = Tenant) -> dict:
    """Archive the current ledger (never deleted) and start an empty graph."""
    return _reset(st)


@app.post("/api/demo/seed")
def seed_demo(st: AppState = Tenant) -> dict:
    """Reset, then replay the scripted T0 → T1 → T2 trajectory."""
    _reset(st)
    decisions: dict[str, int] = {}
    for obs in trajectory():
        st.record(obs.text, obs.timestamp, obs.source, obs.confidence)
        prov = new_uuid()
        for triple in extract_triples(obs.text, confidence=obs.confidence, source=obs.source,
                                      timestamp=obs.timestamp, provenance_id=prov):
            outcome = st.reconciler.ingest(triple)
            DECISIONS.inc(tenant=st.tenant, decision=outcome.decision)
            decisions[outcome.decision] = decisions.get(outcome.decision, 0) + 1
    return {"ledger_records": len(st.ledger), "decisions": decisions, "graph": st.graph.stats()}


# --------------------------------------------------------------------------- #
# A/B Lab
# --------------------------------------------------------------------------- #


@app.get("/api/observations")
def observations(st: AppState = Tenant) -> dict:
    """Raw observations made through this dashboard, in arrival order."""
    return {"count": len(st.journal), "observations": st.journal}


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
        job = registry.ab_jobs.submit(req)
    except ab_service.ObservationError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "errors": exc.errors}) from exc
    except ab_service.BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.view()


@app.get("/api/ab/runs")
def ab_list() -> dict:
    return {"runs": registry.ab_jobs.list()}


@app.get("/api/ab/runs/{job_id}")
def ab_get(job_id: str) -> dict:
    job = registry.ab_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown A/B run {job_id!r}")
    return job.view()


@app.post("/api/ab/runs/{job_id}/cancel")
def ab_cancel(job_id: str) -> dict:
    job = registry.ab_jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown A/B run {job_id!r}")
    return job.view(include_rows=False)
