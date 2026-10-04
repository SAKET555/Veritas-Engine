"""Pydantic v2 data models: graph schema, audit ledger records and API contracts.

All datetimes are normalized to timezone-aware UTC at the model boundary so the
temporal algebra (``t_start <= t < t_end``) is never evaluated across mixed
naive/aware values.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Interpret naive datetimes as UTC and convert aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def entity_id_for(name: str) -> str:
    """Deterministic entity identifier derived from the surface name."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive an entity id from name {name!r}")
    return slug


def new_uuid() -> str:
    return str(uuid.uuid4())


class _UTCModel(BaseModel):
    """Base model that coerces every datetime field to aware UTC."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*", mode="after")
    @classmethod
    def _coerce_datetimes(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return ensure_utc(value)
        return value


# --------------------------------------------------------------------------- #
# Graph schema
# --------------------------------------------------------------------------- #


class EntityNode(_UTCModel):
    id: str
    name: str
    entity_type: str = "Entity"
    attributes: dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=utc_now)


class TemporalEdge(_UTCModel):
    """A bi-temporal fact.

    Valid time  ``[t_start, t_end)``          — when the fact held in the world.
    System time ``[recorded_at, superseded_at)`` — when the system believed it.
    """

    id: str = Field(default_factory=new_uuid)
    source_id: str
    relation: str
    target_id: str
    t_start: datetime
    t_end: Optional[datetime] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    provenance_id: str
    source: str = "agent_observation"
    recorded_at: datetime = Field(default_factory=utc_now)
    superseded_at: Optional[datetime] = None
    invalidated_by: Optional[str] = None
    invalidation_rationale: Optional[str] = None
    corroborating_provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_interval(self) -> "TemporalEdge":
        if self.t_end is not None and self.t_end < self.t_start:
            raise ValueError(
                f"Edge {self.id}: t_end ({self.t_end.isoformat()}) precedes "
                f"t_start ({self.t_start.isoformat()})"
            )
        return self

    @property
    def is_active(self) -> bool:
        return self.t_end is None

    def valid_at(self, ts: datetime) -> bool:
        ts = ensure_utc(ts)
        return self.t_start <= ts and (self.t_end is None or ts < self.t_end)


MutationType = Literal["INSERT", "INVALIDATE", "RECONCILE"]


class MutationAuditRecord(_UTCModel):
    """Immutable, hash-chained ledger entry describing one state transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mutation_id: str = Field(default_factory=new_uuid)
    sequence: int = Field(ge=0)
    transaction_id: str
    timestamp: datetime
    mutation_type: MutationType
    affected_edge_id: str
    entity_ids: list[str]
    deterministic_rationale: str
    prior_state_snapshot: Optional[dict[str, Any]] = None
    new_state_snapshot: dict[str, Any]
    prev_hash: str
    record_hash: str = ""

    @property
    def short_id(self) -> str:
        return self.mutation_id.replace("-", "")[:8]


# --------------------------------------------------------------------------- #
# Internal value objects
# --------------------------------------------------------------------------- #


class TripleObservation(_UTCModel):
    """A single normalized (subject, predicate, object) observation."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source: str = "agent_observation"
    provenance_id: str = Field(default_factory=new_uuid)
    timestamp: datetime = Field(default_factory=utc_now)
    expected_prior_object: Optional[str] = None
    raw_text: Optional[str] = None


ReconciliationDecision = Literal[
    "INSERTED",
    "SUPERSEDED",
    "REINFORCED",
    "RETAINED_INCUMBENT",
    "RETROACTIVE_INSERT",
]


class ReconciliationOutcome(BaseModel):
    triple: TripleObservation
    decision: ReconciliationDecision
    transaction_id: str
    active_edge_id: Optional[str]
    mutations: list[MutationAuditRecord]
    summary: str


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #


class ObservationRequest(_UTCModel):
    """Either free text (parsed by the rule-based extractor) or a structured triple."""

    text: Optional[str] = Field(default=None, description="e.g. 'Alice moved from Mumbai to London'")
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source: str = "user_statement"
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def _require_payload(self) -> "ObservationRequest":
        structured = (self.subject, self.predicate, self.object)
        has_structured = all(structured)
        if not self.text and not has_structured:
            raise ValueError("Provide either 'text' or all of 'subject', 'predicate', 'object'.")
        if any(structured) and not has_structured:
            raise ValueError("Structured observations need 'subject', 'predicate' and 'object'.")
        return self


class ObservationResponse(BaseModel):
    provenance_id: str
    extracted_triples: list[TripleObservation]
    outcomes: list[ReconciliationOutcome]


class GraphNodeView(BaseModel):
    id: str
    name: str
    entity_type: str
    attributes: dict[str, Any]
    first_seen: datetime
    degree_active: int


class GraphEdgeView(BaseModel):
    edge: TemporalEdge
    valid_at_query_time: bool


class GraphResponse(BaseModel):
    timestamp: Optional[datetime]
    as_known_at: Optional[datetime]
    include_history: bool
    nodes: list[GraphNodeView]
    edges: list[GraphEdgeView]
    stats: dict[str, int]


class TimelineResponse(BaseModel):
    timestamps: list[datetime]


class AuditLedgerResponse(BaseModel):
    count: int
    head_hash: str
    records: list[MutationAuditRecord]


class ChainVerification(BaseModel):
    valid: bool
    records_checked: int
    head_hash: str
    first_invalid_sequence: Optional[int] = None
    detail: str


class ProvenanceEntry(BaseModel):
    record: MutationAuditRecord
    narrative: str


class ProvenanceTrail(BaseModel):
    entity: EntityNode
    entries: list[ProvenanceEntry]


class PruningParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k_hops: int = Field(default=2, ge=0, le=6)
    alpha: float = Field(default=0.35, ge=0.0, description="Weight on confidence")
    beta: float = Field(default=0.15, ge=0.0, description="Weight on recency")
    gamma: float = Field(default=0.50, ge=0.0, description="Weight on relational relevance")
    tau: float = Field(default=0.60, ge=0.0, le=1.0, description="Score threshold")
    max_edges: int = Field(default=40, ge=1, le=1000)
    recency_half_life_days: float = Field(default=30.0, gt=0.0)
    hop_decay: float = Field(default=0.6, gt=0.0, le=1.0)
    reverse_edge_penalty: float = Field(default=0.4, gt=0.0, le=1.0)
    include_history: Optional[bool] = Field(
        default=None, description="None = auto-detect historical intent from the question"
    )
    output_format: Literal["markdown", "json"] = "markdown"


PromptStyle = Literal["compact", "standard"]


class QueryRequest(_UTCModel):
    question: str = Field(min_length=1)
    seed_entities: Optional[list[str]] = None
    as_of: Optional[datetime] = None
    pruning: PruningParams = Field(default_factory=PruningParams)
    prompt_style: PromptStyle = Field(
        default="compact",
        description="compact: short instructions + [n] citation tags (ledger-mapped server-side); "
        "standard: full M:id citations plus the rationale audit trail in the prompt",
    )


class ScoredEdge(BaseModel):
    edge_id: str
    source: str
    relation: str
    target: str
    active: bool
    confidence: float
    recency: float
    relevance: float
    score: float
    kept: bool
    reason: str


class TokenMetrics(BaseModel):
    tokenizer: str
    full_graph_tokens: int
    pruned_context_tokens: int
    audit_trail_tokens: int
    prompt_tokens_total: int
    reduction_ratio: float
    full_graph_edges: int
    pruned_edges: int
    model_prompt_tokens: Optional[int] = Field(default=None, description="Prompt tokens under the model's tokenizer")
    model_completion_tokens: Optional[int] = None


class Citation(BaseModel):
    short_id: str
    verified: bool
    alias: Optional[str] = Field(default=None, description="Prompt-local tag, e.g. '2' for [2] (compact style)")
    mutation_id: Optional[str] = None
    mutation_type: Optional[MutationType] = None
    rationale: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    prompt_style: str = "standard"
    latency_ms: float
    grounded: bool
    citations: list[Citation]
    seed_entities: list[str]
    include_history: bool
    pruned_context: str
    audit_context: str
    system_prompt: str
    scored_edges: list[ScoredEdge]
    metrics: TokenMetrics
    model_timings_ms: dict[str, float] = Field(default_factory=dict)
