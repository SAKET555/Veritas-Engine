# VERITAS — Technical Documentation

**Verifiable Knowledge State Editing for Long-Horizon AI Agents**

This document describes the full system: the problem it addresses, the architecture and algorithms, how each component
works, how it was evaluated (including results against a real local small language model), its current limitations,
and directions for future work. Every number quoted here was produced by code in this repository and can be reproduced
with the commands in [§13](#13-reproducing-the-results).

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Design thesis](#2-design-thesis)
3. [System architecture](#3-system-architecture)
4. [Data model](#4-data-model)
5. [Bi-temporal graph engine](#5-bi-temporal-graph-engine)
6. [Reconciliation and the audit ledger](#6-reconciliation-and-the-audit-ledger)
7. [Statement extraction](#7-statement-extraction)
8. [Confidence-driven subgraph pruning](#8-confidence-driven-subgraph-pruning)
9. [Frozen-LLM agent harness](#9-frozen-llm-agent-harness)
10. [API and dashboard](#10-api-and-dashboard)
11. [Evaluation](#11-evaluation)
12. [Current status, limitations and future work](#12-current-status-limitations-and-future-work)
13. [Reproducing the results](#13-reproducing-the-results)
14. [References](#14-references)

---

## 1. Problem statement

A long-horizon agent keeps working in an environment that keeps changing. People relocate, companies move their
headquarters, a server is moved to another rack, and a user changes a preference. The agent must answer questions about
the **current** state of the world, sometimes about **past** states, and ideally explain **why** it believes what it
believes.

The three common ways of giving a frozen language model memory each fail this requirement in a different way.

**Append-only transcripts (full history).** The simplest agent memory pastes everything it was ever told into the
prompt. Nothing is lost, but nothing is resolved either. Old and new values coexist, and the model must reconcile them
at inference time on every call. The prompt grows without limit, so latency and cost grow with it. Model accuracy is
known to degrade when relevant information sits in the middle of long contexts (Liu et al., 2024).

**Dense vector retrieval (RAG).** Retrieval-augmented generation (Lewis et al., 2020) embeds stored text and fetches
the passages most similar to the question. Embeddings capture *semantic* similarity, not *temporal validity*. "Alice
lives in Mumbai" and "Alice lives in London" are near neighbours in embedding space, so the retriever returns both, or
returns only the stale one. Nothing in the index records that one fact superseded the other.

**Parametric knowledge editing.** Methods such as ROME and MEMIT (Meng et al., 2022; 2023) rewrite model weights to
change a stored fact. They are elegant, but they are opaque: you cannot easily list what changed, prove when it changed,
or roll it back. Edits are also known to interact with each other and to degrade the model as they accumulate. They
require white-box access to the weights, which rules out hosted models.

What is missing is an **explicit, inspectable, versioned representation of the world state** that sits outside the
model. It must resolve conflicts deterministically, keep history instead of deleting it, and justify every answer with
evidence that can be verified.

---

## 2. Design thesis

VERITAS separates **execution** (a frozen language model that reads and writes text) from **knowledge state** (an
external, non-parametric store that is edited explicitly). The store has four properties:

1. **Bi-temporal.** Every fact records *when it was true in the world* (valid time) and *when the system learned or
   revised it* (system time). This follows the classic bi-temporal model from temporal databases (Snodgrass, 1999;
   Jensen & Snodgrass, 1999). Its use for agent memory is inspired by Graphiti, the temporal knowledge graph behind
   Zep (Rasmussen et al., 2025).
2. **Reconciled, not accumulated.** When a new observation conflicts with an existing fact, deterministic rules decide
   the outcome. A superseded fact is *closed* (its validity interval gets an end time), never deleted.
3. **Audited.** Every state transition is written to an append-only ledger in which each record carries the SHA-256
   hash of its predecessor, the same chaining idea as Haber and Stornetta's (1991) timestamping scheme. Tampering with
   or deleting any record is detectable, and the graph can be rebuilt from the ledger alone.
4. **Pruned before prompting.** At question time, only a small, relevant, currently valid subgraph is serialized into
   the prompt, and the model must cite the ledger records behind each claim. Those citations are verified after
   generation.

The model is never fine-tuned. Updating knowledge means writing a ledger record, which is cheap, instant, inspectable
and reversible.

---

## 3. System architecture

```
                     ┌────────────────────────────── backend (FastAPI, Python 3.10+) ─────────────────────────────┐
 observation ──────► │ extraction.py ─► reconciliation.py ─► graph_engine.py (NetworkX MultiDiGraph)               │
 "Alice moved to     │  rule-based       rules R1–R5,          valid time [t_start, t_end)                         │
  London"            │  triples          atomic txn            system time [recorded_at, superseded_at)            │
                     │                       │                                                                     │
                     │                       └──► AuditLedger (JSONL, SHA-256 chain, replayable)                   │
                     │                                                                                             │
 question ─────────► │ pruning.py: seeds ─► k-hop ego ─► composite score ─► τ filter ─► serialize (standard|compact)│
                     │ agent.py:   prompt = facts (+ audit trail) + question ─► frozen LLM ─► citation verification │
                     │ ab_eval.py / ab_service.py: A/B harness (full history vs vector RAG vs VERITAS)             │
                     └─────────────────────────────────────────────────────────────────────────────────────────────┘
                                             ▲  REST /api/*
                     ┌───────────────────────┴──────── frontend (React 18, Vite, Tailwind, JavaScript) ────────────┐
                     │ graph canvas · time-travel slider · audit feed · provenance drawer · agent console · A/B Lab │
                     └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

| File | Responsibility |
|---|---|
| `backend/models.py` | Pydantic v2 schema: entities, bi-temporal edges, audit records, API bodies. All datetimes are normalized to UTC. |
| `backend/ontology.py` | Predicate registry: cardinality (single- or multi-valued), types, lexical aliases. |
| `backend/graph_engine.py` | Temporal graph: insert, invalidate, point-in-time and bi-temporal queries, active-edge index. |
| `backend/reconciliation.py` | Conflict rules, transactions with rollback, hash-chained ledger, provenance trail, replay. |
| `backend/extraction.py` | Deterministic sentence-to-triple parser. |
| `backend/pruning.py` | Ego expansion, composite scoring, filtering, standard and compact serializers. |
| `backend/agent.py` | LLM clients (mock, Ollama, OpenAI-compatible), prompt plans, citation verification. |
| `backend/main.py` | FastAPI app, observation journal, A/B Lab endpoints. |
| `backend/benchmark.py` | Offline structural benchmark against dense vector RAG. |
| `backend/ab_eval.py` | Head-to-head harness against a real local SLM. |
| `backend/ab_service.py` | Background A/B jobs and question generation for the dashboard. |
| `backend/scenarios.py` | The shared deterministic T0 → T1 → T2 trajectory. |
| `backend/baselines.py` | Comparison memories: vector, hybrid BM25 + dense, static graph RAG, overwrite memory, full history. |
| `backend/datasets.py` | Synthetic multi-session dataset with independent ground truth (§11.6). |
| `backend/temporal_bench.py` | Offline benchmark of every memory with one shared reader, plus scaling mode (§11.6, §11.7). |
| `backend/entity_linking.py` | Declared and automatic aliases, recorded as `ALIAS` ledger mutations (§7.1). |
| `backend/observability.py` | Prometheus metrics (§10.1). |
| `frontend/src/` | Dashboard (graph view and A/B Lab). |

The codebase has roughly 7,100 lines of Python, 1,900 lines of tests and 2,500 lines of JSX/JS.

---

## 4. Data model

**EntityNode:** `id` (a deterministic slug of the name, e.g. `"Stark Industries"` → `stark-industries`), `name`,
`entity_type`, `attributes`, `first_seen`.

**TemporalEdge**, one fact:

| Field | Meaning |
|---|---|
| `id` | UUID. Parallel facts between the same pair of entities are separate edges. |
| `source_id`, `relation`, `target_id` | The triple. |
| `t_start`, `t_end` | Valid-time interval `[t_start, t_end)`. `t_end = None` means currently true. |
| `recorded_at`, `superseded_at` | System-time interval: when the system learned the fact and when it learned that the fact had ended. |
| `confidence_score` | In [0, 1]. |
| `provenance_id`, `source` | The originating observation and its source class. |
| `invalidated_by`, `invalidation_rationale` | Link to the superseding edge and the reason. |
| `corroborating_provenance` | Later observations that confirmed this fact. |

A model validator rejects `t_end < t_start`. Validity is half-open: a fact that ends at `T` is not valid at `T`, so
consecutive values never overlap.

**MutationAuditRecord**, one ledger entry, immutable (`frozen=True`):

- `mutation_id`, `sequence`, `transaction_id`, `timestamp`
- `mutation_type`: one of `INSERT`, `INVALIDATE`, `RECONCILE`
- `affected_edge_id`, `entity_ids`
- `deterministic_rationale`: a human-readable explanation that includes the numbers used in the decision
- `prior_state_snapshot` and `new_state_snapshot`: the full edge plus both entities, before and after
- `prev_hash` and `record_hash`

---

## 5. Bi-temporal graph engine

`TemporalGraphEngine` wraps a `networkx.MultiDiGraph`. Edges are keyed by their UUID, so the graph can hold a
historical and a current `works_at` edge between the same two nodes. Three indexes are maintained alongside the graph:

- `edge_id → (u, v)`, for O(1) lookup;
- the set of active edge IDs, so `get_active_subgraph()` costs O(active), not O(history);
- `(subject, relation) → edge IDs`, used for conflict detection.

A re-entrant lock serializes all mutations. The reconciler takes the same lock to make multi-step transitions atomic.

**Core operations:**

- `add_temporal_edge(source, relation, target, confidence, provenance_id, timestamp)` opens `[timestamp, ∞)`, or a
  closed interval when `t_end` is given (used for backfilled history).
- `invalidate_edge(edge_id, end_time, rationale)` closes the interval and stamps `superseded_at`. It refuses to close
  an already-closed edge, or to close an edge before its start. The edge and both nodes remain in the graph.
- `query_point_in_time(ts)` returns the subgraph where `t_start ≤ ts < (t_end or ∞)`.
- `query_point_in_time(ts, as_known_at=K)` is a true **bi-temporal** query: what the system *believed at time K* about
  the world at time `ts`. An edge recorded after `K` is invisible. An invalidation learned after `K` is ignored, so the
  fact still appears open.
- `timeline()` returns the sorted distinct transition points (T0 … Tn) that drive the dashboard's time-travel slider.

Entities are refined monotonically. A generic type is upgraded to a specific one (`Entity` → `Organization`) but never
downgraded, and `first_seen` only ever moves earlier.

---

## 6. Reconciliation and the audit ledger

### 6.1 Cardinality

The ontology marks each predicate as **functional** (one current value per subject: `lives_in`, `works_at`,
`has_role`, `reports_to`, `married_to`, `prefers`, `headquartered_in`, `located_in`, `has_status`) or
**multi-valued** (`owns`, `knows`, `member_of`). Unknown predicates default to functional. That is the conservative
choice for a state store, because an attribute like `favorite_color` is far more likely to be single-valued, and the
default guarantees that two contradictory values are never both current.

### 6.2 Decision rules

For an incoming observation `(s, p, o, t, confidence, source)`, the **effective confidence** is
`eff = confidence × authority(source)`. The authority values are:

| Source | Authority |
|---|---|
| `system_of_record` | 1.00 |
| `sensor` | 0.95 |
| `user_statement` | 0.90 |
| `agent_observation` | 0.85 |
| `web` | 0.70 |
| `agent_inference` | 0.60 |
| `hearsay` | 0.40 |
| anything else | 0.75 |

| Rule | Condition | Outcome |
|---|---|---|
| **R1** | The identical fact is already active. | `RECONCILE`: confidence is fused by noisy-OR, `1 − (1 − c_old)(1 − c_new)` (Pearl, 1988), capped at 0.999, and the provenance is recorded as corroboration. |
| **R2** | `p` is multi-valued. | `INSERT`. No conflict is possible. |
| **R3** | `p` is functional and recorded history starts *after* `t` (a late-arriving observation). | If a recorded fact covers `t` with a different object, history wins and the candidate is logged and rejected. Otherwise the fact is inserted as a closed interval `[t, next_start)`. Current state is unchanged. |
| **R4** | `p` is functional and has an active incumbent. | **Newer valid time:** supersede if `eff_new ≥ eff_old × (1 − 0.35)`. **Equal valid time:** supersede only if `eff_new > eff_old`. Superseding emits `INVALIDATE` (old edge, `t_end = t`) and `INSERT` (new edge) in one transaction. Otherwise a `RECONCILE` record logs the rejected candidate with no state change. |
| **R5** | Functional, with no value yet. | `INSERT`. |

The 0.35 tolerance encodes a deliberate prior: newer information should usually win, but not when it comes from a much
weaker source. In the demo trajectory, a hearsay claim that Alice moved to Paris (0.5 × 0.40 = 0.20) cannot displace a
user statement that she lives in London (0.81; the bar is 0.81 × 0.65 = 0.53). A later sensor reading of London
corroborates it, and its confidence rises to 0.995.

If an observation says "moved **from** Mumbai to London", the extractor records Mumbai as the *expected prior value*.
The rationale then states whether the ledger agreed, which surfaces a discrepancy instead of silently overwriting it.

### 6.3 Atomicity

Each `ingest()` builds a transaction: a list of draft records plus compensating *undo* actions (remove the inserted
edge, restore the prior edge, drop newly created orphan entities). The drafts are sealed into the ledger and written
with `flush()` and `fsync()`. If the write fails, the file is truncated back to its previous size and the graph changes
are undone. The graph and the ledger therefore never diverge. A test simulates a full disk to verify this.

### 6.4 Hash chain, verification and replay

Each record is hashed over its canonical JSON (sorted keys, compact separators, with `record_hash` excluded). Its
`prev_hash` is the previous record's hash, and the first record links to 64 zeros. `verify()` checks sequence
continuity, the links and every hash, and reports the first bad record. Editing a rationale, deleting a line or
reordering lines are all detected.

Because every record stores a complete post-state snapshot, `replay()` rebuilds the graph from the ledger alone. The
server does this on every start, after verifying the chain, and refuses to boot on a broken chain. A test confirms that
a replayed graph is byte-identical to the original.

Resetting the dashboard never deletes data: the ledger and the raw observation journal are moved aside as timestamped
archives.

---

## 7. Statement extraction

`extraction.py` maps natural-language observations to triples with an ordered list of regular-expression rules.

**What the rules cover:**

- **Places:** relocation ("moved from X to Y", "relocated to Y"), residence ("lives in", "is now living in"),
  headquarters ("is headquartered in", "moved its headquarters to") and location ("is located in").
- **Work:** job changes ("left X for Y"), employment ("works at", "joined", "is currently working at"), a role at an
  organization ("is a senior data scientist at Acme", which yields *two* triples) and reporting lines.
- **Relationships and possessions:** marriage, ownership, acquaintance and membership.
- **Attributes:** preferences, status and possessive attributes ("Priya's favorite food is dosa").

Rule order matters, and tests pin it down. For example, an early version read "is currently working at Initech" as a
job title of "Working", until the employment rule was moved ahead of the role rule. Anything the rules can't parse can
be entered explicitly as `Subject | predicate | Object`. Determinism is deliberate: the same text must always produce
the same triples, otherwise the ledger would not be reproducible.

### 7.1 Entity linking

Before reconciliation, subject and object pass through `EntityLinker`:

1. A **declared alias** (`POST /api/entities/alias`) maps a surface name to a canonical entity. Aliases never chain:
   aliasing to an alias points at its canonical entity, and cycles are rejected.
2. In **auto mode** (`ENTITY_LINKING=auto`), a name that is not yet an entity is normalized by stripping honorifics
   and legal-form suffixes. If exactly one existing entity has the same normalized form, the name is linked to it.
   Zero or several matches mean no link.

A new link is written as an `ALIAS` mutation in the *same transaction* as the fact that triggered it. If the ledger
write fails, the link is undone with everything else. Replay rebuilds the alias table from those records, and the
pruner resolves aliases when finding seeds in a question.

---

## 8. Confidence-driven subgraph pruning

Serializing the whole graph into every prompt would recreate the append-only problem. The pruner selects a small,
relevant, consistent subgraph instead.

**1. Seed resolution.** Entities named in the question are found by longest-match first, with word boundaries and
possessives ("Alice Smith" beats "Alice"; "Alice's" matches "Alice"). If nothing matches, every node becomes a weak
seed.

**2. Ego expansion with path weights.** A radius-k traversal (default k = 2) gives each node a weight. Seeds weigh 1.
A forward hop multiplies the weight by `hop_decay` (0.6). Walking an edge against its direction multiplies it by
`hop_decay × reverse_edge_penalty` (0.6 × 0.4). This keeps chains the question follows (Alice → Acme → Munich) while
damping fan-in through hubs (everyone else who lives in London).

**3. Composite score.**

```
Score(e)      = (α·Confidence(e) + β·Recency(e) + γ·Relevance(e)) / (α + β + γ)
Recency(e)    = 2^(−age_days / half_life)       age measured from the fact's last moment of validity
Relevance(e)  = structural(e) · (0.3 + 0.7 · lexical(e))
structural(e) = max(w(source), w(target) · reverse_edge_penalty)
lexical(e)    = 1   if the question uses the relation's name or an alias ("employer" → works_at)
                0.5 if the question names the target entity
                0   otherwise (treated as 1 if the question names no relation at all)
```

The defaults are α = 0.35, β = 0.15, γ = 0.5, τ = 0.6, a recency half-life of 30 days and at most 40 edges.

**4. Filtering.** Invalidated facts are dropped unless the question has historical intent. That intent is either set
explicitly or auto-detected from cues such as *before*, *previously* and *why*. Facts scoring below τ are dropped, the
top `max_edges` are kept, and isolated nodes are removed. Every candidate edge is returned with its score components
and the reason it was kept or dropped, and the dashboard can show this breakdown.

**5. Serialization.** There are two formats:

*Standard* (the default):

```
## KG facts as_of now (2)
fmt: subject -relation-> object | conf | valid from..to | cite
- Alice -lives_in-> London | 0.99 | 2026-03-01T09:00Z..now | M:c19ee4e7
- Alice -lives_in-> Mumbai | 0.95 | 2026-01-05T09:00Z..2026-03-01T09:00Z [INVALIDATED] | M:c21e0940
```

This is followed by a *mutation audit trail*: the ledger records behind the facts the question asks about, with
rationales truncated to 240 characters.

*Compact* (opt-in):

```
Facts:
[1] Alice -lives_in-> London (since 2026-03-01)
[2] Alice -lives_in-> Mumbai (ENDED 2026-03-01, was true from 2026-01-05; replaced by London)
```

The compact format drops the confidence column (everything shown already passed τ) and the rationale text (it stays in
the ledger). It replaces 8-hex mutation IDs with prompt-local `[n]` tags and adds explicit "replaced by" links. Past
facts are placed newest-first, right after the current value, so "what came before X" becomes a lookup. On the demo
questions this cuts VERITAS prompts by about 73%. Its accuracy trade-off is discussed in §11.3.

On the demo graph (86 facts, 12 of them historical), the standard pruned context averages **120 tokens, against 3,515
for the full graph, a 96.6% reduction**.

### 8.1 Optional retrieval settings

Three settings are off by default, so the behaviour above and every number measured with it are unchanged.

- **Ego-local candidates** (`candidate_mode="ego"`). The default scores every fact in the history, so query cost grows
  with the whole graph. Since confidence and recency are at most 1, a fact cannot reach τ unless its structural
  weight is at least `w_min = (τ·(α+β+γ) − α − β) / γ` (0.2 with the defaults). The ego walk therefore stops
  expanding at nodes below `w_min`, and only facts touching a node at or above it are scored. The kept set is
  provably identical, and a test checks it on a synthetic graph. Facts skipped this way are counted in
  `edges_unscored`. The engine keeps insertion-order, relation and latest-time indexes so this path never scans the
  history.
- **Hub damping** (`hub_damping`). Walking backwards out of a non-seed node with fan-in `d > 1` is divided by
  `d^hub_damping`. This applies to both the propagated node weights and the structural score of edges into that node.
  At 0.5, a city shared by 900 people contributes 1/30 of its undamped weight to each of them. Seeds are never damped:
  "Who lives in London?" still returns London's residents.
- **Dates in questions** (`QueryRequest.detect_as_of`). ISO dates and "March 3, 2025" / "3 March 2025" forms become
  `as_of`. A bare date means midday UTC; two different dates are treated as ambiguous and ignored.

These three switches are method **B3** in the A/B harness and "VERITAS optimized" in the temporal benchmark.

---

## 9. Frozen-LLM agent harness

`VerifiableAgent.plan(request)` builds a `PromptPlan` containing the system prompt, the user prompt, the fact block, the
audit block, the set of mutation IDs the model may cite and (for compact style) the tag-to-ID alias map.
`answer(request)` then calls the model, verifies citations and returns the answer, the context, per-edge scores and
token metrics.

**System prompt (standard).** Answer strictly from the supplied facts. Never present invalidated facts as current. End
every claim with `[M:xxxxxxxx]`. Reply `INSUFFICIENT_CONTEXT` when the context lacks the answer.

**Citation verification.** Every `M:xxxxxxxx` and every `[n]` tag in the reply is resolved. It is marked *verified*
only if both hold:

1. It resolves to a real ledger record.
2. That record was actually supplied in this prompt.

Rule 2 matters: a model that cites a genuine but irrelevant ledger record is flagged as ungrounded. An answer is
*grounded* if it abstains, or if it has at least one citation and every citation is verified. The parser tolerates
small-model formatting drift, such as `M: AB12CD34`, upper case and `[1, 3]`.

**LLM clients**, all behind one interface, `complete(system, user) → LLMCompletion`:

- **`MockLLMClient`.** A deterministic extractive reader that only sees the prompt text. It parses the serialized
  facts, ranks them lexically against the question, follows one hop for multi-hop questions and emits cited sentences.
  It makes the whole system and test suite runnable offline.
- **`OllamaClient`.** Uses the native `/api/chat` endpoint, or `/api/generate`. It always sets `num_ctx` explicitly,
  because Ollama otherwise silently truncates long prompts. It reports the model's own `prompt_eval_count` and the
  server-side load, prompt-evaluation and generation durations, and flags context saturation. Its warm-up uses the same
  `num_ctx` as real calls, because Ollama reloads the model whenever the context size changes. That pitfall put a
  24-second reload inside the first measured call before it was fixed.
- **`OpenAICompatibleClient`.** For llama.cpp's `llama-server`, OpenAI, LiteLLM or vLLM. It parses `usage` and
  llama.cpp's own `timings` block.

The provider is chosen with `LLM_PROVIDER = mock | ollama | llamacpp | openai | litellm`.

---

## 10. API and dashboard

**REST API (FastAPI):**

| Endpoint | Purpose |
|---|---|
| `POST /api/observe` | Text or a structured triple → reconciliation outcomes. The raw statement is also journaled. |
| `GET /api/graph?timestamp=&include_history=&as_known_at=` | Current, point-in-time or bi-temporal topology. |
| `GET /api/timeline` | The valid-time transition points. |
| `GET /api/audit`, `GET /api/audit/verify` | The ledger and its chain verification. |
| `GET /api/provenance/{entity}` | An entity's chronological mutation trail with narratives. |
| `POST /api/query` | Prune → LLM → cited answer, plus metrics. |
| `POST /api/demo/seed`, `POST /api/reset` | Load the demo trajectory; archive and start empty. |
| `GET /api/observations` | The raw observation journal. |
| `/api/ab/*` | Models, preset, `suggest-queries`, and starting, polling and cancelling runs. |

**Dashboard (React 18, Vite, Tailwind CSS, JavaScript):**

- **Graph view:**
  - a force-directed canvas (`react-force-graph-2d`): active edges emerald, invalidated edges dashed rose, edges that
    were in the last prompt in sky blue with particles;
  - a time-travel slider over T0 … Tn, with replay;
  - an audit feed with snapshots and hash links;
  - a provenance drawer for any node, and an agent console that shows the pruned context, token savings and the cited
    answer side by side.
- **A/B Lab:** users enter their own ordered facts (typed, pasted, imported from the dashboard journal, or the demo)
  and their own questions, or generate questions automatically. They then choose a model and methods and watch
  accuracy, accuracy on changed facts, prompt tokens, median latency, stale-answer rate and verified citations stream
  in. Every answer cell opens the exact prompt and reply.

**Automatic question generation.** The user's facts are reconciled. Every functional slot whose value changed yields a
"current value" question, with every superseded value counted as a wrong answer, and a "what came before" question.
Unchanged slots become stable controls. Ground truth therefore follows VERITAS's own rules, and the UI tells the user
to review it.

### 10.1 Cloud deployment

The deployment features are all optional. With none configured, the server behaves exactly as a single local process.

| Concern | Mechanism |
|---|---|
| Tenancy | `X-Tenant-ID` selects a tenant (1–63 of `a-z 0-9 - _`). Each tenant is booted lazily as its own graph, ledger and journal under `data/tenants/<id>/`, capped by `MAX_TENANTS`. The default tenant keeps the original ledger path. |
| Authentication | `VERITAS_API_KEYS="k1,k2:acme"` requires `X-API-Key` or a bearer token. A key bound to a tenant cannot address another tenant or list tenants. Health and readiness probes stay public. |
| Observability | `GET /api/metrics` in Prometheus text format, with no client library: HTTP request count and latency histogram by route template (bounded cardinality), reconciliation decisions, context tokens sent vs avoided, LLM latency, citation verification, and per-tenant ledger and graph size. |
| Probes | `/api/health` (liveness) and `/api/ready` (ledger loaded and hash chain verified, else 503). |
| Durability | `LEDGER_FSYNC=1` (default) fsyncs every transaction. Setting it to 0 trades crash durability for ingest throughput (§11.7). |
| Packaging | `backend/Dockerfile` (non-root, one worker, health check), `frontend/Dockerfile` (nginx serving the build and proxying `/api`), `docker-compose.yml` (optional Ollama under the `llm` profile). |

Because each tenant's state is an in-memory projection of its ledger, a tenant must be served by exactly one process.
Scale out by routing tenants to replicas, each with its own volume, for example consistent hashing on `X-Tenant-ID` at
the load balancer. The ledger stays the source of truth: a replica that restarts rebuilds its tenants by replay.

---

## 11. Evaluation

### 11.1 Structural benchmark (offline, deterministic)

`benchmark.py` feeds the 88-observation demo trajectory to dense vector RAG and to VERITAS. It covers T0 initial
facts, T1 relocations, a job change, a rumor, a corroboration and a late-arriving historical fact, and T2 a preference
change and an HQ move. The RAG side uses a deterministic hashed dense encoder (384 dimensions; optionally
sentence-transformers) and a top-1 reader. The VERITAS side uses the mock reader.

| Metric | Dense vector RAG | VERITAS |
|---|---|---|
| Accuracy | 0% | 100% |
| Epistemic consistency | 0% | 100% |
| Stale fact presented | 83% | 0% |
| Contradictory context | 83% | 0% |
| Verifiable (cited and chained) | 0% | 100% |

This benchmark demonstrates the *structural* failure, a temporally blind index retrieving stale and fresh facts
together. It is not a fair accuracy comparison, because both readers are simple. The real-model comparison is §11.2.

### 11.2 Head-to-head with a real local SLM

**Setup:** llama3.2 3B (Q4_K_M) via Ollama, CPU inference, temperature 0, seed 42. There are 11 questions: 5 about
changed facts, 1 multi-hop, 3 historical (2 of them added afterwards as held-out questions) and 2 stable controls,
each asked 3 times, for **33 graded answers per method**.

All methods receive the same 88 observations and the same answer format (`ANSWER:` / `EXPLANATION:`). The baselines
are explicitly told that facts change over time and that they must report the current state, so the comparison
measures the memory representation, not weak instructions.

**Measurement controls:**

- **Cold prompts.** Each request starts with a unique ID so the server's prompt cache can't reuse earlier prompt
  prefixes. Without this, Ollama's logs showed thousands of cached tokens being reused between append-only prompts,
  which undercounts their real cost.
- **Rotating order.** The order of methods rotates for each question.
- **Warm-up.** The model is warmed up with identical options before timing starts.
- **Explicit context size.** `num_ctx` is set to the smallest 1024-aligned window that fits every prompt.
- **Strict grading.** An answer is correct only if its `ANSWER:` line contains the true value and none of the stale
  values, so "Berlin or Munich" is wrong.

| Method | Accuracy | Changed facts | Historical | Stale answers | Prompt tokens* | Median latency | Prompt eval |
|---|---|---|---|---|---|---|---|
| A1 Full history | 81.8% | 60.0% | 100% | 18.2% | 1,756 | 21.2 s | 19.0 s |
| A2 Vector RAG | 54.5% | 40.0% | 66.7% | 21.2% | 258 | 4.5 s | 2.4 s |
| **B VERITAS standard** | **90.9%** | **100%** | 66.7% | **9.1%** | 562 | 8.2 s | 5.6 s |
| B2 VERITAS compact | 75.8% | 80.0% | 44.4% | 9.1% | **149** | **2.4 s** | **1.4 s** |

\*Counted by the model's own tokenizer (`prompt_eval_count`).

Multi-hop and stable questions were 100% for every method except A2, which scored 0% on multi-hop. Verified citations
were 90.9% for B and 84.8% for B2.

**Findings:**

1. **VERITAS standard is the most accurate method**, and the only one that answered every question about changed facts
   correctly. It used **68% fewer prompt tokens** and answered **2.6 times faster** than the full history.
2. **The full history fails in exactly the expected way.** It answered "Paris" to "Where does Alice live now?" in
   every repeat. That value came from a low-authority rumor that the transcript still contains and that a later sensor
   reading had contradicted. VERITAS had rejected the rumor at ingestion time.
3. **Vector RAG is the weakest.** It often retrieves the stale value, or misses the relevant fact and says "UNKNOWN".
4. **Historical "immediately before" questions are hard for a 3B model even with clean context.** VERITAS context
   listed Pune → Mumbai → London with dates, yet the model answered "Pune". The full history did better on this one
   category, since chronological narrative is a natural fit for it.
5. **Single runs are noisy.** An earlier 3-method run gave the full history 85.2% and VERITAS 88.9%, and one
   single-repeat run gave the full history 77.8%. The cache-busting ID slightly changes each prompt, so answers can flip
   even at temperature 0. Repeats matter.

### 11.3 The compact prompt: what was learned

The compact format was designed to keep VERITAS's accuracy at a fraction of its tokens. It achieved the token goal
(−73% against standard; 3.5 times faster) but **not** the accuracy goal on this model, so it stays opt-in. Controlled
experiments that held the facts fixed and varied only the wording produced three lessons:

- **The header matters enormously.** A header reading `Facts as of now ("past" = no longer true):` led the 3B model to
  treat *every* fact as no longer true and answer UNKNOWN. Accuracy was 5 of 11; replacing the header with a neutral
  `Facts:` raised it to 12 of 15.
- **Placeholders get copied.** Instructions that said "cite the [n] tag" produced answers like "Acme [n]".
- **Prose is worse than structure for this model.** Across three independent fact sets (the demo, a personal-facts
  set and an operations set with servers and orders; 40 graded answers per variant):
  - `-relation->` arrows scored **88%**;
  - arrows with the relation in plain words scored 82%;
  - full sentences scored 65%.

The remaining compact failures are abstentions ("UNKNOWN") when the context holds only one or two facts. The standard
format, which carries more surrounding text, does not show them.

### 11.4 Infrastructure findings

On the development laptop (RTX 3050, 4 GB VRAM), Ollama could offload only 21 of 29 layers of llama3.2 at an 8k
context. Its llama.cpp CUDA backend, with flash attention on auto, then crashed with `CUDA error: an illegal
instruction was encountered`. Before the crash, generations degenerated into repetition loops (`token repeat limit
reached`). Those numbers were discarded rather than reported. The harness now:

- retries runner crashes up to twice;
- records degenerate generations as failed answers instead of aborting;
- supports `--num-gpu 0` for CPU-only runs.

All results above are CPU results. On the CPU, prompt evaluation runs at about 84 tokens/s and decoding at about
13–19 tokens/s, which makes prompt length the dominant cost.

### 11.5 Test suite

There are 176 `pytest` tests in 16 files, and none needs network access or a model:

- **Graph engine:** interval algebra and bi-temporal visibility.
- **Reconciliation:** every rule, tamper and deletion detection, rollback on a failed write, replay equivalence and
  provenance.
- **Extraction:** 18 sentence patterns.
- **Pruning:** stale and hub-distractor removal, multi-hop retention, thresholds and both serializers.
- **Pruning options:** date parsing, ego-local scoring that keeps exactly the same facts as full scoring, hub damping,
  and checks that the defaults are unchanged.
- **Agent:** grounding, fabricated and out-of-context citations, and compact tags.
- **API:** every endpoint, including restart persistence and CORS.
- **Cloud:** tenant isolation and limits, API keys (including tenant-bound keys), metrics, readiness, alias endpoints,
  and opt-in date detection.
- **Entity linking:** normalization, declared and automatic links, ambiguity, validation, replay, and rollback on a
  failed ledger write.
- **Baselines:** each one's characteristic behaviour and failure mode.
- **Temporal benchmark:** generator determinism, independence of ground truth from rumors, the shared reader, and the
  end-to-end comparison and CLI.
- **SLM connectors:** fake HTTP servers via `httpx.MockTransport`.
- **A/B harness:** grading, rotation, cache busting, crash retry, the CLI, the new methods and the synthetic dataset.
- **Lab service:** jobs, cancellation and question generation.
- **Benchmark:** end to end.

### 11.6 Temporal benchmark with independent ground truth

`datasets.py` simulates people and organizations whose residence, employer, role, preference and headquarters drift
across sessions, and keeps the *true* validity interval of every value. The observation stream adds:

- **rumors:** `hearsay` claims of values that are never true, arriving after the truth;
- **late arrivals:** `system_of_record` entries about the period *before* anything else on file, delivered sessions
  later.

Questions are sampled evenly from six categories (changed, contested, stable, point-in-time, historical, multi-hop).
Their answers come from the simulation, so the evaluation of VERITAS is not circular.

`temporal_bench.py` runs every memory over the same stream and reads every retrieval with **one deterministic
reader**. The reader takes the newest fact for the slot, or the fact valid on the asked date, or the value before the
current one, or two hops for multi-hop questions. This rule is the best a reader can do with timestamped facts, so it
favours the baselines. Results for the default configuration:

| | Full history | Vector RAG | Hybrid BM25+dense | Static graph RAG | Overwrite memory | VERITAS | VERITAS optimized |
|---|---|---|---|---|---|---|---|
| Accuracy | 81.7% | 53.3% | 60.0% | 56.7% | 53.3% | **100%** | **100%** |
| · changed | 100% | 72.7% | 81.8% | 100% | 100% | 100% | 100% |
| · contested (rumor) | 40.0% | 20.0% | 30.0% | 40.0% | 40.0% | **100%** | **100%** |
| · stable | 100% | 88.9% | 88.9% | 100% | 100% | 100% | 100% |
| · point-in-time | 90.0% | 80.0% | 90.0% | 10.0% | 10.0% | **100%** | **100%** |
| · historical | 90.0% | 60.0% | 70.0% | 80.0% | 0.0% | **100%** | **100%** |
| · multi-hop | 70.0% | 0.0% | 0.0% | 10.0% | 70.0% | **100%** | **100%** |
| Stale / wrong value answered | 13.3% | 23.3% | 16.7% | 28.3% | 11.7% | **0%** | **0%** |
| Contradictory context | 100% | 80.0% | 75.0% | 100% | 0% | **0%** | **0%** |
| Verifiable (ledger-cited) | 0% | 0% | 0% | 0% | 0% | **100%** | **100%** |
| Context tokens (mean) | 12,623 | 131 | 135 | 541 | 479 | 552 | 538 |
| Query latency mean | 0.06 ms | 28.2 ms | 28.6 ms | 1.9 ms | 1.6 ms | 13.3 ms | 2.8 ms |
| Ingest per observation | 0.04 ms | 0.23 ms | 0.22 ms | 0.02 ms | 0.01 ms | 3.8 ms (fsync) | shared |

VERITAS's advantage here is structural: its memory never hands the reader a rumor, a stale value or a missing hop.
The categories show which mechanism matters where:

- **Contested:** source-authority reconciliation.
- **Point-in-time:** valid-time intervals combined with dates read from the question.
- **Historical:** closed intervals instead of overwrites.
- **Multi-hop:** the ego graph.

The caveat in §12.2 (8) applies: the generator's rumors are low-authority, which is the assumption VERITAS's rule makes.

### 11.7 Scaling

`temporal_bench.py --scaling` repeats the benchmark at growing world sizes:

| People | Observations | Facts | Ledger records | Ingest mean / p95 | Chain verify | Query default (mean / p95) | Query optimized (mean / p95) | Context tokens default / optimized | Full-history tokens |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 1,533 | 1,214 | 2,294 | 2.8 / 4.6 ms | 190 ms | 34.7 / 46.9 ms | 4.3 / 15.1 ms | 643 / 617 | 32,319 |
| 400 | 5,959 | 4,755 | 8,918 | 3.4 / 5.2 ms | 779 ms | 196.8 / 379.8 ms | 14.7 / 73.5 ms | 934 / 756 | 128,759 |
| 1,000 | 14,787 | 11,758 | 22,142 | 3.4 / 5.2 ms | 1,886 ms | 527.6 / 1,000.4 ms | 56.7 / 230.9 ms | 1,187 / 811 | 337,986 |

- **Ingestion** cost per observation is flat (about 3 ms), dominated by the per-transaction fsync (`LEDGER_FSYNC=0`
  brings it to about 1 ms).
- **Chain verification** is linear: 1.9 s for 22k records.
- **Queries:** the default pruner scores every fact, so its latency grows with the history. The optional ego-local
  mode (§8.1) keeps exactly the same facts at 9x lower mean latency for 11,758 facts.
- **Context tokens:** with hub damping, context grows from 617 to only 811 tokens over a 10x larger graph, while the
  full-history prompt grows to 338k tokens.

### 11.8 All eight methods with a real SLM

Same setup as §11.2 (llama3.2 3B Q4_K_M, CPU, temperature 0, cold prompt cache), one repeat over the 11 demo
questions:

| | A1 full history | A2 vector RAG | A3 hybrid RAG | A4 static graph RAG | A5 overwrite memory | B VERITAS | B2 compact | B3 optimized |
|---|---|---|---|---|---|---|---|---|
| Accuracy (11 questions) | 81.8% | 54.5% | 63.6% | 54.5% | 72.7% | **90.9%** | 81.8% | **90.9%** |
| · changed (5) | 60% | 40% | 40% | 40% | 100% | 100% | 80% | 100% |
| · historical (3) | 100% | 66.7% | 100% | 66.7% | 0% | 66.7% | 66.7% | 66.7% |
| · multi-hop (1) | 100% | 0% | 0% | 0% | 100% | 100% | 100% | 100% |
| Stale / conflicting answers | 18.2% | 18.2% | 18.2% | 36.4% | 0% | 9.1% | 9.1% | 9.1% |
| Verified citations | n/a | n/a | n/a | n/a | n/a | 100% | 90.9% | 100% |
| Prompt tokens (model) | 1,757 | 258 | 261 | 438 | 385 | 561 | **149** | 561 |
| Median latency | 21.4 s | 4.6 s | 4.6 s | 6.0 s | 5.0 s | 8.3 s | **2.3 s** | 8.2 s |

- **VERITAS standard and optimized:** 90.9% each, the highest of the eight methods, with every citation verified
  against the ledger.
- **Overwrite memory:** answers changed facts well (100%), but fails every historical question, because the history is
  gone.
- **Hybrid retrieval:** improves on dense retrieval (63.6% vs 54.5%), but still surfaces stale values.
- **Static graph RAG:** the most stale answers (36.4%), because it presents every value a slot ever had as equally
  current.
- **Full history:** close in accuracy (81.8%), but at 3x VERITAS's prompt tokens and 2.6x its median latency.

With 11 questions and one repeat, differences of one answer (9 points) are within noise.

## 12. Current status, limitations and future work

### 12.1 What works today

- The full pipeline (observe → reconcile → ledger → prune → prompt → verify), with persistence, replay and tamper
  detection.
- Point-in-time and bi-temporal queries, exposed through the API and the time-travel UI.
- Real local SLM inference with honest, reproducible measurement, plus an interactive A/B Lab for arbitrary user facts.
- Offline operation end to end, through the mock reader.

### 12.2 Known limitations

1. **Rule-based extraction.** Coverage is limited to the patterns in §7. Paraphrases ("Alice now calls London home"),
   negation ("Alice no longer works at Acme"), quantities and compound sentences are not handled.
2. **Limited entity resolution.** Declared aliases and the optional honorific/legal-suffix rule (§7.1) cover the
   common cases, but "Alice Smith" and "A. Smith" are still two entities unless an alias is declared. There is no
   coreference resolution, and a link applies only from the moment it is recorded.
3. **Fixed ontology.** Cardinality is static. Unknown predicates default to functional, which is wrong for genuinely
   multi-valued relations the ontology doesn't know.
4. **Heuristic authority model.** Source authorities and the 0.35 tolerance are hand-set constants, not learned or
   calibrated.
5. **Lexical relevance.** Pruning relevance uses alias overlap, so a question phrased with vocabulary outside the alias
   lists can miss the right relation.
6. **"End" events are not modelled.** A fact can only be closed by a conflicting replacement. There is no way to say
   "Alice left Acme" without naming a new employer.
7. **Single-process scale per tenant.** Each tenant's graph is in memory behind one lock and its ledger is one local
   file. Tenants (§10.1) give horizontal scale by sharding, not shared-state replicas. The lab still runs one job at a
   time.
8. **Evaluation scale.** The real-model results use one trajectory, 11 questions and one 3B model, so they are
   indicative, not a benchmark claim. The synthetic benchmark (§11.6) has independent ground truth and scales to
   thousands of entities, but its generator encodes the same assumption VERITAS makes: rumors come from low-authority
   sources. A world where a *high*-authority source lies would defeat the authority rule as it stands. Auto-generated
   Lab questions remain circular and should be checked by a human.
9. **Integrity, not authenticity.** The hash chain proves the ledger was not altered after writing. It does not prove
   *who* wrote it, because records are not signed.

### 12.3 Future work

**Knowledge acquisition**

- LLM-assisted extraction behind the deterministic rules, validated against the ontology: the model proposes triples
  and the rules accept or reject them.
- Explicit end events (`ENDED`/`LEFT` mutations) and negated observations.
- Entity resolution with alias tables and embedding-plus-rule matching, recorded as `MERGE` mutations so merges are
  auditable and reversible.
- Ontology learning: inferring cardinality from data (a predicate that routinely carries several concurrent values is
  multi-valued).

**Reconciliation**

- Calibrated source reliability, learned per source from how often its claims are later corroborated or contradicted,
  in the spirit of truth-discovery methods.
- Uncertainty-preserving states: keep two candidate values open with probabilities when evidence is balanced, instead
  of forcing a winner.
- Retraction and rollback as first-class mutations ("undo transaction X"), with the original records preserved.

**Retrieval and prompting**

- Embedding-based relation matching alongside the alias lists.
- Learned or tuned pruning weights (α, β, γ, τ) per domain, fitted on held-out questions.
- **Adaptive serialization:** use compact format when it is safe and standard when the context is tiny or the question
  is historical, exactly where compact underperformed in §11.3. Test the "abstain only if no fact mentions the subject"
  phrasing on held-out data.
- Successor-chain hints for historical questions ("Pune → Mumbai → London") to help small models with "immediately
  before".

**Evaluation**

- Larger and public benchmarks for temporal and knowledge-update QA. TimeQA (Chen et al., 2021), TempLAMA (Dhingra et
  al., 2022) and the LongMemEval-style agent-memory evaluations used for Zep (Rasmussen et al., 2025) are natural
  candidates.
- More models (Qwen-2.5-3B, Phi-3.5, 7–8B models) and scaling curves with `--scale`, to find where the full history
  breaks down in accuracy and not just cost.
- GPU numbers once Ollama's CUDA backend is stable on the target hardware (or with `OLLAMA_FLASH_ATTENTION=0`).
- Human-verified ground truth for auto-generated questions.

**Systems**

- A persistent graph store, such as Neo4j/Memgraph or SQLite with interval indexes, with the ledger as the source of
  truth and the graph as a rebuildable projection.
- Signed ledger records (Ed25519 per writer) and periodic anchoring of the head hash, so the log proves *who* wrote
  each change as well as integrity.
- Merkle-tree checkpoints, so verification and proofs for single records are logarithmic instead of linear.
- Multi-tenant isolation and authentication for the API, plus a job queue for parallel lab runs on separate model
  instances.

---

## 13. Reproducing the results

```bash
# backend
cd backend
pip install -r requirements-dev.txt
python -m pytest                         # 176 tests, offline
python benchmark.py                      # structural benchmark (§11.1)
python benchmark.py --all-baselines      # plus hybrid, static graph and overwrite baselines
python temporal_bench.py                 # synthetic benchmark, independent ground truth (§11.6)
python temporal_bench.py --scaling 100,400,1000 --questions 30   # scaling (§11.7)

# real-SLM comparison (§11.2), requires Ollama
ollama pull llama3.2
python ab_eval.py --model llama3.2 --repeats 3 --num-gpu 0
python ab_eval.py --model llama3.2 --num-gpu 0 --methods A1,A2,A3,A4,A5,B,B2,B3   # all methods (§11.8)
python ab_eval.py --model llama3.2 --dataset synthetic --methods A1,A3,A5,B,B3   # synthetic questions
#   reports → backend/reports/ab_<model>_<timestamp>/{report.json, report.md, veritas_ledger.jsonl}

# app
uvicorn main:app --port 8000             # LLM_PROVIDER=ollama to use the SLM in the dashboard
cd ../frontend && npm install && npm run dev   # http://localhost:5173 → "A/B Lab"
```

The harness options are `--methods` (any of A1, A2, A3, A4, A5, B, B2, B3; default A1,A2,B,B2), `--dataset demo|synthetic`, `--scale N` (a longer history), `--allow-prompt-cache`,
`--num-ctx`, `--provider llamacpp --base-url …` and `--model qwen2.5:3b`.

---

## 14. References

- Chen, W., Wang, X., & Wang, W. Y. (2021). *A Dataset for Answering Time-Sensitive Questions* (TimeQA). NeurIPS
  Datasets and Benchmarks.
- Dhingra, B., Cole, J. R., Eisenschlos, J. M., Gillick, D., Eisenstein, J., & Cohen, W. W. (2022). *Time-Aware
  Language Models as Temporal Knowledge Bases* (TempLAMA). Transactions of the ACL.
- Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.
  arXiv:2404.16130.
- Haber, S., & Stornetta, W. S. (1991). *How to Time-Stamp a Digital Document*. Journal of Cryptology, 3(2).
- Jensen, C. S., & Snodgrass, R. T. (1999). *Temporal Data Management*. IEEE Transactions on Knowledge and Data
  Engineering, 11(1).
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. Transactions of the ACL.
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT* (ROME).
  NeurIPS.
- Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023). *Mass-Editing Memory in a Transformer*
  (MEMIT). ICLR.
- Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann. (Noisy-OR combination.)
- Rasmussen, P., et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*. arXiv:2501.13956.
  Graphiti, the open-source implementation, is at https://github.com/getzep/graphiti. **This is the primary prior work
  VERITAS builds on.** VERITAS is an independent implementation and uses none of its code.
- Snodgrass, R. T. (1999). *Developing Time-Oriented Database Applications in SQL*. Morgan Kaufmann.

**Software:**

- FastAPI, Pydantic v2, NetworkX and httpx (backend).
- React 18, Vite, Tailwind CSS, `react-force-graph-2d` and Lucide (frontend).
- Ollama and llama.cpp (local inference).
- Meta Llama 3.2 3B Instruct (evaluation model).
