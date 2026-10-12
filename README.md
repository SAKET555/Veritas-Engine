# VERITAS — Verifiable Knowledge State Editing for Long-Horizon AI Agents

A frozen LLM paired with an external **bi-temporal knowledge graph**. Every state change is decided by deterministic
rules, applied atomically, and committed to a **hash-chained JSONL audit ledger**. At query time a
**confidence-driven pruner** hands the model a minimal, epistemically consistent subgraph, and the model must cite
the mutation IDs that justify each claim.

```
observation ─► extraction ─► reconciliation (R1–R5) ─► graph (valid time + system time)
                                   │                          │
                                   └─► audit ledger (SHA-256 chain, replayable)
question ─► seed resolution ─► k-hop ego ─► score/prune ─► compact context + audit trail ─► frozen LLM ─► citation check
```

## Prior work and credits

The central idea is **not new**. It comes from **Graphiti**, the open-source temporal knowledge graph behind
**Zep's** agent memory:

- Rasmussen, P., et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*. arXiv:2501.13956.
- Graphiti: https://github.com/getzep/graphiti

That idea is to store agent memory as a knowledge graph whose facts carry validity intervals (valid time) and
ingestion times (system time), and to close superseded facts instead of deleting them.

VERITAS is an **independent implementation written from scratch**. No Graphiti or Zep code is used. It builds on that
idea with:

- deterministic, authority-weighted conflict rules instead of LLM-decided edge invalidation;
- a SHA-256 hash-chained, replayable audit ledger;
- confidence-driven subgraph pruning;
- post-generation verification of the model's citations against the ledger;
- an evaluation harness and A/B Lab focused on small local models.

The bi-temporal model itself comes from temporal-database research (Snodgrass, 1999; Jensen & Snodgrass, 1999). See
[TECHNICAL.md](TECHNICAL.md) for the full reference list.

## Layout

| Path | Purpose |
|---|---|
| `backend/models.py` | Pydantic v2 schema: `EntityNode`, `TemporalEdge`, `MutationAuditRecord`, API bodies |
| `backend/ontology.py` | Predicate cardinality (functional vs multi-valued) and lexical aliases |
| `backend/graph_engine.py` | `networkx.MultiDiGraph` engine: `add_temporal_edge`, `invalidate_edge`, `query_point_in_time`, `get_active_subgraph` |
| `backend/reconciliation.py` | Conflict rules, atomic transactions with rollback, hash-chained ledger, `get_provenance_trail`, replay |
| `backend/extraction.py` | Deterministic rule-based statement → triple parser (`Alice moved from Mumbai to London`, `A \| rel \| B`) |
| `backend/pruning.py` | Ego expansion, composite scoring, τ filtering, compact Markdown/JSON serialization |
| `backend/agent.py` | Frozen-LLM harness: mock reader, OpenAI / Ollama / LiteLLM clients, citation verification |
| `backend/main.py` | FastAPI app |
| `backend/benchmark.py` | TKG vs dense vector RAG on a T0 → T1 → T2 trajectory (`--all-baselines` adds three more) |
| `backend/scenarios.py` | The shared demo/benchmark trajectory |
| `backend/baselines.py` | Comparison memories: vector RAG, hybrid BM25 + dense RAG, static graph RAG, overwrite memory, full history |
| `backend/datasets.py` | Synthetic multi-session benchmark with independent ground truth (rumors, late arrivals, dated questions) |
| `backend/temporal_bench.py` | Offline benchmark of VERITAS vs every baseline, plus a latency/token scaling mode |
| `backend/entity_linking.py` | Auditable entity linking (declared aliases, optional automatic normalization) |
| `backend/observability.py` | Prometheus metrics without extra dependencies |
| `docker-compose.yml`, `*/Dockerfile` | Container deployment (API, dashboard behind nginx, optional Ollama) |
| `frontend/` | React 18 + Vite + Tailwind (JavaScript/JSX) dashboard |

## Setup

### Backend (Python 3.10+)

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs are served at http://localhost:8000/docs. The ledger is stored at `backend/data/audit_log.jsonl`, and on
startup the graph is rebuilt from it after the hash chain is verified.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, `openai`, `ollama` or `litellm` |
| `LLM_MODEL` | provider-specific | e.g. `gpt-4o-mini`, `llama3.1:8b` |
| `LLM_BASE_URL` | provider-specific | any OpenAI-compatible `/v1` endpoint |
| `OPENAI_API_KEY` / `LITELLM_API_KEY` | — | credentials |
| `AUDIT_LOG_PATH` | `backend/data/audit_log.jsonl` | ledger location |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allowed browser origins |
| `VERITAS_API_KEYS` | unset (no auth) | `key1,key2:acme`: require `X-API-Key`; a `key:tenant` key is bound to that tenant |
| `MAX_TENANTS` | `100` | tenants one process will load |
| `ENTITY_LINKING` | `explicit` | `explicit` (declared aliases only) or `auto` (also link "Acme Inc." to an existing "Acme") |
| `LEDGER_FSYNC` | `1` | `0` skips the per-transaction fsync (faster ingest, weaker crash durability) |

For exact BPE token counts, run `pip install tiktoken`. Without it, a deterministic estimator is used.

### Frontend (Node 18+)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 — /api is proxied to :8000
```

Click **Load demo trajectory**, then scrub the timeline, toggle **history**, click nodes to see their provenance,
submit observations, and ask questions. For a production build, run `npm run build`. If the API is on another
origin, set `VITE_API_BASE=https://api.example.com` at build time.

### Tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest            # 176 tests, runs in about 15 seconds, no network or API keys needed
```

The suite covers the graph engine (interval algebra, bi-temporal visibility), every reconciliation rule, ledger
tamper and deletion detection, rollback on a failed ledger write, replay equivalence, extraction rules, pruning
behaviour, citation verification (fabricated and out-of-context IDs), every API endpoint including restart
persistence, and the end-to-end benchmark. It also covers the comparison baselines, the synthetic dataset and its
reader, entity linking (including rollback and replay), tenants, API keys, metrics, and the opt-in retrieval options.
One test checks that ego-local scoring keeps exactly the same facts as full scoring.

### Benchmark

```bash
cd backend
python benchmark.py            # summary table
python benchmark.py -v         # plus retrieved chunks and pruned contexts
python benchmark.py --json out.json
python benchmark.py --embedder st   # sentence-transformers encoder, if installed
python benchmark.py --all-baselines # also hybrid BM25+dense RAG, static graph RAG, overwrite memory
```

The exit code is 0 only when the TKG reaches 100% epistemic consistency and the ledger chain verifies.

## Local SLM (Ollama / llama.cpp)

**Run the dashboard and API on a real local model:**

```bash
ollama pull llama3.2                     # or qwen2.5:3b, phi3.5
# PowerShell: $env:LLM_PROVIDER="ollama"; $env:LLM_MODEL="llama3.2"
LLM_PROVIDER=ollama LLM_MODEL=llama3.2 uvicorn main:app --port 8000
```

- `OllamaClient` (in `agent.py`) calls the native `/api/chat` endpoint, or `/api/generate` if selected.
- It always sets `num_ctx` explicitly. Otherwise Ollama silently truncates long prompts.
- It reports the model's own `prompt_eval_count` and the server-side load, prompt-evaluation and generation times.
  The dashboard shows these under **Token efficiency**.
- For a llama.cpp `llama-server`, set `LLM_PROVIDER=llamacpp` and `LLM_BASE_URL=http://localhost:8080/v1`.

### A/B Lab (in the dashboard)

Open the dashboard and switch to **A/B Lab** in the header.

1. **Facts:** enter facts in the order an agent learns them. You can type them, paste lines (`[2026-01-05] Priya lives
   in Pune`), import everything you observed on the Graph page, or load the demo. Each fact can have an optional time,
   a source and a confidence.
2. **Questions:** write test questions, or click **Auto-generate from my facts**. This builds "what is it now", "what
   was it before" and stable-control questions, taking the correct answers from VERITAS's reconciliation of your facts.
   Review them before running.
3. **Run:** pick a local model (from `ollama list`), CPU-only if you like, the number of repeats and the methods, then
   run.

Results stream in live as side-by-side bars for accuracy, accuracy on changed facts, prompt tokens, median latency,
stale answers and verified citations. A table shows every answer; click a cell to see the exact prompt and reply.

Runs execute one at a time in the backend and are saved to `backend/reports/lab_*`.

### Compact prompt (opt-in)

`QueryRequest.prompt_style` is `standard` by default. `compact` sends the same pruned facts, written for a small
model:

- **Short instructions.**
- **`[1]`-style citation tags.** The server maps each tag back to its ledger mutation ID and verifies it, so citations
  stay tamper-evident.
- **`ENDED … replaced by X` notes** on old facts.
- **No rationale text.** The rationale stays in the ledger instead of the prompt.

Measured on llama3.2-3B (CPU, 11 questions × 3 repeats = 33 graded answers per method):

| | Accuracy | Prompt tokens | Median latency |
|---|---|---|---|
| A1 full history | 81.8% | 1,756 | 21.2 s |
| A2 vector RAG | 54.5% | 258 | 4.5 s |
| B VERITAS standard | **90.9%** | 562 | 8.2 s |
| B2 VERITAS compact | 75.8% | **149** | **2.4 s** |

Compact is 3.5 times faster than standard and uses 73% fewer tokens, but it does **not** keep standard's accuracy on
this 3B model. The model sometimes answers UNKNOWN when it is given only one or two facts. That is why standard stays
the default. Measured wording lessons, recorded in the code:

- A header that defined "past" made the model treat every fact as no longer true.
- A `[n]` placeholder in the instructions got copied literally.
- Writing relations as plain sentences scored worse (65%) than `-relation->` arrows (88%) across three separate fact
  sets.

### A/B evaluation from the command line: append-only history vs vector RAG vs VERITAS

```bash
cd backend
python ab_eval.py --model llama3.2 --repeats 3           # GPU if available
python ab_eval.py --model llama3.2 --repeats 3 --num-gpu 0   # CPU only (stable on small or unstable GPUs)
python ab_eval.py --model qwen2.5:3b --scale 4            # 4x longer accumulated history
python ab_eval.py --provider llamacpp --base-url http://localhost:8080/v1 --model local
```

All three methods see the same observation stream and are queried by the same frozen SLM with the same answer
format. Only the memory placed in the prompt differs:

| Method | Memory in the prompt |
|---|---|
| **A1** append-only history | every observation ever made, timestamped, oldest first |
| **A2** vector RAG | the top-k observations by dense similarity, with no invalidation |
| **B** VERITAS | the pruned active subgraph plus the audit trail, from the real reconciliation engine and hash-chained ledger |

The baselines are explicitly told that facts change and that they must report the current state.

**Measurement controls:**

- The model is warmed up with the same `num_ctx` before timing starts.
- Each request starts with a unique ID, so the server's prompt cache can't reuse earlier prompts and every call is
  measured cold. Use `--allow-prompt-cache` to measure with caching instead.
- The order of methods rotates for each query.
- `num_ctx` is automatically the smallest window that fits every prompt.

**Grading and reliability:**

- Grading reads the `ANSWER:` line. An answer counts as correct only if it contains the true current value and no
  stale value, so a hedged answer like "Berlin or Munich" is wrong.
- A generation the server aborts (for example `token repeat limit reached`) is recorded as a failed answer.
- If the inference runner crashes (for example a CUDA fault), the call is retried up to `--retries` times.

Reports are written to `backend/reports/ab_<model>_<timestamp>/`:

- `report.json`: every row, with the full prompts and raw outputs
- `report.md`: summary tables
- `veritas_ledger.jsonl`: the verified ledger that method B answered from

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/observe` | `{text}` or `{subject, predicate, object}` plus `confidence`, `source`, `timestamp` → reconciliation outcomes |
| `GET` | `/api/graph?timestamp=&include_history=&as_known_at=` | current, point-in-time or bi-temporal topology |
| `GET` | `/api/timeline` | discrete valid-time transition points T0…Tn |
| `GET` | `/api/audit?entity_id=` | chronological ledger |
| `GET` | `/api/audit/verify` | hash-chain verification |
| `GET` | `/api/provenance/{entity_id}` | an entity's mutation trail with narratives |
| `POST` | `/api/query` | prune → LLM → cited answer, pruned context, token metrics, per-edge scores |
| `POST` | `/api/demo/seed`, `/api/reset` | load the demo trajectory / archive the ledger and start empty |
| `POST` | `/api/entities/alias` | `{alias, canonical}` → an `ALIAS` ledger record; later observations naming the alias resolve to the entity |
| `GET` | `/api/entities/aliases` | the alias table and linking mode |
| `GET` | `/api/metrics` | Prometheus metrics (requests, latency, reconciliation decisions, tokens, citations, per-tenant state) |
| `GET` | `/api/ready` | readiness probe: ledger loaded and hash chain verified |
| `GET` | `/api/tenants` | tenants loaded in this process |

Every endpoint accepts an optional `X-Tenant-ID` header (see [Cloud deployment](#cloud-deployment)).

## Reconciliation rules

For an incoming `(s, p, o, t, confidence, source)` with `eff = confidence × authority(source)`:

- **R1**: the same active fact already exists → `RECONCILE`. Confidence is fused by noisy-OR.
- **R2**: `p` is multi-valued → `INSERT`.
- **R3**: `p` is functional and recorded history for `(s, p)` starts after `t` (a late-arriving observation) → insert
  a closed historical interval. If it contradicts recorded history, the history wins.
- **R4**: `p` is functional with an active incumbent:
  - with a newer valid time, the new fact supersedes when `eff_new ≥ eff_old × (1 − 0.35)`;
  - at equal valid time, it supersedes only when `eff_new > eff_old`;
  - superseding emits `INVALIDATE` (old `t_end = t`) plus `INSERT` (new `[t, ∞)`) in one transaction;
  - otherwise a `RECONCILE` records that the candidate was rejected.
- **R5**: no value exists yet → `INSERT`.

## Pruning score

`Score(e) = (α·Confidence + β·Recency + γ·Relevance) / (α+β+γ)` with defaults α = 0.35, β = 0.15, γ = 0.5, τ = 0.6,
k = 2.

- **Recency** halves every 30 days.
- **Relevance** = path weight × (0.3 + 0.7 × lexical match). The path weight decays by 0.6 per hop and by a further
  0.4 when walking an edge backwards, which suppresses fan-in through hub nodes.
- Invalidated edges are dropped unless the question shows historical intent (auto-detected, or set with
  `include_history`).

## Benchmark caveats

The vector-RAG baseline uses a deterministic hashed dense encoder (no model download), and its reader takes the
top-ranked retrieved chunk for the asked (subject, predicate). The default agent reader is the deterministic
`MockLLMClient`, which reads only the prompt text. Absolute numbers will differ with a real embedding model or LLM;
the structural failure, where a temporally blind index retrieves stale and fresh facts side by side, does not.

## Optional retrieval settings

All off by default, so existing behaviour and results are unchanged.

| Setting | Where | Effect |
|---|---|---|
| `pruning.candidate_mode = "ego"` | `QueryRequest` | Score only facts that can still reach τ from the question's ego graph. The kept facts are identical to the default `"all"` (a test checks this); facts that cannot qualify are counted in `metrics.edges_unscored` instead of being scored one by one. Query cost then depends on the neighbourhood, not on the size of the whole history. |
| `pruning.hub_damping = 0.5` | `QueryRequest` | Walking backwards out of a crowded non-seed node (a city thousands of people live in) is divided by `fan_in ** hub_damping`, so hub neighbours stop crowding the prompt. A hub the question is *about* is never damped. |
| `detect_as_of = true` | `QueryRequest` | A date in the question ("on March 3, 2025", "2026-03-01", "15 March 2026") becomes `as_of`, so the answer is read from the graph as it was on that date. |
| `--all-baselines` | `benchmark.py` | Adds the three extra baselines to the structural benchmark. |
| `A3`, `A4`, `A5`, `B3` | A/B Lab and `ab_eval.py --methods` | Extra methods (see below). The default selection is still A1, A2, B, B2. |

## Additional comparison baselines

`baselines.py` adds the baselines the evaluation design calls for. Every one ingests the same extracted triples in the
same order:

| Key | Method | What it stands for |
|---|---|---|
| A3 | Hybrid BM25 + dense RAG | Okapi BM25 and dense retrieval fused by reciprocal rank fusion, the usual production RAG stack. No invalidation. |
| A4 | Static graph RAG | The same triples in a knowledge graph with no time dimension; k-hop ego graph around the question's entities. Every value a fact ever had is present and none is marked old. |
| A5 | Overwrite memory | Mutable slot memory without audit-trailed reconciliation: the last-arriving value overwrites, no history, no source authority, no ledger. |
| B3 | VERITAS (optimized) | Method B's prompt over the same graph, with ego-local scoring, hub damping and date detection switched on. |

## Temporal benchmark with independent ground truth

```bash
cd backend
python temporal_bench.py                              # 40 people, 6 sessions, 60 questions
python temporal_bench.py --people 120 --sessions 8    # a bigger world
python temporal_bench.py --scaling 100,400,1000       # latency and tokens as the graph grows
python temporal_bench.py --json r.json --markdown r.md --export-dataset ds.json
python ab_eval.py --dataset synthetic --model llama3.2 --methods A1,A2,A3,A4,A5,B,B3   # same data, real SLM
```

`datasets.py` simulates a world that drifts across sessions and records the true validity interval of every value.
The agent receives accurate updates plus **rumors** (low-authority claims of false values that never become true) and
**late arrivals** (records about the period before anything else on file, delivered sessions later). Questions cover
six categories: changed, contested (a rumor contradicted the fact), stable, point-in-time, historical and multi-hop.
Ground truth comes from the simulation, not from VERITAS's rules, which removes the circularity of the A/B Lab's
auto-generated questions.

Every method's retrieval is read by the same deterministic reader, and that reader is generous to the baselines. It
always takes the newest fact it was given, so a baseline loses only when its *memory* hands over stale, rumored or
missing facts. No LLM is involved.

Default run (40 people, 8 organizations, 6 sessions, 597 observations including 135 rumors, 60 questions, seed 7):

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

Each baseline fails in its own characteristic way:

- **Full history** contains every answer but also every contradiction; it costs 23 times VERITAS's tokens and is
  still fooled by rumors.
- **Vector and hybrid RAG** miss the second hop of multi-hop questions.
- **Static graph RAG** and **overwrite memory** cannot answer dated questions.
- **Overwrite memory** keeps no history.
- **Every baseline** believes newer rumors.

**Real model on the demo questions** (llama3.2 3B, CPU, `python ab_eval.py --methods A1,A2,A3,A4,A5,B,B2,B3`, one
repeat):

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

**Scaling** (`python temporal_bench.py --scaling 100,400,1000 --questions 30`, ledger fsync on):

| People | Observations | Facts | Ledger records | Ingest mean / p95 | Chain verify | Query default (mean / p95) | Query optimized (mean / p95) | Context tokens default / optimized | Full-history tokens |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 1,533 | 1,214 | 2,294 | 2.8 / 4.6 ms | 190 ms | 34.7 / 46.9 ms | 4.3 / 15.1 ms | 643 / 617 | 32,319 |
| 400 | 5,959 | 4,755 | 8,918 | 3.4 / 5.2 ms | 779 ms | 196.8 / 379.8 ms | 14.7 / 73.5 ms | 934 / 756 | 128,759 |
| 1,000 | 14,787 | 11,758 | 22,142 | 3.4 / 5.2 ms | 1,886 ms | 527.6 / 1,000.4 ms | 56.7 / 230.9 ms | 1,187 / 811 | 337,986 |

Every size kept 100% accuracy and a valid hash chain. Reconciliation cost per observation stays flat (about 3 ms,
mostly the fsync), while full-history context grows linearly to 338k tokens. With the default settings, query time
grows with the history, because every fact is scored. The optional ego-local mode keeps exactly the same facts and is
9x faster at 11,758 facts, and hub damping holds context growth to 811 tokens instead of 1,187. Numbers are from one
Windows laptop, CPU only; absolute values will vary, but the trends do not.

## Entity linking

Proposal phase 1 is "ingestion and entity linking". Entity ids are already slugs of the name, so case and punctuation
variants merge. On top of that:

- **Declared aliases** (always available): `POST /api/entities/alias {"alias": "Bob", "canonical": "Robert Smith"}`.
  Later observations naming *Bob* are recorded against *Robert Smith*, and questions naming *Bob* seed *Robert Smith*.
- **Automatic linking** (`ENTITY_LINKING=auto`): honorifics and legal-form suffixes are stripped ("Dr.", "Prof.",
  "Inc.", "Ltd", "GmbH", …). The name is linked only when exactly one *existing* entity has the same normalized form;
  ambiguous names are left alone.

Every link is an `ALIAS` record in the hash-chained ledger with a deterministic rationale, so merges are as auditable
as fact changes. Links are restored by replay and rolled back with their transaction. Past facts are never rewritten:
a link applies from the moment it is recorded.

## Cloud deployment

```bash
docker compose up --build                  # API on :8000, dashboard on :8080 (mock reader)
docker compose --profile llm up --build    # plus Ollama; then: docker compose exec ollama ollama pull llama3.2
```

- **Tenants.** An `X-Tenant-ID` header gives each tenant its own graph, ledger and journal under
  `data/tenants/<id>/`. With no header the `default` tenant is used, whose ledger is `AUDIT_LOG_PATH`: single-tenant
  mode is unchanged. The dashboard has a tenant/API-key control in the header.
- **API keys.** Set `VERITAS_API_KEYS`. `/api/health` and `/api/ready` stay public for orchestrator probes.
- **Metrics.** `GET /api/metrics` serves the Prometheus text format, scrapable by Prometheus, Grafana Agent, the
  OpenTelemetry Collector or managed cloud scrapers. It covers request count and latency by route, reconciliation
  decisions, context tokens sent vs avoided, LLM latency, citation verification, and per-tenant ledger and graph size.
- **Scaling model.** State is in memory and rebuilt from the ledger on boot, so run **one process per data volume**
  (the image uses `--workers 1`). Scale out by sharding tenants across replicas, each with its own volume, never by
  pointing two processes at one ledger file.

## How the code maps to the book chapter

| Chapter section | Code |
|---|---|
| 1. Agentic memory and the non-stationary state problem | `scenarios.py`, `datasets.py` (drift, rumors, late arrivals) |
| 2. Foundations: cloud RAG, dense retrieval, knowledge graphs | `baselines.py` (vector, hybrid BM25 + dense, static graph) |
| 3. Limits of vector and graph memory (bloat, stale retrieval, latency) | `temporal_bench.py` (stale answers, contradictions, tokens, `--scaling`) |
| 4. Non-parametric persistence over a frozen model | `graph_engine.py`, `models.py`, `agent.py` |
| 5. Audit-trailed reconciliation and provenance | `reconciliation.py` (R1–R5, hash chain, replay), `entity_linking.py` |
| 6. Confidence-driven pruning and token minimization | `pruning.py` (scoring, τ, ego-local mode, hub damping, compact prompt) |
| 7. Benchmark evaluation vs baseline RAG | `benchmark.py`, `temporal_bench.py`, `ab_eval.py`, A/B Lab |
| 8. Practical cloud-native implementation | `main.py` (tenants, keys, metrics), `observability.py`, Docker / compose |

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Saket Hegde. See [NOTICE](NOTICE) for
attribution.
