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
| `backend/benchmark.py` | TKG vs dense vector RAG on a T0 → T1 → T2 trajectory |
| `backend/scenarios.py` | The shared demo/benchmark trajectory |
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
python -m pytest            # about 90 tests, runs in a few seconds, no network or API keys needed
```

The suite covers the graph engine (interval algebra, bi-temporal visibility), every reconciliation rule, ledger
tamper and deletion detection, rollback on a failed ledger write, replay equivalence, extraction rules, pruning
behaviour, citation verification (fabricated and out-of-context IDs), every API endpoint including restart
persistence, and the end-to-end benchmark.

### Benchmark

```bash
cd backend
python benchmark.py            # summary table
python benchmark.py -v         # plus retrieved chunks and pruned contexts
python benchmark.py --json out.json
python benchmark.py --embedder st   # sentence-transformers encoder, if installed
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

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Saket Hegde. See [NOTICE](NOTICE) for
attribution.
