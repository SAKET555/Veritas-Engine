import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Link2, ShieldAlert, ShieldCheck } from "lucide-react";
import { formatTs } from "../api.js";
import { MUTATION_STYLES } from "../theme.js";

function AuditRecord({ record, highlighted, onSelectEntity }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const snap = record.new_state_snapshot;
  const edge = snap.edge;
  const shortId = record.mutation_id.replace(/-/g, "").slice(0, 8);

  useEffect(() => {
    if (highlighted) ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [highlighted]);

  return (
    <li
      ref={ref}
      className={`rounded-lg border p-3 transition ${
        highlighted ? "border-sky-500/60 bg-sky-500/10" : "border-slate-800 bg-slate-950/40"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className={`chip ${MUTATION_STYLES[record.mutation_type]}`}>{record.mutation_type}</span>
        <span className="chip bg-slate-800 text-slate-300" title={record.mutation_id}>
          M:{shortId}
        </span>
        <span className="font-mono text-[11px] text-slate-500">#{record.sequence}</span>
        <span className="ml-auto font-mono text-[11px] text-slate-500">{formatTs(record.timestamp)}</span>
      </div>

      <div className="mt-2 text-sm">
        <button type="button" className="font-medium text-sky-300 hover:underline" onClick={() => onSelectEntity(edge.source_id)}>
          {snap.source_entity.name}
        </button>
        <span className="mx-1.5 font-mono text-xs text-slate-400">—{edge.relation}→</span>
        <button type="button" className="font-medium text-violet-300 hover:underline" onClick={() => onSelectEntity(edge.target_id)}>
          {snap.target_entity.name}
        </button>
        <span className="ml-2 font-mono text-[11px] text-slate-500">
          [{formatTs(edge.t_start)} → {edge.t_end ? formatTs(edge.t_end) : "∞"})
        </span>
      </div>

      <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{record.deterministic_rationale}</p>

      <button
        type="button"
        className="mt-2 flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        state snapshots & hash chain
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          <div className="grid gap-1 font-mono text-[10px] text-slate-500">
            <span className="flex items-center gap-1">
              <Link2 className="h-3 w-3" /> prev {record.prev_hash.slice(0, 24)}…
            </span>
            <span>hash {record.record_hash.slice(0, 24)}…</span>
            <span>txn {record.transaction_id}</span>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            <pre className="max-h-48 overflow-auto rounded bg-slate-950 p-2 text-[10px] text-rose-200/80">
              {record.prior_state_snapshot ? JSON.stringify(record.prior_state_snapshot.edge, null, 1) : "∅ (new fact)"}
            </pre>
            <pre className="max-h-48 overflow-auto rounded bg-slate-950 p-2 text-[10px] text-emerald-200/80">
              {JSON.stringify(edge, null, 1)}
            </pre>
          </div>
        </div>
      )}
    </li>
  );
}

export default function AuditFeed({ records, verification, highlightedIds, onSelectEntity }) {
  const [filter, setFilter] = useState("ALL");
  const [search, setSearch] = useState("");

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    return [...records]
      .reverse()
      .filter((r) => filter === "ALL" || r.mutation_type === filter)
      .filter(
        (r) =>
          !q ||
          r.mutation_id.replace(/-/g, "").startsWith(q.replace(/^m:/, "")) ||
          r.entity_ids.some((id) => id.includes(q)) ||
          r.deterministic_rationale.toLowerCase().includes(q),
      );
  }, [records, filter, search]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div
        className={`flex items-start gap-2 rounded-lg px-3 py-2 text-xs ring-1 ${
          verification?.valid
            ? "bg-emerald-500/10 text-emerald-300 ring-emerald-500/30"
            : "bg-rose-500/10 text-rose-300 ring-rose-500/30"
        }`}
      >
        {verification?.valid ? <ShieldCheck className="h-4 w-4 shrink-0" /> : <ShieldAlert className="h-4 w-4 shrink-0" />}
        <div>
          <div className="font-medium">{verification?.detail || "Verifying ledger…"}</div>
          {verification && (
            <div className="mt-0.5 font-mono text-[10px] opacity-75">head {verification.head_hash.slice(0, 32)}…</div>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        <input className="input !py-1.5" placeholder="Filter: M:id, entity, text…" value={search} onChange={(e) => setSearch(e.target.value)} />
        <select className="input !w-auto !py-1.5" value={filter} onChange={(e) => setFilter(e.target.value)}>
          {["ALL", "INSERT", "INVALIDATE", "RECONCILE"].map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
      </div>

      <ul className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {visible.map((r) => (
          <AuditRecord
            key={r.mutation_id}
            record={r}
            highlighted={highlightedIds.has(r.mutation_id.replace(/-/g, "").slice(0, 8))}
            onSelectEntity={onSelectEntity}
          />
        ))}
        {visible.length === 0 && <li className="py-8 text-center text-sm text-slate-500">No mutations match.</li>}
      </ul>
    </div>
  );
}
