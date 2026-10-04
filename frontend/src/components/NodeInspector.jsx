import { useEffect, useState } from "react";
import { History, Loader2, X } from "lucide-react";
import { api, formatTs } from "../api.js";
import { MUTATION_STYLES, entityColor } from "../theme.js";

// Provenance drawer: entity attributes plus the full chronological mutation trail.
export default function NodeInspector({ entityId, refreshKey, onClose, onSelectEntity }) {
  const [trail, setTrail] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!entityId) return undefined;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .provenance(entityId)
      .then((data) => !cancelled && setTrail(data))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [entityId, refreshKey]);

  if (!entityId) return null;
  const entity = trail?.entity;

  return (
    <aside className="panel absolute inset-y-3 right-3 z-20 flex w-[min(380px,calc(100%-1.5rem))] flex-col shadow-2xl shadow-black/50">
      <div className="flex items-start justify-between gap-2 border-b border-slate-800 p-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full" style={{ background: entityColor(entity?.entity_type) }} />
            <h3 className="text-base font-semibold text-slate-100">{entity?.name || entityId}</h3>
          </div>
          <div className="mt-1 font-mono text-[11px] text-slate-500">
            {entity?.entity_type} · id={entityId} · first seen {entity ? formatTs(entity.first_seen) : "…"}
          </div>
        </div>
        <button type="button" onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-800" aria-label="Close">
          <X className="h-4 w-4" />
        </button>
      </div>

      {entity && Object.keys(entity.attributes).length > 0 && (
        <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 border-b border-slate-800 px-4 py-3 text-xs">
          {Object.entries(entity.attributes).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-slate-500">{k}</dt>
              <dd className="font-mono text-slate-300">{typeof v === "object" ? JSON.stringify(v) : String(v)}</dd>
            </div>
          ))}
        </dl>
      )}

      <div className="panel-title px-4 pt-3">
        <History className="h-3.5 w-3.5" /> Provenance trail ({trail?.entries.length ?? 0})
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4 pt-2">
        {loading && <Loader2 className="mx-auto mt-6 h-5 w-5 animate-spin text-slate-500" />}
        {error && <p className="text-sm text-rose-400">{error}</p>}
        <ol className="relative space-y-3 border-l border-slate-800 pl-4">
          {trail?.entries.map(({ record }) => {
            const snap = record.new_state_snapshot;
            const entityLink = (e) =>
              e.id === entityId ? (
                <span className="font-medium text-slate-100">{e.name}</span>
              ) : (
                <button type="button" className="text-sky-300 hover:underline" onClick={() => onSelectEntity(e.id)}>
                  {e.name}
                </button>
              );
            return (
              <li key={record.mutation_id} className="relative">
                <span className="absolute -left-[21px] top-1 h-2.5 w-2.5 rounded-full border-2 border-slate-900 bg-slate-500" />
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className={`chip ${MUTATION_STYLES[record.mutation_type]}`}>{record.mutation_type}</span>
                  <span className="chip bg-slate-800 text-slate-400">M:{record.mutation_id.replace(/-/g, "").slice(0, 8)}</span>
                  <span className="font-mono text-[10px] text-slate-500">#{record.sequence}</span>
                </div>
                <div className="mt-1 text-xs text-slate-300">
                  {entityLink(snap.source_entity)} <span className="font-mono text-slate-500">—{snap.edge.relation}→</span>{" "}
                  {entityLink(snap.target_entity)}
                  <span className="ml-1 font-mono text-[10px] text-slate-500">
                    [{formatTs(snap.edge.t_start)} → {snap.edge.t_end ? formatTs(snap.edge.t_end) : "∞"})
                  </span>
                </div>
                <p className="mt-1 text-[11px] leading-relaxed text-slate-500">{record.deterministic_rationale}</p>
              </li>
            );
          })}
        </ol>
      </div>
    </aside>
  );
}
