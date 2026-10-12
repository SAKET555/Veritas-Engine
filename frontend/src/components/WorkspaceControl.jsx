import { useState } from "react";
import { Building2, Check, X } from "lucide-react";
import { getWorkspace, setWorkspace } from "../api.js";

// Optional cloud settings: which tenant's knowledge graph to use, and the API key the server expects
// (only needed when the backend runs with VERITAS_API_KEYS). Empty = single-tenant local mode.
export default function WorkspaceControl({ tenant, onChange }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(() => getWorkspace());

  const apply = () => {
    const next = { tenant: (draft.tenant || "").trim().toLowerCase(), apiKey: (draft.apiKey || "").trim() };
    setWorkspace(next);
    setOpen(false);
    onChange?.(next);
  };

  return (
    <div className="relative">
      <button
        type="button"
        className="btn-ghost"
        onClick={() => {
          setDraft(getWorkspace());
          setOpen((v) => !v);
        }}
        title="Tenant and API key (for multi-tenant / authenticated deployments)"
      >
        <Building2 className="h-4 w-4" />
        {tenant || "default"}
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-2 w-72 rounded-lg border border-slate-800 bg-slate-900 p-3 shadow-xl">
          <label className="block text-[11px] text-slate-400">
            Tenant
            <input
              className="input mt-1"
              placeholder="default"
              value={draft.tenant || ""}
              onChange={(e) => setDraft((d) => ({ ...d, tenant: e.target.value }))}
            />
          </label>
          <label className="mt-2 block text-[11px] text-slate-400">
            API key (only if the server requires one)
            <input
              className="input mt-1"
              type="password"
              autoComplete="off"
              value={draft.apiKey || ""}
              onChange={(e) => setDraft((d) => ({ ...d, apiKey: e.target.value }))}
            />
          </label>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            Each tenant has its own graph and hash-chained ledger. Stored in this browser only.
          </p>
          <div className="mt-2 flex justify-end gap-2">
            <button type="button" className="btn-ghost !py-1 text-xs" onClick={() => setOpen(false)}>
              <X className="h-3.5 w-3.5" /> Cancel
            </button>
            <button type="button" className="btn-ghost !py-1 text-xs text-emerald-300" onClick={apply}>
              <Check className="h-3.5 w-3.5" /> Apply
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
