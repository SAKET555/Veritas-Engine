// Shared visual vocabulary for entity types, edge states and mutation types.

export const ENTITY_COLORS = {
  Person: "#38bdf8",
  Location: "#a78bfa",
  Organization: "#f59e0b",
  Role: "#f472b6",
  Preference: "#2dd4bf",
  Status: "#facc15",
  Entity: "#94a3b8",
};

export const entityColor = (type) => ENTITY_COLORS[type] || ENTITY_COLORS.Entity;

export const EDGE_COLORS = {
  active: "#10b981",
  activeFaded: "rgba(16,185,129,0.35)",
  highlighted: "#38bdf8",
  invalidated: "#f43f5e",
  invalidatedFaded: "rgba(244,63,94,0.45)",
};

export const MUTATION_STYLES = {
  INSERT: "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30",
  INVALIDATE: "bg-rose-500/15 text-rose-300 ring-1 ring-rose-500/30",
  RECONCILE: "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/30",
  ALIAS: "bg-fuchsia-500/15 text-fuchsia-300 ring-1 ring-fuchsia-500/30",
};

export const DECISION_STYLES = {
  INSERTED: "bg-emerald-500/15 text-emerald-300",
  SUPERSEDED: "bg-amber-500/15 text-amber-300",
  REINFORCED: "bg-sky-500/15 text-sky-300",
  RETAINED_INCUMBENT: "bg-slate-500/20 text-slate-300",
  RETROACTIVE_INSERT: "bg-violet-500/15 text-violet-300",
};
