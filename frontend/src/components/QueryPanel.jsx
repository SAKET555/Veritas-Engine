import { useState } from "react";
import { Loader2, Search, SlidersHorizontal } from "lucide-react";
import { api } from "../api.js";

const EXAMPLES = [
  "Where does Alice live?",
  "Which city is Alice's employer headquartered in?",
  "Where did Alice live before London and why did it change?",
  "What is Alice's role?",
  "Tell me about Bob",
];

const DEFAULT_PARAMS = {
  k_hops: 2,
  alpha: 0.35,
  beta: 0.15,
  gamma: 0.5,
  tau: 0.6,
  max_edges: 40,
  include_history: null,
  output_format: "markdown",
};

function NumberField({ label, value, onChange, step = 0.05, min = 0, max = 1, hint }) {
  return (
    <label className="space-y-1" title={hint}>
      <span className="flex justify-between text-slate-500">
        {label} <span className="font-mono text-slate-300">{value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-500"
      />
    </label>
  );
}

// Natural-language query form with tunable pruning parameters.
// asOf: ISO timestamp of the time-travel slider (null = now) — queries follow the slider.
export default function QueryPanel({ asOf, onResult }) {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [showParams, setShowParams] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const set = (key) => (value) => setParams((p) => ({ ...p, [key]: value }));

  const submit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.query({ question: question.trim(), as_of: asOf || undefined, pruning: params });
      onResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="flex gap-2">
        <input className="input" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask the agent…" />
        <button type="submit" className="btn-primary !bg-sky-600 hover:!bg-sky-500" disabled={busy || !question.trim()}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
        </button>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setQuestion(ex)}
            className="rounded-md bg-slate-800/80 px-2 py-0.5 text-[11px] text-slate-400 hover:bg-slate-700 hover:text-slate-200"
          >
            {ex}
          </button>
        ))}
      </div>
      {asOf && (
        <p className="rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-300">
          Querying the graph as of {asOf.slice(0, 16).replace("T", " ")}Z (timeline position).
        </p>
      )}

      <button
        type="button"
        onClick={() => setShowParams((v) => !v)}
        className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200"
      >
        <SlidersHorizontal className="h-3.5 w-3.5" /> Pruning parameters
      </button>
      {showParams && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 rounded-lg border border-slate-800 p-3 text-xs">
          <NumberField label="α confidence" value={params.alpha} onChange={set("alpha")} />
          <NumberField label="β recency" value={params.beta} onChange={set("beta")} />
          <NumberField label="γ relevance" value={params.gamma} onChange={set("gamma")} />
          <NumberField label="τ threshold" value={params.tau} onChange={set("tau")} />
          <NumberField label="k hops" value={params.k_hops} onChange={set("k_hops")} step={1} min={0} max={4} />
          <NumberField label="max edges" value={params.max_edges} onChange={set("max_edges")} step={1} min={1} max={100} />
          <label className="space-y-1">
            <span className="text-slate-500">History</span>
            <select
              className="input !py-1"
              value={params.include_history === null ? "auto" : String(params.include_history)}
              onChange={(e) =>
                set("include_history")(e.target.value === "auto" ? null : e.target.value === "true")
              }
            >
              <option value="auto">auto-detect</option>
              <option value="true">include invalidated</option>
              <option value="false">active only</option>
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-slate-500">Serialization</span>
            <select className="input !py-1" value={params.output_format} onChange={(e) => set("output_format")(e.target.value)}>
              <option value="markdown">markdown</option>
              <option value="json">compact JSON</option>
            </select>
          </label>
          <button type="button" className="btn-ghost col-span-2 !py-1 text-xs" onClick={() => setParams(DEFAULT_PARAMS)}>
            Reset defaults
          </button>
        </div>
      )}

      {error && <p className="rounded-lg bg-rose-500/10 p-2 text-xs text-rose-300">{error}</p>}
    </form>
  );
}
