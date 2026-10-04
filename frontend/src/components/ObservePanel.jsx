import { useState } from "react";
import { Loader2, Send } from "lucide-react";
import { api } from "../api.js";
import { DECISION_STYLES } from "../theme.js";

const EXAMPLES = [
  "Alice moved from London to Lisbon",
  "Alice left Acme for Globex",
  "Alice lives in Paris",
  "Bob is a principal engineer at Acme",
  "Alice's favorite city is Kyoto",
  "Server-7 | located_in | Rack 4",
];

const SOURCES = [
  ["user_statement", "user statement (0.90)"],
  ["sensor", "sensor (0.95)"],
  ["system_of_record", "system of record (1.00)"],
  ["agent_observation", "agent observation (0.85)"],
  ["web", "web (0.70)"],
  ["agent_inference", "agent inference (0.60)"],
  ["hearsay", "hearsay (0.40)"],
];

// Submits a non-stationary observation and shows the deterministic reconciliation decision(s).
export default function ObservePanel({ onCommitted }) {
  const [text, setText] = useState("");
  const [confidence, setConfidence] = useState(0.9);
  const [source, setSource] = useState("user_statement");
  const [validFrom, setValidFrom] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.observe({
        text: text.trim(),
        confidence,
        source,
        timestamp: validFrom ? new Date(validFrom).toISOString() : undefined,
      });
      setResult(res);
      setText("");
      onCommitted(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <textarea
        className="input min-h-[64px] resize-y"
        placeholder='e.g. "Alice moved from London to Lisbon"'
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
        }}
      />
      <div className="flex flex-wrap gap-1.5">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setText(ex)}
            className="rounded-md bg-slate-800/80 px-2 py-0.5 text-[11px] text-slate-400 hover:bg-slate-700 hover:text-slate-200"
          >
            {ex}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <label className="space-y-1">
          <span className="text-slate-500">Source authority</span>
          <select className="input !py-1.5" value={source} onChange={(e) => setSource(e.target.value)}>
            {SOURCES.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1">
          <span className="text-slate-500">Valid from (blank = now)</span>
          <input type="datetime-local" className="input !py-1.5" value={validFrom} onChange={(e) => setValidFrom(e.target.value)} />
        </label>
        <label className="col-span-2 space-y-1">
          <span className="flex justify-between text-slate-500">
            Confidence <span className="font-mono text-slate-300">{confidence.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.05}
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </label>
      </div>

      <button type="submit" className="btn-primary w-full" disabled={busy || !text.trim()}>
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        Observe & reconcile
      </button>

      {error && <p className="rounded-lg bg-rose-500/10 p-2 text-xs text-rose-300">{error}</p>}

      {result && (
        <ul className="space-y-2">
          {result.outcomes.map((o) => (
            <li key={o.transaction_id} className="rounded-lg border border-slate-800 bg-slate-950/50 p-2.5 text-xs">
              <div className="flex items-center gap-2">
                <span className={`chip ${DECISION_STYLES[o.decision]}`}>{o.decision}</span>
                <span className="font-mono text-[10px] text-slate-500">txn {o.transaction_id.slice(0, 8)}</span>
              </div>
              <p className="mt-1.5 text-slate-300">{o.summary}</p>
              <p className="mt-1 font-mono text-[10px] text-slate-500">
                {o.mutations.map((m) => `${m.mutation_type} M:${m.mutation_id.replace(/-/g, "").slice(0, 8)}`).join(" · ")}
              </p>
            </li>
          ))}
        </ul>
      )}
    </form>
  );
}
