import { useMemo, useState } from "react";
import { BadgeCheck, Bot, ChevronDown, ChevronRight, FileText, Gauge, ShieldAlert } from "lucide-react";

const CITATION_SPLIT = /(\[M:[0-9a-f]{8}\])/g;

function AnswerText({ text, citations, onCitationClick }) {
  const byId = useMemo(() => Object.fromEntries(citations.map((c) => [c.short_id, c])), [citations]);
  return (
    <p className="text-sm leading-relaxed text-slate-200">
      {text.split(CITATION_SPLIT).map((part, i) => {
        const m = part.match(/^\[M:([0-9a-f]{8})\]$/);
        if (!m) return <span key={i}>{part}</span>;
        const cite = byId[m[1]];
        const ok = cite?.verified;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onCitationClick(m[1])}
            title={cite?.rationale || "Unknown mutation"}
            className={`chip mx-0.5 align-middle ${
              ok ? "bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25" : "bg-rose-500/20 text-rose-300"
            }`}
          >
            M:{m[1]}
          </button>
        );
      })}
    </p>
  );
}

function TokenBar({ label, value, max, color }) {
  const pct = max ? Math.max(1.5, (value / max) * 100) : 0;
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-200">{value.toLocaleString()}</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// Side-by-side: (1) pruned prompt context, (2) token efficiency, (3) cited answer.
export default function QueryResults({ result, onCitationClick }) {
  const [showScores, setShowScores] = useState(false);
  const m = result.metrics;
  const kept = result.scored_edges.filter((e) => e.kept);
  const dropped = result.scored_edges.length - kept.length;

  return (
    <section className="grid gap-3 lg:grid-cols-3">
      <div className="panel flex min-h-0 flex-col p-4">
        <div className="panel-title mb-2">
          <FileText className="h-3.5 w-3.5" /> Pruned context injected into prompt
        </div>
        <div className="mb-2 flex flex-wrap gap-1.5 text-[11px]">
          <span className="chip bg-slate-800 text-slate-300">seeds: {result.seed_entities.join(", ") || "none (global)"}</span>
          <span className={`chip ${result.include_history ? "bg-amber-500/15 text-amber-300" : "bg-slate-800 text-slate-300"}`}>
            history {result.include_history ? "included" : "pruned"}
          </span>
        </div>
        <pre className="max-h-72 min-h-0 flex-1 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-950 p-3 font-mono text-[11px] leading-relaxed text-sky-100/90">
          {result.pruned_context}
          {"\n\n"}
          <span className="text-slate-400">{result.audit_context}</span>
        </pre>
      </div>

      <div className="panel flex flex-col p-4">
        <div className="panel-title mb-3">
          <Gauge className="h-3.5 w-3.5" /> Token efficiency
        </div>
        <div className="mb-4 flex items-baseline gap-2">
          <span className="text-4xl font-semibold text-emerald-400">{(m.reduction_ratio * 100).toFixed(1)}%</span>
          <span className="text-xs text-slate-400">fewer graph-context tokens than the unpruned graph</span>
        </div>
        <div className="space-y-3">
          <TokenBar label={`Unpruned full graph (${m.full_graph_edges} facts)`} value={m.full_graph_tokens} max={m.full_graph_tokens} color="bg-slate-500" />
          <TokenBar label={`Pruned subgraph (${m.pruned_edges} facts)`} value={m.pruned_context_tokens} max={m.full_graph_tokens} color="bg-emerald-500" />
          <TokenBar label="Audit trail" value={m.audit_trail_tokens} max={m.full_graph_tokens} color="bg-sky-500" />
          <TokenBar label="Total prompt (system + context + audit + question)" value={m.prompt_tokens_total} max={m.full_graph_tokens} color="bg-violet-500" />
        </div>
        <p className="mt-3 text-[11px] text-slate-500">
          tokenizer: <span className="font-mono">{m.tokenizer}</span> · {kept.length} kept / {dropped} pruned candidates
        </p>
        {m.model_prompt_tokens != null && (
          <div className="mt-3 grid grid-cols-2 gap-2 rounded-lg border border-slate-800 p-2 text-[11px]">
            <span className="text-slate-500">model prompt tokens</span>
            <span className="text-right font-mono text-slate-200">{m.model_prompt_tokens.toLocaleString()}</span>
            <span className="text-slate-500">completion tokens</span>
            <span className="text-right font-mono text-slate-200">{m.model_completion_tokens ?? "–"}</span>
            {["prompt_eval", "generation", "load"].map(
              (k) =>
                result.model_timings_ms?.[k] != null && (
                  <div key={k} className="contents">
                    <span className="text-slate-500">{k.replace("_", " ")} (ms)</span>
                    <span className="text-right font-mono text-slate-200">
                      {Math.round(result.model_timings_ms[k]).toLocaleString()}
                    </span>
                  </div>
                ),
            )}
            {result.model_timings_ms?.tokens_per_s != null && (
              <>
                <span className="text-slate-500">decode speed</span>
                <span className="text-right font-mono text-slate-200">{result.model_timings_ms.tokens_per_s} tok/s</span>
              </>
            )}
          </div>
        )}
        <button
          type="button"
          className="mt-2 flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200"
          onClick={() => setShowScores((v) => !v)}
        >
          {showScores ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          Composite scores (α·conf + β·recency + γ·relevance)
        </button>
        {showScores && (
          <div className="mt-2 max-h-56 overflow-auto rounded-lg border border-slate-800">
            <table className="w-full text-left text-[10px]">
              <thead className="sticky top-0 bg-slate-900 text-slate-500">
                <tr>
                  <th className="px-2 py-1">fact</th>
                  <th className="px-1">conf</th>
                  <th className="px-1">rec</th>
                  <th className="px-1">rel</th>
                  <th className="px-1">score</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {result.scored_edges
                  .filter((e) => e.kept || e.relevance > 0)
                  .slice(0, 60)
                  .map((e) => (
                    <tr key={e.edge_id} className={e.kept ? "text-emerald-300" : "text-slate-500"} title={e.reason}>
                      <td className="px-2 py-0.5">
                        {e.source} -{e.relation}→ {e.target}
                        {!e.active && " ✕"}
                      </td>
                      <td className="px-1">{e.confidence.toFixed(2)}</td>
                      <td className="px-1">{e.recency.toFixed(2)}</td>
                      <td className="px-1">{e.relevance.toFixed(2)}</td>
                      <td className="px-1">{e.score.toFixed(3)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="panel flex flex-col p-4">
        <div className="panel-title mb-3">
          <Bot className="h-3.5 w-3.5" /> Agent answer · {result.model}
        </div>
        <div
          className={`mb-3 flex items-center gap-2 rounded-lg px-3 py-2 text-xs ring-1 ${
            result.grounded
              ? "bg-emerald-500/10 text-emerald-300 ring-emerald-500/30"
              : "bg-rose-500/10 text-rose-300 ring-rose-500/30"
          }`}
        >
          {result.grounded ? <BadgeCheck className="h-4 w-4" /> : <ShieldAlert className="h-4 w-4" />}
          {result.grounded
            ? `Grounded: ${result.citations.length} citation(s), all resolve in the hash-chained ledger`
            : "Ungrounded: missing or unverifiable citations"}
          <span className="ml-auto font-mono text-[10px] opacity-70">{result.latency_ms} ms</span>
        </div>
        <div className="question mb-2 text-xs text-slate-500">Q: {result.question}</div>
        <AnswerText text={result.answer} citations={result.citations} onCitationClick={onCitationClick} />
        <ul className="mt-4 space-y-1.5">
          {result.citations.map((c) => (
            <li key={c.short_id} className="flex gap-2 text-[11px]">
              <span className={`chip shrink-0 ${c.verified ? "bg-emerald-500/15 text-emerald-300" : "bg-rose-500/20 text-rose-300"}`}>
                {c.verified ? "✓" : "✗"} {c.mutation_type || "?"}
              </span>
              <span className="truncate font-mono text-slate-500" title={c.mutation_id || ""}>
                {c.mutation_id || `M:${c.short_id} not found in context/ledger`}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
