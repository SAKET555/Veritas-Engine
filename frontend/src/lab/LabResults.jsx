import { useMemo, useState } from "react";
import { Check, Trophy, X } from "lucide-react";
import { METHODS, METRICS, formatMetric, metricValue } from "./labMeta.js";

function MetricCard({ metric, methods, summary }) {
  const values = methods.map((k) => ({ key: k, value: metricValue(summary[k], metric.key) })).filter((v) => v.value !== null);
  if (!values.length) return null;
  const max = Math.max(...values.map((v) => v.value), metric.unit === "%" ? 1 : 0) || 1;
  const best =
    values.length > 1
      ? values.reduce((a, b) => ((metric.better === "higher" ? b.value > a.value : b.value < a.value) ? b : a)).key
      : null;
  const tied = best && values.filter((v) => v.value === values.find((x) => x.key === best).value).length > 1;

  return (
    <div className="panel p-3">
      <div className="mb-0.5 flex items-center justify-between">
        <h4 className="text-sm font-medium text-slate-100">{metric.label}</h4>
        <span className="text-[10px] text-slate-500">{metric.better} is better</span>
      </div>
      <p className="mb-3 text-[11px] leading-snug text-slate-500">{metric.explain}</p>
      <div className="space-y-2">
        {values.map(({ key, value }) => (
          <div key={key}>
            <div className="mb-0.5 flex items-center justify-between text-[11px]">
              <span className={`flex items-center gap-1 ${METHODS[key].text}`}>
                {METHODS[key].short} · {METHODS[key].name}
                {key === best && !tied && <Trophy className="h-3 w-3 text-yellow-300" />}
              </span>
              <span className="font-mono text-slate-200">{formatMetric(value, metric.unit)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-800">
              <div className={`h-full rounded-full ${METHODS[key].bar}`} style={{ width: `${Math.max(2, (value / max) * 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RowDetail({ row, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="panel flex max-h-[85vh] w-full max-w-3xl flex-col !bg-slate-900 p-4" onClick={(e) => e.stopPropagation()}>
        <div className="mb-2 flex items-start justify-between gap-3">
          <div>
            <div className={`text-xs ${METHODS[row.method].text}`}>
              {row.method} · {METHODS[row.method].name}
            </div>
            <h3 className="text-sm font-medium text-slate-100">{row.query}</h3>
            <p className="text-[11px] text-slate-500">
              expected <b className="text-slate-300">{row.expected}</b> · {row.correct ? "correct" : "wrong"} ·{" "}
              {(row.model_prompt_tokens ?? row.est_prompt_tokens).toLocaleString()} prompt tokens · {Math.round(row.wall_ms).toLocaleString()} ms
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-800" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Model reply</div>
        <pre className="mb-3 whitespace-pre-wrap rounded-lg bg-slate-950 p-3 text-xs text-slate-100">{row.raw_output}</pre>
        {row.extra?.citations?.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-1.5 text-[11px]">
            {row.extra.citations.map((c, i) => (
              <span key={i} className={`chip ${c.verified ? "bg-emerald-500/15 text-emerald-300" : "bg-rose-500/20 text-rose-300"}`}>
                {c.verified ? "✓" : "✗"} {c.tag ? `[${c.tag}] → ` : ""}M:{c.id} {c.type || ""}
              </span>
            ))}
          </div>
        )}
        <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">What the model was given (memory + question)</div>
        <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-950 p-3 font-mono text-[11px] text-sky-100/80">{row.prompt}</pre>
      </div>
    </div>
  );
}

export default function LabResults({ job }) {
  const [detail, setDetail] = useState(null);
  const summary = job.summary || {};
  const methods = (job.config?.methods || job.request_methods || Object.keys(summary)).filter((k) => METHODS[k]);
  const rows = job.rows || [];

  const questions = useMemo(() => {
    const order = [];
    rows.forEach((r) => {
      if (!order.find((q) => q.query === r.query)) order.push({ query: r.query, expected: r.expected, category: r.category });
    });
    return order;
  }, [rows]);

  const cell = (q, m) => rows.filter((r) => r.query === q.query && r.method === m);

  return (
    <div className="space-y-3">
      {Object.keys(summary).length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {METRICS.map((metric) => (
            <MetricCard key={metric.key} metric={metric} methods={methods} summary={summary} />
          ))}
        </div>
      )}

      {questions.length > 0 && (
        <div className="panel overflow-x-auto p-3">
          <h4 className="mb-2 text-sm font-medium text-slate-100">Answers per question</h4>
          <p className="mb-2 text-[11px] text-slate-500">Click any cell to see exactly what the model was given and what it replied.</p>
          <table className="w-full min-w-[720px] text-left text-xs">
            <thead className="text-[10px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="py-1 pr-2">Question</th>
                <th className="px-2">Correct</th>
                {methods.map((m) => (
                  <th key={m} className={`px-2 ${METHODS[m].text}`}>
                    {m} · {METHODS[m].name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {questions.map((q) => (
                <tr key={q.query} className="border-t border-slate-800/70 align-top">
                  <td className="py-1.5 pr-2">
                    <div className="text-slate-200">{q.query}</div>
                    <span className="chip mt-0.5 bg-slate-800 text-slate-400">{q.category}</span>
                  </td>
                  <td className="px-2 py-1.5 font-medium text-slate-200">{q.expected}</td>
                  {methods.map((m) => {
                    const runs = cell(q, m);
                    if (!runs.length) return <td key={m} className="px-2 py-1.5 text-slate-600">…</td>;
                    const r = runs[0];
                    const ok = runs.filter((x) => x.correct).length;
                    return (
                      <td key={m} className="px-2 py-1.5">
                        <button
                          type="button"
                          onClick={() => setDetail(r)}
                          className={`w-full rounded-md px-2 py-1 text-left ring-1 transition hover:brightness-125 ${
                            r.correct ? "bg-emerald-500/10 ring-emerald-500/30" : "bg-rose-500/10 ring-rose-500/30"
                          }`}
                        >
                          <span className="flex items-center gap-1">
                            {r.correct ? <Check className="h-3 w-3 text-emerald-300" /> : <X className="h-3 w-3 text-rose-300" />}
                            <span className="truncate text-slate-100">{r.answer || "—"}</span>
                          </span>
                          <span className="font-mono text-[10px] text-slate-500">
                            {(r.model_prompt_tokens ?? r.est_prompt_tokens).toLocaleString()} tok · {(r.wall_ms / 1000).toFixed(1)} s
                            {runs.length > 1 && ` · ${ok}/${runs.length} runs`}
                          </span>
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {detail && <RowDetail row={detail} onClose={() => setDetail(null)} />}
    </div>
  );
}
