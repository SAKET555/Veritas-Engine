import { Loader2, Plus, Sparkles, Trash2, Wand2, X } from "lucide-react";
import { CATEGORY_HELP } from "./labMeta.js";

// Editable test questions with ground truth: the expected answer and the values that count as wrong.
export default function QueryEditor({ rows, onChange, onSuggest, onLoadDemo, suggesting, busy, note }) {
  const update = (i, patch) => onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const remove = (i) => onChange(rows.filter((_, j) => j !== i));
  const add = () => onChange([...rows, { question: "", expected: "", wrong: [], category: "changed" }]);

  return (
    <div className="flex min-h-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className="btn-primary !py-1 text-xs" onClick={onSuggest} disabled={busy || suggesting}>
          {suggesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
          Auto-generate from my facts
        </button>
        <button type="button" className="btn-ghost !py-1 text-xs" onClick={onLoadDemo} disabled={busy}>
          <Sparkles className="h-3.5 w-3.5" /> Demo questions
        </button>
        <button type="button" className="btn-ghost !py-1 text-xs" onClick={() => onChange([])} disabled={busy || !rows.length}>
          <Trash2 className="h-3.5 w-3.5" /> Clear
        </button>
        <span className="ml-auto text-[11px] text-slate-500">{rows.length} questions</span>
      </div>
      {note && <p className="rounded-md bg-sky-500/10 px-2 py-1 text-[11px] text-sky-200">{note}</p>}

      <div className="min-h-0 flex-1 overflow-auto rounded-lg border border-slate-800">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 z-10 bg-slate-900 text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1.5">Question</th>
              <th className="w-32 px-1 py-1.5">Correct answer</th>
              <th className="w-40 px-1 py-1.5" title="If the answer contains any of these, it is marked wrong (stale)">
                Wrong if it says
              </th>
              <th className="w-28 px-1 py-1.5">Type</th>
              <th className="w-6" />
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-t border-slate-800/70">
                <td className="px-2 py-1">
                  <input className="input !px-2 !py-1 text-xs" value={row.question} placeholder="Where does Priya live now?" onChange={(e) => update(i, { question: e.target.value })} disabled={busy} />
                </td>
                <td className="px-1">
                  <input className="input !px-2 !py-1 text-xs" value={row.expected} placeholder="Bengaluru" onChange={(e) => update(i, { expected: e.target.value })} disabled={busy} />
                </td>
                <td className="px-1">
                  <input
                    className="input !px-2 !py-1 text-xs"
                    value={row.wrong.join(", ")}
                    placeholder="Pune, Delhi"
                    onChange={(e) => update(i, { wrong: e.target.value.split(",").map((v) => v.trimStart()) })}
                    disabled={busy}
                  />
                </td>
                <td className="px-1">
                  <select className="input !px-1.5 !py-1 text-[11px]" value={row.category} title={CATEGORY_HELP[row.category]} onChange={(e) => update(i, { category: e.target.value })} disabled={busy}>
                    {Object.keys(CATEGORY_HELP).map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="pr-1">
                  <button type="button" onClick={() => remove(i)} className="rounded p-1 text-slate-500 hover:bg-slate-800 hover:text-rose-300" disabled={busy} aria-label="Remove question">
                    <X className="h-3.5 w-3.5" />
                  </button>
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={5} className="px-3 py-8 text-center text-slate-500">
                  Click “Auto-generate from my facts”, or write your own questions.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <button type="button" className="btn-ghost !py-1 text-xs" onClick={add} disabled={busy}>
        <Plus className="h-3.5 w-3.5" /> Add question
      </button>
    </div>
  );
}
