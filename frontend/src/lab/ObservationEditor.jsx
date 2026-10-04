import { useState } from "react";
import { ClipboardPaste, Download, Plus, Sparkles, Trash2, X } from "lucide-react";

const SOURCES = ["user_statement", "sensor", "system_of_record", "agent_observation", "web", "agent_inference", "hearsay"];

const toLocalInput = (iso) => (iso ? new Date(iso).toISOString().slice(0, 16) : "");
const fromLocalInput = (value) => (value ? new Date(`${value}Z`).toISOString() : null);

// "[2026-01-05] Alice lives in Mumbai" or "2026-01-05 09:00 | Alice lives in Mumbai" or plain text.
function parsePasted(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const m = line.match(/^\[?(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?)\]?\s*(?:[|:-]\s+)?(.*)$/);
      if (m && m[2]) {
        return { text: m[2], timestamp: new Date(`${m[1].replace(" ", "T")}${m[1].length > 10 ? "" : "T09:00"}Z`).toISOString() };
      }
      return { text: line, timestamp: null };
    });
}

// Editable, ordered list of observations. Order = the order the agent learned them.
export default function ObservationEditor({ rows, onChange, errors, onLoadDemo, onImport, busy }) {
  const [pasting, setPasting] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const errorByIndex = Object.fromEntries((errors || []).map((e) => [e.index, e.error]));

  const update = (i, patch) => onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const remove = (i) => onChange(rows.filter((_, j) => j !== i));
  const add = () => onChange([...rows, { text: "", timestamp: null, source: "user_statement", confidence: 0.9 }]);

  return (
    <div className="flex min-h-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className="btn-ghost !py-1 text-xs" onClick={onLoadDemo} disabled={busy}>
          <Sparkles className="h-3.5 w-3.5" /> Demo facts
        </button>
        <button type="button" className="btn-ghost !py-1 text-xs" onClick={onImport} disabled={busy} title="Everything you observed on the Graph page">
          <Download className="h-3.5 w-3.5" /> Import from dashboard
        </button>
        <button type="button" className="btn-ghost !py-1 text-xs" onClick={() => setPasting((v) => !v)} disabled={busy}>
          <ClipboardPaste className="h-3.5 w-3.5" /> Paste lines
        </button>
        <button type="button" className="btn-ghost !py-1 text-xs" onClick={() => onChange([])} disabled={busy || !rows.length}>
          <Trash2 className="h-3.5 w-3.5" /> Clear
        </button>
        <span className="ml-auto text-[11px] text-slate-500">{rows.length} facts · top = learned first</span>
      </div>

      {pasting && (
        <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-2">
          <textarea
            className="input min-h-[120px] font-mono text-xs"
            placeholder={"One statement per line, oldest first. Dates are optional:\n[2026-01-05] Priya lives in Pune\n[2026-03-01] Priya moved to Bengaluru\nPriya works at Zoho"}
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
          />
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              className="btn-primary !py-1 text-xs"
              onClick={() => {
                const parsed = parsePasted(pasteText).map((r) => ({ ...r, source: "user_statement", confidence: 0.9 }));
                onChange([...rows, ...parsed]);
                setPasteText("");
                setPasting(false);
              }}
              disabled={!pasteText.trim()}
            >
              Add {parsePasted(pasteText).length || ""} lines
            </button>
            <button type="button" className="btn-ghost !py-1 text-xs" onClick={() => setPasting(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto rounded-lg border border-slate-800">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 z-10 bg-slate-900 text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="w-8 px-2 py-1.5">#</th>
              <th className="px-1 py-1.5">Statement</th>
              <th className="w-40 px-1 py-1.5">When (UTC, optional)</th>
              <th className="w-32 px-1 py-1.5">Source</th>
              <th className="w-16 px-1 py-1.5">Conf.</th>
              <th className="w-6" />
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className={`border-t border-slate-800/70 ${errorByIndex[i] ? "bg-rose-500/10" : ""}`}>
                <td className="px-2 font-mono text-[10px] text-slate-500">{i + 1}</td>
                <td className="px-1 py-1">
                  <input
                    className="input !px-2 !py-1 text-xs"
                    value={row.text}
                    placeholder="e.g. Priya moved to Bengaluru"
                    onChange={(e) => update(i, { text: e.target.value })}
                    disabled={busy}
                  />
                  {errorByIndex[i] && <div className="mt-0.5 text-[10px] text-rose-300">{errorByIndex[i].split(". Try")[0]}</div>}
                </td>
                <td className="px-1">
                  <input
                    type="datetime-local"
                    className="input !px-1.5 !py-1 text-[11px]"
                    value={toLocalInput(row.timestamp)}
                    onChange={(e) => update(i, { timestamp: fromLocalInput(e.target.value) })}
                    disabled={busy}
                  />
                </td>
                <td className="px-1">
                  <select className="input !px-1.5 !py-1 text-[11px]" value={row.source} onChange={(e) => update(i, { source: e.target.value })} disabled={busy}>
                    {SOURCES.map((s) => (
                      <option key={s} value={s}>
                        {s.replace(/_/g, " ")}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="px-1">
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    className="input !px-1.5 !py-1 text-[11px]"
                    value={row.confidence}
                    onChange={(e) => update(i, { confidence: Number(e.target.value) })}
                    disabled={busy}
                  />
                </td>
                <td className="pr-1">
                  <button type="button" onClick={() => remove(i)} className="rounded p-1 text-slate-500 hover:bg-slate-800 hover:text-rose-300" disabled={busy} aria-label="Remove fact">
                    <X className="h-3.5 w-3.5" />
                  </button>
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={6} className="px-3 py-8 text-center text-slate-500">
                  Add facts in the order your agent learns them — or load the demo.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <button type="button" className="btn-ghost !py-1 text-xs" onClick={add} disabled={busy}>
        <Plus className="h-3.5 w-3.5" /> Add fact
      </button>
    </div>
  );
}
