import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Cpu, FlaskConical, HelpCircle, Loader2, Play, Square } from "lucide-react";
import { labApi } from "../api.js";
import { METHODS } from "./labMeta.js";
import ObservationEditor from "./ObservationEditor.jsx";
import QueryEditor from "./QueryEditor.jsx";
import LabResults from "./LabResults.jsx";

const STORAGE_KEY = "veritas.lab.v1";

function loadSaved() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function save(state) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* storage unavailable (private mode) — the lab still works, it just won't remember */
  }
}

function Explainer() {
  const [open, setOpen] = useState(false);
  return (
    <div className="panel p-3">
      <button type="button" onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-2 text-left text-sm text-slate-200">
        <HelpCircle className="h-4 w-4 text-sky-400" />
        How does this comparison work?
        {open ? <ChevronDown className="ml-auto h-4 w-4" /> : <ChevronRight className="ml-auto h-4 w-4" />}
      </button>
      {open && (
        <div className="mt-3 space-y-3 text-xs leading-relaxed text-slate-400">
          <p>
            You give a list of facts in the order an agent would learn them (people move, change jobs, change their
            minds) and some test questions. Every method below is fed <b className="text-slate-200">the same facts</b>{" "}
            and asked <b className="text-slate-200">the same questions</b> by <b className="text-slate-200">the same model</b>.
            The only difference is how each one remembers — i.e. what text it puts in front of the model.
          </p>
          <div className="grid gap-2 md:grid-cols-2">
            {Object.entries(METHODS).map(([key, m]) => (
              <div key={key} className="rounded-lg border border-slate-800 p-2">
                <div className={`mb-1 text-xs font-medium ${m.text}`}>
                  {key} · {m.name}
                </div>
                <p>{m.explain}</p>
              </div>
            ))}
          </div>
          <p>
            An answer counts as correct only if its <code>ANSWER:</code> line contains the correct value and none of the
            “wrong if it says” values — so hedging (“Berlin or Munich”) is marked wrong. Auto-generated questions take
            their correct answers from VERITAS’s own reconciliation of your facts (newest valid fact wins unless its
            source is much less trustworthy), so glance over them before running.
          </p>
          <p>
            Tip: with only a handful of facts, “Full history” is already short, so VERITAS mainly wins on accuracy; its
            token and speed advantage grows as the history gets longer. Running on CPU is slower but avoids GPU driver
            crashes on small GPUs.
          </p>
        </div>
      )}
    </div>
  );
}

export default function ABLab() {
  const saved = useRef(loadSaved());
  const [observations, setObservations] = useState(saved.current?.observations || []);
  const [queries, setQueries] = useState(saved.current?.queries || []);
  const [settings, setSettings] = useState(
    saved.current?.settings || { model: "llama3.2", cpuOnly: true, repeats: 1, methods: ["A1", "A2", "B", "B2"] },
  );
  const [models, setModels] = useState({ ollama_available: false, models: [] });
  const [obsErrors, setObsErrors] = useState([]);
  const [error, setError] = useState(null);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestNote, setSuggestNote] = useState(null);
  const [job, setJob] = useState(null);
  const [starting, setStarting] = useState(false);

  const running = job && (job.status === "queued" || job.status === "running");

  useEffect(() => {
    save({ observations, queries, settings });
  }, [observations, queries, settings]);

  useEffect(() => {
    labApi.models().then(setModels).catch(() => setModels({ ollama_available: false, models: [] }));
    if (!saved.current) {
      labApi
        .preset()
        .then((p) => {
          setObservations(p.observations);
          setQueries(p.queries);
        })
        .catch((err) => setError(err.message));
    }
    // Re-attach to a run that is still going (e.g. after switching pages).
    labApi
      .list()
      .then(({ runs }) => {
        const latest = runs[0];
        if (latest) labApi.get(latest.id).then(setJob).catch(() => {});
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(() => {
      labApi.get(job.id).then(setJob).catch((err) => setError(err.message));
    }, 1500);
    return () => clearInterval(id);
  }, [running, job?.id]);

  const cleanObservations = useCallback(
    () =>
      observations
        .filter((o) => o.text && o.text.trim())
        .map((o) => ({ text: o.text.trim(), timestamp: o.timestamp || null, source: o.source || "user_statement", confidence: Number(o.confidence) || 0.9 })),
    [observations],
  );

  const handleObsError = (err) => {
    setError(err.message);
    if (err.errors) {
      // Server indexes refer to the cleaned list; map back to the editor rows.
      const nonEmpty = observations.map((o, i) => (o.text && o.text.trim() ? i : null)).filter((i) => i !== null);
      setObsErrors(err.errors.map((e) => ({ ...e, index: nonEmpty[e.index] })));
    }
  };

  const loadDemo = async (what) => {
    try {
      const p = await labApi.preset();
      if (what !== "queries") setObservations(p.observations);
      if (what !== "facts") setQueries(p.queries);
      setObsErrors([]);
    } catch (err) {
      setError(err.message);
    }
  };

  const importDashboard = async () => {
    try {
      const { observations: obs } = await labApi.dashboardObservations();
      if (!obs.length) {
        setError("No observations recorded on the Graph page yet. Add some there (or load the demo trajectory) first.");
        return;
      }
      setObservations(obs);
      setObsErrors([]);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const suggest = async () => {
    setSuggesting(true);
    setError(null);
    setObsErrors([]);
    try {
      const res = await labApi.suggest(cleanObservations(), 12);
      setQueries(res.queries);
      const a = res.available;
      setSuggestNote(
        res.queries.length
          ? `Generated ${res.queries.length} questions (found ${a.changed} changed, ${a.historical} history, ${a.stable} stable facts). Correct answers come from VERITAS's reconciliation — check them.`
          : "No single-valued facts found to ask about. Add facts like 'X lives in Y' / 'X works at Z', including some that change.",
      );
    } catch (err) {
      handleObsError(err);
    } finally {
      setSuggesting(false);
    }
  };

  const start = async () => {
    setStarting(true);
    setError(null);
    setObsErrors([]);
    const cleanQueries = queries
      .filter((q) => q.question.trim() && q.expected.trim())
      .map((q) => ({ ...q, wrong: q.wrong.map((w) => w.trim()).filter(Boolean) }));
    if (!cleanQueries.length) {
      setError("Add at least one question with a correct answer.");
      setStarting(false);
      return;
    }
    const mock = settings.model === "__mock__";
    try {
      const created = await labApi.start({
        observations: cleanObservations(),
        queries: cleanQueries,
        methods: settings.methods,
        provider: mock ? "mock" : "ollama",
        model: mock ? "mock" : settings.model,
        num_gpu: !mock && settings.cpuOnly ? 0 : null,
        repeats: settings.repeats,
      });
      setJob(created);
    } catch (err) {
      handleObsError(err);
    } finally {
      setStarting(false);
    }
  };

  const toggleMethod = (key) =>
    setSettings((s) => {
      const has = s.methods.includes(key);
      const next = has ? s.methods.filter((m) => m !== key) : [...s.methods, key];
      return { ...s, methods: Object.keys(METHODS).filter((m) => next.includes(m)) };
    });

  const progress = job?.progress?.total ? job.progress.done / job.progress.total : 0;
  const modelOptions = models.models.length ? models.models : ["llama3.2"];

  return (
    <div className="space-y-3">
      <div className="panel flex flex-wrap items-center gap-x-5 gap-y-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-emerald-400" />
          <div>
            <h2 className="text-sm font-semibold text-slate-100">A/B Lab</h2>
            <p className="text-[11px] text-slate-500">Same facts, same questions, same model — only the memory differs.</p>
          </div>
        </div>

        <label className="flex items-center gap-2 text-xs text-slate-400">
          Model
          <select className="input !w-auto !py-1 text-xs" value={settings.model} onChange={(e) => setSettings((s) => ({ ...s, model: e.target.value }))} disabled={running}>
            {modelOptions.map((m) => (
              <option key={m} value={m.replace(/:latest$/, "")}>
                {m}
              </option>
            ))}
            <option value="__mock__">offline mock (no model — only meaningful for VERITAS)</option>
          </select>
        </label>
        {!models.ollama_available && <span className="text-[11px] text-amber-300">Ollama not detected on :11434</span>}

        <label className="flex items-center gap-1.5 text-xs text-slate-400" title="Run the model on CPU only (slower, but avoids GPU driver crashes)">
          <input type="checkbox" checked={settings.cpuOnly} onChange={(e) => setSettings((s) => ({ ...s, cpuOnly: e.target.checked }))} disabled={running} className="accent-emerald-500" />
          <Cpu className="h-3.5 w-3.5" /> CPU only
        </label>

        <label className="flex items-center gap-2 text-xs text-slate-400" title="Repeat every question this many times for steadier numbers">
          Repeats
          <select className="input !w-auto !py-1 text-xs" value={settings.repeats} onChange={(e) => setSettings((s) => ({ ...s, repeats: Number(e.target.value) }))} disabled={running}>
            {[1, 2, 3, 5].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>

        <div className="flex flex-wrap items-center gap-1.5">
          {Object.entries(METHODS).map(([key, m]) => (
            <button
              key={key}
              type="button"
              onClick={() => toggleMethod(key)}
              disabled={running}
              title={m.explain}
              className={`chip !px-2 !py-1 ring-1 transition ${
                settings.methods.includes(key) ? `${m.text} bg-slate-800 ring-slate-600` : "text-slate-600 ring-slate-800"
              }`}
            >
              {key} {m.name}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {running ? (
            <button type="button" className="btn-ghost !border-rose-500/50 text-rose-300" onClick={() => labApi.cancel(job.id).then(() => labApi.get(job.id)).then(setJob)}>
              <Square className="h-4 w-4" /> Stop
            </button>
          ) : (
            <button type="button" className="btn-primary" onClick={start} disabled={starting || !settings.methods.length || !observations.length}>
              {starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Run comparison
            </button>
          )}
        </div>
      </div>

      <Explainer />

      {error && <div className="rounded-lg bg-rose-500/10 px-4 py-2 text-sm text-rose-300 ring-1 ring-rose-500/30">{error}</div>}

      <div className="grid gap-3 xl:grid-cols-2">
        <section className="panel flex h-[440px] flex-col p-3">
          <h3 className="panel-title mb-2">1 · Facts your agent learns (in order)</h3>
          <ObservationEditor
            rows={observations}
            onChange={(rows) => {
              setObservations(rows);
              setObsErrors([]);
            }}
            errors={obsErrors}
            onLoadDemo={() => loadDemo("facts")}
            onImport={importDashboard}
            busy={running}
          />
        </section>
        <section className="panel flex h-[440px] flex-col p-3">
          <h3 className="panel-title mb-2">2 · Test questions</h3>
          <QueryEditor rows={queries} onChange={setQueries} onSuggest={suggest} onLoadDemo={() => loadDemo("queries")} suggesting={suggesting} busy={running} note={suggestNote} />
        </section>
      </div>

      {job && (
        <section className="space-y-3">
          <div className="panel p-3">
            <div className="mb-1.5 flex flex-wrap items-center gap-2 text-xs">
              <h3 className="panel-title">3 · Results</h3>
              <span className={`chip ${job.status === "done" ? "bg-emerald-500/15 text-emerald-300" : job.status === "error" ? "bg-rose-500/20 text-rose-300" : "bg-sky-500/15 text-sky-300"}`}>
                {job.phase}
              </span>
              <span className="text-slate-500">
                {job.progress.done}/{job.progress.total} answers · {job.config?.model || ""} ·{" "}
                {job.config?.observations ?? "…"} facts · {job.config?.queries ?? "…"} questions
                {job.config?.ledger_verification && ` · ledger: ${job.config.ledger_verification}`}
              </span>
              {running && <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-400" />}
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-sky-500 transition-all" style={{ width: `${progress * 100}%` }} />
            </div>
            {job.error && <p className="mt-2 text-xs text-rose-300">{job.error}</p>}
            {job.report_dir && <p className="mt-2 text-[11px] text-slate-500">Saved report: <code>{job.report_dir}</code></p>}
          </div>
          <LabResults job={job} />
        </section>
      )}
    </div>
  );
}
