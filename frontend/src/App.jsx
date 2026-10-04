import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Database,
  Eye,
  EyeOff,
  GitBranch,
  Loader2,
  MessageSquarePlus,
  RotateCcw,
  ScrollText,
  Sparkles,
} from "lucide-react";
import { api } from "./api.js";
import GraphCanvas from "./components/GraphCanvas.jsx";
import TimelineSlider from "./components/TimelineSlider.jsx";
import AuditFeed from "./components/AuditFeed.jsx";
import NodeInspector from "./components/NodeInspector.jsx";
import ObservePanel from "./components/ObservePanel.jsx";
import QueryPanel from "./components/QueryPanel.jsx";
import QueryResults from "./components/QueryResults.jsx";

function Stat({ label, value, tone = "text-slate-100" }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <span className={`font-mono text-sm ${tone}`}>{value ?? "–"}</span>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [sliderIndex, setSliderIndex] = useState(0);
  const [showHistory, setShowHistory] = useState(false);
  const [graph, setGraph] = useState({ nodes: [], edges: [] });
  const [audit, setAudit] = useState([]);
  const [verification, setVerification] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [queryResult, setQueryResult] = useState(null);
  const [highlightedMutations, setHighlightedMutations] = useState(new Set());
  const [tab, setTab] = useState("console");
  const [revision, setRevision] = useState(0);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const followNow = useRef(true);

  const isNow = sliderIndex >= timeline.length;
  const asOf = isNow ? null : timeline[sliderIndex];

  // Ledger, timeline and health change only when state is mutated.
  useEffect(() => {
    let cancelled = false;
    Promise.all([api.timeline(), api.audit(), api.verify(), api.health()])
      .then(([tl, au, ver, h]) => {
        if (cancelled) return;
        setTimeline(tl.timestamps);
        setAudit(au.records);
        setVerification(ver);
        setHealth(h);
        setError(null);
        if (followNow.current) setSliderIndex(tl.timestamps.length);
      })
      .catch((err) => !cancelled && setError(`Backend unreachable: ${err.message}`));
    return () => {
      cancelled = true;
    };
  }, [revision]);

  useEffect(() => {
    followNow.current = isNow;
  }, [isNow]);

  // Graph view follows the timeline position and the history toggle.
  useEffect(() => {
    let cancelled = false;
    setLoadingGraph(true);
    api
      .graph({ timestamp: asOf, includeHistory: showHistory })
      .then((g) => !cancelled && setGraph(g))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoadingGraph(false));
    return () => {
      cancelled = true;
    };
  }, [asOf, showHistory, revision]);

  const refresh = useCallback(() => setRevision((r) => r + 1), []);

  const runAdmin = async (kind) => {
    if (kind === "reset" && !window.confirm("Archive the current ledger and start with an empty graph?")) return;
    setBusy(kind);
    try {
      if (kind === "seed") await api.seedDemo();
      else await api.reset();
      followNow.current = true;
      setQueryResult(null);
      setSelectedId(null);
      setHighlightedMutations(new Set());
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  };

  const highlightedEdgeIds = useMemo(
    () => new Set(queryResult ? queryResult.scored_edges.filter((e) => e.kept).map((e) => e.edge_id) : []),
    [queryResult],
  );

  const handleQueryResult = (res) => {
    setQueryResult(res);
    setHighlightedMutations(new Set(res.citations.map((c) => c.short_id)));
  };

  const focusCitation = (shortId) => {
    setHighlightedMutations(new Set([shortId]));
    setTab("audit");
  };

  const stats = health?.graph;

  return (
    <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col gap-3 p-3 lg:p-4">
      <header className="panel flex flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-emerald-500/15 p-2 text-emerald-400">
            <GitBranch className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-100">VERITAS</h1>
            <p className="text-[11px] text-slate-500">Verifiable knowledge-state editing for long-horizon agents</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-5">
          <Stat label="entities" value={stats?.entities} />
          <Stat label="active facts" value={stats?.edges_active} tone="text-emerald-400" />
          <Stat label="historical" value={stats?.edges_historical} tone="text-rose-400" />
          <Stat label="ledger" value={health?.ledger_records} tone="text-sky-400" />
          <Stat label="frozen LLM" value={health?.llm_model} tone="text-violet-300" />
        </div>
        <div className="ml-auto flex gap-2">
          <button type="button" className="btn-ghost" onClick={() => runAdmin("seed")} disabled={!!busy}>
            {busy === "seed" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            Load demo trajectory
          </button>
          <button type="button" className="btn-ghost" onClick={() => runAdmin("reset")} disabled={!!busy}>
            {busy === "reset" ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
            Reset
          </button>
        </div>
      </header>

      {error && (
        <div className="rounded-lg bg-rose-500/10 px-4 py-2 text-sm text-rose-300 ring-1 ring-rose-500/30">
          {error} — start the API with <code className="font-mono">uvicorn main:app --port 8000</code> in backend/.
        </div>
      )}

      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[minmax(0,1fr)_420px]">
        <div className="flex min-h-0 flex-col gap-3">
          <div className="panel relative h-[58vh] min-h-[420px] overflow-hidden">
            <div className="absolute right-3 top-3 z-10 flex items-center gap-2">
              {loadingGraph && <Loader2 className="h-4 w-4 animate-spin text-slate-500" />}
              <span className="chip bg-slate-950/80 text-slate-400 ring-1 ring-slate-800">
                {graph.nodes.length} nodes · {graph.edges.filter((e) => e.valid_at_query_time).length} valid
                {showHistory && ` · ${graph.edges.filter((e) => !e.valid_at_query_time).length} invalidated`}
              </span>
              <button
                type="button"
                className={`btn-ghost !px-2 !py-1 text-xs ${showHistory ? "!border-rose-500/50 !text-rose-300" : ""}`}
                onClick={() => setShowHistory((v) => !v)}
                title="Toggle invalidated (historical) edges"
              >
                {showHistory ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
                history
              </button>
            </div>
            <GraphCanvas
              nodes={graph.nodes}
              edges={graph.edges}
              selectedId={selectedId}
              onSelectNode={setSelectedId}
              highlightedEdgeIds={highlightedEdgeIds}
              showHistory={showHistory}
            />
            <NodeInspector
              entityId={selectedId}
              refreshKey={revision}
              onClose={() => setSelectedId(null)}
              onSelectEntity={setSelectedId}
            />
            {health && health.ledger_records === 0 && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-slate-950/70">
                <Database className="h-8 w-8 text-slate-600" />
                <p className="text-sm text-slate-400">The knowledge graph is empty.</p>
                <button type="button" className="btn-primary" onClick={() => runAdmin("seed")} disabled={!!busy}>
                  <Sparkles className="h-4 w-4" /> Load the T0 → T1 → T2 demo trajectory
                </button>
              </div>
            )}
          </div>
          <TimelineSlider timestamps={timeline} index={Math.min(sliderIndex, timeline.length)} onChange={setSliderIndex} />
        </div>

        <aside className="panel flex min-h-[520px] flex-col lg:h-[calc(58vh+7.5rem)]">
          <div className="flex border-b border-slate-800">
            {[
              ["console", "Agent console", Bot],
              ["audit", `Audit ledger (${audit.length})`, ScrollText],
            ].map(([key, label, Icon]) => (
              <button
                key={key}
                type="button"
                onClick={() => setTab(key)}
                className={`flex flex-1 items-center justify-center gap-2 px-3 py-2.5 text-sm transition ${
                  tab === key ? "border-b-2 border-emerald-500 text-slate-100" : "text-slate-500 hover:text-slate-300"
                }`}
              >
                <Icon className="h-4 w-4" /> {label}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {tab === "console" ? (
              <div className="space-y-6">
                <section>
                  <h2 className="panel-title mb-3">
                    <MessageSquarePlus className="h-3.5 w-3.5" /> Observe a non-stationary fact
                  </h2>
                  <ObservePanel onCommitted={refresh} />
                </section>
                <section>
                  <h2 className="panel-title mb-3">
                    <Bot className="h-3.5 w-3.5" /> Ask the frozen agent
                  </h2>
                  <QueryPanel asOf={asOf} onResult={handleQueryResult} />
                </section>
              </div>
            ) : (
              <AuditFeed
                records={audit}
                verification={verification}
                highlightedIds={highlightedMutations}
                onSelectEntity={setSelectedId}
              />
            )}
          </div>
        </aside>
      </div>

      {queryResult && <QueryResults result={queryResult} onCitationClick={focusCitation} />}
    </div>
  );
}
