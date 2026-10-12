// Shared vocabulary for the A/B Lab: method identities and plain-language metric definitions.

export const METHODS = {
  A1: {
    name: "Full history",
    short: "A1",
    color: "#f59e0b",
    bar: "bg-amber-500",
    text: "text-amber-300",
    explain:
      "Keeps every fact ever said and pastes the whole list into the prompt, oldest first. Nothing is ever removed, so old and new values sit side by side and the model has to work out which is current.",
  },
  A2: {
    name: "Search (vector RAG)",
    short: "A2",
    color: "#a78bfa",
    bar: "bg-violet-500",
    text: "text-violet-300",
    explain:
      "Stores every fact and, for each question, pulls out the few that look most similar to the question. It is cheap, but it does not know which facts are out of date, and it can miss the one that matters.",
  },
  A3: {
    name: "Hybrid search (BM25 + vector)",
    short: "A3",
    color: "#f472b6",
    bar: "bg-pink-500",
    text: "text-pink-300",
    explain:
      "The setup most production RAG systems use: keyword search (BM25) and vector search, merged by rank. It finds exact names better than vector search alone, but still has no idea which facts are out of date.",
  },
  A4: {
    name: "Graph RAG (no time)",
    short: "A4",
    color: "#fb923c",
    bar: "bg-orange-500",
    text: "text-orange-300",
    explain:
      "Puts the same facts in a knowledge graph and hands the model the neighbourhood of the people and things in the question. The graph has no notion of time, so every value a fact ever had is listed and none is marked as old.",
  },
  A5: {
    name: "Overwrite memory (no audit)",
    short: "A5",
    color: "#94a3b8",
    bar: "bg-slate-400",
    text: "text-slate-300",
    explain:
      "Keeps one current value per fact and simply overwrites it with whatever arrived last, like many agent-memory layers. No history, no source trust, no ledger: a late rumor or an old record replaces the truth, and nothing can be cited.",
  },
  B: {
    name: "VERITAS (standard)",
    short: "B",
    color: "#38bdf8",
    bar: "bg-sky-500",
    text: "text-sky-300",
    explain:
      "Updates a knowledge graph as facts arrive: a new value closes the old one instead of piling up. The prompt gets only the current, relevant facts plus the written reasons for each change and long ledger IDs to cite.",
  },
  B2: {
    name: "VERITAS (compact)",
    short: "B2",
    color: "#10b981",
    bar: "bg-emerald-500",
    text: "text-emerald-300",
    explain:
      "Exactly the same facts as standard VERITAS, written more briefly: short instructions, [1]-style citation tags that the server maps back to the ledger, and \"replaced by\" notes on old facts. The written reasons stay in the ledger instead of the prompt.",
  },
};

METHODS.B3 = {
  name: "VERITAS (optimized)",
  short: "B3",
  color: "#22d3ee",
  bar: "bg-cyan-400",
  text: "text-cyan-300",
  explain:
    "Standard VERITAS with the optional retrieval settings switched on: only the question's neighbourhood is scored (faster on big graphs), crowded hubs such as a city everyone lives in are damped, and a date in the question (\"on March 3, 2025\") is answered as of that date.",
};

export const METRICS = [
  {
    key: "accuracy",
    label: "Accuracy",
    unit: "%",
    better: "higher",
    explain: "Out of all questions, how many the model answered correctly.",
  },
  {
    key: "changed_accuracy",
    label: "Accuracy on changed facts",
    unit: "%",
    better: "higher",
    explain:
      "Only the questions about facts that changed over time (someone moved, switched jobs…). This is where outdated memory hurts most.",
  },
  {
    key: "prompt_tokens_mean",
    label: "Prompt tokens",
    unit: "tok",
    better: "lower",
    explain:
      "How much text the model has to read per question (a token is roughly ¾ of a word). Fewer tokens means cheaper and faster, and leaves room for other work.",
  },
  {
    key: "latency_ms_median",
    label: "Median latency",
    unit: "ms",
    better: "lower",
    explain: "The typical time from sending the question to getting the full answer back.",
  },
  {
    key: "stale_answer_rate",
    label: "Stale answers",
    unit: "%",
    better: "lower",
    explain: "How often the answer contained an out-of-date or wrong value, such as an old city.",
  },
  {
    key: "citation_verified_rate",
    label: "Verified citations",
    unit: "%",
    better: "higher",
    explain:
      "VERITAS only: how often every source the model cited was checked against the tamper-proof ledger and found to be real and actually supplied.",
  },
];

export const CATEGORY_HELP = {
  changed: "The fact changed over time; asks for the current value.",
  historical: "Asks what a value was before a change.",
  stable: "The fact never changed (a control question).",
  "multi-hop": "Needs two facts joined together.",
  contested: "A low-trust rumor contradicted the fact; asks for the true current value.",
  "point-in-time": "Asks for the value on a specific past date.",
};

export function metricValue(summary, key) {
  if (!summary) return null;
  if (key === "changed_accuracy") return summary.accuracy_by_category?.changed ?? null;
  const v = summary[key];
  return v === undefined ? null : v;
}

export function formatMetric(value, unit) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (unit === "%") return `${(value * 100).toFixed(0)}%`;
  if (unit === "ms") return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
  return Math.round(value).toLocaleString();
}
