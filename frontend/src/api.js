// Thin fetch wrapper around the FastAPI backend.
// In dev, Vite proxies /api → http://localhost:8000; set VITE_API_BASE for other deployments.

const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// Optional workspace (multi-tenant / authenticated deployments). Kept in this browser only.
const WORKSPACE_KEY = "veritas.workspace";

export function getWorkspace() {
  try {
    return JSON.parse(window.localStorage.getItem(WORKSPACE_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

export function setWorkspace(workspace) {
  try {
    window.localStorage.setItem(WORKSPACE_KEY, JSON.stringify(workspace || {}));
  } catch {
    // storage unavailable (private mode): the setting lasts until reload
  }
}

function workspaceHeaders() {
  const { tenant, apiKey } = getWorkspace();
  const headers = {};
  if (tenant) headers["X-Tenant-ID"] = tenant;
  if (apiKey) headers["X-API-Key"] = apiKey;
  return headers;
}

async function request(path, { method = "GET", body, params } = {}) {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, String(value));
    });
  }
  const res = await fetch(url, {
    method,
    headers: { ...(body ? { "Content-Type": "application/json" } : {}), ...workspaceHeaders() },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ")
      : detail && typeof detail === "object"
        ? detail.message
        : detail || `${res.status} ${res.statusText}`;
    const error = new ApiError(res.status, message);
    error.errors = detail && typeof detail === "object" && !Array.isArray(detail) ? detail.errors : undefined;
    throw error;
  }
  return data;
}

export const api = {
  health: () => request("/api/health"),
  graph: ({ timestamp, includeHistory, asKnownAt } = {}) =>
    request("/api/graph", {
      params: { timestamp, include_history: includeHistory ? "true" : undefined, as_known_at: asKnownAt },
    }),
  timeline: () => request("/api/timeline"),
  audit: (entityId) => request("/api/audit", { params: { entity_id: entityId } }),
  verify: () => request("/api/audit/verify"),
  provenance: (entityId) => request(`/api/provenance/${encodeURIComponent(entityId)}`),
  observe: (payload) => request("/api/observe", { method: "POST", body: payload }),
  query: (payload) => request("/api/query", { method: "POST", body: payload }),
  reset: () => request("/api/reset", { method: "POST" }),
  seedDemo: () => request("/api/demo/seed", { method: "POST" }),
  aliases: () => request("/api/entities/aliases"),
  declareAlias: (alias, canonical) => request("/api/entities/alias", { method: "POST", body: { alias, canonical } }),
};

export const shortId = (uuid) => (uuid || "").replace(/-/g, "").slice(0, 8);

export function formatTs(iso, { withTime = true } = {}) {
  if (!iso) return "now";
  const d = new Date(iso);
  const date = d.toISOString().slice(0, 10);
  if (!withTime) return date;
  const time = d.toISOString().slice(11, 16);
  return time === "00:00" ? date : `${date} ${time}Z`;
}

// ---- A/B Lab ---------------------------------------------------------------
export const labApi = {
  models: () => request("/api/ab/models"),
  preset: () => request("/api/ab/preset"),
  dashboardObservations: () => request("/api/observations"),
  suggest: (observations, maxQuestions = 12) =>
    request("/api/ab/suggest-queries", { method: "POST", body: { observations, max_questions: maxQuestions } }),
  start: (payload) => request("/api/ab/runs", { method: "POST", body: payload }),
  get: (id) => request(`/api/ab/runs/${id}`),
  list: () => request("/api/ab/runs"),
  cancel: (id) => request(`/api/ab/runs/${id}/cancel`, { method: "POST" }),
};
