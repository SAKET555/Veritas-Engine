// Thin fetch wrapper around the FastAPI backend.
// In dev, Vite proxies /api → http://localhost:8000; set VITE_API_BASE for other deployments.

const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
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
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ")
      : detail || `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, message);
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
