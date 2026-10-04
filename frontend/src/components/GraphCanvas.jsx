import { useCallback, useEffect, useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { Crosshair } from "lucide-react";
import { useElementSize } from "../hooks/useElementSize.js";
import { EDGE_COLORS, ENTITY_COLORS, entityColor } from "../theme.js";

// Mirrors force-graph's quadratic-bezier control point so labels sit on curved links.
function linkMidpoint(link) {
  const s = link.source;
  const t = link.target;
  if (typeof s !== "object" || typeof t !== "object") return null;
  const mx = (s.x + t.x) / 2;
  const my = (s.y + t.y) / 2;
  if (!link.curvature) return { x: mx, y: my, angle: Math.atan2(t.y - s.y, t.x - s.x) };
  const dx = t.x - s.x;
  const dy = t.y - s.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const a = Math.atan2(dy, dx);
  const d = len * link.curvature;
  const cx = mx + d * Math.cos(a - Math.PI / 2);
  const cy = my + d * Math.sin(a - Math.PI / 2);
  return { x: 0.25 * s.x + 0.5 * cx + 0.25 * t.x, y: 0.25 * s.y + 0.5 * cy + 0.25 * t.y, angle: a };
}

export default function GraphCanvas({ nodes, edges, selectedId, onSelectNode, highlightedEdgeIds, showHistory }) {
  const [containerRef, size] = useElementSize();
  const fgRef = useRef(null);
  const nodeCache = useRef(new Map());

  const graphData = useMemo(() => {
    // Reuse node objects across refreshes so the simulation keeps positions stable.
    const cache = nodeCache.current;
    const nextNodes = nodes.map((n) => {
      const existing = cache.get(n.id);
      if (existing) return Object.assign(existing, n);
      const created = { ...n };
      cache.set(n.id, created);
      return created;
    });
    const nodeIds = new Set(nodes.map((n) => n.id));

    const visible = edges.filter(
      (e) => nodeIds.has(e.edge.source_id) && nodeIds.has(e.edge.target_id) && (showHistory || e.valid_at_query_time),
    );
    const groups = new Map();
    visible.forEach((e) => {
      const key = [e.edge.source_id, e.edge.target_id].sort().join("|");
      groups.set(key, (groups.get(key) || 0) + 1);
    });
    const seen = new Map();
    const links = visible.map((e) => {
      const key = [e.edge.source_id, e.edge.target_id].sort().join("|");
      const count = groups.get(key);
      const idx = seen.get(key) || 0;
      seen.set(key, idx + 1);
      const flip = e.edge.source_id > e.edge.target_id ? -1 : 1;
      return {
        id: e.edge.id,
        source: e.edge.source_id,
        target: e.edge.target_id,
        relation: e.edge.relation,
        valid: e.valid_at_query_time,
        edge: e.edge,
        curvature: count > 1 ? flip * (idx - (count - 1) / 2) * 0.35 : 0,
      };
    });
    return { nodes: nextNodes, links };
  }, [nodes, edges, showHistory]);

  const neighborIds = useMemo(() => {
    if (!selectedId) return null;
    const ids = new Set([selectedId]);
    graphData.links.forEach((l) => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      if (s === selectedId) ids.add(t);
      if (t === selectedId) ids.add(s);
    });
    return ids;
  }, [selectedId, graphData]);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength(-140);
    fg.d3Force("link")?.distance(55);
  }, []);

  const linkColor = useCallback(
    (link) => {
      if (highlightedEdgeIds?.has(link.id)) return EDGE_COLORS.highlighted;
      const faded = neighborIds && !(neighborIds.has(link.source.id) && neighborIds.has(link.target.id));
      if (!link.valid) return faded ? "rgba(244,63,94,0.15)" : EDGE_COLORS.invalidatedFaded;
      return faded ? "rgba(16,185,129,0.12)" : EDGE_COLORS.active;
    },
    [highlightedEdgeIds, neighborIds],
  );

  const drawNode = useCallback(
    (node, ctx, scale) => {
      const isSelected = node.id === selectedId;
      const dimmed = neighborIds && !neighborIds.has(node.id);
      const radius = 3 + Math.min(6, Math.sqrt(node.degree_active || 1) * 1.6);
      ctx.globalAlpha = dimmed ? 0.2 : 1;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = entityColor(node.entity_type);
      ctx.fill();
      if (isSelected) {
        ctx.lineWidth = 2 / scale;
        ctx.strokeStyle = "#f8fafc";
        ctx.stroke();
      }
      if (scale > 0.9 || isSelected || (neighborIds && neighborIds.has(node.id))) {
        const fontSize = Math.max(10 / scale, 2.5);
        ctx.font = `${isSelected ? 600 : 500} ${fontSize}px Inter, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = "#e2e8f0";
        ctx.fillText(node.name, node.x, node.y + radius + 1.5);
      }
      ctx.globalAlpha = 1;
    },
    [selectedId, neighborIds],
  );

  const drawLinkLabel = useCallback(
    (link, ctx, scale) => {
      const highlighted = highlightedEdgeIds?.has(link.id);
      const focused = neighborIds && neighborIds.has(link.source.id) && neighborIds.has(link.target.id);
      if (!(highlighted || focused || scale > 2.2)) return;
      const mid = linkMidpoint(link);
      if (!mid) return;
      const fontSize = Math.max(8 / scale, 1.8);
      let angle = mid.angle;
      if (angle > Math.PI / 2 || angle < -Math.PI / 2) angle += Math.PI;
      ctx.save();
      ctx.translate(mid.x, mid.y);
      ctx.rotate(angle);
      ctx.font = `500 ${fontSize}px "JetBrains Mono", monospace`;
      const label = link.valid ? link.relation : `${link.relation} ✕`;
      const w = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(2,6,23,0.85)";
      ctx.fillRect(-w / 2 - 1.5, -fontSize / 2 - 1, w + 3, fontSize + 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = highlighted ? EDGE_COLORS.highlighted : link.valid ? "#6ee7b7" : "#fda4af";
      ctx.fillText(label, 0, 0);
      ctx.restore();
    },
    [highlightedEdgeIds, neighborIds],
  );

  return (
    <div className="relative h-full w-full" ref={containerRef}>
      {size.width > 0 && (
        <ForceGraph2D
          ref={fgRef}
          width={size.width}
          height={size.height}
          graphData={graphData}
          backgroundColor="rgba(0,0,0,0)"
          nodeId="id"
          nodeLabel={(n) => `${n.name} · ${n.entity_type}`}
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI);
            ctx.fill();
          }}
          linkColor={linkColor}
          linkWidth={(l) => (highlightedEdgeIds?.has(l.id) ? 2.6 : l.valid ? 1.3 : 1)}
          linkLineDash={(l) => (l.valid ? null : [4, 3])}
          linkCurvature="curvature"
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={0.92}
          linkDirectionalArrowColor={linkColor}
          linkDirectionalParticles={(l) => (highlightedEdgeIds?.has(l.id) ? 2 : 0)}
          linkDirectionalParticleWidth={2.2}
          linkDirectionalParticleColor={() => EDGE_COLORS.highlighted}
          linkLabel={(l) =>
            `<div style="font:12px Inter;padding:4px 6px;background:#0f172a;border:1px solid #334155;border-radius:6px">
              <b>${l.source.name}</b> —${l.relation}→ <b>${l.target.name}</b><br/>
              conf ${l.edge.confidence_score.toFixed(3)} · ${l.edge.t_start.slice(0, 16)} → ${
                l.edge.t_end ? l.edge.t_end.slice(0, 16) : "∞"
              }</div>`
          }
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={drawLinkLabel}
          onNodeClick={(node) => onSelectNode(node.id === selectedId ? null : node.id)}
          onBackgroundClick={() => onSelectNode(null)}
          cooldownTicks={120}
          d3VelocityDecay={0.3}
        />
      )}

      <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-x-3 gap-y-1 rounded-lg bg-slate-950/80 px-3 py-2 text-[11px] text-slate-400 ring-1 ring-slate-800">
        {Object.entries(ENTITY_COLORS).map(([type, color]) => (
          <span key={type} className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full" style={{ background: color }} />
            {type}
          </span>
        ))}
        <span className="flex items-center gap-1">
          <span className="h-0.5 w-4" style={{ background: EDGE_COLORS.active }} /> active
        </span>
        <span className="flex items-center gap-1">
          <span className="h-0 w-4 border-t border-dashed" style={{ borderColor: EDGE_COLORS.invalidated }} /> invalidated
        </span>
        <span className="flex items-center gap-1">
          <span className="h-0.5 w-4" style={{ background: EDGE_COLORS.highlighted }} /> in last prompt
        </span>
      </div>

      <button
        type="button"
        onClick={() => fgRef.current?.zoomToFit(400, 40)}
        className="btn-ghost absolute bottom-3 right-3 !px-2"
        title="Fit graph to view"
      >
        <Crosshair className="h-4 w-4" />
      </button>

      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-500">
          No facts valid at this point in time.
        </div>
      )}
    </div>
  );
}
