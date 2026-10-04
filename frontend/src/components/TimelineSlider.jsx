import { useEffect, useRef, useState } from "react";
import { Clock, Pause, Play, SkipForward } from "lucide-react";
import { formatTs } from "../api.js";

// index === timestamps.length means "NOW" (current active state, fast path).
export default function TimelineSlider({ timestamps, index, onChange }) {
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);
  const max = timestamps.length;
  const isNow = index >= max;

  // Advance one discrete timestamp per tick; stop on reaching NOW.
  useEffect(() => {
    if (!playing) return undefined;
    if (index >= max) {
      setPlaying(false);
      return undefined;
    }
    timer.current = setTimeout(() => onChange(index + 1), 900);
    return () => clearTimeout(timer.current);
  }, [playing, index, max, onChange]);

  const startPlayback = () => {
    if (isNow) onChange(0);
    setPlaying(true);
  };

  return (
    <div className="panel px-4 py-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="panel-title">
          <Clock className="h-3.5 w-3.5" /> Time-travel · valid time
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`chip ${isNow ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-500/15 text-amber-300"}`}
          >
            {isNow ? "T_now · active subgraph" : `T${index} · ${formatTs(timestamps[index])}`}
          </span>
          <button
            type="button"
            className="btn-ghost !px-2 !py-1"
            onClick={() => (playing ? setPlaying(false) : startPlayback())}
            disabled={max === 0}
            title={playing ? "Pause replay" : "Replay history from T0"}
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>
          <button
            type="button"
            className="btn-ghost !px-2 !py-1"
            onClick={() => {
              setPlaying(false);
              onChange(max);
            }}
            disabled={isNow}
            title="Jump to now"
          >
            <SkipForward className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      <input
        type="range"
        min={0}
        max={max}
        step={1}
        value={Math.min(index, max)}
        onChange={(e) => {
          setPlaying(false);
          onChange(Number(e.target.value));
        }}
        className="w-full accent-emerald-500"
        disabled={max === 0}
        aria-label="Timeline position"
      />

      <div className="relative mt-1 h-4 text-[10px] text-slate-500">
        {max > 0 &&
          [0, Math.floor(max / 4), Math.floor(max / 2), Math.floor((3 * max) / 4)]
            .filter((v, i, arr) => arr.indexOf(v) === i)
            .map((i) => (
              <span
                key={i}
                className="absolute -translate-x-1/2 whitespace-nowrap font-mono"
                style={{ left: `${(i / max) * 100}%` }}
              >
                T{i} {formatTs(timestamps[i], { withTime: false })}
              </span>
            ))}
        <span className="absolute right-0 font-mono text-emerald-400">NOW</span>
      </div>
    </div>
  );
}
