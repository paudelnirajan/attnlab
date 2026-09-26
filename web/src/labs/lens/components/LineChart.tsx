import { useLayoutEffect, useMemo, useRef, useState } from "react";

// A small line chart over the lens rows: one x position per residual-stream
// row, one line per series. SVG, sized to its container, with a crosshair and
// tooltip on hover (the dataviz rules: 2px lines, one y-axis, recessive grid,
// legend for >= 2 series plus direct end labels, text in ink colours).

export interface Series {
  id: string;
  label: string;
  values: (number | null)[];
  /** a CSS colour, normally var(--series-N) */
  color: string;
  dashed?: boolean;
}

interface Props {
  xLabels: string[];
  series: Series[];
  yLabel: string;
  format: (v: number) => string;
  /** log scale (ranks, KL); values must be > 0 */
  log?: boolean;
  /** draw larger values lower (ranks: 1 at the top) */
  invert?: boolean;
  yMin?: number;
  yMax?: number;
  /** a horizontal reference line, e.g. 0 for a logit difference */
  zeroLine?: number;
  highlight?: number | null;
  onPick?: (i: number) => void;
  height?: number;
  ariaLabel: string;
}

const PAD = { top: 12, right: 88, bottom: 56, left: 52 };

function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (el.clientWidth) setW(el.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function niceTicks(lo: number, hi: number, log: boolean): number[] {
  if (log) {
    const out: number[] = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(10 ** e);
    return out.filter((t) => t >= lo * 0.999 && t <= hi * 1.001);
  }
  const span = hi - lo || 1;
  const step = 10 ** Math.floor(Math.log10(span / 4));
  const mult = [1, 2, 2.5, 5, 10].find((m) => span / (m * step) <= 5) ?? 10;
  const s = mult * step;
  const out: number[] = [];
  for (let t = Math.ceil(lo / s) * s; t <= hi + 1e-9; t += s) out.push(Number(t.toFixed(10)));
  return out;
}

export function LineChart({
  xLabels,
  series,
  yLabel,
  format,
  log = false,
  invert = false,
  yMin,
  yMax,
  zeroLine,
  highlight = null,
  onPick,
  height = 240,
  ariaLabel,
}: Props) {
  const [wrapRef, width] = useWidth<HTMLDivElement>(640);
  const [hover, setHover] = useState<number | null>(null);
  const n = xLabels.length;
  const plotW = Math.max(40, width - PAD.left - PAD.right);
  const plotH = height - PAD.top - PAD.bottom;

  const [lo, hi] = useMemo(() => {
    const vals = series.flatMap((s) => s.values).filter((v): v is number => v !== null && Number.isFinite(v));
    let a = yMin ?? Math.min(...vals, zeroLine ?? Infinity);
    let b = yMax ?? Math.max(...vals, zeroLine ?? -Infinity);
    if (!vals.length) [a, b] = [0, 1];
    if (log) {
      a = Math.max(yMin ?? 1e-9, Math.min(a, b));
      b = Math.max(b, a * 10);
      a = 10 ** Math.floor(Math.log10(a));
      b = 10 ** Math.ceil(Math.log10(b));
    } else if (a === b) {
      a -= 1;
      b += 1;
    }
    return [a, b];
  }, [series, yMin, yMax, log, zeroLine]);

  const x = (i: number) => PAD.left + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const t = (v: number) => (log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo));
  const y = (v: number) => {
    const f = Math.min(1, Math.max(0, t(v)));
    return PAD.top + (invert ? f : 1 - f) * plotH;
  };
  const ticks = niceTicks(lo, hi, log);
  const labelEvery = Math.max(1, Math.ceil(n / Math.max(1, Math.floor(plotW / 34))));

  const paths = series.map((s) => {
    let d = "";
    s.values.forEach((v, i) => {
      if (v === null || !Number.isFinite(v) || (log && v <= 0)) return;
      d += `${d && s.values[i - 1] !== null ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    });
    return d;
  });

  // direct end labels, nudged apart so two series ending together don't overlap
  const ends = series
    .map((s, k) => {
      const i = s.values.map((v, j) => (v !== null ? j : -1)).filter((j) => j >= 0).pop();
      return i === undefined ? null : { k, y: y(s.values[i] as number) };
    })
    .filter((e): e is { k: number; y: number } => e !== null)
    .sort((a, b) => a.y - b.y);
  for (let j = 1; j < ends.length; j++) if (ends[j].y - ends[j - 1].y < 13) ends[j].y = ends[j - 1].y + 13;

  function indexAt(clientX: number, rect: DOMRect): number {
    const px = ((clientX - rect.left) / rect.width) * width;
    return Math.round(Math.min(1, Math.max(0, (px - PAD.left) / plotW)) * (n - 1));
  }

  return (
    <div className="linechart" ref={wrapRef}>
      {series.length >= 2 && (
        <div className="legend" aria-hidden="true">
          {series.map((s) => (
            <span key={s.id} className="legend__item">
              <span className={s.dashed ? "legend__swatch legend__swatch--dashed" : "legend__swatch"} style={{ borderColor: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={ariaLabel}
        onMouseMove={(e) => setHover(indexAt(e.clientX, e.currentTarget.getBoundingClientRect()))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => onPick?.(indexAt(e.clientX, e.currentTarget.getBoundingClientRect()))}
        style={{ cursor: onPick ? "pointer" : undefined }}
      >
        {ticks.map((tk) => (
          <g key={tk}>
            <line className="linechart__grid" x1={PAD.left} x2={PAD.left + plotW} y1={y(tk)} y2={y(tk)} />
            <text className="linechart__tick" x={PAD.left - 6} y={y(tk)} dy="0.32em" textAnchor="end">
              {format(tk)}
            </text>
          </g>
        ))}
        {zeroLine !== undefined && (
          <line className="linechart__zero" x1={PAD.left} x2={PAD.left + plotW} y1={y(zeroLine)} y2={y(zeroLine)} />
        )}
        <text className="linechart__ytitle" transform={`translate(12 ${PAD.top + plotH / 2}) rotate(-90)`} textAnchor="middle">
          {yLabel}
        </text>
        {xLabels.map((lab, i) =>
          i % labelEvery === 0 || i === n - 1 ? (
            <text
              key={i}
              className={i === highlight ? "linechart__tick linechart__tick--on" : "linechart__tick"}
              transform={`translate(${x(i)} ${PAD.top + plotH + 8}) rotate(45)`}
              textAnchor="start"
            >
              {lab}
            </text>
          ) : null,
        )}
        {highlight !== null && highlight >= 0 && highlight < n && (
          <line className="linechart__pick" x1={x(highlight)} x2={x(highlight)} y1={PAD.top} y2={PAD.top + plotH} />
        )}
        {series.map((s, k) => (
          <path
            key={s.id}
            d={paths[k]}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? "5 4" : undefined}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        {series.map((s) =>
          s.values.map((v, i) =>
            v === null || (log && v <= 0) || n > 40 ? null : (
              <circle key={`${s.id}-${i}`} cx={x(i)} cy={y(v)} r={2.5} fill={s.color} className="linechart__dot" />
            ),
          ),
        )}
        {series.length >= 2 &&
          series.length <= 4 &&
          ends.map((e) => (
            <text key={series[e.k].id} className="linechart__end" x={PAD.left + plotW + 8} y={e.y} dy="0.32em">
              {series[e.k].label}
            </text>
          ))}
        {hover !== null && (
          <g pointerEvents="none">
            <line className="linechart__cross" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + plotH} />
            {series.map((s) => {
              const v = s.values[hover];
              return v === null || (log && v <= 0) ? null : (
                <circle key={s.id} cx={x(hover)} cy={y(v)} r={4.5} fill={s.color} className="linechart__ring" />
              );
            })}
          </g>
        )}
      </svg>
      {hover !== null && (
        <div
          className="tooltip linechart__tooltip"
          style={{ left: Math.min(x(hover) + 12, width - 190), top: PAD.top }}
          role="status"
        >
          <strong>{xLabels[hover]}</strong>
          {series.map((s) => (
            <div key={s.id} className="linechart__tiprow">
              <span className="legend__swatch" style={{ borderColor: s.color }} />
              <span className="linechart__tiplabel">{s.label}</span>
              <span className="tooltip__val">{s.values[hover] === null ? "–" : format(s.values[hover] as number)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
