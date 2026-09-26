// Number formatting and cell shading shared by every view in the lab.

import { getSequentialLut } from "../../render/colormap";
import type { Theme } from "../../theme";
import type { LensCells } from "./api";
import type { Metric } from "./store";

/** 1..999 as-is, then 1.6k, 24k — keeps a rank to four characters so a cell stays narrow. */
export function fmtRank(r: number | null): string {
  if (r === null) return "–";
  if (r < 1000) return String(r);
  if (r < 10_000) return `${(r / 1000).toFixed(1)}k`;
  return `${Math.round(r / 1000)}k`;
}

export function fmtProb(p: number | null): string {
  if (p === null) return "–";
  if (p >= 0.995) return "1.00";
  if (p >= 0.01) return p.toFixed(2);
  if (p >= 0.001) return p.toFixed(3);
  return p === 0 ? "0" : p.toExponential(0);
}

export function fmtPct(p: number | null): string {
  return p === null ? "–" : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
}

export function fmtNum(x: number | null, digits = 2): string {
  if (x === null || !Number.isFinite(x)) return "–";
  return x.toFixed(digits);
}

export function fmtSigned(x: number, digits = 2): string {
  return `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
}

/** The value a metric shows for one cell, as text. */
export function cellValue(metric: Metric, cells: LensCells, r: number, c: number): string {
  switch (metric) {
    case "top1":
      return fmtProb(cells.top_p[r][c][0]);
    case "p_next":
      return fmtProb(cells.p_next[r][c]);
    case "rank_next":
      return cells.rank_next[r][c] === null ? "–" : `#${fmtRank(cells.rank_next[r][c])}`;
    case "rank_final":
      return `#${fmtRank(cells.rank_final[r][c])}`;
    case "entropy":
      return fmtNum(cells.entropy[r][c], 1);
    case "kl":
      return fmtNum(cells.kl[r][c], 2);
  }
}

/**
 * How strongly a cell is shaded, in [0, 1], or null when the metric has no value there.
 * One rule for every metric: stronger = this row has settled (more confident,
 * nearer rank 1, nearer the output). That way the same shade reads the same
 * across metrics, and the eye can find "where it decided" without the legend.
 */
export function cellShade(metric: Metric, cells: LensCells, r: number, c: number, dVocab: number): number | null {
  const logV = Math.log(dVocab);
  switch (metric) {
    case "top1":
      return cells.top_p[r][c][0];
    case "p_next":
      return cells.p_next[r][c];
    case "rank_next": {
      const k = cells.rank_next[r][c];
      return k === null ? null : 1 - Math.log(k) / logV;
    }
    case "rank_final":
      return 1 - Math.log(cells.rank_final[r][c]) / logV;
    case "entropy":
      return Math.max(0, 1 - cells.entropy[r][c] / logV);
    case "kl":
      return Math.exp(-Math.max(0, cells.kl[r][c]));
  }
}

export interface Shade {
  background: string;
  color: string;
}

/** Background from the app's sequential blue ramp, and an ink that stays readable on it. */
export function shadeStyle(t: number | null, theme: Theme): Shade | undefined {
  if (t === null) return undefined;
  const lut = getSequentialLut(theme);
  const i = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3;
  const [r, g, b] = [lut[i], lut[i + 1], lut[i + 2]];
  const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return { background: `rgb(${r},${g},${b})`, color: lum < 0.5 ? "#ffffff" : "#0b0b0b" };
}
