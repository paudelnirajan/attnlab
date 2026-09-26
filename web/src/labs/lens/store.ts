import { create } from "zustand";
import type { ModelInfo } from "../../api/types";
import type { LensName, LensRunResponse, LensSummary } from "./api";

// The logit lens lab's store. Same rule as the other labs: everything a
// permalink should reproduce (view, model, text, lens, the selected cell, the
// tokens being followed) lives in the URL, read once before the first render
// and written back with replaceState on every change.

export type View = "grid" | "trajectory" | "attribution" | "layers" | "hood";
export const VIEWS: View[] = ["grid", "trajectory", "attribution", "layers", "hood"];

/** blocks: the stream between blocks (the notebook's rows); sub: also after each attention sub-layer */
export type RowMode = "blocks" | "sub";

export type Metric = "top1" | "p_next" | "rank_next" | "rank_final" | "entropy" | "kl";
export const METRICS: Metric[] = ["top1", "p_next", "rank_next", "rank_final", "entropy", "kl"];

/** A token chosen by the learner: a known id (clicked in the grid), or a string to tokenize. */
export type TokenRef = { id: number; label: string } | { str: string };

export const DEFAULT_MODEL = "gpt2-small";
export const DEFAULT_TEXT =
  "Sometimes, when people say plasma, they mean a state of matter. Other times, when people say plasma, they mean";

export interface LensPermalink {
  view: View;
  model: string;
  text: string;
  lens: LensName;
  rowMode: RowMode;
  metric: Metric;
  /** selected position (column); null = the last one */
  pos: number | null;
  /** selected row, as an index into the run's full row list */
  row: number | null;
  track: TokenRef[];
  target: TokenRef | null;
  contrast: TokenRef | null;
  bos: boolean;
}

function intParam(raw: string | null): number | null {
  if (raw === null || raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isInteger(n) && n >= 0 ? n : null;
}

function isRef(x: unknown): x is TokenRef {
  if (typeof x !== "object" || x === null) return false;
  const o = x as Record<string, unknown>;
  return (typeof o.id === "number" && typeof o.label === "string") || typeof o.str === "string";
}

function refParam(raw: string | null): TokenRef | null {
  if (!raw) return null;
  try {
    const v = JSON.parse(raw);
    return isRef(v) ? v : null;
  } catch {
    return null;
  }
}

function refsParam(raw: string | null): TokenRef[] {
  if (!raw) return [];
  try {
    const v = JSON.parse(raw);
    return Array.isArray(v) ? v.filter(isRef).slice(0, 4) : [];
  } catch {
    return [];
  }
}

export function refKey(r: TokenRef): string {
  return "id" in r ? `#${r.id}` : `s:${r.str}`;
}

export function readLensUrl(search: string = window.location.search): LensPermalink {
  const p = new URLSearchParams(search);
  const view = p.get("view");
  const lens = p.get("lens");
  const metric = p.get("metric");
  return {
    view: VIEWS.includes(view as View) ? (view as View) : "grid",
    model: p.get("model") ?? DEFAULT_MODEL,
    // "prompt" is what the attention lab calls it, "text" the tokenizer lab
    text: p.get("prompt") ?? p.get("text") ?? DEFAULT_TEXT,
    lens: lens === "plain" ? "plain" : "ln_final",
    rowMode: p.get("rows") === "sub" ? "sub" : "blocks",
    metric: METRICS.includes(metric as Metric) ? (metric as Metric) : "top1",
    pos: intParam(p.get("pos")),
    row: intParam(p.get("row")),
    track: refsParam(p.get("track")),
    target: refParam(p.get("target")),
    contrast: refParam(p.get("contrast")),
    bos: p.get("bos") !== "0",
  };
}

export function writeLensUrl(s: LensPermalink): void {
  const p = new URLSearchParams();
  if (s.view !== "grid") p.set("view", s.view);
  p.set("model", s.model);
  p.set("prompt", s.text);
  if (s.lens !== "ln_final") p.set("lens", s.lens);
  if (s.rowMode !== "blocks") p.set("rows", s.rowMode);
  if (s.metric !== "top1") p.set("metric", s.metric);
  if (s.pos !== null) p.set("pos", String(s.pos));
  if (s.row !== null) p.set("row", String(s.row));
  if (s.track.length) p.set("track", JSON.stringify(s.track));
  if (s.target) p.set("target", JSON.stringify(s.target));
  if (s.contrast) p.set("contrast", JSON.stringify(s.contrast));
  if (!s.bos) p.set("bos", "0");
  window.history.replaceState(null, "", `${window.location.pathname}?${p.toString()}`);
}

export type Phase = "idle" | "loading" | "error";

interface LensState extends LensPermalink {
  models: ModelInfo[];
  modelsError: string | null;
  run: LensRunResponse | null;
  runPhase: Phase;
  runError: string | null;
  /** grids already fetched for this run, by lens — switching back is free */
  summaries: Partial<Record<LensName, LensSummary>>;
  /** bumped to force a fresh forward pass (e.g. the stored run expired) */
  runNonce: number;

  setView: (v: View) => void;
  setModel: (m: string) => void;
  setText: (t: string) => void;
  setLens: (l: LensName) => void;
  setRowMode: (m: RowMode) => void;
  setMetric: (m: Metric) => void;
  select: (pos: number | null, row?: number | null) => void;
  setBos: (b: boolean) => void;
  addTrack: (r: TokenRef) => void;
  removeTrack: (r: TokenRef) => void;
  setTarget: (r: TokenRef | null) => void;
  setContrast: (r: TokenRef | null) => void;
  loadExample: (e: { text: string; track?: string[]; target?: string; contrast?: string; model?: string }) => void;
  setModels: (m: ModelInfo[], error?: string | null) => void;
  beginRun: () => void;
  setRun: (r: LensRunResponse) => void;
  failRun: (e: string) => void;
  cacheSummary: (lens: LensName, s: LensSummary) => void;
  rerun: () => void;
}

const initial = readLensUrl();

export const useLens = create<LensState>((set) => ({
  ...initial,
  models: [],
  modelsError: null,
  run: null,
  runPhase: "idle",
  runError: null,
  summaries: {},
  runNonce: 0,

  setView: (view) => set({ view }),
  // a new text or model invalidates positions, but tracked tokens still make sense
  setModel: (model) => set({ model, pos: null, row: null }),
  setText: (text) => set({ text, pos: null, row: null }),
  setLens: (lens) => set({ lens }),
  setRowMode: (rowMode) => set({ rowMode }),
  setMetric: (metric) => set({ metric }),
  select: (pos, row) => set((s) => ({ pos, row: row === undefined ? s.row : row })),
  setBos: (bos) => set({ bos, pos: null, row: null }),
  addTrack: (r) =>
    set((s) => (s.track.some((t) => refKey(t) === refKey(r)) || s.track.length >= 4 ? s : { track: [...s.track, r] })),
  removeTrack: (r) => set((s) => ({ track: s.track.filter((t) => refKey(t) !== refKey(r)) })),
  setTarget: (target) => set({ target }),
  setContrast: (contrast) => set({ contrast }),
  loadExample: (e) =>
    set((s) => ({
      text: e.text,
      model: e.model ?? s.model,
      pos: null,
      row: null,
      track: (e.track ?? []).map((str) => ({ str })),
      target: e.target ? { str: e.target } : null,
      contrast: e.contrast ? { str: e.contrast } : null,
    })),
  setModels: (models, modelsError = null) => set({ models, modelsError }),
  beginRun: () => set({ runPhase: "loading" }),
  setRun: (run) =>
    set((s) => ({
      run,
      runPhase: "idle",
      runError: null,
      summaries: { [run.lens]: run },
      // a lens this model doesn't have (plain, on a model with no output norm)
      lens: run.lenses.includes(s.lens) ? s.lens : run.lens,
      pos: s.pos !== null && s.pos < run.tokens.length ? s.pos : null,
      row: s.row !== null && s.row < run.rows.length ? s.row : null,
    })),
  failRun: (runError) => set({ runPhase: "error", runError }),
  cacheSummary: (lens, sm) => set((s) => ({ summaries: { ...s.summaries, [lens]: sm } })),
  rerun: () => set((s) => ({ runNonce: s.runNonce + 1 })),
}));

/** The selected position, resolved: an explicit pick, else the last token. */
export function resolvedPos(pos: number | null, run: LensRunResponse | null): number {
  if (!run) return 0;
  return pos !== null && pos < run.tokens.length ? pos : run.tokens.length - 1;
}

/** Row indices to draw for the chosen mode; the output row is always included. */
export function visibleRows(rows: { block_end: boolean }[], mode: RowMode): number[] {
  return rows.map((r, i) => ({ r, i })).filter(({ r }) => mode === "sub" || r.block_end).map(({ i }) => i);
}
