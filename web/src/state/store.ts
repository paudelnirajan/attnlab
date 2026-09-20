import { create } from "zustand";
import type { DecodedPatterns } from "../api/patterns";
import type { ModelInfo, RunResponse, TokenizeResponse } from "../api/types";
import { applyThemeAttr, persistMode, readStoredMode, resolveTheme, type Theme, type ThemeMode } from "../theme";
import { DEFAULT_DIRECTION, DEFAULT_MODEL, DEFAULT_PROMPT, type Direction } from "./defaults";
import { readPermalinkFromUrl } from "./urlSync";

// Pure state + setters only — no fetching here. Async orchestration (calling
// the API when model/prompt/layer change, decoding patterns) lives in App.tsx's
// effects, so this store stays trivially testable and the URL-sync logic
// (state/urlSync.ts) has one place to read from.

export type ApiPhase = "idle" | "loading" | "error";

/**
 * Which way round the hover-linking reads — the same choice CircuitsVis
 * offers in ARENA 1.2 by showing paired Destination/Source token columns.
 *
 *  dest2src: a matrix ROW.    "at this token, what does the model look back at?"
 *            Sums to 1 (softmax over sources).
 *  src2dest: a matrix COLUMN. "which later tokens look back at this one?"
 *            Does NOT sum to 1 — it's attention *received*, not a distribution.
 */
export type { Direction };

interface AttnlabState {
  // Permalink state — docs/PLAN.md: "model + prompt + layer + head in the URL;
  // the URL is the single source of truth."
  model: string;
  prompt: string;
  selectedLayer: number;
  /** null = grid view (all heads of selectedLayer); a number = expanded single head */
  selectedHead: number | null;
  direction: Direction;

  // Non-permalink UI state. "Clicking a chip sets the selected position
  // globally" (docs/PLAN.md) is distinct from hover: hover previews
  // (ephemeral, wins while active), click pins (persists after the mouse moves
  // away). Components read hoveredTokenIdx ?? selectedTokenIdx.
  hoveredTokenIdx: number | null;
  selectedTokenIdx: number | null;

  themeMode: ThemeMode;
  theme: Theme;

  // Server data
  models: ModelInfo[];
  budget: { limit_mb: number; used_mb: number } | null;
  tokenizeResult: TokenizeResponse | null;
  runResult: RunResponse | null;
  patternsByLayer: Map<number, DecodedPatterns>;

  // Status
  modelsPhase: ApiPhase;
  runPhase: ApiPhase;
  errorMessage: string | null;

  setModel: (id: string) => void;
  setPrompt: (text: string) => void;
  setSelectedLayer: (layer: number) => void;
  setSelectedHead: (head: number | null) => void;
  setDirection: (d: Direction) => void;
  setHoveredToken: (idx: number | null) => void;
  setSelectedToken: (idx: number | null) => void;
  setThemeMode: (mode: ThemeMode) => void;
  setResolvedTheme: (theme: Theme) => void;

  setModels: (models: ModelInfo[], budget: AttnlabState["budget"], phase: ApiPhase) => void;
  setTokenizeResult: (r: TokenizeResponse | null) => void;
  /** Marks a run in flight WITHOUT clearing the previous one — the old view
   * stays on screen (dimmed) instead of the page flashing empty on every
   * debounced keystroke. */
  beginRun: () => void;
  setRunResult: (r: RunResponse) => void;
  failRun: () => void;
  clearRun: () => void;
  cachePatterns: (layer: number, decoded: DecodedPatterns) => void;
  setError: (message: string | null) => void;

}

export { DEFAULT_DIRECTION, DEFAULT_MODEL, DEFAULT_PROMPT };

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

const initialMode = readStoredMode();
// Read the permalink once, here, before the first render. Doing it in an
// effect meant the first paint used DEFAULT_PROMPT and fired a throwaway run
// against it before the URL's real prompt reached the debounce.
const initial = readPermalinkFromUrl();
// Same reason, for the very first paint.
applyThemeAttr(initialMode);

export const useStore = create<AttnlabState>((set) => ({
  model: initial.model,
  prompt: initial.prompt,
  selectedLayer: initial.selectedLayer,
  selectedHead: initial.selectedHead,
  direction: initial.direction,

  hoveredTokenIdx: null,
  selectedTokenIdx: null,

  themeMode: initialMode,
  theme: resolveTheme(initialMode),

  models: [],
  budget: null,
  tokenizeResult: null,
  runResult: null,
  patternsByLayer: new Map(),

  modelsPhase: "idle",
  runPhase: "idle",
  errorMessage: null,

  setModel: (id) =>
    set({
      model: id,
      selectedLayer: 0,
      selectedHead: null,
      hoveredTokenIdx: null,
      selectedTokenIdx: null,
      runResult: null,
      tokenizeResult: null,
      patternsByLayer: new Map(),
      errorMessage: null,
    }),
  setPrompt: (text) => set({ prompt: text }),
  // Deliberately keeps selectedHead: head *index* is meaningful across layers
  // ("is head 4 an induction head in every layer?"), and dropping it made the
  // up/down arrow shortcut close the detail view on every press.
  setSelectedLayer: (layer) => set({ selectedLayer: layer }),
  setSelectedHead: (head) => set({ selectedHead: head }),
  setDirection: (direction) => set({ direction }),
  setHoveredToken: (idx) => set({ hoveredTokenIdx: idx }),
  setSelectedToken: (idx) => set({ selectedTokenIdx: idx }),
  // The [data-theme] stamp has to land BEFORE any component re-renders, not in
  // an effect afterwards. Child effects run before parent effects, so a canvas
  // would otherwise repaint itself reading the *old* theme's CSS variables.
  setThemeMode: (themeMode) => {
    applyThemeAttr(themeMode);
    persistMode(themeMode);
    set({ themeMode, theme: resolveTheme(themeMode) });
  },
  setResolvedTheme: (theme) => set({ theme }),

  setModels: (models, budget, phase) => set({ models, budget, modelsPhase: phase }),
  setTokenizeResult: (r) => set({ tokenizeResult: r }),

  beginRun: () => set({ runPhase: "loading" }),
  setRunResult: (r) =>
    set((s) => {
      // A permalink can name a layer/head that doesn't exist on this model
      // (?layer=9 on a 2-layer model), and a shorter prompt can strand a token
      // selection past the end. Clamp on arrival rather than 404-ing later.
      const seq = r.tokens.length;
      return {
        runResult: r,
        runPhase: "idle",
        patternsByLayer: new Map(),
        selectedLayer: clamp(s.selectedLayer, 0, Math.max(0, r.n_layers - 1)),
        selectedHead: s.selectedHead === null ? null : clamp(s.selectedHead, 0, Math.max(0, r.n_heads - 1)),
        hoveredTokenIdx: s.hoveredTokenIdx === null ? null : clamp(s.hoveredTokenIdx, 0, seq - 1),
        selectedTokenIdx: s.selectedTokenIdx === null ? null : clamp(s.selectedTokenIdx, 0, seq - 1),
      };
    }),
  failRun: () => set({ runPhase: "error" }),
  clearRun: () => set({ runResult: null, runPhase: "idle", patternsByLayer: new Map() }),

  cachePatterns: (layer, decoded) =>
    set((s) => {
      const next = new Map(s.patternsByLayer);
      next.set(layer, decoded);
      return { patternsByLayer: next };
    }),
  setError: (message) => set({ errorMessage: message }),

}));
