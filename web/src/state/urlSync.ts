// The URL is the single source of truth for permalink state (model, prompt,
// layer, head, direction) — docs/PLAN.md Stage 1. Read once on mount to
// hydrate the store; thereafter, any change to that state rewrites the URL via
// replaceState (no history spam from typing or clicking heads — a permalink is
// meant to be copied at a point in time, not to make every click a
// back-button stop).

import { DEFAULT_DIRECTION, DEFAULT_MODEL, DEFAULT_PROMPT, type Direction } from "./defaults";

export interface PermalinkState {
  model: string;
  prompt: string;
  selectedLayer: number;
  selectedHead: number | null;
  direction: Direction;
}

/** Non-negative integer or null — rejects "abc", "-1", "1.5" and "" alike,
 * all of which Number() would otherwise turn into a plausible-looking index. */
function intParam(raw: string | null): number | null {
  if (raw === null || raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isInteger(n) && n >= 0 ? n : null;
}

export function readPermalinkFromUrl(): PermalinkState {
  const params = new URLSearchParams(window.location.search);
  const dir = params.get("dir");
  return {
    model: params.get("model") ?? DEFAULT_MODEL,
    // `?? DEFAULT` not `|| DEFAULT`: an explicitly empty ?prompt= means the
    // user cleared the box, and reloading should not resurrect the default.
    prompt: params.get("prompt") ?? DEFAULT_PROMPT,
    selectedLayer: intParam(params.get("layer")) ?? 0,
    selectedHead: intParam(params.get("head")),
    direction: dir === "src2dest" || dir === "dest2src" ? dir : DEFAULT_DIRECTION,
  };
}

export function writePermalinkToUrl(state: PermalinkState): void {
  const params = new URLSearchParams();
  params.set("model", state.model);
  params.set("prompt", state.prompt);
  params.set("layer", String(state.selectedLayer));
  // Omitted when at their defaults so a freshly loaded page has a short,
  // legible URL rather than five params of noise.
  if (state.selectedHead !== null) params.set("head", String(state.selectedHead));
  if (state.direction !== DEFAULT_DIRECTION) params.set("dir", state.direction);
  window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
}
