import { create } from "zustand";
import type { TokenizerInfo } from "./api";

// The Tokenizer lab's own store. Same rule as the attention lab's: the URL is
// the source of truth for everything a permalink should reproduce (view,
// tokenizer, text, comparison set, …), read once before the first render and
// written back with replaceState on every change.

export type View = "inspect" | "compare" | "languages" | "bpe" | "vocab";
export const VIEWS: View[] = ["inspect", "compare", "languages", "bpe", "vocab"];

/** How each chip labels its token. */
export type ChipMode = "text" | "ids" | "bytes" | "vocab";

export const DEFAULT_TOKENIZER = "gpt2";
export const DEFAULT_TEXT = "Tokenizers don't read words. They read pieces: 12345, unbelievably, नेपाल, 🙂";
export const DEFAULT_COMPARE = ["gpt2", "llama2", "bloom", "o200k"];
export const DEFAULT_WORD = " unbelievably";

export interface TokLabPermalink {
  view: View;
  tokenizer: string;
  text: string;
  compare: string[];
  special: boolean;
  word: string;
  query: string;
  /** an attention-lab model id to resolve to its tokenizer once the catalogue loads */
  fromModel: string | null;
}

export function readTokLabUrl(search: string = window.location.search): TokLabPermalink {
  const p = new URLSearchParams(search);
  const view = p.get("view");
  const cmp = p.get("cmp");
  return {
    view: VIEWS.includes(view as View) ? (view as View) : "inspect",
    tokenizer: p.get("tok") ?? DEFAULT_TOKENIZER,
    // "prompt" is what the attention lab calls it; accept either so a link
    // from there needs no translation.
    text: p.get("text") ?? p.get("prompt") ?? DEFAULT_TEXT,
    compare: cmp ? cmp.split(",").filter(Boolean).slice(0, 6) : DEFAULT_COMPARE,
    special: p.get("special") === "1",
    word: p.get("word") ?? DEFAULT_WORD,
    query: p.get("q") ?? "",
    fromModel: p.has("tok") ? null : p.get("model"),
  };
}

export function writeTokLabUrl(s: Omit<TokLabPermalink, "fromModel">): void {
  const p = new URLSearchParams();
  if (s.view !== "inspect") p.set("view", s.view);
  p.set("tok", s.tokenizer);
  p.set("text", s.text);
  if (s.compare.join(",") !== DEFAULT_COMPARE.join(",")) p.set("cmp", s.compare.join(","));
  if (s.special) p.set("special", "1");
  if (s.word !== DEFAULT_WORD) p.set("word", s.word);
  if (s.query) p.set("q", s.query);
  window.history.replaceState(null, "", `${window.location.pathname}?${p.toString()}`);
}

interface TokLabState extends Omit<TokLabPermalink, "fromModel"> {
  fromModel: string | null;
  tokenizers: TokenizerInfo[];
  catalogError: string | null;
  chipMode: ChipMode;
  /** token index under the pointer in the Inspect view */
  hovered: number | null;
  /** search-by-script filter chosen from the vocabulary breakdown */
  scriptFilter: string | null;

  setView: (v: View) => void;
  setTokenizer: (id: string) => void;
  setText: (t: string) => void;
  toggleCompare: (id: string) => void;
  setSpecial: (b: boolean) => void;
  setWord: (w: string) => void;
  setQuery: (q: string) => void;
  setScriptFilter: (s: string | null) => void;
  setChipMode: (m: ChipMode) => void;
  setHovered: (i: number | null) => void;
  setTokenizers: (list: TokenizerInfo[]) => void;
  setCatalogError: (e: string | null) => void;
  /** jump to Inspect with this text and tokenizer — the hand-off every other view uses */
  inspect: (text: string, tokenizer?: string) => void;
}

const initial = readTokLabUrl();

export const useTokLab = create<TokLabState>((set) => ({
  ...initial,
  tokenizers: [],
  catalogError: null,
  chipMode: "text",
  hovered: null,
  scriptFilter: null,

  setView: (view) => set({ view, hovered: null }),
  setTokenizer: (tokenizer) => set({ tokenizer, hovered: null }),
  setText: (text) => set({ text }),
  toggleCompare: (id) =>
    set((s) => {
      if (s.compare.includes(id)) return s.compare.length > 1 ? { compare: s.compare.filter((x) => x !== id) } : s;
      return s.compare.length < 6 ? { compare: [...s.compare, id] } : s;
    }),
  setSpecial: (special) => set({ special }),
  setWord: (word) => set({ word }),
  setQuery: (query) => set({ query }),
  setScriptFilter: (scriptFilter) => set({ scriptFilter }),
  setChipMode: (chipMode) => set({ chipMode }),
  setHovered: (hovered) => set({ hovered }),
  setTokenizers: (tokenizers) =>
    set((s) => {
      // Resolve ?model=… (a link from the attention lab) to the tokenizer that
      // model actually uses, now that we know which that is.
      const fromModel = s.fromModel ? tokenizers.find((t) => t.models.includes(s.fromModel!)) : undefined;
      const known = new Set(tokenizers.map((t) => t.id));
      return {
        tokenizers,
        fromModel: null,
        tokenizer: fromModel?.id ?? (known.has(s.tokenizer) ? s.tokenizer : DEFAULT_TOKENIZER),
        compare: s.compare.filter((id) => known.has(id)).length ? s.compare.filter((id) => known.has(id)) : DEFAULT_COMPARE,
      };
    }),
  setCatalogError: (catalogError) => set({ catalogError }),
  inspect: (text, tokenizer) => set((s) => ({ view: "inspect", text, tokenizer: tokenizer ?? s.tokenizer, hovered: null })),
}));
