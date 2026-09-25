import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { currentLabId, labHref } from "../labs/registry";
import { TokenLab } from "../labs/tokens/TokenLab";
import {
  DEFAULT_COMPARE,
  DEFAULT_TEXT,
  readTokLabUrl,
  useTokLab,
  writeTokLabUrl,
  type TokLabPermalink,
} from "../labs/tokens/store";
import type { AnalyzeResult, TokenizerInfo } from "../labs/tokens/api";
// Real responses captured from the running API (toklab-fixtures.json was
// produced by calling the endpoints, not written by hand), so the shapes here
// are the backend's actual shapes. Regenerate it if toklab.py's output changes.
import FX from "./toklab-fixtures.json";

const TOKENIZERS = FX.tokenizers.tokenizers as TokenizerInfo[];
const ANALYZE = FX.analyze as Record<string, AnalyzeResult>;
const FRAG_TEXT = "aन b";

const PRISTINE = { ...useTokLab.getState() };
let calls: { method: string; url: string; body: any }[] = [];

function jsonRes(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

function installFetch() {
  calls = [];
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({ method: init?.method ?? "GET", url, body });
    if (url.startsWith("/api/tokenizers")) return jsonRes(FX.tokenizers);
    if (url.startsWith("/api/toklab/analyze")) {
      return jsonRes({
        results: body.tokenizers.map(
          (t: string) => ANALYZE[`${t}|${body.text}`] ?? { ...ANALYZE[`gpt2|${FRAG_TEXT}`], tokenizer: t },
        ),
      });
    }
    if (url.startsWith("/api/toklab/count")) {
      // a deterministic stand-in: gpt2 pays one token per code point, everyone else a third of that
      return jsonRes({
        texts: body.texts.map((t: string) => ({ n_chars: t.length, n_graphemes: t.length, n_bytes: t.length, n_words: 1 })),
        results: body.tokenizers.map((tok: string) => ({
          tokenizer: tok,
          counts: body.texts.map((t: string) => Math.max(1, Math.round(t.length / (tok === "gpt2" ? 1 : 3)))),
        })),
      });
    }
    if (url.startsWith("/api/toklab/trace")) return jsonRes(FX.trace["gpt2| unbelievably"]);
    if (url.startsWith("/api/toklab/vocab")) {
      return jsonRes({ tokenizer: "gpt2", size: 50257, n_rows: 50257, n_merges: 50000, kinds: {}, by_script: [], longest: [], special: [] });
    }
    throw new Error(`unmocked fetch: ${url}`);
  });
}

function reset(over: Partial<ReturnType<typeof useTokLab.getState>> = {}) {
  useTokLab.setState({ ...PRISTINE, view: "inspect", tokenizer: "gpt2", text: FRAG_TEXT, tokenizers: [], hovered: null, ...over }, true);
}

beforeEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState(null, "", "/tokens");
  installFetch();
  reset();
});

describe("permalink", () => {
  it("falls back to defaults on an empty URL", () => {
    expect(readTokLabUrl("")).toMatchObject({ view: "inspect", tokenizer: "gpt2", text: DEFAULT_TEXT, compare: DEFAULT_COMPARE });
  });

  it("accepts the attention lab's ?prompt= and ?model= so a hand-off needs no translation", () => {
    const p = readTokLabUrl("?model=pythia-160m&prompt=hello");
    expect(p.text).toBe("hello");
    expect(p.fromModel).toBe("pythia-160m");
    // an explicit tokenizer wins over the model
    expect(readTokLabUrl("?model=pythia-160m&tok=bloom").fromModel).toBeNull();
  });

  it("rejects an unknown view", () => {
    expect(readTokLabUrl("?view=nope").view).toBe("inspect");
  });

  it("round-trips, omitting defaults", () => {
    const state: Omit<TokLabPermalink, "fromModel"> = {
      view: "bpe",
      tokenizer: "llama2",
      text: "x y",
      compare: ["gpt2", "bloom"],
      special: true,
      word: " cat",
      query: "",
    };
    writeTokLabUrl(state);
    expect(window.location.search).not.toContain("q=");
    const { fromModel: _f, ...back } = readTokLabUrl();
    expect(back).toEqual(state);
  });

  it("resolves a model to the tokenizer it actually uses once the catalogue arrives", () => {
    reset({ fromModel: "pythia-160m", tokenizer: "gpt2" });
    act(() => useTokLab.getState().setTokenizers(TOKENIZERS));
    expect(useTokLab.getState().tokenizer).toBe("gpt-neox");
  });
});

describe("lab registry", () => {
  it("maps paths to labs", () => {
    expect(currentLabId("/tokens")).toBe("tokens");
    expect(currentLabId("/attention/")).toBe("attention");
    expect(currentLabId("/")).toBe("home");
  });

  it("builds hand-off links that drop empty params", () => {
    expect(labHref("attention", { model: "gpt2-small", prompt: "a b", head: undefined })).toBe(
      "/attention?model=gpt2-small&prompt=a+b",
    );
  });
});

describe("inspect", () => {
  it("draws a split character as one joined group of its byte tokens", async () => {
    render(<TokenLab />);
    await waitFor(() => expect(document.querySelectorAll(".chip").length).toBe(4));
    const frag = document.querySelector(".cluster--frag");
    expect(frag?.querySelectorAll(".chip")).toHaveLength(2);
    expect([...document.querySelectorAll(".chip")].map((c) => c.textContent)).toEqual(["a", "न", "⋯", "·b"]);
  });

  it("explains the hovered token and highlights its whole character in the text", async () => {
    render(<TokenLab />);
    await waitFor(() => expect(document.querySelectorAll(".chip").length).toBe(4));
    fireEvent.mouseEnter(screen.getByRole("button", { name: /token 2 of 2 for "न"/ }));
    expect(screen.getByText(/that together spell/)).toBeInTheDocument();
    expect(document.querySelector(".token-detail__source mark")?.textContent).toBe("न");
  });

  it("switches the chips to ids and bytes", async () => {
    render(<TokenLab />);
    await waitFor(() => expect(document.querySelectorAll(".chip").length).toBe(4));
    fireEvent.click(screen.getByRole("button", { name: "Bytes" }));
    expect(screen.getByRole("button", { name: /token 1 of 2 for "न"/ })).toHaveTextContent("e0 a4");
    fireEvent.click(screen.getByRole("button", { name: "IDs" }));
    expect(screen.getByRole("button", { name: /token 1 of 2 for "न"/ })).toHaveTextContent("11976");
  });

  it("hands the same text to the attention lab with a model that reads these tokens", async () => {
    render(<TokenLab />);
    const link = await screen.findByRole("link", { name: /gpt2-small attends over exactly these tokens/ });
    expect(link.getAttribute("href")).toBe(labHref("attention", { model: "gpt2-small", prompt: FRAG_TEXT }));
  });

  it("loads a quirk and says what to look for", async () => {
    render(<TokenLab />);
    await waitFor(() => expect(useTokLab.getState().tokenizers.length).toBeGreaterThan(0));
    fireEvent.click(screen.getByRole("button", { name: "A glitch token" }));
    expect(useTokLab.getState().text).toBe(" SolidGoldMagikarp");
    expect(screen.getByText(/What to notice/)).toBeInTheDocument();
  });
});

describe("languages", () => {
  it("counts every language in one request and reports the premium against English", async () => {
    reset({ view: "languages", compare: ["gpt2", "bloom"] });
    render(<TokenLab />);
    await waitFor(() => expect(document.querySelector(".langtable")).not.toBeNull());
    const count = calls.filter((c) => c.url.startsWith("/api/toklab/count"));
    expect(count).toHaveLength(1);
    expect(count[0].body.texts).toHaveLength(32);
    const english = screen.getByRole("button", { name: "English" }).closest("tr")!;
    expect(english.textContent).toContain("×1.0");
    expect(screen.getByText(/the same meaning costs/)).toBeInTheDocument();
  });

  it("clicking a language opens it in Inspect", async () => {
    reset({ view: "languages", compare: ["gpt2", "bloom"] });
    render(<TokenLab />);
    fireEvent.click(await screen.findByRole("button", { name: "Nepali" }));
    expect(useTokLab.getState().view).toBe("inspect");
    expect(useTokLab.getState().text).toMatch(/[ऀ-ॿ]/);
    // let the Inspect view's own analyze request land before the test ends
    await waitFor(() => expect(document.querySelectorAll(".chip").length).toBeGreaterThan(0));
  });
});

describe("bpe step-through", () => {
  it("walks the merges in rank order and ends on the verified result", async () => {
    reset({ view: "bpe", word: " unbelievably" });
    render(<TokenLab />);
    const next = await screen.findByRole("button", { name: "next →" });
    fireEvent.click(next);
    expect(screen.getByText("#50", { selector: "strong" })).toBeInTheDocument();
    const slider = screen.getByRole("slider", { name: "Step" });
    fireEvent.change(slider, { target: { value: slider.getAttribute("max") } });
    expect(screen.getByText(/Matches the real tokenizer/)).toBeInTheDocument();
  });
});
