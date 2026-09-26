import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { LABS, currentLabId, labHref } from "../labs/registry";
import { LensLab } from "../labs/lens/LensLab";
import { cellShade, fmtRank } from "../labs/lens/format";
import { readLensUrl, useLens, visibleRows, writeLensUrl, type LensPermalink } from "../labs/lens/store";
import type { LensRunResponse } from "../labs/lens/api";
// Real responses from the running API on attn-only-2l-demo (lens-fixtures.json
// was produced by calling the endpoints, not written by hand). Regenerate it
// if lens.py's output shape changes.
import FX from "./lens-fixtures.json";

const RUN = FX.run as unknown as LensRunResponse;
const PRISTINE = { ...useLens.getState() };
let calls: { url: string; body: any }[] = [];

function jsonRes(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

function installFetch() {
  calls = [];
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({ url, body });
    if (url.startsWith("/api/models")) return jsonRes(FX.models);
    if (url.startsWith("/api/lens/run")) return jsonRes(FX.run);
    if (url.startsWith("/api/lens/layers")) return jsonRes(FX.layers);
    if (url.startsWith("/api/lens/position")) return jsonRes(FX.position);
    if (url.startsWith("/api/lens/attribution")) return jsonRes(FX.attribution);
    throw new Error(`unmocked fetch: ${url}`);
  });
}

function mount(search: string) {
  window.history.replaceState(null, "", `/logit-lens${search}`);
  useLens.setState({ ...PRISTINE, ...readLensUrl() }, true);
  installFetch();
  return render(<LensLab />);
}

beforeEach(() => vi.useRealTimers());

describe("registry", () => {
  it("the logit lens is a live step 4", () => {
    const lab = LABS.find((l) => l.id === "logit-lens")!;
    expect(lab.status).toBe("live");
    expect(lab.step).toBe(4);
    expect(currentLabId("/logit-lens")).toBe("logit-lens");
    expect(labHref("logit-lens", { prompt: "x" })).toBe("/logit-lens?prompt=x");
  });
});

describe("URL state", () => {
  it("round-trips every permalink field", () => {
    const s: LensPermalink = {
      view: "attribution",
      model: "gpt2-small",
      text: "When Mary and John went to the store, John gave a drink to",
      lens: "plain",
      rowMode: "sub",
      metric: "rank_next",
      pos: 14,
      row: 3,
      track: [{ str: " Mary" }, { id: 1757, label: "·John" }],
      target: { str: " Mary" },
      contrast: { id: 1757, label: "·John" },
      bos: false,
    };
    window.history.replaceState(null, "", "/logit-lens");
    writeLensUrl(s);
    expect(readLensUrl(window.location.search)).toEqual(s);
  });

  it("falls back on junk and accepts the other labs' names for the text", () => {
    const s = readLensUrl("?view=nope&lens=raw&metric=x&pos=-1&track=%7Bbad&text=hello");
    expect(s.view).toBe("grid");
    expect(s.lens).toBe("ln_final");
    expect(s.metric).toBe("top1");
    expect(s.pos).toBeNull();
    expect(s.track).toEqual([]);
    expect(s.text).toBe("hello");
    expect(readLensUrl("?prompt=a&text=b").text).toBe("a");
  });
});

describe("helpers", () => {
  it("formats ranks to four characters", () => {
    expect([1, 999, 1600, 24553, null].map(fmtRank)).toEqual(["1", "999", "1.6k", "25k", "–"]);
  });

  it("shades rank 1 fully and the last rank not at all", () => {
    const cells = { ...RUN.cells, rank_next: [[1, RUN.anatomy.d_vocab]] };
    expect(cellShade("rank_next", cells, 0, 0, RUN.anatomy.d_vocab)).toBe(1);
    expect(cellShade("rank_next", cells, 0, 1, RUN.anatomy.d_vocab)).toBeCloseTo(0);
  });

  it("blocks mode keeps only block-end rows (and the output)", () => {
    const rows = [{ block_end: true }, { block_end: false }, { block_end: true }, { block_end: true }];
    expect(visibleRows(rows, "blocks")).toEqual([0, 2, 3]);
    expect(visibleRows(rows, "sub")).toEqual([0, 1, 2, 3]);
  });
});

describe("the lab", () => {
  it("runs the model once and draws one cell per (row, position), output on top", async () => {
    const { container } = mount("");
    await waitFor(() => expect(container.querySelectorAll(".lgrid__cell").length).toBeGreaterThan(0));
    const seq = RUN.tokens.length;
    expect(container.querySelectorAll(".lgrid__cell")).toHaveLength(RUN.rows.length * seq);
    const labels = [...container.querySelectorAll(".lgrid__rowlabel")].map((e) => e.textContent);
    expect(labels).toEqual(["output", "L1 +attn", "L0 +attn", "embed"]);
    expect(calls.filter((c) => c.url === "/api/lens/run")).toHaveLength(1);
    // this model has no final norm: one lens, labelled for what it is
    expect(screen.getByRole("button", { name: /no norm/ })).toBeInTheDocument();
  });

  it("clicking a cell selects it and writes the permalink", async () => {
    const { container } = mount("");
    await waitFor(() => expect(container.querySelector(".lgrid__cell")).not.toBeNull());
    const cell = container.querySelector('.lgrid__cell[data-r="1"][data-c="3"]')!;
    fireEvent.click(cell);
    expect(useLens.getState().pos).toBe(3);
    expect(useLens.getState().row).toBe(1);
    await waitFor(() => expect(window.location.search).toContain("pos=3"));
    expect(screen.getByText(/L0 \+attn, position 3/)).toBeInTheDocument();
  });

  it("changing the metric re-labels cells without another request", async () => {
    const { container } = mount("");
    await waitFor(() => expect(container.querySelector(".lgrid__cell")).not.toBeNull());
    const before = calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Rank of final answer" }));
    const out = container.querySelector(`.lgrid__cell[data-r="${RUN.rows.length - 1}"][data-c="0"]`)!;
    expect(out.textContent).toContain("#1"); // the output row ranks its own answer first
    expect(calls.length).toBe(before);
  });

  it("attribution shows the decomposition and that it sums", async () => {
    const { container } = mount("?view=attribution");
    await waitFor(() => expect(container.querySelector(".equation")).not.toBeNull());
    expect(container.querySelector(".equation")!.textContent).toMatch(/sums to the model's logit/);
    expect(container.querySelectorAll(".headmap__cell")).toHaveLength(RUN.anatomy.n_layers * RUN.anatomy.n_heads);
    const req = calls.find((c) => c.url === "/api/lens/attribution")!;
    expect(req.body.pos).toBe(RUN.tokens.length - 1);
  });

  it("trajectory asks for the default and tracked tokens", async () => {
    mount('?view=trajectory&track=[{"str":" mat"}]');
    await waitFor(() => expect(calls.some((c) => c.url === "/api/lens/position")).toBe(true));
    const req = calls.find((c) => c.url === "/api/lens/position")!;
    expect(req.body.track).toEqual([" mat"]);
    await waitFor(() => expect(screen.getByText("Rank at every row")).toBeInTheDocument());
  });

  it("layers and under-the-hood render from the real responses", async () => {
    mount("?view=layers");
    await waitFor(() => expect(screen.getByText("Agrees with the output's top-1")).toBeInTheDocument());
    act(() => useLens.getState().setView("hood"));
    await waitFor(() => expect(screen.getByText("Checked on this run")).toBeInTheDocument());
    expect(document.querySelectorAll(".checks__item")).toHaveLength(RUN.checks.length);
  });
});
