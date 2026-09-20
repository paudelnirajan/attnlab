import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import { DEFAULT_PROMPT } from "../state/defaults";
import { useStore } from "../state/store";
import { FRAGMENTED_TOKENS, installFetchMock, N_HEADS, SEQ, type FetchMock } from "./fixtures";
import { stubRect } from "./setup";

// The store is a module singleton; snapshot it once so each test starts clean
// without resetting the module registry (which would load a second React).
const PRISTINE = { ...useStore.getState() };

let mock: FetchMock;

function resetStore() {
  useStore.setState(
    {
      ...PRISTINE,
      model: "attn-only-2l-demo",
      prompt: DEFAULT_PROMPT,
      selectedLayer: 0,
      selectedHead: null,
      direction: "dest2src",
      hoveredTokenIdx: null,
      selectedTokenIdx: null,
      models: [],
      budget: null,
      tokenizeResult: null,
      runResult: null,
      patternsByLayer: new Map(),
      modelsPhase: "idle",
      runPhase: "idle",
      errorMessage: null,
    },
    true,
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState(null, "", "/");
  resetStore();
  mock = installFetchMock();
});

/** Renders and waits for the first run to land and its patterns to decode. */
async function renderReady() {
  const utils = render(<App />);
  await waitFor(() => expect(document.querySelectorAll(".chip").length).toBeGreaterThan(0), { timeout: 3000 });
  await waitFor(() => expect(useStore.getState().patternsByLayer.size).toBeGreaterThan(0));
  return utils;
}

describe("boot", () => {
  it("loads the catalogue, tokenizes and runs the default prompt", async () => {
    await renderReady();
    expect(mock.calls.some((c) => c.startsWith("GET /api/models"))).toBe(true);
    expect(mock.calls.some((c) => c.startsWith("POST /api/tokenize"))).toBe(true);
    expect(mock.calls.some((c) => c === "POST /api/run")).toBe(true);
    for (const t of ["The", "·quick", "·brown", "·fox"]) {
      expect(screen.getByRole("button", { name: t })).toBeInTheDocument();
    }
  });

  it("does not fire a throwaway run for the default prompt when the URL names another", async () => {
    // seeded in the store's initialiser, so this asserts the parser only
    window.history.replaceState(null, "", "/?prompt=zzz&layer=1&head=3&dir=src2dest");
    const { readPermalinkFromUrl } = await import("../state/urlSync");
    expect(readPermalinkFromUrl()).toMatchObject({ prompt: "zzz", selectedLayer: 1, selectedHead: 3, direction: "src2dest" });
  });

  it("renders one tile per head and a pill per layer", async () => {
    await renderReady();
    expect(screen.getAllByTitle(/^Head \d+$/)).toHaveLength(N_HEADS);
    expect(screen.getByRole("button", { name: "L0" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "L1" })).toHaveAttribute("aria-pressed", "false");
  });

  it("fetches the selected layer and prefetches its neighbour, once each", async () => {
    await renderReady();
    await waitFor(() => expect(useStore.getState().patternsByLayer.size).toBe(2));
    const l0 = mock.calls.filter((c) => c.includes("patterns?layers=0"));
    const l1 = mock.calls.filter((c) => c.includes("patterns?layers=1"));
    expect(l0).toHaveLength(1);
    expect(l1).toHaveLength(1);
  });
});

describe("hover linking", () => {
  it("tints the other tokens by attention weight when a token is hovered", async () => {
    const user = userEvent.setup();
    await renderReady();
    const fox = screen.getByRole("button", { name: "·fox" }); // last position
    await user.hover(fox);

    await waitFor(() => expect(useStore.getState().hoveredTokenIdx).toBe(3));
    // every earlier position is a legal source and should carry a tint
    for (const name of ["The", "·quick", "·brown"]) {
      expect(screen.getByRole("button", { name }).style.background).toMatch(/color-mix/);
    }
  });

  it("pins on click and releases on a second click", async () => {
    const user = userEvent.setup();
    await renderReady();
    const brown = screen.getByRole("button", { name: "·brown" });
    await user.click(brown);
    expect(useStore.getState().selectedTokenIdx).toBe(2);
    expect(brown.className).toContain("chip--pinned");
    await user.click(brown);
    expect(useStore.getState().selectedTokenIdx).toBeNull();
  });

  it("dims positions the causal mask makes unreachable, and flips which ones when direction flips", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.hover(screen.getByRole("button", { name: "·quick" })); // position 1
    await waitFor(() => expect(useStore.getState().hoveredTokenIdx).toBe(1));
    // dest2src: later positions are unreachable
    expect(screen.getByRole("button", { name: "·fox" }).className).toContain("chip--masked");
    expect(screen.getByRole("button", { name: "The" }).className).not.toContain("chip--masked");

    await user.click(screen.getByRole("button", { name: "Source → Destination" }));
    await user.hover(screen.getByRole("button", { name: "·quick" }));
    await waitFor(() => expect(useStore.getState().direction).toBe("src2dest"));
    // src2dest: earlier positions are now the unreachable ones
    expect(screen.getByRole("button", { name: "The" }).className).toContain("chip--masked");
    expect(screen.getByRole("button", { name: "·fox" }).className).not.toContain("chip--masked");
  });
});

describe("direction toggle", () => {
  it("flips the pressed state, the caption and the URL", async () => {
    const user = userEvent.setup();
    await renderReady();
    const d2s = screen.getByRole("button", { name: "Destination → Source" });
    const s2d = screen.getByRole("button", { name: "Source → Destination" });
    expect(d2s).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("what does this token attend to?")).toBeInTheDocument();
    expect(window.location.search).not.toContain("dir=");

    await user.click(s2d);
    expect(s2d).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("which tokens attend to this one?")).toBeInTheDocument();
    await waitFor(() => expect(window.location.search).toContain("dir=src2dest"));
  });

  it("is reachable from the keyboard with d", async () => {
    await renderReady();
    fireEvent.keyDown(window, { key: "d" });
    expect(useStore.getState().direction).toBe("src2dest");
    fireEvent.keyDown(window, { key: "d" });
    expect(useStore.getState().direction).toBe("dest2src");
  });
});

describe("head detail", () => {
  it("opens on click, states the direction's caveat, and closes again", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByTitle("Head 3"));
    expect(await screen.findByText("Layer 0 · Head 3")).toBeInTheDocument();
    expect(screen.getByText(/softmax distribution, so these weights sum to 1/)).toBeInTheDocument();
    await waitFor(() => expect(window.location.search).toContain("head=3"));

    await user.click(screen.getByRole("button", { name: "Close head detail" }));
    expect(screen.queryByText("Layer 0 · Head 3")).not.toBeInTheDocument();
    await waitFor(() => expect(window.location.search).not.toContain("head="));
  });

  it("ranks the top sources strongest-first for the last position by default", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByTitle("Head 0"));
    // scope to the detail card — the prediction panel has a ranked list too
    const detail = (await screen.findByText("Layer 0 · Head 0")).closest(".card") as HTMLElement;
    const values = within(detail)
      .getAllByText(/^0\.\d{4}$/)
      .map((n) => Number(n.textContent));
    expect(values).toHaveLength(SEQ); // all four positions are sources for the last token
    expect([...values].sort((a, b) => b - a)).toEqual(values);
  });

  it("switches the caption to destinations when the direction flips", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByTitle("Head 0"));
    expect(await screen.findByText(/Top sources for/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Source → Destination" }));
    expect(await screen.findByText(/Top destinations for/)).toBeInTheDocument();
    expect(screen.getByText(/do NOT sum to 1/)).toBeInTheDocument();
  });

  it("shows a value tooltip on the matrix and says so when a cell is masked", async () => {
    const user = userEvent.setup();
    const { container } = await renderReady();
    await user.click(screen.getByTitle("Head 0"));
    const canvas = container.querySelector(".heat__canvas") as HTMLCanvasElement;
    stubRect(canvas, { width: 400, height: 400 });

    // below the diagonal: dest 3, src 1 -> a real weight
    fireEvent.mouseMove(canvas, { clientX: 150, clientY: 350 });
    expect(await screen.findByText("dest 3")).toBeInTheDocument();
    expect(screen.getByText("src 1")).toBeInTheDocument();

    // above the diagonal: dest 0, src 3 -> impossible under causal masking
    fireEvent.mouseMove(canvas, { clientX: 350, clientY: 50 });
    expect(await screen.findByText(/can't attend to a later position/)).toBeInTheDocument();
  });
});

describe("keyboard navigation", () => {
  it("moves layers without closing the open head", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByTitle("Head 2"));
    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(useStore.getState().selectedLayer).toBe(1);
    expect(useStore.getState().selectedHead).toBe(2);
    expect(await screen.findByText("Layer 1 · Head 2")).toBeInTheDocument();
  });

  it("clamps at the ends instead of wrapping", async () => {
    await renderReady();
    fireEvent.keyDown(window, { key: "ArrowUp" });
    expect(useStore.getState().selectedLayer).toBe(0);
    fireEvent.keyDown(window, { key: "ArrowDown" });
    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(useStore.getState().selectedLayer).toBe(1);
  });

  it("steps through heads with left/right", async () => {
    await renderReady();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(useStore.getState().selectedHead).toBe(0);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(useStore.getState().selectedHead).toBe(1);
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(useStore.getState().selectedHead).toBe(0);
  });

  it("Escape unwinds one step at a time: pinned token, then the head view", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByTitle("Head 1"));
    await user.click(screen.getByRole("button", { name: "·brown" }));
    expect(useStore.getState().selectedTokenIdx).toBe(2);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(useStore.getState().selectedTokenIdx).toBeNull();
    expect(useStore.getState().selectedHead).toBe(1);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(useStore.getState().selectedHead).toBeNull();
  });

  it("stays out of the way while typing in the prompt", async () => {
    const user = userEvent.setup();
    await renderReady();
    const box = screen.getByLabelText(/Prompt/);
    await user.click(box);
    fireEvent.keyDown(box, { key: "ArrowDown" });
    fireEvent.keyDown(box, { key: "d" });
    expect(useStore.getState().selectedLayer).toBe(0);
    expect(useStore.getState().direction).toBe("dest2src");
  });

  it("/ focuses the prompt box", async () => {
    await renderReady();
    fireEvent.keyDown(window, { key: "/" });
    expect(document.activeElement).toBe(screen.getByLabelText(/Prompt/));
  });
});

describe("modals", () => {
  it("opens shortcuts, traps nothing behind it, closes on Escape and restores focus", async () => {
    const user = userEvent.setup();
    await renderReady();
    const trigger = screen.getByRole("button", { name: /Shortcuts/ });
    await user.click(trigger);

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveAccessibleName("Keyboard shortcuts");
    expect(document.body.style.overflow).toBe("hidden");

    // shortcuts are suspended while the dialog owns the keyboard
    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(useStore.getState().selectedLayer).toBe(0);

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(document.body.style.overflow).toBe("");
    expect(document.activeElement).toBe(trigger);
  });

  it("closes on a backdrop click but not on a click inside the panel", async () => {
    const user = userEvent.setup();
    const { container } = await renderReady();
    await user.click(screen.getByRole("button", { name: /How to read this/ }));
    const dialog = await screen.findByRole("dialog");

    await user.click(within(dialog).getByText("The matrix"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    fireEvent.mouseDown(container.ownerDocument.querySelector(".modal-backdrop")!);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("lists models with their memory cost and lets you switch from there", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.click(screen.getByRole("button", { name: "details" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/6144/)).toBeInTheDocument();
    expect(within(dialog).getByText(/12 layers × 12 heads/)).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Use" }));
    expect(useStore.getState().model).toBe("gpt2-small");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});

describe("loading and errors", () => {
  it("keeps the previous run on screen while the next one is in flight", async () => {
    const user = userEvent.setup();
    await renderReady();

    let release!: () => void;
    mock.gateRun(new Promise<void>((r) => (release = r)));
    await user.type(screen.getByLabelText(/Prompt/), "!");

    await waitFor(() => expect(useStore.getState().runPhase).toBe("loading"), { timeout: 2000 });
    // the point of stale-while-revalidate: the old view is still there
    expect(screen.getByRole("button", { name: "·quick" })).toBeInTheDocument();
    // the dim is deliberately delayed, so a fast run never flashes grey
    expect(document.querySelector(".is-stale")).toBeNull();
    await waitFor(() => expect(document.querySelector('[aria-busy="true"]')).toHaveClass("is-stale"), {
      timeout: 2000,
    });

    release();
    await waitFor(() => expect(useStore.getState().runPhase).toBe("idle"));
    expect(document.querySelector(".is-stale")).toBeNull();
  });

  it("surfaces an API error in a live region", async () => {
    const user = userEvent.setup();
    await renderReady();
    mock.failRunWith({ code: "seq_too_long", message: "Prompt is 900 tokens; this model caps at 512." });
    await user.type(screen.getByLabelText(/Prompt/), "!");

    const alert = await screen.findByRole("alert", {}, { timeout: 2000 });
    expect(alert).toHaveTextContent("this model caps at 512");
    expect(useStore.getState().runPhase).toBe("error");
  });

  it("clears back to the empty state when the prompt is emptied", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.clear(screen.getByLabelText(/Prompt/));
    expect(await screen.findByText(/Type a prompt above/, {}, { timeout: 2000 })).toBeInTheDocument();
  });
});

describe("cost and prediction panels", () => {
  it("reports the wire size against the float32 figure for the visible layer", async () => {
    await renderReady();
    expect(screen.getByText("Cost")).toBeInTheDocument();
    // 2 layers x 8 heads x 4^2 x 4B = 1024 B
    expect(screen.getByText("1.00 KB")).toBeInTheDocument();
    expect(screen.getByText(/Sent to your browser \(layer 0\)/)).toBeInTheDocument();
  });

  it("shows the next-token prediction and marks the one that actually followed", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.hover(screen.getByRole("button", { name: "·brown" })); // position 2
    expect(await screen.findByText(/after position 2/)).toBeInTheDocument();
    expect(screen.getByText("actual")).toBeInTheDocument(); // " fox" follows position 2
    expect(screen.getByText(/loss/)).toBeInTheDocument();
  });
});

describe("byte-level fragmentation (non-Latin scripts)", () => {
  beforeEach(() => {
    // GPT-2 has no merges for Devanagari, so each character costs two tokens
    // and neither decodes to a character alone.
    mock.useTokens(FRAGMENTED_TOKENS);
  });

  it("never renders the replacement character, showing the source character instead", async () => {
    await renderReady();
    const strip = document.querySelector(".chipstrip")!;
    expect(strip.textContent).not.toContain("\ufffd");
    expect(screen.getByRole("button", { name: /byte 1 of 2 of "म"/ })).toHaveTextContent("म");
  });

  it("joins the tokens of one character into a single marked group", async () => {
    await renderReady();
    const groups = document.querySelectorAll(".cluster--frag");
    expect(groups).toHaveLength(2); // म and ल; ा is a whole token
    expect(groups[0].querySelectorAll(".chip")).toHaveLength(2);
    // the continuation carries a marker, not a duplicate of the character
    expect(groups[0].querySelectorAll(".chip")[1]).toHaveTextContent("⋯");
  });

  it("does not group a character the tokenizer encodes in one token", async () => {
    await renderReady();
    const whole = screen.getByTitle(/id 48077/);
    expect(whole).toHaveTextContent("ा");
    expect(whole.closest(".cluster")).not.toHaveClass("cluster--frag");
  });

  it("reports the raw bytes and the fragment position in the tooltip", async () => {
    await renderReady();
    expect(screen.getByRole("button", { name: /byte 1 of 2 of "म"/ })).toHaveAttribute(
      "title",
      expect.stringContaining("bytes 0xe0a4"),
    );
    expect(screen.getByRole("button", { name: /byte 2 of 2 of "म"/ })).toHaveAttribute(
      "title",
      expect.stringContaining("bytes 0xae"),
    );
  });

  it("explains the fragmentation rather than leaving the reader to guess", async () => {
    await renderReady();
    expect(screen.getByText(/4 of 5 tokens are byte fragments/)).toBeInTheDocument();
    expect(screen.getByText(/no merges for this script/)).toBeInTheDocument();
  });

  it("keeps every fragment individually hoverable — each is its own position", async () => {
    const user = userEvent.setup();
    await renderReady();
    await user.hover(screen.getByRole("button", { name: /byte 2 of 2 of "म"/ }));
    await waitFor(() => expect(useStore.getState().hoveredTokenIdx).toBe(1));
  });

  it("shows the tokens-per-character ratio", async () => {
    await renderReady();
    expect(screen.getByText(/1\.67 tokens\/char/)).toBeInTheDocument();
  });
});

describe("Latin text stays unfragmented", () => {
  it("shows no cluster grouping and no fragmentation notice", async () => {
    await renderReady();
    expect(document.querySelectorAll(".cluster--frag")).toHaveLength(0);
    expect(screen.queryByText(/byte fragments/)).not.toBeInTheDocument();
  });
});
