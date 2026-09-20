import { beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_MODEL, DEFAULT_PROMPT } from "../state/defaults";
import { readPermalinkFromUrl, writePermalinkToUrl } from "../state/urlSync";

function setUrl(search: string) {
  window.history.replaceState(null, "", `/${search}`);
}

describe("permalink", () => {
  beforeEach(() => setUrl(""));

  it("falls back to the defaults on a bare URL", () => {
    expect(readPermalinkFromUrl()).toEqual({
      model: DEFAULT_MODEL,
      prompt: DEFAULT_PROMPT,
      selectedLayer: 0,
      selectedHead: null,
      direction: "dest2src",
    });
  });

  it("reads every field back", () => {
    setUrl("?model=gpt2-small&prompt=hello%20there&layer=3&head=7&dir=src2dest");
    expect(readPermalinkFromUrl()).toEqual({
      model: "gpt2-small",
      prompt: "hello there",
      selectedLayer: 3,
      selectedHead: 7,
      direction: "src2dest",
    });
  });

  it("treats an explicitly empty prompt as empty, not as the default", () => {
    setUrl("?prompt=");
    expect(readPermalinkFromUrl().prompt).toBe("");
  });

  it("rejects junk indices instead of turning them into plausible ones", () => {
    setUrl("?layer=abc&head=-1");
    expect(readPermalinkFromUrl().selectedLayer).toBe(0);
    expect(readPermalinkFromUrl().selectedHead).toBeNull();
    setUrl("?layer=1.5&head=2.7");
    expect(readPermalinkFromUrl().selectedLayer).toBe(0);
    expect(readPermalinkFromUrl().selectedHead).toBeNull();
  });

  it("ignores an unknown direction", () => {
    setUrl("?dir=sideways");
    expect(readPermalinkFromUrl().direction).toBe("dest2src");
  });

  it("omits head and direction when they are at their defaults", () => {
    writePermalinkToUrl({ model: "m", prompt: "p", selectedLayer: 2, selectedHead: null, direction: "dest2src" });
    expect(window.location.search).toBe("?model=m&prompt=p&layer=2");
  });

  it("round-trips a full state through the URL", () => {
    const state = {
      model: "gpt2-small",
      prompt: "a b c",
      selectedLayer: 4,
      selectedHead: 2,
      direction: "src2dest" as const,
    };
    writePermalinkToUrl(state);
    expect(readPermalinkFromUrl()).toEqual(state);
  });

  it("does not add history entries", () => {
    const before = window.history.length;
    writePermalinkToUrl({ model: "m", prompt: "x", selectedLayer: 0, selectedHead: null, direction: "dest2src" });
    writePermalinkToUrl({ model: "m", prompt: "y", selectedLayer: 0, selectedHead: null, direction: "dest2src" });
    expect(window.history.length).toBe(before);
  });
});
