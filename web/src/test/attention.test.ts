import { describe, expect, it } from "vitest";
import { decodePatterns } from "../api/patterns";
import { attentionVector, isMasked, maxOf, topK } from "../lib/attention";
import { encodeAtnp, expectedWeight, N_HEADS, SEQ } from "./fixtures";

const decoded = decodePatterns(encodeAtnp([0]));

describe("ATNP decoder", () => {
  it("reads the header the spec describes", () => {
    expect(decoded.layerIds).toEqual([0]);
    expect(decoded.nHeads).toBe(N_HEADS);
    expect(decoded.seq).toBe(SEQ);
    expect(decoded.packed).toBe(true);
    expect(decoded.byteLength).toBe(16 + 2 + N_HEADS * ((SEQ * (SEQ + 1)) / 2));
  });

  it("rejects a payload that isn't ATNP", () => {
    expect(() => decodePatterns(new ArrayBuffer(32))).toThrow(/bad magic/);
  });

  it("round-trips within the companding error bound (docs/01-wire-format.md)", () => {
    for (let h = 0; h < N_HEADS; h++) {
      for (let dest = 0; dest < SEQ; dest++) {
        for (let src = 0; src <= dest; src++) {
          const want = expectedWeight(h, dest, src);
          const got = decoded.at(0, h, dest, src);
          expect(Math.abs(got - want)).toBeLessThanOrEqual(0.004);
          if (want >= 0.01) expect(Math.abs(got - want) / want).toBeLessThanOrEqual(0.04);
        }
      }
    }
  });

  it("returns 0 above the diagonal, where causal masking makes a cell impossible", () => {
    expect(decoded.at(0, 0, 1, 2)).toBe(0);
    expect(decoded.at(0, 0, 0, 3)).toBe(0);
  });
});

describe("read direction", () => {
  it("masks later positions when reading destination -> source", () => {
    expect(isMasked("dest2src", 2, 3)).toBe(true);
    expect(isMasked("dest2src", 2, 1)).toBe(false);
  });

  it("masks earlier positions when reading source -> destination", () => {
    expect(isMasked("src2dest", 2, 1)).toBe(true);
    expect(isMasked("src2dest", 2, 3)).toBe(false);
  });

  it("dest2src returns the matrix row, and it sums to 1", () => {
    const v = attentionVector(decoded, 0, 0, SEQ - 1, "dest2src");
    // tolerance is the wire format's own quantisation bound, not an
    // arbitrary decimal place — see docs/01-wire-format.md
    for (let src = 0; src < SEQ; src++) {
      expect(Math.abs(v[src] - expectedWeight(0, SEQ - 1, src))).toBeLessThanOrEqual(0.004);
    }
    expect(v.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 2);
  });

  it("src2dest returns the matrix column, which is NOT a distribution", () => {
    const anchor = 0;
    const v = attentionVector(decoded, 0, 0, anchor, "src2dest");
    for (let dest = 0; dest < SEQ; dest++) {
      expect(Math.abs(v[dest] - expectedWeight(0, dest, anchor))).toBeLessThanOrEqual(0.004);
    }
    // every position attends to position 0, so the column total exceeds 1
    expect(v.reduce((a, b) => a + b, 0)).toBeGreaterThan(1);
  });

  it("the two directions read the same cell from opposite ends", () => {
    const row = attentionVector(decoded, 0, 3, 3, "dest2src");
    const col = attentionVector(decoded, 0, 3, 1, "src2dest");
    expect(row[1]).toBeCloseTo(col[3], 6);
  });

  it("a null head averages over every head in the layer", () => {
    const mean = attentionVector(decoded, 0, null, 3, "dest2src");
    const perHead = Array.from({ length: N_HEADS }, (_, h) => attentionVector(decoded, 0, h, 3, "dest2src"));
    for (let i = 0; i < SEQ; i++) {
      const avg = perHead.reduce((a, v) => a + v[i], 0) / N_HEADS;
      expect(mean[i]).toBeCloseTo(avg, 6);
    }
    // and it is genuinely different from any single head, or the test is vacuous
    expect(mean[0]).not.toBeCloseTo(perHead[0][0], 6);
  });
});

describe("topK", () => {
  it("ranks strongest first and drops masked positions", () => {
    const v = attentionVector(decoded, 0, 0, 2, "dest2src");
    const ranked = topK(v, 10);
    expect(ranked.map((r) => r.idx)).toEqual([2, 1, 0]); // weight grows with src here
    expect(ranked[0].value).toBeGreaterThan(ranked[1].value);
  });

  it("maxOf finds the largest weight", () => {
    const v = attentionVector(decoded, 0, 0, 2, "dest2src");
    expect(maxOf(v)).toBeCloseTo(Math.max(...Array.from(v)), 9);
  });
});
