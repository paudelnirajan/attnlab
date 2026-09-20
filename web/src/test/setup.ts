import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(cleanup);

// jsdom has no matchMedia; theme.ts reads it at module load.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

// jsdom has no canvas backend. A minimal 2D context means the real rendering
// loop in render/heatmap.ts actually executes under test (LUT lookups, the
// causal-mask branch, bounds) instead of being skipped by its `if (!ctx)`
// guard — the pixels just go nowhere.
HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, kind: string) {
  if (kind !== "2d") return null;
  return {
    imageSmoothingEnabled: true,
    createImageData: (w: number, h: number) => ({
      width: w,
      height: h,
      data: new Uint8ClampedArray(w * h * 4),
      colorSpace: "srgb" as PredefinedColorSpace,
    }),
    putImageData: () => {},
  } as unknown as CanvasRenderingContext2D;
} as typeof HTMLCanvasElement.prototype.getContext;

/** jsdom gives every element a zero-size rect, which makes pointer->cell maths
 * produce NaN. Tests that click the matrix stamp a real box on it. */
export function stubRect(el: Element, box: { left?: number; top?: number; width: number; height: number }) {
  const { left = 0, top = 0, width, height } = box;
  vi.spyOn(el, "getBoundingClientRect").mockReturnValue({
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
    x: left,
    y: top,
    toJSON: () => ({}),
  } as DOMRect);
}
