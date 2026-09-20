import type { DecodedPatterns } from "../api/patterns";
import type { Theme } from "../theme";
import { getSequentialLut } from "./colormap";

/** Reads the mask colour off <html> on every render. Deliberately NOT cached:
 * a cache keyed by theme once stored a value read a tick too early (before the
 * [data-theme] stamp landed) and then served it forever. One custom-property
 * lookup per canvas is far cheaper than the pixel loop it precedes. */
function maskRgb(theme: Theme): [number, number, number] {
  const raw = getComputedStyle(document.documentElement).getPropertyValue("--heat-mask").trim();
  const m = /^#([0-9a-f]{6})$/i.exec(raw);
  const n = m ? parseInt(m[1], 16) : theme === "dark" ? 0x141413 : 0xf3f3f0;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Renders one head's attention matrix directly from its raw quantized bytes
 * (decoded.rawAt) through a precomputed colormap LUT — no float round-trip, no
 * per-pixel interpolation. Canvas intrinsic size is exactly seq x seq; visual
 * upscaling is a CSS concern (`.pixel-canvas { image-rendering: pixelated }`),
 * per docs/PLAN.md Stage 1.
 *
 * Cells above the diagonal (src > dest) are structurally impossible under
 * causal masking, not "attention of zero". They're painted in the page's mask
 * colour so the matrix reads as a triangle; painting them ramp-minimum instead
 * made half the plot a solid block of pale blue that looked like data.
 */
export function renderHeadToCanvas(
  canvas: HTMLCanvasElement,
  decoded: DecodedPatterns,
  layerIndexInResponse: number,
  head: number,
  theme: Theme,
): void {
  const { seq } = decoded;
  if (canvas.width !== seq) canvas.width = seq;
  if (canvas.height !== seq) canvas.height = seq;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;

  const lut = getSequentialLut(theme);
  const [mr, mg, mb] = maskRgb(theme);
  const imageData = ctx.createImageData(seq, seq);
  const data = imageData.data;

  for (let dest = 0; dest < seq; dest++) {
    const rowOff = dest * seq;
    for (let src = 0; src < seq; src++) {
      const p = (rowOff + src) * 4;
      if (src > dest) {
        data[p] = mr;
        data[p + 1] = mg;
        data[p + 2] = mb;
      } else {
        const lutOff = decoded.rawAt(layerIndexInResponse, head, dest, src) * 3;
        data[p] = lut[lutOff];
        data[p + 1] = lut[lutOff + 1];
        data[p + 2] = lut[lutOff + 2];
      }
      data[p + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

/** Maps a pointer event's position over a (CSS-scaled) canvas back to
 * (dest, src) matrix coordinates. */
export function canvasEventToCell(
  e: { clientX: number; clientY: number },
  canvas: HTMLCanvasElement,
  seq: number,
): { dest: number; src: number } {
  const rect = canvas.getBoundingClientRect();
  const src = Math.max(0, Math.min(seq - 1, Math.floor(((e.clientX - rect.left) / rect.width) * seq)));
  const dest = Math.max(0, Math.min(seq - 1, Math.floor(((e.clientY - rect.top) / rect.height) * seq)));
  return { dest, src };
}
