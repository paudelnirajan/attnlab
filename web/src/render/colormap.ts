// Sequential single-hue (blue) colormap per the dataviz skill's reference
// palette (references/palette.md): "Sequential = one hue, light->dark."
// The 13 named steps there are a discrete ramp for a legend; a heatmap
// needs a continuous one, so this interpolates across those exact stops
// and bakes the result into a 256-entry LUT per theme — rendering then
// costs one array index per pixel, no interpolation at paint time.
//
// Dark mode reverses the stop order rather than inventing new hex values
// (the palette file gives one ramp, not a separate light/dark pair for
// sequential data): step 700 (near-black-blue) sits near the dark
// surface at the low end, and the palest step reads as a bright,
// high-contrast "hot" color against a dark background — the same
// "recede toward this theme's own surface at zero" rule the palette
// describes for the light direction, applied in the other direction.

const BLUE_RAMP_LIGHT_TO_DARK: readonly string[] = [
  "#cde2fb",
  "#b7d3f6",
  "#9ec5f4",
  "#86b6ef",
  "#6da7ec",
  "#5598e7",
  "#3987e5",
  "#2a78d6",
  "#256abf",
  "#1c5cab",
  "#184f95",
  "#104281",
  "#0d366b",
];

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function buildLut(stopsLowToHigh: readonly string[]): Uint8ClampedArray {
  const stops = stopsLowToHigh.map(hexToRgb);
  const lut = new Uint8ClampedArray(256 * 3);
  const lastIdx = stops.length - 1;
  for (let byte = 0; byte < 256; byte++) {
    const t = (byte / 255) * lastIdx;
    const i0 = Math.floor(t);
    const i1 = Math.min(i0 + 1, lastIdx);
    const frac = t - i0;
    const [r0, g0, b0] = stops[i0];
    const [r1, g1, b1] = stops[i1];
    lut[byte * 3] = r0 + (r1 - r0) * frac;
    lut[byte * 3 + 1] = g0 + (g1 - g0) * frac;
    lut[byte * 3 + 2] = b0 + (b1 - b0) * frac;
  }
  return lut;
}

// byte 0 ("near zero") -> palest step, near the light surface.
export const SEQUENTIAL_LUT_LIGHT = buildLut(BLUE_RAMP_LIGHT_TO_DARK);
// byte 0 -> darkest step, near the dark surface (#1a1a19); see file header.
export const SEQUENTIAL_LUT_DARK = buildLut([...BLUE_RAMP_LIGHT_TO_DARK].reverse());

export function getSequentialLut(theme: "light" | "dark"): Uint8ClampedArray {
  return theme === "dark" ? SEQUENTIAL_LUT_DARK : SEQUENTIAL_LUT_LIGHT;
}
