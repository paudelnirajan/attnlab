import type { DecodedPatterns } from "../api/patterns";
import type { Direction } from "../state/store";

// One place that knows what "direction" means numerically, so the token strip,
// the head tiles, the detail crosshair and the top-k list can never disagree
// about which way round they're reading the matrix.

/** Human-facing copy for each direction, used in labels, tooltips and help. */
export const DIRECTION_COPY: Record<
  Direction,
  { short: string; arrow: string; anchorRole: string; otherRole: string; question: string; note: string }
> = {
  dest2src: {
    short: "Destination → Source",
    arrow: "→",
    anchorRole: "destination",
    otherRole: "source",
    question: "what does this token attend to?",
    note: "A row of the matrix — a softmax distribution, so these weights sum to 1.",
  },
  src2dest: {
    short: "Source → Destination",
    arrow: "←",
    anchorRole: "source",
    otherRole: "destination",
    question: "which tokens attend to this one?",
    note: "A column of the matrix — attention received, not a distribution, so these do NOT sum to 1.",
  },
};

/** True when position `i` can't participate given the anchor and direction,
 * because causal masking forbids it (not because the weight happens to be 0). */
export function isMasked(direction: Direction, anchor: number, i: number): boolean {
  return direction === "dest2src" ? i > anchor : i < anchor;
}

/**
 * The attention weights linked to `anchor`, one per token position.
 *
 *  dest2src -> row    A[anchor][*]  (sums to 1)
 *  src2dest -> column A[*][anchor]  (does not sum to 1)
 *
 * `head === null` averages over every head in the layer, which is the sensible
 * thing to show while the grid is open and no single head is expanded.
 */
export function attentionVector(
  decoded: DecodedPatterns,
  layerIndexInResponse: number,
  head: number | null,
  anchor: number,
  direction: Direction,
): Float32Array {
  const { seq, nHeads } = decoded;
  const out = new Float32Array(seq);
  const heads = head === null ? Array.from({ length: nHeads }, (_, h) => h) : [head];
  const scale = 1 / heads.length;

  for (let i = 0; i < seq; i++) {
    if (isMasked(direction, anchor, i)) continue;
    let acc = 0;
    for (const h of heads) {
      acc +=
        direction === "dest2src"
          ? decoded.at(layerIndexInResponse, h, anchor, i)
          : decoded.at(layerIndexInResponse, h, i, anchor);
    }
    out[i] = acc * scale;
  }
  return out;
}

export function maxOf(v: Float32Array): number {
  let m = 0;
  for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
  return m;
}

export interface RankedEntry {
  idx: number;
  value: number;
}

/** Top-k positions by weight, strongest first, skipping exact zeros. */
export function topK(v: Float32Array, k: number): RankedEntry[] {
  const entries: RankedEntry[] = [];
  for (let i = 0; i < v.length; i++) if (v[i] > 0) entries.push({ idx: i, value: v[i] });
  entries.sort((a, b) => b.value - a.value);
  return entries.slice(0, k);
}
