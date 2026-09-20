import { useMemo } from "react";
import { attentionVector, maxOf } from "../lib/attention";
import { useStore } from "../state/store";

export interface LinkedWeights {
  /** the token position everything is linked to */
  anchor: number;
  /** one weight per position, already direction-aware and causally masked */
  vector: Float32Array;
  /** largest weight in `vector`, for normalising tints and bars */
  max: number;
  /** null when the head grid is open — the vector is then a mean over heads */
  head: number | null;
}

/**
 * The single source of "what is hover-linked right now". Hover wins while
 * active; a clicked token stays pinned once the pointer leaves. Returns null
 * when there is nothing to link (no patterns yet, or no token engaged).
 */
export function useLinkedWeights(): LinkedWeights | null {
  const decoded = useStore((s) => s.patternsByLayer.get(s.selectedLayer));
  const hovered = useStore((s) => s.hoveredTokenIdx);
  const selected = useStore((s) => s.selectedTokenIdx);
  const head = useStore((s) => s.selectedHead);
  const direction = useStore((s) => s.direction);

  const anchor = hovered ?? selected;

  return useMemo(() => {
    if (!decoded || anchor === null || anchor < 0 || anchor >= decoded.seq) return null;
    const vector = attentionVector(decoded, 0, head, anchor, direction);
    return { anchor, vector, max: maxOf(vector), head };
  }, [decoded, anchor, head, direction]);
}
