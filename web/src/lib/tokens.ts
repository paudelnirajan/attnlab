import type { TokenInfo } from "../api/types";

/**
 * What to call a token outside the token strip.
 *
 * A continuation token's `display` is "⋯", which is right *inside* the strip —
 * its neighbours give it context there — and useless anywhere else. "Top
 * sources for ⋯ at position 56" tells the reader nothing. Elsewhere, name a
 * fragment by the character it helps spell.
 */
export function tokenLabel(t: TokenInfo | undefined): string {
  if (!t) return "∅";
  if (t.cluster_size > 1) return t.cluster_text || "∅";
  return t.display || "∅";
}

/** "2/3" when the token is one byte of a multi-token character, else null. */
export function tokenFragmentNote(t: TokenInfo | undefined): string | null {
  if (!t || t.cluster_size <= 1) return null;
  return `${t.cluster_index + 1}/${t.cluster_size}`;
}
