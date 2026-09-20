import { useMemo } from "react";
import type { TokenInfo } from "../api/types";
import { DIRECTION_COPY, isMasked } from "../lib/attention";
import { useStore } from "../state/store";
import { useLinkedWeights } from "../hooks/useLinkedWeights";

/** Share of tokens that are fragments of a character rather than a word piece,
 * above which the panel explains what the reader is looking at. */
const FRAGMENT_NOTICE_THRESHOLD = 0.2;

interface Group {
  cluster: number;
  size: number;
  text: string;
  members: { token: TokenInfo; index: number }[];
}

/** Consecutive tokens covering one indivisible piece of source text are drawn
 * as one joined unit, so "this single character cost three tokens" is visible
 * at a glance instead of inferable from a row of identical-looking chips. */
function groupByCluster(tokens: TokenInfo[]): Group[] {
  const groups: Group[] = [];
  tokens.forEach((token, index) => {
    const last = groups[groups.length - 1];
    if (last && last.cluster === token.cluster && token.cluster_index > 0) {
      last.members.push({ token, index });
    } else {
      groups.push({ cluster: token.cluster, size: token.cluster_size, text: token.cluster_text, members: [{ token, index }] });
    }
  });
  return groups;
}

function tooltip(t: TokenInfo, index: number, weightNote: string): string {
  const parts = [`position ${index}`, `id ${t.id}`];
  if (t.cluster_size > 1) {
    parts.push(`byte ${t.cluster_index + 1} of ${t.cluster_size} of "${t.cluster_text}"`);
  }
  if (t.byte_hex) parts.push(`bytes 0x${t.byte_hex}`);
  if (weightNote) parts.push(weightNote);
  return parts.join(" · ");
}

/**
 * Token chips, tinted live by attention weight — docs/PLAN.md Stage 1:
 * "hovering a destination token highlights its source distribution both on the
 * heatmap and inline in the text. This is the feature that makes it feel
 * alive." A binary highlight can't show *how much*; the tint can.
 *
 * Tokens come from the run, not from /tokenize, so the strip can never
 * disagree with the heatmap about what index `i` means while a new tokenize
 * response and a slower run response are in flight at the same time.
 */
export function TokenizerPanel() {
  const runResult = useStore((s) => s.runResult);
  const hovered = useStore((s) => s.hoveredTokenIdx);
  const selected = useStore((s) => s.selectedTokenIdx);
  const setHoveredToken = useStore((s) => s.setHoveredToken);
  const setSelectedToken = useStore((s) => s.setSelectedToken);
  const direction = useStore((s) => s.direction);
  const linked = useLinkedWeights();

  const tokens = runResult?.tokens;
  const groups = useMemo(() => (tokens ? groupByCluster(tokens) : []), [tokens]);
  const fragmentCount = useMemo(() => (tokens ? tokens.filter((t) => t.cluster_size > 1).length : 0), [tokens]);

  if (!runResult || !tokens) return null;
  const anchor = linked?.anchor ?? hovered ?? selected;
  const copy = DIRECTION_COPY[direction];
  const heavilyFragmented = fragmentCount / tokens.length > FRAGMENT_NOTICE_THRESHOLD;

  return (
    <>
      <div className="chipstrip" onMouseLeave={() => setHoveredToken(null)}>
        {groups.map((group) => (
          <span
            key={`${group.cluster}-${group.members[0].index}`}
            className={group.members.length > 1 ? "cluster cluster--frag" : "cluster"}
          >
            {group.members.map(({ token, index }) => {
              const isAnchor = index === anchor;
              const masked = anchor !== null && isMasked(direction, anchor, index);
              const weight = linked ? linked.vector[index] : 0;
              // Normalised against the row/column max so the strongest link is
              // always fully saturated; sqrt keeps mid-weights visible. Mixing
              // toward --surface-2 means "no attention" is literally the chip's
              // resting colour, so only real signal shows up as colour.
              const pct = linked && linked.max > 0 && !masked ? Math.round(Math.sqrt(weight / linked.max) * 92) : 0;

              const classes = ["chip"];
              if (isAnchor) classes.push("chip--anchor");
              if (index === selected) classes.push("chip--pinned");
              if (token.cluster_index > 0) classes.push("chip--cont");
              // a lone unrenderable token, with no cluster to carry the marker
              if (token.is_byte_fallback && token.cluster_size === 1) classes.push("chip--bytefallback");
              if (masked && !isAnchor) classes.push("chip--masked");

              return (
                <button
                  key={index}
                  type="button"
                  className={classes.join(" ")}
                  aria-pressed={index === selected}
                  // Only fragments get an explicit label. A whole token's
                  // visible text IS its name; overriding that would hide the
                  // token from anyone querying or reading by its text.
                  aria-label={
                    token.cluster_size > 1
                      ? `byte ${token.cluster_index + 1} of ${token.cluster_size} of "${token.cluster_text}", position ${index}`
                      : undefined
                  }
                  style={{
                    background: pct > 0 ? `color-mix(in oklab, var(--accent) ${pct}%, var(--surface-2))` : undefined,
                    color: pct > 55 ? "var(--accent-ink)" : undefined,
                  }}
                  title={tooltip(token, index, linked && !masked ? `${copy.otherRole} weight ${weight.toFixed(4)}` : "")}
                  onMouseEnter={() => setHoveredToken(index)}
                  onFocus={() => setHoveredToken(index)}
                  onClick={() => setSelectedToken(selected === index ? null : index)}
                >
                  {token.display || "∅"}
                </button>
              );
            })}
          </span>
        ))}
      </div>

      <p className="muted" style={{ fontSize: "var(--t-sm)", marginTop: "var(--s2)" }}>
        {anchor !== null && tokens[anchor] ? (
          <>
            <strong className="mono">{tokens[anchor].cluster_text || tokens[anchor].display || "∅"}</strong> at
            position {anchor} as {copy.anchorRole}
            {tokens[anchor].cluster_size > 1 && (
              <> (byte {tokens[anchor].cluster_index + 1} of {tokens[anchor].cluster_size})</>
            )}{" "}
            — tint shows the {copy.otherRole} weights
            {linked?.head === null ? ", averaged over all heads in this layer" : ` for head ${linked?.head}`}.
            {selected === anchor ? " Pinned; click again to release." : ""}
          </>
        ) : (
          <>Hover a token to link it across the heads; click to pin it.</>
        )}
      </p>

      {heavilyFragmented && (
        <p className="banner banner--info" style={{ marginTop: "var(--s3)", fontSize: "var(--t-sm)" }}>
          <span>
            <strong>{fragmentCount} of {tokens.length} tokens are byte fragments.</strong> This model's tokenizer has
            no merges for this script, so it falls back to raw UTF-8 bytes — a single character costs two or three
            tokens, and none of them is a word piece. Underlined groups above are the tokens that together spell one
            character. Much of what you see the attention heads doing here is reassembling the text encoding rather
            than modelling the language.
          </span>
        </p>
      )}
    </>
  );
}
