import { useMemo } from "react";
import type { LabToken } from "../api";
import { KIND_COPY } from "../copy";
import type { ChipMode } from "../store";

interface Group {
  cluster: number;
  members: LabToken[];
}

/** Consecutive tokens that jointly spell one character are drawn as one joined
 * unit — the same convention as the attention lab's token strip, so a learner
 * who has seen one can read the other. */
function groupByCluster(tokens: LabToken[]): Group[] {
  const groups: Group[] = [];
  for (const t of tokens) {
    const last = groups[groups.length - 1];
    if (last && last.cluster === t.cluster && t.cluster_index > 0) last.members.push(t);
    else groups.push({ cluster: t.cluster, members: [t] });
  }
  return groups;
}

function spacedHex(hex: string): string {
  return hex.replace(/(..)(?!$)/g, "$1 ");
}

export function chipLabel(t: LabToken, mode: ChipMode): string {
  switch (mode) {
    case "ids":
      return String(t.id);
    case "bytes":
      return t.byte_hex ? spacedHex(t.byte_hex) : t.piece;
    case "vocab":
      return t.piece;
    default:
      return t.display || "∅";
  }
}

export function tokenTooltip(t: LabToken): string {
  const parts = [`#${t.index}`, `id ${t.id}`, KIND_COPY[t.kind]];
  if (t.cluster_size > 1) parts.push(`token ${t.cluster_index + 1} of ${t.cluster_size} for "${t.cluster_text}"`);
  if (t.byte_hex) parts.push(`bytes ${spacedHex(t.byte_hex)}`);
  if (t.piece !== t.display) parts.push(`vocab string ${JSON.stringify(t.piece)}`);
  return parts.join(" · ");
}

interface Props {
  tokens: LabToken[];
  mode: ChipMode;
  hovered?: number | null;
  onHover?: (index: number | null) => void;
  /** a shorter scroll box, for stacking several strips (Compare) */
  compact?: boolean;
  label?: string;
}

export function TokenStrip({ tokens, mode, hovered = null, onHover, compact, label }: Props) {
  const groups = useMemo(() => groupByCluster(tokens), [tokens]);
  const hoveredCluster = hovered !== null ? tokens[hovered]?.cluster : undefined;

  return (
    <div
      className={compact ? "chipstrip chipstrip--compact" : "chipstrip"}
      role="list"
      aria-label={label ?? "Tokens"}
      onMouseLeave={() => onHover?.(null)}
    >
      {groups.map((g) => (
        <span
          key={`${g.cluster}-${g.members[0].index}`}
          role="listitem"
          className={g.members.length > 1 ? "cluster cluster--frag" : "cluster"}
        >
          {g.members.map((t) => {
            const classes = ["chip", `chip--${t.kind}`];
            if (mode !== "text") classes.push("chip--mono");
            if (t.cluster_index > 0 && mode === "text") classes.push("chip--cont");
            if (hovered === t.index) classes.push("chip--anchor");
            else if (hoveredCluster === t.cluster && hovered !== null) classes.push("chip--sibling");
            return (
              <button
                key={t.index}
                type="button"
                className={classes.join(" ")}
                title={tokenTooltip(t)}
                aria-label={
                  t.cluster_size > 1
                    ? `token ${t.cluster_index + 1} of ${t.cluster_size} for "${t.cluster_text}", id ${t.id}`
                    : undefined
                }
                onMouseEnter={() => onHover?.(t.index)}
                onFocus={() => onHover?.(t.index)}
              >
                {chipLabel(t, mode)}
              </button>
            );
          })}
        </span>
      ))}
    </div>
  );
}

const MODES: { id: ChipMode; label: string; hint: string }[] = [
  { id: "text", label: "Text", hint: "what each token spells" },
  { id: "ids", label: "IDs", hint: "the integers the model actually receives" },
  { id: "bytes", label: "Bytes", hint: "the UTF-8 bytes each token covers" },
  { id: "vocab", label: "Vocab string", hint: "how the token is written in the tokenizer's vocabulary file" },
];

export function ChipModeToggle({ mode, onChange }: { mode: ChipMode; onChange: (m: ChipMode) => void }) {
  return (
    <div className="segmented" role="group" aria-label="Show tokens as">
      {MODES.map((m) => (
        <button
          key={m.id}
          type="button"
          className="segmented__opt"
          aria-pressed={mode === m.id}
          title={m.hint}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
