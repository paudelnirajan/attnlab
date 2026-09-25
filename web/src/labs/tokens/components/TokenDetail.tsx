import type { ReactNode } from "react";
import type { AnalyzeResult, Algorithm } from "../api";
import { KIND_COPY } from "../copy";

/**
 * The hovered token, in full, and where it came from in the text. Hovering a
 * fragment highlights the whole character it belongs to, because the
 * character is the smallest unit of the text that fragment can point at.
 */
export function TokenDetail({
  result,
  text,
  hovered,
  algorithm,
  nMerges,
}: {
  result: AnalyzeResult;
  /** the text `result` was computed from — not the live text box, which may be ahead of it */
  text: string;
  hovered: number | null;
  algorithm: Algorithm;
  nMerges: number;
}) {
  const t = hovered !== null ? result.tokens[hovered] : undefined;
  if (!t) {
    return (
      <p className="muted token-detail token-detail--empty">
        Hover a token to see its id, its bytes, and where it came from in your text.
      </p>
    );
  }

  const members = result.tokens.filter((u) => u.cluster === t.cluster);
  const lo = Math.min(...members.map((u) => u.start));
  const hi = Math.max(...members.map((u) => u.end));
  const before = text.slice(Math.max(0, lo - 24), lo);
  const inside = text.slice(lo, hi);
  const after = text.slice(hi, hi + 24);

  const facts: [string, ReactNode][] = [
    ["position", <span className="mono">{t.index}</span>],
    ["id", <span className="mono">{t.id}</span>],
    ["vocab string", <span className="mono">{JSON.stringify(t.piece)}</span>],
    ["kind", KIND_COPY[t.kind]],
  ];
  if (t.byte_hex) facts.push(["bytes", <span className="mono">{t.byte_hex.replace(/(..)(?!$)/g, "$1 ")}</span>]);
  if (t.cluster_size > 1) {
    facts.push([
      "part of",
      <>
        token {t.cluster_index + 1} of {t.cluster_size} that together spell{" "}
        <strong className="mono">{t.cluster_text}</strong>
      </>,
    ]);
  }
  if (t.rank !== null && nMerges > 0) {
    const pct = (100 * t.rank) / nMerges;
    facts.push([
      "learned at",
      <>
        merge <span className="mono">#{t.rank.toLocaleString()}</span> of {nMerges.toLocaleString()}{" "}
        <span className="muted">
          ({pct < 10 ? "early: a very common pair" : pct > 70 ? "late: a rarer pair" : "mid-way"})
        </span>
      </>,
    ]);
  } else if (t.rank === null && (algorithm === "byte-bpe" || algorithm === "sp-bpe") && t.kind !== "special" && t.kind !== "added") {
    facts.push(["learned at", <span className="muted">not a merge — a base symbol of the alphabet</span>]);
  }
  if (t.score !== null) facts.push(["log-prob", <span className="mono">{t.score.toFixed(3)}</span>]);

  return (
    <div className="token-detail">
      {hi > lo ? (
        <p className="token-detail__source mono">
          {lo > 24 && "…"}
          {before.replace(/\n/g, "⏎")}
          <mark>{inside.replace(/\n/g, "⏎")}</mark>
          {after.replace(/\n/g, "⏎")}
          {hi + 24 < text.length && "…"}
        </p>
      ) : (
        <p className="token-detail__source muted">Covers no characters of your text — inserted by the tokenizer.</p>
      )}
      <dl className="token-detail__facts">
        {facts.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
