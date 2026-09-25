import type { AnalyzeResult } from "../api";

function ratio(a: number, b: number, digits = 2): string {
  return b > 0 ? (a / b).toFixed(digits) : "—";
}

/**
 * The numbers that make tokenization concrete. Four different "lengths" of the
 * same text — what you see, code points, bytes, tokens — are shown side by
 * side on purpose: a learner who expects them to agree finds out, on their
 * own text, that they don't.
 */
export function StatsCard({ result, algorithm }: { result: AnalyzeResult; algorithm: string }) {
  const s = result.stats;
  const textTokens = s.n_tokens - s.n_special;
  const rows: [string, string, string][] = [
    ["Tokens", String(s.n_tokens), s.n_special ? `${s.n_special} of them special` : "positions the model will see"],
    ["Characters you see", String(s.n_graphemes), "grapheme clusters: what a reader counts"],
    ["Unicode code points", String(s.n_chars), "what a program counts as the string's length"],
    ["UTF-8 bytes", String(s.n_bytes), "what byte-level BPE starts from"],
    ["Tokens per word", ratio(textTokens, s.n_words), `${s.n_words} whitespace-separated words`],
    ["Bytes per token", ratio(s.n_bytes, textTokens), "higher = the vocabulary compresses this text better"],
  ];
  if (algorithm === "byte-bpe" || s.n_fragment_tokens > 0) {
    rows.push([
      "Fragment tokens",
      `${s.n_fragment_tokens} (${textTokens ? Math.round((100 * s.n_fragment_tokens) / textTokens) : 0}%)`,
      "tokens that are only part of a character",
    ]);
  }
  if (s.n_byte_tokens > 0 && algorithm === "sp-bpe") {
    rows.push(["Byte-fallback tokens", String(s.n_byte_tokens), "characters missing from the vocabulary, spelled as <0xNN> bytes"]);
  }
  if (s.n_unk > 0 || algorithm === "wordpiece" || algorithm === "unigram") {
    rows.push(["Unknown tokens", String(s.n_unk), "text this tokenizer cannot represent at all"]);
  }

  return (
    <section className="card">
      <div className="card__head">
        <h2 className="card__title">By the numbers</h2>
      </div>
      <div className="card__body">
        <dl className="statlist">
          {rows.map(([label, value, note]) => (
            <div key={label} style={{ display: "contents" }}>
              <dt>
                {label}
                <br />
                <span className="muted" style={{ fontSize: "var(--t-xs)" }}>
                  {note}
                </span>
              </dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>

        <p className={s.roundtrip ? "verdict verdict--ok" : "verdict verdict--warn"}>
          <span aria-hidden="true">{s.roundtrip ? "✓" : "!"}</span>
          {s.roundtrip ? (
            <span>
              <strong>Lossless.</strong> Decoding these tokens gives back your exact text.
            </span>
          ) : (
            <span>
              <strong>Not lossless.</strong> Decoding gives back{" "}
              <span className="mono">{JSON.stringify(result.decoded)}</span> — this tokenizer changed your text before
              splitting it (see the pipeline).
            </span>
          )}
        </p>
      </div>
    </section>
  );
}
