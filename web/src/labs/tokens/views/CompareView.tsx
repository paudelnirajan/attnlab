import { useState } from "react";
import { labHref } from "../../registry";
import { toklab, type AnalyzeResult } from "../api";
import { TokenStrip, ChipModeToggle } from "../components/TokenStrip";
import { TokenizerPills } from "../components/TokenizerPicker";
import { ALGORITHM_COPY } from "../copy";
import { useRemote } from "../hooks";
import { useTokLab } from "../store";
import { TextBox } from "./TextBox";

/**
 * The same text through several tokenizers, stacked. The count bars answer
 * "which is cheapest"; the strips underneath answer "why" — where each one
 * cuts, and what it had to fall back to.
 */
export function CompareView() {
  const compare = useTokLab((s) => s.compare);
  const tokenizers = useTokLab((s) => s.tokenizers);
  const text = useTokLab((s) => s.text);
  const chipMode = useTokLab((s) => s.chipMode);
  const setChipMode = useTokLab((s) => s.setChipMode);
  const inspect = useTokLab((s) => s.inspect);
  const [hovered, setHovered] = useState<{ tok: string; index: number } | null>(null);

  const remote = useRemote<AnalyzeResult[]>(
    JSON.stringify([compare, text]),
    () => toklab.analyze(compare, text, false).then((r) => r.results),
    { enabled: tokenizers.length > 0 && text.length > 0 },
  );
  const results = remote.data ?? [];
  const byId = new Map(tokenizers.map((t) => [t.id, t]));
  const max = Math.max(1, ...results.map((r) => r.stats.n_tokens));
  const fewest = Math.min(...results.map((r) => r.stats.n_tokens));

  return (
    <div className="stack">
      <div className="setup setup--wide">
        <TokenizerPills />
        <TextBox />
      </div>

      {remote.error && (
        <div className="banner banner--error" role="alert">
          <span>{remote.error}</span>
        </div>
      )}

      {results.length > 0 && (
        <div className={remote.loading ? "stack is-stale" : "stack"}>
          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Token count</h2>
              <span className="card__hint">same text, same meaning — the only thing that differs is the tokenizer</span>
            </div>
            <div className="card__body">
              <ol className="ranked">
                {[...results]
                  .sort((a, b) => a.stats.n_tokens - b.stats.n_tokens)
                  .map((r) => {
                    const t = byId.get(r.tokenizer);
                    return (
                      <li key={r.tokenizer} className="ranked__row" title={t?.blurb}>
                        <span className="ranked__bar" style={{ width: `${(100 * r.stats.n_tokens) / max}%` }} />
                        <span className="ranked__label ranked__label--sans">
                          {t?.label ?? r.tokenizer}
                          <span className="muted"> · {t ? ALGORITHM_COPY[t.algorithm].name : ""}</span>
                        </span>
                        <span className="ranked__value">
                          {r.stats.n_tokens}
                          <span className="muted">
                            {" "}
                            {r.stats.n_tokens === fewest ? "fewest" : `×${(r.stats.n_tokens / Math.max(1, fewest)).toFixed(2)}`}
                          </span>
                        </span>
                      </li>
                    );
                  })}
              </ol>
            </div>
          </section>

          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Where each one cuts</h2>
              <span className="spacer" />
              <ChipModeToggle mode={chipMode} onChange={setChipMode} />
            </div>
            <div className="card__body compare">
              {results.map((r) => {
                const t = byId.get(r.tokenizer);
                const h = hovered?.tok === r.tokenizer ? hovered.index : null;
                const s = r.stats;
                return (
                  <div key={r.tokenizer} className="compare__row">
                    <div className="compare__head">
                      <strong>{t?.label ?? r.tokenizer}</strong>
                      <span className="muted mono">
                        {s.n_tokens} tokens
                        {s.n_fragment_tokens > 0 && ` · ${s.n_fragment_tokens} fragments`}
                        {s.n_unk > 0 && ` · ${s.n_unk} unknown`}
                        {!s.roundtrip && " · not lossless"}
                      </span>
                      <span className="spacer" />
                      <button type="button" className="linkbtn" onClick={() => inspect(text, r.tokenizer)}>
                        inspect
                      </button>
                      {t?.models[0] && (
                        <a className="linkbtn" href={labHref("attention", { model: t.models[0], prompt: text })}>
                          attention lab →
                        </a>
                      )}
                    </div>
                    <TokenStrip
                      tokens={r.tokens}
                      mode={chipMode}
                      compact
                      label={`Tokens from ${t?.label ?? r.tokenizer}`}
                      hovered={h}
                      onHover={(index) => setHovered(index === null ? null : { tok: r.tokenizer, index })}
                    />
                    {!s.roundtrip && (
                      <p className="muted" style={{ fontSize: "var(--t-xs)" }}>
                        decodes to <span className="mono">{JSON.stringify(r.decoded)}</span>
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
