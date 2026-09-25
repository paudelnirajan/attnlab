import { labHref } from "../../registry";
import { toklab, type AnalyzeResult } from "../api";
import { ChipModeToggle, TokenStrip } from "../components/TokenStrip";
import { PipelineCard } from "../components/PipelineCard";
import { StatsCard } from "../components/StatsCard";
import { TokenDetail } from "../components/TokenDetail";
import { TokenizerSelect } from "../components/TokenizerPicker";
import { QUIRKS } from "../copy";
import { useRemote } from "../hooks";
import { useTokLab } from "../store";
import { TextBox } from "./TextBox";

/** Share of text tokens that are byte fragments above which the view says,
 * in words, what the learner is looking at. Same threshold as the attention lab. */
const FRAGMENT_NOTICE = 0.2;

export function InspectView() {
  const tokenizer = useTokLab((s) => s.tokenizer);
  const tokenizers = useTokLab((s) => s.tokenizers);
  const text = useTokLab((s) => s.text);
  const special = useTokLab((s) => s.special);
  const setSpecial = useTokLab((s) => s.setSpecial);
  const chipMode = useTokLab((s) => s.chipMode);
  const setChipMode = useTokLab((s) => s.setChipMode);
  const hovered = useTokLab((s) => s.hovered);
  const setHovered = useTokLab((s) => s.setHovered);
  const setText = useTokLab((s) => s.setText);
  const setView = useTokLab((s) => s.setView);

  const info = tokenizers.find((t) => t.id === tokenizer);
  const remote = useRemote<{ text: string; result: AnalyzeResult }>(
    JSON.stringify([tokenizer, text, special]),
    () => toklab.analyze([tokenizer], text, special).then((r) => ({ text, result: r.results[0] })),
    { enabled: tokenizers.length > 0 && text.length > 0 },
  );
  const quirk = QUIRKS.find((q) => q.text === text);
  const result = remote.data?.result.tokenizer === tokenizer ? remote.data.result : null;
  const s = result?.stats;
  const heavy = s && s.n_tokens > 0 && s.n_fragment_tokens / (s.n_tokens - s.n_special || 1) > FRAGMENT_NOTICE;

  return (
    <div className="stack">
      <section className="card">
        <div className="card__head">
          <h2 className="card__title">Try a quirk</h2>
          <span className="card__hint">each one is a real behaviour worth knowing — click to load it</span>
        </div>
        <div className="card__body">
          <div className="pillrow" role="group" aria-label="Examples">
            {QUIRKS.map((q) => (
              <button
                key={q.id}
                type="button"
                className="pill pill--wide pill--sans"
                aria-pressed={quirk?.id === q.id}
                onClick={() => setText(q.text)}
              >
                {q.title}
              </button>
            ))}
          </div>
          {quirk && (
            <p className="notice">
              <strong>What to notice.</strong> {quirk.notice}
              {quirk.also === "compare" && (
                <>
                  {" "}
                  <button type="button" className="linkbtn" onClick={() => setView("compare")}>
                    Compare it across tokenizers →
                  </button>
                </>
              )}
            </p>
          )}
        </div>
      </section>

      <div className="setup">
        <TokenizerSelect />
        <TextBox />
      </div>

      {remote.error && (
        <div className="banner banner--error" role="alert">
          <span>{remote.error}</span>
        </div>
      )}

      {!text ? (
        <div className="card">
          <div className="empty">Type something above to see how it's tokenized.</div>
        </div>
      ) : !result || !info ? (
        <div className="card">
          <div className="empty">
            <span className="spinner" style={{ display: "inline-block", verticalAlign: "-2px" }} /> tokenizing… (the
            first use of a tokenizer downloads it)
          </div>
        </div>
      ) : (
        <div className={remote.loading ? "stack is-stale" : "stack"}>
          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Tokens</h2>
              <span className="card__hint">
                {result.stats.n_tokens} tokens · {result.stats.n_graphemes} characters ·{" "}
                {(result.stats.n_tokens / Math.max(1, result.stats.n_graphemes)).toFixed(2)} tokens/char
              </span>
              <span className="spacer" />
              <label className="check">
                <input type="checkbox" checked={special} onChange={(e) => setSpecial(e.target.checked)} />
                add special tokens
              </label>
              <ChipModeToggle mode={chipMode} onChange={setChipMode} />
            </div>
            <div className="card__body card__body--tight">
              <TokenStrip tokens={result.tokens} mode={chipMode} hovered={hovered} onHover={setHovered} />
              <TokenDetail
                result={result}
                text={remote.data!.text}
                hovered={hovered}
                algorithm={info.algorithm}
                nMerges={result.pipeline.n_merges}
              />
              {heavy && (
                <p className="banner banner--info" style={{ marginTop: "var(--s3)", fontSize: "var(--t-sm)" }}>
                  <span>
                    <strong>
                      {result.stats.n_fragment_tokens} of {result.stats.n_tokens} tokens are byte fragments.
                    </strong>{" "}
                    This vocabulary has few merges for this script, so characters fall apart into raw UTF-8 bytes — the
                    underlined groups are several tokens spelling one character. A model reading this spends its early
                    layers reassembling the encoding before it can do anything with the language.
                  </span>
                </p>
              )}
              {special && (
                <p className="muted" style={{ fontSize: "var(--t-xs)", marginTop: "var(--s2)" }}>
                  Special tokens are added the Hugging Face way for this tokenizer. TransformerLens, which the
                  attention lab uses, prepends a BOS token for most models regardless.
                </p>
              )}
              {info.models.length > 0 && (
                <p className="handoff">
                  {info.models.map((m) => (
                    <a key={m} href={labHref("attention", { model: m, prompt: text })}>
                      See how <span className="mono">{m}</span> attends over exactly these tokens →
                    </a>
                  ))}
                </p>
              )}
            </div>
          </section>

          <div className="grid2">
            <StatsCard result={result} algorithm={info.algorithm} />
            <PipelineCard result={result} algorithm={info.algorithm} />
          </div>
        </div>
      )}
    </div>
  );
}
