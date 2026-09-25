import type { TokenizerInfo } from "../api";
import { ALGORITHM_COPY } from "../copy";
import { useTokLab } from "../store";

function optionLabel(t: TokenizerInfo): string {
  return `${t.label} · ${ALGORITHM_COPY[t.algorithm].name}`;
}

/** Single-choice tokenizer select, with the chosen one's one-line story under it. */
export function TokenizerSelect({ id = "tokenizer-picker" }: { id?: string }) {
  const tokenizers = useTokLab((s) => s.tokenizers);
  const tokenizer = useTokLab((s) => s.tokenizer);
  const setTokenizer = useTokLab((s) => s.setTokenizer);
  const current = tokenizers.find((t) => t.id === tokenizer);

  return (
    <div className="field">
      <label className="label" htmlFor={id}>
        Tokenizer
      </label>
      {tokenizers.length === 0 ? (
        <div className="skeleton" style={{ height: "var(--control-h)" }} />
      ) : (
        <select id={id} className="select" value={tokenizer} onChange={(e) => setTokenizer(e.target.value)}>
          {tokenizers.map((t) => (
            <option key={t.id} value={t.id}>
              {optionLabel(t)}
            </option>
          ))}
        </select>
      )}
      {current && <TokenizerBlurb t={current} />}
    </div>
  );
}

export function TokenizerBlurb({ t }: { t: TokenizerInfo }) {
  return (
    <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
      {t.blurb}{" "}
      <span className="nowrap">
        · {t.year} · <span className="mono">{t.hf_name}</span>
        {t.source === "port" && <> · community conversion</>}
      </span>
    </p>
  );
}

/** Multi-choice, for the views that compare tokenizers. Order is click order. */
export function TokenizerPills({ max = 6 }: { max?: number }) {
  const tokenizers = useTokLab((s) => s.tokenizers);
  const compare = useTokLab((s) => s.compare);
  const toggle = useTokLab((s) => s.toggleCompare);

  return (
    <div className="field">
      <span className="label">
        Tokenizers
        <span className="label__hint">
          {compare.length} of up to {max} · click to add or remove
        </span>
      </span>
      <div className="pillrow" role="group" aria-label="Tokenizers to compare">
        {tokenizers.map((t) => {
          const on = compare.includes(t.id);
          return (
            <button
              key={t.id}
              type="button"
              className="pill pill--wide"
              aria-pressed={on}
              disabled={!on && compare.length >= max}
              title={`${optionLabel(t)} — ${t.blurb}`}
              onClick={() => toggle(t.id)}
            >
              {t.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
