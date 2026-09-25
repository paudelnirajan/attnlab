import type { AnalyzeResult, Algorithm } from "../api";
import { ALGORITHM_COPY, COMPONENT_COPY } from "../copy";

function Components({ names, empty }: { names: string[]; empty: string }) {
  if (names.length === 0) return <p className="muted">{empty}</p>;
  return (
    <ul className="pipeline__list">
      {names.map((n, i) => (
        <li key={`${n}-${i}`}>
          <span className="mono pipeline__name">{n}</span> {COMPONENT_COPY[n] ?? ""}
        </li>
      ))}
    </ul>
  );
}

/**
 * The four stages every Hugging Face tokenizer runs, in order, with what each
 * one did to THIS text. Most "why did it split there?" questions are answered
 * by stage 2: BPE merges never cross a pre-token boundary.
 */
export function PipelineCard({ result, algorithm }: { result: AnalyzeResult; algorithm: Algorithm }) {
  const p = result.pipeline;
  const algo = ALGORITHM_COPY[algorithm];
  const ids = result.tokens.map((t) => t.id);

  return (
    <section className="card">
      <div className="card__head">
        <h2 className="card__title">The pipeline</h2>
        <span className="card__hint">what happened to your text, in order</span>
      </div>
      <ol className="pipeline">
        <li className="pipeline__stage">
          <h3 className="pipeline__title">
            <span className="pipeline__n">1</span> Normalize
          </h3>
          <Components names={p.normalizers} empty="No normalizer: the text goes through exactly as typed." />
          {p.normalizers.length > 0 &&
            (p.normalized_changed ? (
              <p className="pipeline__out">
                became <span className="mono pipeline__text">{p.normalized.replace(/ /g, "·")}</span>
              </p>
            ) : (
              <p className="muted">Your text was already normalized — nothing changed.</p>
            ))}
        </li>

        <li className="pipeline__stage">
          <h3 className="pipeline__title">
            <span className="pipeline__n">2</span> Pre-tokenize
            <span className="muted pipeline__count">
              {p.pretokens.length}
              {p.pretokens_truncated ? "+" : ""} chunks
            </span>
          </h3>
          <Components names={p.pre_tokenizers} empty="No pre-tokenizer: the whole text is one chunk." />
          <div className="pretokens" aria-label="Pre-tokens">
            {p.pretokens.map((pt, i) => (
              <span key={i} className="pretoken mono" title={`raw ${JSON.stringify(pt.raw)} · characters ${pt.start}–${pt.end}`}>
                {pt.display}
              </span>
            ))}
          </div>
          {p.split_patterns.length > 0 && (
            <details className="pipeline__details">
              <summary>The split regex</summary>
              {p.split_patterns.map((pat, i) => (
                <pre key={i} className="pipeline__regex">
                  {pat}
                </pre>
              ))}
            </details>
          )}
        </li>

        <li className="pipeline__stage">
          <h3 className="pipeline__title">
            <span className="pipeline__n">3</span> Model: {algo.name}
          </h3>
          <p>
            {algo.how}
            {p.n_merges > 0 && (
              <>
                {" "}
                This tokenizer learned <strong className="mono">{p.n_merges.toLocaleString()}</strong> merges.
              </>
            )}
          </p>
          <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
            Step through it on this text in the <strong>BPE step-through</strong> tab.
          </p>
        </li>

        <li className="pipeline__stage">
          <h3 className="pipeline__title">
            <span className="pipeline__n">4</span> IDs
            <span className="muted pipeline__count">{ids.length} integers</span>
          </h3>
          <p className="mono pipeline__ids">[{ids.join(", ")}]</p>
          <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
            This list is all the model ever receives. Each id selects one row of its embedding matrix.
          </p>
        </li>
      </ol>
    </section>
  );
}
