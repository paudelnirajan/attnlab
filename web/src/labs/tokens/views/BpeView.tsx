import { useEffect, useState } from "react";
import { toklab, type TraceResult, type TraceWord } from "../api";
import { TokenizerSelect } from "../components/TokenizerPicker";
import { ALGORITHM_COPY } from "../copy";
import { useRemote } from "../hooks";
import { useTokLab } from "../store";

const PLAY_MS = 650;

function Symbols({ symbols, highlight }: { symbols: string[]; highlight: number[] }) {
  return (
    <div className="symrow" aria-label="Current symbols">
      {symbols.map((s, i) => (
        <span key={i} className={highlight.includes(i) ? "sym sym--new" : "sym"}>
          {s}
        </span>
      ))}
    </div>
  );
}

function WordTrace({ word, algorithm, nMerges }: { word: TraceWord; algorithm: TraceResult["algorithm"]; nMerges: number }) {
  // step 0 = before anything happened; step k = after steps[k-1]
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const n = word.steps.length;

  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [word]);

  useEffect(() => {
    if (!playing) return;
    if (step >= n) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setStep((s) => s + 1), PLAY_MS);
    return () => clearTimeout(t);
  }, [playing, step, n]);

  const current = step === 0 ? null : word.steps[step - 1];
  const symbols = current ? current.symbols : word.initial;
  const wordpiece = algorithm === "wordpiece";

  if (algorithm === "unigram") {
    return (
      <div className="stack">
        <p className="notice">{word.notes[0]}</p>
        <div className="symrow">
          {word.actual.map((p, i) => (
            <span key={i} className="sym" title={`log-prob ${word.scores?.[i]?.toFixed(3)}`}>
              {p}
              <small className="sym__score">{word.scores?.[i]?.toFixed(2)}</small>
            </span>
          ))}
        </div>
        <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
          Numbers are each piece's log-probability; the chosen segmentation has the highest total. A longer piece is one
          term instead of several negative ones, which is why common words stay whole.
        </p>
      </div>
    );
  }

  return (
    <div className="stack bpe">
      <div className="bpe__controls">
        <button type="button" className="btn btn--sm" onClick={() => setStep(0)} disabled={step === 0}>
          ⏮ start
        </button>
        <button type="button" className="btn btn--sm" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
          ← back
        </button>
        <button
          type="button"
          className="btn btn--sm btn--primary"
          onClick={() => {
            if (step >= n) setStep(0);
            setPlaying((p) => !p);
          }}
          disabled={n === 0}
        >
          {playing ? "pause" : step >= n ? "replay" : "play"}
        </button>
        <button type="button" className="btn btn--sm" onClick={() => setStep((s) => Math.min(n, s + 1))} disabled={step >= n}>
          next →
        </button>
        <input
          type="range"
          className="bpe__slider"
          min={0}
          max={n}
          value={step}
          onChange={(e) => {
            setPlaying(false);
            setStep(Number(e.target.value));
          }}
          aria-label="Step"
        />
        <span className="mono muted">
          step {step} / {n}
        </span>
      </div>

      <p className="bpe__caption">
        {step === 0 ? (
          wordpiece ? (
            <>Start with the cursor at the beginning of the word.</>
          ) : (
            <>
              Start: every {algorithm === "byte-bpe" ? "byte" : "character"} is its own symbol ({word.initial.length}{" "}
              symbols). Hex in ‹ › is a byte that isn't a whole character.
            </>
          )
        ) : wordpiece ? (
          <>
            Longest vocabulary entry at the cursor: <strong className="mono">{current!.merged}</strong>
            {current!.tried ? <> — after trying {current!.tried} longer candidates that aren't in the vocabulary</> : null}
          </>
        ) : (
          <>
            Merge <strong className="mono">#{current!.rank!.toLocaleString()}</strong>
            <span className="muted"> of {nMerges.toLocaleString()}</span>:{" "}
            <span className="mono sym sym--inline">{current!.left}</span> +{" "}
            <span className="mono sym sym--inline">{current!.right}</span> →{" "}
            <span className="mono sym sym--inline sym--new">{current!.merged}</span>
            {current!.at.length > 1 && <> (applied at {current!.at.length} places)</>}
          </>
        )}
      </p>

      <Symbols symbols={symbols} highlight={current?.at ?? []} />

      {n > 0 && (
        <ol className="bpe__steps" aria-label="All steps">
          {word.steps.map((s, i) => (
            <li key={i}>
              <button
                type="button"
                className="bpe__step"
                aria-current={step === i + 1 ? "step" : undefined}
                onClick={() => {
                  setPlaying(false);
                  setStep(i + 1);
                }}
              >
                {s.rank !== null ? <span className="mono muted">#{s.rank}</span> : <span className="mono muted">{i + 1}</span>}{" "}
                <span className="mono">{wordpiece ? s.merged : `${s.left} + ${s.right}`}</span>
              </button>
            </li>
          ))}
        </ol>
      )}

      {step >= n && (
        <p className={word.verified ? "verdict verdict--ok" : "verdict verdict--warn"}>
          <span aria-hidden="true">{word.verified ? "✓" : "!"}</span>
          <span>
            {word.verified ? (
              <>
                <strong>Matches the real tokenizer:</strong>{" "}
              </>
            ) : (
              <>
                <strong>This replay differs from the real tokenizer</strong>, which produced{" "}
                <span className="mono">{word.actual.join(" ")}</span>. Final replay:{" "}
              </>
            )}
            {word.final.map((p, i) => (
              <span key={i} className="mono sym sym--inline">
                {p}
              </span>
            ))}
            {word.final_ids.some((x) => x !== null) && (
              <span className="muted mono"> = [{word.final_ids.map((x) => x ?? "?").join(", ")}]</span>
            )}
          </span>
        </p>
      )}
      {word.notes.map((note) => (
        <p key={note} className="muted" style={{ fontSize: "var(--t-sm)" }}>
          Note: {note}.
        </p>
      ))}
    </div>
  );
}

/**
 * Replays the tokenizer's own algorithm on a word, one decision at a time.
 * The backend checks every replay against the real (Rust) tokenizer, and the
 * view says so — this is a claim about how the tokenizer works, so it has to
 * be true for the word on screen, not just for the examples we picked.
 */
export function BpeView() {
  const tokenizer = useTokLab((s) => s.tokenizer);
  const tokenizers = useTokLab((s) => s.tokenizers);
  const word = useTokLab((s) => s.word);
  const setWord = useTokLab((s) => s.setWord);
  const [pick, setPick] = useState(0);
  const info = tokenizers.find((t) => t.id === tokenizer);

  const remote = useRemote<TraceResult>(JSON.stringify([tokenizer, word]), () => toklab.trace(tokenizer, word), {
    enabled: tokenizers.length > 0 && word.length > 0,
  });
  const trace = remote.data?.tokenizer === tokenizer ? remote.data : null;
  useEffect(() => setPick(0), [trace]);
  const chosen = trace?.words[Math.min(pick, (trace?.words.length ?? 1) - 1)];

  return (
    <div className="stack">
      <div className="setup">
        <TokenizerSelect id="bpe-tokenizer" />
        <div className="field">
          <label className="label" htmlFor="bpe-word">
            Word or short phrase
            <span className="label__hint">a leading space matters — try with and without</span>
          </label>
          <input
            id="bpe-word"
            className="input"
            value={word}
            onChange={(e) => setWord(e.target.value)}
            spellCheck={false}
          />
          <div className="pillrow">
            {[" unbelievably", " tokenization", "unbelievably", " नेपाल", " 1234567", " SolidGoldMagikarp", " Straße"].map((w) => (
              <button key={w} type="button" className="pill" aria-pressed={word === w} onClick={() => setWord(w)}>
                {w.replace(/^ /, "·")}
              </button>
            ))}
          </div>
        </div>
      </div>

      {info && (
        <section className="card">
          <div className="card__head">
            <h2 className="card__title">How {info.label} tokenizes: {ALGORITHM_COPY[info.algorithm].name}</h2>
          </div>
          <div className="card__body">
            <p>{ALGORITHM_COPY[info.algorithm].how}</p>
            {(info.algorithm === "byte-bpe" || info.algorithm === "sp-bpe") && (
              <p className="muted" style={{ fontSize: "var(--t-sm)", marginTop: "var(--s2)" }}>
                A merge's <strong>rank</strong> is when training learned it. Training merges the most frequent pair
                first, so a low rank means a very common pair and a high rank a rare one. At every step the replay
                applies the lowest-ranked merge available.
              </p>
            )}
          </div>
        </section>
      )}

      {remote.error && (
        <div className="banner banner--error" role="alert">
          <span>{remote.error}</span>
        </div>
      )}

      {trace && trace.words.length > 1 && (
        <div className="field">
          <span className="label">
            Pre-tokens
            <span className="label__hint">merges never cross these boundaries, so each one is traced on its own</span>
          </span>
          <div className="pillrow" role="group" aria-label="Pre-token to trace">
            {trace.words.map((w, i) => (
              <button key={i} type="button" className="pill" aria-pressed={i === pick} onClick={() => setPick(i)}>
                {w.display}
              </button>
            ))}
          </div>
        </div>
      )}

      {trace && chosen && (
        <section className={remote.loading ? "card is-stale" : "card"}>
          <div className="card__head">
            <h2 className="card__title">
              Tracing <span className="mono">{chosen.display}</span>
            </h2>
            <span className="card__hint">
              {chosen.steps.length} {trace.algorithm === "wordpiece" ? "matches" : "merges"} →{" "}
              {chosen.actual.length} token{chosen.actual.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="card__body">
            <WordTrace word={chosen} algorithm={trace.algorithm} nMerges={trace.n_merges} />
          </div>
        </section>
      )}
    </div>
  );
}
