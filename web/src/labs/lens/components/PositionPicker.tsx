import type { LensRunResponse } from "../api";

/**
 * The input tokens as a row of chips; clicking one picks the position whose
 * prediction the view is about. Shows what that position is predicting, since
 * "position i" always means "the guess for token i+1".
 */
export function PositionPicker({
  run,
  pos,
  onPick,
}: {
  run: LensRunResponse;
  pos: number;
  onPick: (pos: number) => void;
}) {
  const labels = run.input_labels;
  const next = run.next_labels[pos];
  return (
    <div className="pospick">
      <div className="chipstrip chipstrip--compact" role="listbox" aria-label="Position">
        {labels.map((l, i) => (
          <button
            key={i}
            type="button"
            role="option"
            aria-selected={i === pos}
            className={i === pos ? "chip chip--anchor" : "chip"}
            title={`position ${i} · predicts ${run.next_labels[i] ?? "the next token (not in the text)"}`}
            onClick={() => onPick(i)}
          >
            {l || "∅"}
          </button>
        ))}
      </div>
      <p className="pospick__caption">
        Position <span className="mono">{pos}</span>, after <span className="tok">{labels[pos]}</span>, predicting{" "}
        {next !== null ? (
          <>
            the actual next token <span className="tok">{next}</span>
          </>
        ) : (
          <>the token after the text, which we can't know</>
        )}
        .
      </p>
    </div>
  );
}
