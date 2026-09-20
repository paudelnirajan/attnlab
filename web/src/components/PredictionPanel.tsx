import { tokenLabel } from "../lib/tokens";
import { useStore } from "../state/store";

/**
 * What the model actually predicted at the linked position, plus the loss it
 * took there. /api/run already returns both (top_logits length seq,
 * loss_per_token length seq-1 — index i scores the prediction of token i+1),
 * and without showing them the attention plots float free of any outcome.
 */
export function PredictionPanel() {
  const runResult = useStore((s) => s.runResult);
  const hovered = useStore((s) => s.hoveredTokenIdx);
  const selected = useStore((s) => s.selectedTokenIdx);

  if (!runResult) return null;
  const tokens = runResult.tokens;
  const pos = hovered ?? selected ?? tokens.length - 1;
  const preds = runResult.top_logits[pos];
  if (!preds || preds.length === 0) return null;

  const actual = tokens[pos + 1];
  const loss = pos < runResult.loss_per_token.length ? runResult.loss_per_token[pos] : null;
  const maxProb = preds[0]?.prob ?? 1;

  return (
    <div className="card">
      <div className="card__head">
        <h2 className="card__title">Prediction</h2>
        <span className="card__hint">
          after position {pos} · <span className="mono">{tokenLabel(tokens[pos])}</span>
        </span>
      </div>
      <div className="card__body">
        <ol className="ranked">
          {preds.map((p) => {
            const hit = actual !== undefined && p.id === actual.id;
            return (
              <li key={p.id} className="ranked__row" title={`token id ${p.id} · logit ${p.logit.toFixed(2)}`}>
                <span className="ranked__bar" style={{ width: `${maxProb > 0 ? (p.prob / maxProb) * 100 : 0}%` }} />
                <span className="ranked__label">
                  {p.str.replace(/ /g, "·").replace(/\n/g, "⏎") || "∅"}
                  {hit && <span className="badge badge--good" style={{ marginLeft: 6 }}>actual</span>}
                </span>
                <span className="ranked__value">{(p.prob * 100).toFixed(1)}%</span>
              </li>
            );
          })}
        </ol>
        <p className="muted" style={{ fontSize: "var(--t-xs)", marginTop: "var(--s2)" }}>
          {actual ? (
            <>
              Next token was <span className="mono">{tokenLabel(actual)}</span>
              {loss !== null && (
                <>
                  {" "}
                  · loss <span className="mono">{loss.toFixed(3)}</span>
                </>
              )}
            </>
          ) : (
            <>Last position — nothing follows it to score against.</>
          )}
        </p>
      </div>
    </div>
  );
}
