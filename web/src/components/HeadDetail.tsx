import { useEffect, useMemo, useRef, useState } from "react";
import { attentionVector, DIRECTION_COPY, maxOf, topK } from "../lib/attention";
import { useLinkedWeights } from "../hooks/useLinkedWeights";
import { canvasEventToCell, renderHeadToCanvas } from "../render/heatmap";
import { tokenFragmentNote, tokenLabel } from "../lib/tokens";
import { useStore } from "../state/store";
import { Colorbar } from "./Colorbar";

const TOP_K = 6;
/** Above this many tokens, per-token axis labels stop being legible and the
 * axes fall back to position numbers at intervals. */
const LABEL_LIMIT = 28;

interface HoverCell {
  dest: number;
  src: number;
  x: number;
  y: number;
}

function truncate(s: string, n = 10): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

export function HeadDetail() {
  const runResult = useStore((s) => s.runResult);
  const selectedLayer = useStore((s) => s.selectedLayer);
  const selectedHead = useStore((s) => s.selectedHead);
  const setSelectedHead = useStore((s) => s.setSelectedHead);
  const setHoveredToken = useStore((s) => s.setHoveredToken);
  const setSelectedToken = useStore((s) => s.setSelectedToken);
  const decoded = useStore((s) => s.patternsByLayer.get(s.selectedLayer));
  const direction = useStore((s) => s.direction);
  const theme = useStore((s) => s.theme);
  const linked = useLinkedWeights();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverCell, setHoverCell] = useState<HoverCell | null>(null);

  useEffect(() => {
    if (!decoded || selectedHead === null || !canvasRef.current) return;
    renderHeadToCanvas(canvasRef.current, decoded, 0, selectedHead, theme);
  }, [decoded, selectedHead, theme]);

  // Clear any stale crosshair when the head, layer or run changes underneath.
  useEffect(() => setHoverCell(null), [selectedHead, selectedLayer, decoded]);

  const copy = DIRECTION_COPY[direction];
  const seq = decoded?.seq ?? 0;

  // With nothing hovered or pinned, fall back to the last position — "what does
  // the token the model is about to predict from look at?" is the question
  // people actually arrive with.
  const fallbackAnchor = seq > 0 ? seq - 1 : null;
  const anchor = linked?.anchor ?? fallbackAnchor;

  const vector = useMemo(() => {
    if (linked) return linked.vector;
    if (!decoded || selectedHead === null || anchor === null) return null;
    return attentionVector(decoded, 0, selectedHead, anchor, direction);
  }, [linked, decoded, selectedHead, anchor, direction]);

  const ranked = useMemo(() => (vector ? topK(vector, TOP_K) : []), [vector]);
  const vecMax = useMemo(() => (vector ? maxOf(vector) : 0), [vector]);

  if (selectedHead === null || !runResult) return null;
  const tokens = runResult.tokens;

  const pct = (n: number) => `${(n / seq) * 100}%`;
  const thickness = `max(2px, ${(1 / seq) * 100}%)`;
  const showLabels = seq > 0 && seq <= LABEL_LIMIT;
  // Only emit the ticks that carry a label. Rendering one span per position
  // meant 512 empty spans per axis on a long prompt, for nothing.
  const tickStep = showLabels ? 1 : Math.max(1, Math.ceil(seq / 12));
  const tickPositions: number[] = [];
  for (let i = 0; i < seq; i += tickStep) tickPositions.push(i);

  return (
    <div className="card">
      <div className="card__head">
        <h2 className="card__title">
          Layer {selectedLayer} · Head {selectedHead}
        </h2>
        <span className="spacer" />
        <Colorbar />
        <button type="button" className="btn btn--ghost btn--icon" onClick={() => setSelectedHead(null)} aria-label="Close head detail">
          ✕
        </button>
      </div>

      <div className="card__body">
        {!decoded ? (
          <div className="skeleton" style={{ aspectRatio: "1" }} />
        ) : (
          <div className="heat">
            <div className="heat__corner" />
            <div className="heat__xtitle">source (attended to) →</div>
            <div className="heat__xticks" style={{ gridTemplateColumns: `repeat(${seq}, minmax(0, 1fr))` }}>
              {tickPositions.map((i) => (
                <span
                  key={i}
                  className={`heat__xtick${hoverCell?.src === i || (direction === "src2dest" && anchor === i) ? " heat__tick--on" : ""}`}
                  style={{ gridColumn: i + 1 }}
                >
                  {showLabels ? truncate(tokens[i]?.display ?? "") : i}
                </span>
              ))}
            </div>

            <div className="heat__ytitle">← destination (attending)</div>
            <div className="heat__yticks" style={{ width: showLabels ? 86 : 30 }}>
              {tickPositions.map((i) => (
                <span
                  key={i}
                  className={`heat__ytick${hoverCell?.dest === i || (direction === "dest2src" && anchor === i) ? " heat__tick--on" : ""}`}
                  // centre of row i, as a fraction of the plot height
                  style={{ top: `${((i + 0.5) / seq) * 100}%` }}
                >
                  {showLabels ? truncate(tokens[i]?.display ?? "") : i}
                </span>
              ))}
            </div>

            <div className="heat__plot heat__plot--framed">
              <canvas
                ref={canvasRef}
                className="pixel-canvas heat__canvas"
                onMouseMove={(e) => {
                  if (!canvasRef.current) return;
                  const cell = canvasEventToCell(e, canvasRef.current, seq);
                  const rect = canvasRef.current.getBoundingClientRect();
                  setHoverCell({ ...cell, x: e.clientX - rect.left, y: e.clientY - rect.top });
                  // Drive the shared hover link from the canvas too, so moving
                  // over the matrix lights up the token strip and the other
                  // tiles — whichever axis the current direction reads from.
                  setHoveredToken(direction === "dest2src" ? cell.dest : cell.src);
                }}
                onMouseLeave={() => {
                  setHoverCell(null);
                  setHoveredToken(null);
                }}
                onClick={(e) => {
                  if (!canvasRef.current) return;
                  const cell = canvasEventToCell(e, canvasRef.current, seq);
                  setSelectedToken(direction === "dest2src" ? cell.dest : cell.src);
                }}
              />

              {/* the persistent linked band (row or column, by direction) */}
              {anchor !== null &&
                (direction === "dest2src" ? (
                  <div
                    className="heat__band"
                    style={{ left: 0, width: pct(anchor + 1), top: pct(anchor), height: thickness }}
                  />
                ) : (
                  <div
                    className="heat__band"
                    style={{ left: pct(anchor), width: thickness, top: pct(anchor), height: pct(seq - anchor) }}
                  />
                ))}

              {/* the transient crosshair under the cursor */}
              {hoverCell && (
                <>
                  <div className="heat__rule heat__rule--h" style={{ top: pct(hoverCell.dest) }} />
                  <div className="heat__rule heat__rule--v" style={{ left: pct(hoverCell.src) }} />
                </>
              )}

              {hoverCell && tokens[hoverCell.dest] && tokens[hoverCell.src] && (
                <div
                  className="tooltip"
                  style={{
                    // flip to the other side of the cursor near the right/bottom
                    // edge so the tooltip is never clipped by the plot box
                    left: hoverCell.x > 220 ? undefined : hoverCell.x + 14,
                    right: hoverCell.x > 220 ? 8 : undefined,
                    top: hoverCell.y > 120 ? undefined : hoverCell.y + 14,
                    bottom: hoverCell.y > 120 ? 8 : undefined,
                  }}
                >
                  {hoverCell.src > hoverCell.dest ? (
                    <span className="muted">masked — a token can't attend to a later position</span>
                  ) : (
                    <>
                      <div>
                        <span className="muted">dest {hoverCell.dest}</span>{" "}
                        <strong className="mono">{tokenLabel(tokens[hoverCell.dest])}</strong>
                      </div>
                      <div>
                        <span className="muted">src {hoverCell.src}</span>{" "}
                        <strong className="mono">{tokenLabel(tokens[hoverCell.src])}</strong>
                      </div>
                      <div className="tooltip__val">
                        {decoded.at(0, selectedHead, hoverCell.dest, hoverCell.src).toFixed(4)}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="card__body" style={{ paddingTop: 0 }}>
        <p className="secondary" style={{ fontSize: "var(--t-sm)", marginBottom: "var(--s2)" }}>
          {anchor !== null && tokens[anchor] ? (
            <>
              Top {copy.otherRole}s for <strong className="mono">{tokenLabel(tokens[anchor])}</strong> at position{" "}
              {anchor}
              {tokenFragmentNote(tokens[anchor]) && (
                <span className="muted"> (byte {tokenFragmentNote(tokens[anchor])})</span>
              )}
            </>
          ) : (
            <>Hover the matrix or a token</>
          )}
        </p>
        {ranked.length === 0 ? (
          <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
            Nothing to show — position {anchor} has no {copy.otherRole} under causal masking.
          </p>
        ) : (
          <ol className="ranked">
            {ranked.map(({ idx, value }) => (
              <li
                key={idx}
                className="ranked__row"
                onMouseEnter={() => setHoveredToken(idx)}
                onMouseLeave={() => setHoveredToken(null)}
              >
                <span className="ranked__bar" style={{ width: `${vecMax > 0 ? (value / vecMax) * 100 : 0}%` }} />
                <span className="ranked__label">
                  {tokenLabel(tokens[idx])}{" "}
                  <span className="ranked__pos">
                    #{idx}
                    {tokenFragmentNote(tokens[idx]) ? ` · byte ${tokenFragmentNote(tokens[idx])}` : ""}
                  </span>
                </span>
                <span className="ranked__value">{value.toFixed(4)}</span>
              </li>
            ))}
          </ol>
        )}
        <p className="muted" style={{ fontSize: "var(--t-xs)", marginTop: "var(--s2)" }}>
          {copy.note}
        </p>
      </div>
    </div>
  );
}
