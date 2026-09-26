import { memo, useMemo, useState } from "react";
import { useStore } from "../../../state/store";
import type { LensRunResponse, LensSummary } from "../api";
import { GLOSSARY, LENS_COPY, METRIC_COPY, lensLabel } from "../copy";
import { cellShade, cellValue, fmtNum, fmtProb, fmtRank, shadeStyle } from "../format";
import { METRICS, resolvedPos, useLens, visibleRows, type Metric } from "../store";
import type { Theme } from "../../../theme";

export function LensControls({ run }: { run: LensRunResponse }) {
  const lens = useLens((s) => s.lens);
  const setLens = useLens((s) => s.setLens);
  const rowMode = useLens((s) => s.rowMode);
  const setRowMode = useLens((s) => s.setRowMode);
  return (
    <div className="toolbar">
      <span className="toolbar__label">Lens</span>
      <div className="segmented" role="group" aria-label="Lens">
        {run.lenses.map((l) => (
          <button
            key={l}
            type="button"
            className="segmented__opt"
            aria-pressed={lens === l}
            title={LENS_COPY[l].short}
            onClick={() => setLens(l)}
          >
            {lensLabel(l, !!run.anatomy.raw?.norm)}
          </button>
        ))}
      </div>
      {run.lenses.length === 1 && (
        <span className="muted" style={{ fontSize: "var(--t-sm)" }}>
          this model has no final norm, so there is only one lens
        </span>
      )}
      <span className="toolbar__label">Rows</span>
      <div className="segmented" role="group" aria-label="Rows">
        <button type="button" className="segmented__opt" aria-pressed={rowMode === "blocks"} onClick={() => setRowMode("blocks")}>
          blocks
        </button>
        <button
          type="button"
          className="segmented__opt"
          aria-pressed={rowMode === "sub"}
          title="also read the stream after each attention sub-layer, so you can see whether attention or the MLP moved it"
          onClick={() => setRowMode("sub")}
        >
          attn + mlp
        </button>
      </div>
    </div>
  );
}

interface CellProps {
  summary: LensSummary;
  r: number;
  c: number;
  metric: Metric;
  dVocab: number;
  theme: Theme;
  nextId: number | null;
  selected: boolean;
  isOutput: boolean;
}

const Cell = memo(function Cell({ summary, r, c, metric, dVocab, theme, nextId, selected, isOutput }: CellProps) {
  const cells = summary.cells;
  const top = summary.strings[cells.top[r][c][0]];
  const hit = nextId !== null && top.id === nextId;
  const style = shadeStyle(cellShade(metric, cells, r, c, dVocab), theme);
  const cls = ["lgrid__cell"];
  if (selected) cls.push("is-selected");
  if (isOutput) cls.push("lgrid__cell--output");
  return (
    <button type="button" className={cls.join(" ")} style={style} data-r={r} data-c={c}>
      <span className="lgrid__tok">{top.label || "∅"}</span>
      <span className="lgrid__val">
        {cellValue(metric, cells, r, c)}
        {hit && <span className="lgrid__hit" aria-label="top guess is the actual next token"> ✓</span>}
      </span>
    </button>
  );
});

function CellTip({ summary, r, c }: { summary: LensSummary; r: number; c: number }) {
  const cells = summary.cells;
  const row = summary.rows[r];
  return (
    <div className="lgrid__tip tooltip" role="status">
      <strong>
        {row.label} · position {c}
      </strong>
      <div className="muted">
        after <span className="tok">{summary.input_labels[c]}</span>
        {summary.next_labels[c] !== null && (
          <>
            {" "}
            → actual next <span className="tok">{summary.next_labels[c]}</span>
          </>
        )}
      </div>
      <ol className="lgrid__tiplist">
        {cells.top[r][c].map((sid, k) => (
          <li key={k}>
            <span className="tok">{summary.strings[sid].label || "∅"}</span>
            <span className="tooltip__val">{fmtProb(cells.top_p[r][c][k])}</span>
          </li>
        ))}
      </ol>
      <div className="lgrid__tipfacts">
        <span>P(next) {fmtProb(cells.p_next[r][c])}</span>
        <span>rank(next) {fmtRank(cells.rank_next[r][c])}</span>
        <span>rank(final) {fmtRank(cells.rank_final[r][c])}</span>
        <span>H {fmtNum(cells.entropy[r][c], 2)}</span>
        <span>KL {fmtNum(cells.kl[r][c], 2)}</span>
      </div>
    </div>
  );
}

export function GridView({ run, summary }: { run: LensRunResponse; summary: LensSummary }) {
  const metric = useLens((s) => s.metric);
  const setMetric = useLens((s) => s.setMetric);
  const rowMode = useLens((s) => s.rowMode);
  const lens = useLens((s) => s.lens);
  const pos = useLens((s) => s.pos);
  const row = useLens((s) => s.row);
  const select = useLens((s) => s.select);
  const setView = useLens((s) => s.setView);
  const addTrack = useLens((s) => s.addTrack);
  const setTarget = useLens((s) => s.setTarget);
  const theme = useStore((s) => s.theme);
  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null);

  const rows = useMemo(() => visibleRows(summary.rows, rowMode).reverse(), [summary.rows, rowMode]); // output on top
  const seq = summary.input_labels.length;
  const nextIds = useMemo(() => run.tokens.map((_, i) => (i + 1 < seq ? run.tokens[i + 1].id : null)), [run.tokens, seq]);
  const dVocab = run.anatomy.d_vocab;
  const selPos = resolvedPos(pos, run);
  const selRow = row ?? summary.rows.length - 1;

  function cellFrom(e: React.MouseEvent): { r: number; c: number } | null {
    const el = (e.target as HTMLElement).closest<HTMLElement>("[data-r]");
    return el ? { r: Number(el.dataset.r), c: Number(el.dataset.c) } : null;
  }

  return (
    <div className="stack">
      <div className="toolbar">
        <span className="toolbar__label">Colour by</span>
        <div className="segmented segmented--wrap" role="group" aria-label="Metric">
          {METRICS.map((m) => (
            <button
              key={m}
              type="button"
              className="segmented__opt"
              aria-pressed={metric === m}
              title={METRIC_COPY[m].question}
              onClick={() => setMetric(m)}
            >
              {METRIC_COPY[m].label}
            </button>
          ))}
        </div>
      </div>

      <section className="card">
        <div className="card__head">
          <h2 className="card__title">
            {METRIC_COPY[metric].question}
          </h2>
          <span className="card__hint">
            {lensLabel(lens, !!run.anatomy.raw?.norm)} lens · cell = top-1 guess and {METRIC_COPY[metric].value} ·{" "}
            {METRIC_COPY[metric].color}
          </span>
        </div>
        <div className="card__body card__body--tight">
          <div
            className="lgrid"
            style={{ gridTemplateColumns: `var(--lgrid-label) repeat(${seq}, var(--lgrid-col))` }}
            onMouseMove={(e) => {
              const hit = cellFrom(e);
              setHover(hit ? { ...hit, x: e.clientX, y: e.clientY } : null);
            }}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const hit = cellFrom(e);
              if (hit) select(hit.c, hit.r);
            }}
            role="grid"
            aria-label="Logit lens: rows are layers, columns are positions"
          >
            <div className="lgrid__corner lgrid__axis">actual next →</div>
            {summary.next_labels.map((l, c) => (
              <div key={c} className={c === selPos ? "lgrid__head is-on" : "lgrid__head"} title={`token ${c + 1}`}>
                {l ?? "???"}
              </div>
            ))}
            {rows.map((r) => {
              const info = summary.rows[r];
              const isOutput = info.kind === "output";
              return [
                <div
                  key={`l${r}`}
                  className={[
                    "lgrid__rowlabel",
                    isOutput ? "lgrid__rowlabel--output" : "",
                    info.block_end && rowMode === "sub" && !isOutput ? "lgrid__rowlabel--block" : "",
                  ].join(" ")}
                  title={isOutput ? GLOSSARY.output : `${info.id}${info.real ? "" : " — computed, the model never forms this state"}`}
                >
                  {info.label}
                  {!info.real && "*"}
                </div>,
                ...Array.from({ length: seq }, (_, c) => (
                  <Cell
                    key={`${r}-${c}`}
                    summary={summary}
                    r={r}
                    c={c}
                    metric={metric}
                    dVocab={dVocab}
                    theme={theme}
                    nextId={nextIds[c]}
                    selected={r === selRow && c === selPos}
                    isOutput={isOutput}
                  />
                )),
              ];
            })}
            <div className="lgrid__corner lgrid__axis">input →</div>
            {summary.input_labels.map((l, c) => (
              <div key={c} className={c === selPos ? "lgrid__foot is-on" : "lgrid__foot"} title={`position ${c}`}>
                {l || "∅"}
              </div>
            ))}
          </div>
          {hover && (
            <div className="lgrid__tipwrap" style={{ left: hover.x + 14, top: hover.y + 14 }}>
              <CellTip summary={summary} r={hover.r} c={hover.c} />
            </div>
          )}
          <p className="muted lgrid__note">
            {GLOSSARY.predicts} {GLOSSARY.output}
            {summary.rows.some((r) => !r.real) && " * = the stream before the MLP in a parallel block, which the model never forms."}
          </p>
        </div>
      </section>

      <CellDetail
        run={run}
        summary={summary}
        r={selRow}
        c={selPos}
        onFollow={() => setView("trajectory")}
        onAttribute={() => setView("attribution")}
        onTrack={(id, label) => addTrack({ id, label })}
        onTarget={(id, label) => {
          setTarget({ id, label });
          setView("attribution");
        }}
      />
    </div>
  );
}

function CellDetail({
  run,
  summary,
  r,
  c,
  onFollow,
  onAttribute,
  onTrack,
  onTarget,
}: {
  run: LensRunResponse;
  summary: LensSummary;
  r: number;
  c: number;
  onFollow: () => void;
  onAttribute: () => void;
  onTrack: (id: number, label: string) => void;
  onTarget: (id: number, label: string) => void;
}) {
  const cells = summary.cells;
  const row = summary.rows[r];
  if (!row) return null;
  const final = summary.strings[summary.final_top[c]];
  const next = summary.next_labels[c];
  return (
    <section className="card">
      <div className="card__head">
        <h2 className="card__title">
          {row.label}, position {c}
        </h2>
        <span className="card__hint mono">{row.kind === "output" ? "model logits" : row.id}</span>
        <span className="spacer" />
        <button type="button" className="linkbtn" onClick={onFollow}>
          Follow this position through the layers →
        </button>
        <button type="button" className="linkbtn" onClick={onAttribute}>
          Which components wrote it? →
        </button>
      </div>
      <div className="card__body grid2">
        <div>
          <p className="muted" style={{ marginTop: 0 }}>
            After <span className="tok">{summary.input_labels[c]}</span>, this row's top guesses (click one to track it):
          </p>
          <div className="ranked">
            {cells.top[r][c].map((sid, k) => {
              const s = summary.strings[sid];
              const p = cells.top_p[r][c][k];
              return (
                <button key={k} type="button" className="ranked__row ranked__row--button" onClick={() => onTrack(s.id, s.label)}>
                  <span className="ranked__bar" style={{ width: `${Math.max(1, p * 100)}%` }} />
                  <span className="ranked__label">
                    <span className="ranked__pos">{k + 1} </span>
                    {s.label || "∅"}
                  </span>
                  <span className="ranked__value">{fmtProb(p)}</span>
                </button>
              );
            })}
          </div>
        </div>
        <dl className="factlist">
          <dt>actual next</dt>
          <dd>
            {next !== null ? (
              <>
                <span className="tok">{next}</span> · P = {fmtProb(cells.p_next[r][c])} · rank {fmtRank(cells.rank_next[r][c])}
              </>
            ) : (
              "not in the text"
            )}
          </dd>
          <dt>model's final answer</dt>
          <dd>
            <button type="button" className="linkbtn" onClick={() => onTarget(final.id, final.label)} title="attribute this logit">
              <span className="tok">{final.label}</span>
            </button>{" "}
            · here P = {fmtProb(cells.p_final[r][c])} · rank {fmtRank(cells.rank_final[r][c])}
          </dd>
          <dt>entropy</dt>
          <dd>
            {fmtNum(cells.entropy[r][c], 2)} nats{" "}
            <span className="muted">(uniform over the vocabulary = {Math.log(run.anatomy.d_vocab).toFixed(2)})</span>
          </dd>
          <dt>KL(output ‖ this row)</dt>
          <dd>{fmtNum(cells.kl[r][c], 3)} nats</dd>
          {cells.norm[r][c] !== null && (
            <>
              <dt>‖residual‖</dt>
              <dd>{fmtNum(cells.norm[r][c], 1)}</dd>
            </>
          )}
        </dl>
      </div>
    </section>
  );
}
