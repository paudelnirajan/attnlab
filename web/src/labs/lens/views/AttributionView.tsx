import { useMemo, useState } from "react";
import { labHref } from "../../registry";
import { lensApi, type Component, type LensRunResponse, type LensSummary } from "../api";
import { LineChart } from "../components/LineChart";
import { PositionPicker } from "../components/PositionPicker";
import { fmtSigned } from "../format";
import { useRunQuery } from "../hooks";
import { resolvedPos, useLens, type TokenRef } from "../store";

function refLabel(r: TokenRef): string {
  return "id" in r ? r.label : JSON.stringify(r.str);
}

function asQuery(r: TokenRef | null): { id?: number; str?: string } | null {
  if (!r) return null;
  return "id" in r ? { id: r.id } : { str: r.str };
}

/** Diverging shade: blue for a push toward the target, red for a push against, gray at zero. */
function divStyle(v: number, max: number): React.CSSProperties {
  const t = Math.min(1, Math.abs(v) / (max || 1));
  const pole = v >= 0 ? "var(--div-pos)" : "var(--div-neg)";
  return { background: `color-mix(in oklab, ${pole} ${Math.round(t * 100)}%, var(--div-mid))`, color: t > 0.55 ? "#fff" : undefined };
}

function TokenPicker({
  label,
  value,
  options,
  onChange,
  allowNone,
}: {
  label: string;
  value: TokenRef | null;
  options: { key: string; label: string; ref: TokenRef | null }[];
  onChange: (r: TokenRef | null) => void;
  allowNone?: boolean;
}) {
  const [draft, setDraft] = useState("");
  const current = value ? refLabel(value) : null;
  return (
    <div className="trackbar">
      <span className="toolbar__label">{label}</span>
      <div className="segmented segmented--wrap" role="group" aria-label={label}>
        {allowNone && (
          <button type="button" className="segmented__opt" aria-pressed={value === null} onClick={() => onChange(null)}>
            none
          </button>
        )}
        {options.map((o) => (
          <button
            key={o.key}
            type="button"
            className="segmented__opt"
            aria-pressed={(o.ref === null && value === null && !allowNone) || (o.ref !== null && value !== null && refLabel(o.ref) === current)}
            onClick={() => onChange(o.ref)}
          >
            {o.label}
          </button>
        ))}
      </div>
      <form
        className="trackbar__add"
        onSubmit={(e) => {
          e.preventDefault();
          if (draft) onChange({ str: draft });
          setDraft("");
        }}
      >
        <input className="input" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="or type a token" aria-label={`${label} token`} />
        <button type="submit" className="btn btn--sm" disabled={!draft}>
          Use
        </button>
      </form>
    </div>
  );
}

export function AttributionView({ run, summary }: { run: LensRunResponse; summary: LensSummary }) {
  const posSel = useLens((s) => s.pos);
  const select = useLens((s) => s.select);
  const target = useLens((s) => s.target);
  const contrast = useLens((s) => s.contrast);
  const setTarget = useLens((s) => s.setTarget);
  const setContrast = useLens((s) => s.setContrast);
  const model = useLens((s) => s.model);
  const text = useLens((s) => s.text);
  const [picked, setPicked] = useState<string | null>(null);

  const pos = resolvedPos(posSel, run);
  const out = summary.cells.top.length - 1;
  const modelTop = summary.cells.top[out][pos].slice(0, 3).map((sid) => summary.strings[sid]);
  const nextLabel = summary.next_labels[pos];

  const remote = useRunQuery(
    JSON.stringify([run.run_id, pos, target, contrast]),
    () => lensApi.attribution(run.run_id, pos, asQuery(target) ?? {}, asQuery(contrast)),
  );
  const a = remote.data && remote.data.pos === pos && remote.data.run_id === run.run_id ? remote.data : null;

  const layers = run.anatomy.n_layers;
  const heads = run.anatomy.n_heads;
  const agg = useMemo(() => {
    if (!a) return null;
    const attn = Array(layers).fill(0);
    const mlp = Array(layers).fill(0);
    const headGrid: (Component | null)[][] = Array.from({ length: layers }, () => Array(heads).fill(null));
    let embed = 0;
    for (const c of a.components) {
      if (c.kind === "head") {
        attn[c.layer] += c.value;
        headGrid[c.layer][c.head!] = c;
      } else if (c.kind === "attn_bias") attn[c.layer] += c.value;
      else if (c.kind === "mlp") mlp[c.layer] += c.value;
      else embed += c.value;
    }
    const hasMlp = a.components.some((c) => c.kind === "mlp");
    // running total, in the order the stream is built: bias and embeddings, then each sub-layer
    const steps: { label: string; value: number }[] = [{ label: "embed + bias", value: embed + a.bias }];
    for (let l = 0; l < layers; l++) {
      steps.push({ label: `L${l} attn`, value: attn[l] });
      if (hasMlp) steps.push({ label: `L${l} mlp`, value: mlp[l] });
    }
    let acc = 0;
    const running = steps.map((s) => (acc += s.value));
    const maxBar = Math.max(...attn.map(Math.abs), ...mlp.map(Math.abs), Math.abs(embed), 1e-6);
    const maxHead = Math.max(...a.components.filter((c) => c.kind === "head").map((c) => Math.abs(c.value)), 1e-6);
    const ranked = [...a.components].sort((x, y) => Math.abs(y.value) - Math.abs(x.value));
    return { attn, mlp, embed, headGrid, hasMlp, steps, running, maxBar, maxHead, ranked };
  }, [a, layers, heads]);

  const detail = a && agg ? (a.components.find((c) => c.id === picked) ?? agg.ranked[0]) : null;
  const what = a ? (a.contrast ? `logit(${a.target.label}) − logit(${a.contrast.label})` : `logit(${a.target.label})`) : "";

  return (
    <div className="stack">
      <section className="card">
        <div className="card__head">
          <h2 className="card__title">What to explain</h2>
          <span className="card__hint">direct logit attribution always reads the model's real output, whatever lens is selected</span>
        </div>
        <div className="card__body">
          <PositionPicker run={run} pos={pos} onPick={(p) => select(p)} />
          <TokenPicker
            label="Target"
            value={target}
            onChange={setTarget}
            options={[
              { key: "default", label: nextLabel !== null ? `actual next (${nextLabel})` : `model's #1 (${modelTop[0].label})`, ref: null },
              ...modelTop.map((t, i) => ({ key: `m${i}`, label: `#${i + 1} ${t.label}`, ref: { id: t.id, label: t.label } as TokenRef })),
            ]}
          />
          <TokenPicker
            label="minus"
            value={contrast}
            onChange={setContrast}
            allowNone
            options={modelTop.map((t, i) => ({ key: `c${i}`, label: `#${i + 1} ${t.label}`, ref: { id: t.id, label: t.label } as TokenRef }))}
          />
          {a?.notes.map((n) => (
            <p key={n} className="banner banner--info" style={{ fontSize: "var(--t-sm)" }}>
              <span>{n}</span>
            </p>
          ))}
        </div>
      </section>

      {remote.error && (
        <div className="banner banner--error" role="alert">
          <span>{remote.error}</span>
        </div>
      )}

      {a && agg && (
        <div className={remote.loading ? "stack is-stale" : "stack"}>
          <section className="card">
            <div className="card__body">
              <p className="equation">
                <span className="mono">{what}</span> = <strong>{a.actual.toFixed(3)}</strong> ={" "}
                <span className="mono">Σ components</span> {(a.total - a.bias).toFixed(3)} +{" "}
                <span className="mono">bias</span> {a.bias.toFixed(3)}{" "}
                <span className={a.error < 1e-3 ? "okmark okmark--ok" : "okmark okmark--warn"}>
                  {a.error < 1e-3 ? "✓" : "!"} sums to the model's logit (error {a.error.toExponential(1)})
                </span>
              </p>
              <p className="muted" style={{ fontSize: "var(--t-sm)", marginBottom: 0 }}>
                Every component's output is divided by ln_final's scale at this position ({a.scale.toFixed(2)}), frozen
                at its real value, and projected on{" "}
                {a.contrast ? "the difference of the two unembedding columns" : "the target's unembedding column"}. That
                makes the split exact. It is not a claim about what would happen without a component: removing one
                changes the scale and everything that read it later. That question is ablation's.
              </p>
            </div>
          </section>

          <div className="grid2">
            <section className="card">
              <div className="card__head">
                <h2 className="card__title">By layer</h2>
                <span className="card__hint">blue pushes toward the target, red against · click a bar for details</span>
              </div>
              <div className="card__body">
                <div className="dla">
                  <DlaBar label="embed" value={agg.embed} max={agg.maxBar} kind="embed" onClick={() => setPicked("embed")} />
                  {agg.attn.map((v, l) => (
                    <div key={l} className="dla__layer">
                      <DlaBar label={`L${l} attn`} value={v} max={agg.maxBar} kind="attn" />
                      {agg.hasMlp && (
                        <DlaBar label={`L${l} mlp`} value={agg.mlp[l]} max={agg.maxBar} kind="mlp" onClick={() => setPicked(`L${l}.mlp`)} />
                      )}
                    </div>
                  ))}
                </div>
                <p className="muted" style={{ fontSize: "var(--t-xs)" }}>
                  A layer's attention bar is its heads plus its output bias b_O.
                </p>
              </div>
            </section>

            <section className="card">
              <div className="card__head">
                <h2 className="card__title">Running total</h2>
                <span className="card__hint">the target's logit as the stream is built, with the final scale held fixed</span>
              </div>
              <div className="card__body">
                <LineChart
                  ariaLabel="running total of the attribution"
                  xLabels={agg.steps.map((s) => s.label)}
                  series={[{ id: "run", label: what, values: agg.running, color: "var(--series-1)" }]}
                  yLabel={a.contrast ? "logit difference" : "logit"}
                  zeroLine={0}
                  format={(v) => fmtSigned(v, 1)}
                  height={260}
                />
              </div>
            </section>
          </div>

          <div className="grid2">
            <section className="card">
              <div className="card__head">
                <h2 className="card__title">Every head</h2>
                <span className="card__hint">rows = layers, columns = heads · click one</span>
              </div>
              <div className="card__body">
                <div className="headmap" style={{ gridTemplateColumns: `var(--headmap-label) repeat(${heads}, 1fr)` }}>
                  <span />
                  {Array.from({ length: heads }, (_, h) => (
                    <span key={h} className="headmap__col">
                      {h}
                    </span>
                  ))}
                  {agg.headGrid.map((row, l) => [
                    <span key={`l${l}`} className="headmap__row">
                      L{l}
                    </span>,
                    ...row.map((c, h) =>
                      c ? (
                        <button
                          key={`${l}-${h}`}
                          type="button"
                          className={detail?.id === c.id ? "headmap__cell is-selected" : "headmap__cell"}
                          style={divStyle(c.value, agg.maxHead)}
                          title={`L${l}H${h}: ${fmtSigned(c.value)}`}
                          onClick={() => setPicked(c.id)}
                        >
                          {Math.abs(c.value) >= agg.maxHead * 0.3 ? c.value.toFixed(1) : ""}
                        </button>
                      ) : (
                        <span key={`${l}-${h}`} />
                      ),
                    ),
                  ])}
                </div>
                <div className="divlegend" aria-hidden="true">
                  <span>{fmtSigned(-agg.maxHead, 1)}</span>
                  <span className="divlegend__ramp" />
                  <span>{fmtSigned(agg.maxHead, 1)}</span>
                </div>
              </div>
            </section>

            <section className="card">
              <div className="card__head">
                <h2 className="card__title">Largest contributions</h2>
                <span className="card__hint">by size, either sign</span>
              </div>
              <div className="card__body card__body--tight">
                <div className="ranked">
                  {agg.ranked.slice(0, 12).map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      className={detail?.id === c.id ? "ranked__row ranked__row--button is-selected" : "ranked__row ranked__row--button"}
                      onClick={() => setPicked(c.id)}
                    >
                      <span className="ranked__label ranked__label--sans">{c.label}</span>
                      <span className="ranked__value" style={{ color: c.value >= 0 ? "var(--div-pos)" : "var(--div-neg)" }}>
                        {fmtSigned(c.value)}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </section>
          </div>

          {detail && (
            <section className="card">
              <div className="card__head">
                <h2 className="card__title">
                  {detail.label}: {fmtSigned(detail.value)} toward {what}
                </h2>
                {detail.kind === "head" && (
                  <>
                    <span className="spacer" />
                    <a
                      className="linkbtn"
                      href={labHref("attention", { model, prompt: text, layer: String(detail.layer), head: String(detail.head) })}
                    >
                      Where does L{detail.layer}H{detail.head} look? Attention patterns →
                    </a>
                  </>
                )}
              </div>
              <div className="card__body">
                {detail.top_up ? (
                  <div className="grid2">
                    <div>
                      <p className="muted" style={{ marginTop: 0 }}>
                        Pushes up the most, over the whole vocabulary
                      </p>
                      <TokenList items={detail.top_up} />
                    </div>
                    <div>
                      <p className="muted" style={{ marginTop: 0 }}>
                        Pushes down the most
                      </p>
                      <TokenList items={detail.top_down ?? []} />
                    </div>
                  </div>
                ) : (
                  <p className="muted">
                    Not decoded: only the {a.decoded_heads} largest heads are projected onto the whole vocabulary for this
                    model.
                  </p>
                )}
                <p className="muted" style={{ fontSize: "var(--t-xs)", marginBottom: 0 }}>
                  This component's output at this position, divided by the final scale and multiplied by W_U: the direct
                  effect it has on every logit. The vocabulary view is relative, since W_U is centred.
                </p>
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

function DlaBar({
  label,
  value,
  max,
  kind,
  onClick,
}: {
  label: string;
  value: number;
  max: number;
  kind: "embed" | "attn" | "mlp";
  onClick?: () => void;
}) {
  const w = (Math.abs(value) / max) * 50;
  return (
    <button type="button" className={`dla__row dla__row--${kind}`} onClick={onClick} disabled={!onClick} title={`${label}: ${fmtSigned(value, 3)}`}>
      <span className="dla__label">{label}</span>
      <span className="dla__track">
        <span
          className={value >= 0 ? "dla__bar dla__bar--pos" : "dla__bar dla__bar--neg"}
          style={value >= 0 ? { left: "50%", width: `${w}%` } : { left: `${50 - w}%`, width: `${w}%` }}
        />
        <span className="dla__axis" />
      </span>
      <span className="dla__value">{fmtSigned(value)}</span>
    </button>
  );
}

function TokenList({ items }: { items: { label: string; value: number }[] }) {
  return (
    <ol className="toklist">
      {items.map((t, i) => (
        <li key={i}>
          <span className="tok">{t.label || "∅"}</span>
          <span className="muted mono">{fmtSigned(t.value)}</span>
        </li>
      ))}
    </ol>
  );
}
