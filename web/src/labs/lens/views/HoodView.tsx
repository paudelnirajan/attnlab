import type { LensRunResponse } from "../api";
import { LENS_COPY } from "../copy";
import { fmtNum, fmtPct } from "../format";

/**
 * The computation, spelled out with this run's real shapes, and the checks
 * that were re-run on it. Nothing here is illustrative: every number comes
 * from the server for this model and this text.
 */
export function HoodView({ run }: { run: LensRunResponse }) {
  const a = run.anatomy;
  const raw = a.raw;
  const seq = a.seq;
  const rows = a.n_rows;
  const V = a.d_vocab.toLocaleString();
  const D = a.d_model;

  return (
    <div className="stack">
      <section className="card">
        <div className="card__head">
          <h2 className="card__title">What one lens row is</h2>
          <span className="card__hint">{run.model} · this text</span>
        </div>
        <div className="card__body">
          <pre className="codeblock">
{`tokens                       [${seq}]            ${run.prepend_bos ? "BOS prepended" : "no BOS"}
x = W_E[tokens]${a.positional === "standard" ? " + W_pos   " : "           "}   [${seq}, ${D}]        ${a.positional === "standard" ? "row 0: resid_pre 0" : `row 0 (${a.positional} positions are not added to the stream)`}
for each block L:
  x = x + attn_L(x)          [${seq}, ${D}]        ${a.attn_only ? "row: resid_post L (no MLPs)" : a.parallel_attn_mlp ? "row: resid_pre L + attn_out (parallel block, computed)" : "row: resid_mid L"}${a.attn_only ? "" : `
  x = x + mlp_L(x)           [${seq}, ${D}]        row: resid_post L`}

every row, stacked            [${rows}, ${seq}, ${D}]
lens(row) = W_U · norm(row) + b_U     [${rows}, ${seq}, ${D}] @ [${D}, ${V}] → [${rows}, ${seq}, ${V}]
softmax over the last axis → one distribution per (row, position)`}
          </pre>
          <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
            The full stack of lens logits would be {rows} × {seq} × {V} floats ={" "}
            {((rows * seq * a.d_vocab * 4) / 2 ** 20).toFixed(0)} MB, so the server computes it one row at a time and
            sends only summaries: the top 5 per cell, the actual next token's probability and rank, entropy and KL. It keeps
            the residual stream itself ({run.stored_mb} MB) for about ten minutes, which is what the Trajectory and
            Attribution views read.
          </p>
        </div>
      </section>

      <div className="grid2">
        <section className="card">
          <div className="card__head">
            <h2 className="card__title">Checked on this run</h2>
            <span className="card__hint">recomputed every time the model runs</span>
          </div>
          <div className="card__body">
            <ul className="checks">
              {run.checks.map((c) => (
                <li key={c.id} className={c.ok ? "checks__item" : "checks__item checks__item--fail"}>
                  <span className={c.ok ? "okmark okmark--ok" : "okmark okmark--warn"}>{c.ok ? "✓" : "✗"}</span>
                  <div>
                    <div className="mono">{c.label}</div>
                    <div className="muted" style={{ fontSize: "var(--t-xs)" }}>
                      {c.detail}
                      {c.value !== null && <> · measured {c.value === 0 ? "0 (bit-exact)" : c.value.toExponential(1)}</>}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className="card">
          <div className="card__head">
            <h2 className="card__title">The two lenses</h2>
          </div>
          <div className="card__body">
            {run.lenses.map((l) => (
              <div key={l} style={{ marginBottom: "var(--s3)" }}>
                <strong>{LENS_COPY[l].label}</strong> <span className="muted">· {LENS_COPY[l].short}</span>
                <p style={{ margin: "var(--s1) 0 0", fontSize: "var(--t-sm)" }}>{LENS_COPY[l].how}</p>
              </div>
            ))}
            {!raw?.norm && (
              <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
                This model has no final normalization: its output is W_U · x + b_U directly, so every row is read that way
                and there is nothing for a second lens to change. Its raw stream norms are the reason other models have one.
              </p>
            )}
          </div>
        </section>
      </div>

      {raw && raw.norm && (
        <div className="grid2">
          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Why the plain lens fails</h2>
              <span className="card__hint">the dimensions that dominate the last row, and what ln_final does to them</span>
            </div>
            <div className="card__body card__body--tight tablewrap">
              <table className="langtable">
                <thead>
                  <tr>
                    <th>dim</th>
                    <th>share of ‖x‖²</th>
                    <th>ln_final.w</th>
                  </tr>
                </thead>
                <tbody>
                  {a.outlier_dims.map((d) => (
                    <tr key={d.dim}>
                      <td className="mono">{d.dim}</td>
                      <td className="mono">{fmtPct(d.share)}</td>
                      <td className="mono">{d.w.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted" style={{ fontSize: "var(--t-sm)", padding: "0 var(--s4)" }}>
                A handful of {D} dimensions hold a large share of the residual's size. Plain normalization keeps them
                loud. ln_final's learned scale w ranges from {fmtNum(raw.w_min, 3)} to {fmtNum(raw.w_max, 2)} (mean{" "}
                {fmtNum(raw.w_mean, 2)}): where it is near zero it mutes a dimension before anything reaches W_U.
              </p>
            </div>
          </section>

          <section className="card">
            <div className="card__head">
              <h2 className="card__title">The output path</h2>
            </div>
            <div className="card__body">
              <dl className="factlist">
                <dt>trained with</dt>
                <dd>{raw.norm === "LN" ? "LayerNorm (centre, scale, then w and b)" : "RMSNorm (scale, then w)"}</dd>
                <dt>tied embeddings</dt>
                <dd>
                  {raw.tied === null ? "–" : raw.tied ? "yes: W_U = W_Eᵀ in the released weights, so the embedding row mostly predicts the input token itself" : "no: W_U and W_E are separate"}
                </dd>
                <dt>ln_final.b alone pushes up</dt>
                <dd>
                  {raw.bias_prior.map((t, i) => (
                    <span key={i} className="tok" style={{ marginRight: 4 }}>
                      {t}
                    </span>
                  ))}{" "}
                  <span className="muted">a frequent-word prior added to every prediction</span>
                </dd>
                <dt>smallest w</dt>
                <dd className="mono">{raw.smallest_w.map((d) => `${d.dim}: ${d.w.toFixed(4)}`).join(" · ")}</dd>
                <dt>as loaded here</dt>
                <dd>
                  {raw.folded
                    ? "w and b are folded into W_U (TransformerLens's default processing). The plain lens is rebuilt from the folded weights exactly; the check on the left confirms the algebra."
                    : "ln_final keeps its own w and b (this model's weights can't be folded)."}
                </dd>
              </dl>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
