import { lensApi, type LensName, type LensRunResponse, type LensSummary } from "../api";
import { LENS_COPY } from "../copy";
import { LineChart, type Series } from "../components/LineChart";
import { fmtNum, fmtPct } from "../format";
import { useRunQuery } from "../hooks";
import { useLens, visibleRows } from "../store";

const LENS_COLOR: Record<LensName, string> = { ln_final: "var(--series-1)", plain: "var(--series-2)" };

/** Per-layer averages over every position, for every lens at once. */
export function LayersView({ run, summary }: { run: LensRunResponse; summary: LensSummary }) {
  const rowMode = useLens((s) => s.rowMode);
  const remote = useRunQuery(run.run_id, () => lensApi.layers(run.run_id));
  const d = remote.data?.run_id === run.run_id ? remote.data : null;
  const rows = visibleRows(summary.rows, rowMode);
  const xLabels = rows.map((r) => summary.rows[r].label);
  const lenses = run.lenses;
  const hasNext = run.tokens.length > 1;

  function series(key: "agree_final" | "agree_next" | "ce_next" | "kl_final" | "entropy"): Series[] {
    if (!d) return [];
    return lenses.map((l) => ({
      id: l,
      label: LENS_COPY[l].label,
      values: rows.map((r) => d.lenses[l]?.[key][r] ?? null),
      color: LENS_COLOR[l],
    }));
  }

  if (remote.error)
    return (
      <div className="banner banner--error" role="alert">
        <span>{remote.error}</span>
      </div>
    );
  if (!d)
    return (
      <div className="card">
        <div className="empty">
          <span className="spinner" style={{ display: "inline-block", verticalAlign: "-2px" }} /> reading every row under
          every lens…
        </div>
      </div>
    );

  const final = d.lenses.ln_final;
  const firstAgree = final ? rows.find((r) => (final.agree_final[r] ?? 0) >= 0.5) : undefined;

  return (
    <div className={remote.loading ? "stack is-stale" : "stack"}>
      <p className="notice">
        <strong>Averaged over all {run.tokens.length} positions of this text.</strong> One text is a small sample: the
        curves are for building intuition, not for measuring a model.
        {firstAgree !== undefined && (
          <>
            {" "}
            Under ln_final, a majority of positions already have the final answer on top at{" "}
            <strong>{summary.rows[firstAgree].label}</strong>.
          </>
        )}
      </p>
      <div className="grid2">
        <Chart
          title="Agrees with the output's top-1"
          hint="fraction of positions where this row's #1 is the model's final #1"
          series={series("agree_final")}
          xLabels={xLabels}
          yLabel="agreement"
          format={fmtPct}
          yMin={0}
          yMax={1}
        />
        <Chart
          title="KL to the output"
          hint="KL(output ‖ row) in nats, mean over positions · lower = closer"
          series={series("kl_final")}
          xLabels={xLabels}
          yLabel="nats"
          format={(v) => fmtNum(v, 1)}
          yMin={0}
        />
        {hasNext && (
          <Chart
            title="Loss on the actual next token"
            hint="cross-entropy −log P(next), mean over positions · the output row is the model's own loss"
            series={series("ce_next")}
            xLabels={xLabels}
            yLabel="nats"
            format={(v) => fmtNum(v, 1)}
            yMin={0}
          />
        )}
        {hasNext && (
          <Chart
            title="Top-1 is the actual next token"
            hint="fraction of positions where this row's #1 is what really comes next"
            series={series("agree_next")}
            xLabels={xLabels}
            yLabel="accuracy"
            format={fmtPct}
            yMin={0}
            yMax={1}
          />
        )}
        <Chart
          title="Entropy"
          hint={`mean nats; a uniform guess over ${run.anatomy.d_vocab.toLocaleString()} tokens is ${Math.log(run.anatomy.d_vocab).toFixed(1)}`}
          series={series("entropy")}
          xLabels={xLabels}
          yLabel="nats"
          format={(v) => fmtNum(v, 1)}
          yMin={0}
        />
        <Chart
          title="Residual stream norm"
          hint="‖x‖ at each row · the output row is not a residual"
          series={[
            { id: "mean", label: "mean", values: rows.slice(0, -1).map((r) => d.norm_mean[r]), color: "var(--series-1)" },
            ...(d.norm_mean_excl_first
              ? [
                  {
                    id: "excl",
                    label: "mean without position 0",
                    values: rows.slice(0, -1).map((r) => d.norm_mean_excl_first![r]),
                    color: "var(--series-3)",
                  },
                ]
              : []),
          ]}
          xLabels={xLabels.slice(0, -1)}
          yLabel="norm"
          format={(v) => fmtNum(v, 0)}
          yMin={0}
          foot="Every block adds to the stream, so it grows. A logit is a dot product, so a longer vector means sharper probabilities. Without normalization, 'confidence rising with depth' would mostly be the vector growing, which is why every lens normalizes first. Position 0 is often an attention sink with a far larger norm."
        />
      </div>
    </div>
  );
}

type ChartProps = Omit<React.ComponentProps<typeof LineChart>, "ariaLabel"> & { title: string; hint: string; foot?: string };

function Chart({ title, hint, foot, ...chart }: ChartProps) {
  return (
    <section className="card">
      <div className="card__head">
        <h2 className="card__title">{title}</h2>
        <span className="card__hint">{hint}</span>
      </div>
      <div className="card__body">
        <LineChart ariaLabel={title} {...chart} />
        {foot && (
          <p className="muted" style={{ fontSize: "var(--t-xs)", marginBottom: 0 }}>
            {foot}
          </p>
        )}
      </div>
    </section>
  );
}
