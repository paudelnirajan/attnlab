import { useMemo, useState } from "react";
import { lensApi, type LensRunResponse, type LensSummary, type Tracked } from "../api";
import { lensLabel } from "../copy";
import { LineChart, type Series } from "../components/LineChart";
import { PositionPicker } from "../components/PositionPicker";
import { fmtProb, fmtRank, fmtSigned } from "../format";
import { useRunQuery } from "../hooks";
import { refKey, resolvedPos, useLens, visibleRows, type TokenRef } from "../store";

const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)"];

interface Line {
  key: string;
  label: string;
  id: number;
  role: string;
  color: string;
  data: Tracked;
}

/** "Follow one position": every row's prediction for one column, and the tokens you chose to watch. */
export function TrajectoryView({ run, summary }: { run: LensRunResponse; summary: LensSummary }) {
  const lens = useLens((s) => s.lens);
  const rowMode = useLens((s) => s.rowMode);
  const posSel = useLens((s) => s.pos);
  const select = useLens((s) => s.select);
  const track = useLens((s) => s.track);
  const addTrack = useLens((s) => s.addTrack);
  const removeTrack = useLens((s) => s.removeTrack);
  const [draft, setDraft] = useState("");
  const [diff, setDiff] = useState<[string, string] | null>(null);

  const pos = resolvedPos(posSel, run);
  const nextId = pos + 1 < run.tokens.length ? run.tokens[pos + 1].id : null;
  const finalId = summary.strings[summary.final_top[pos]].id;
  const autoIds = [nextId, finalId].filter((x): x is number => x !== null);
  const strs = track.filter((t): t is { str: string } => "str" in t).map((t) => t.str);
  const ids = [...autoIds, ...track.filter((t): t is { id: number; label: string } => "id" in t).map((t) => t.id)];

  const remote = useRunQuery(
    JSON.stringify([run.run_id, pos, lens, strs, ids]),
    () => lensApi.position(run.run_id, pos, lens, 10, strs, ids),
  );
  const d = remote.data && remote.data.pos === pos && remote.data.run_id === run.run_id ? remote.data : null;

  const lines: Line[] = useMemo(() => {
    if (!d) return [];
    const byId = new Map(d.tracked.map((t) => [t.id, t]));
    const out: Line[] = [];
    const push = (key: string, id: number | undefined, role: string) => {
      const t = id === undefined ? undefined : byId.get(id);
      if (!t || out.some((l) => l.id === t.id)) return;
      out.push({ key, id: t.id, label: t.label, role, color: COLORS[out.length % COLORS.length], data: t });
    };
    if (nextId !== null) push("next", nextId, nextId === finalId ? "actual next = final answer" : "actual next");
    push("final", finalId, "final answer");
    for (const t of track) {
      const id = "id" in t ? t.id : d.resolved.find((r) => r.query === t.str)?.id;
      push(refKey(t), id, "tracked");
    }
    return out;
  }, [d, nextId, finalId, track]);

  const rows = visibleRows(summary.rows, rowMode);
  const xLabels = rows.map((r) => summary.rows[r].label);
  const pick = (vals: number[]) => rows.map((r) => vals[r]);
  const series = (f: (t: Tracked) => number[]): Series[] =>
    lines.map((l) => ({ id: l.key, label: l.label || "∅", values: pick(f(l.data)), color: l.color }));

  const pair = diff ?? (lines.length >= 2 ? [lines[0].key, lines[1].key] : null);
  const A = lines.find((l) => l.key === pair?.[0]);
  const B = lines.find((l) => l.key === pair?.[1]);

  return (
    <div className="stack">
      <section className="card">
        <div className="card__head">
          <h2 className="card__title">Pick a position</h2>
          <span className="card__hint">{lensLabel(lens, !!run.anatomy.raw?.norm)} lens · switch lens and rows in the toolbar above</span>
        </div>
        <div className="card__body">
          <PositionPicker run={run} pos={pos} onPick={(p) => select(p)} />
          <div className="trackbar">
            <span className="toolbar__label">Following</span>
            {lines.map((l) => (
              <span key={l.key} className="trackchip">
                <span className="legend__swatch" style={{ borderColor: l.color }} />
                <span className="tok">{l.label || "∅"}</span>
                <span className="muted">{l.role}</span>
                {l.role === "tracked" && (
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    aria-label={`stop tracking ${l.label}`}
                    onClick={() => {
                      const ref = track.find((t) => refKey(t) === l.key);
                      if (ref) removeTrack(ref);
                    }}
                  >
                    ×
                  </button>
                )}
              </span>
            ))}
            <form
              className="trackbar__add"
              onSubmit={(e) => {
                e.preventDefault();
                if (draft) addTrack({ str: draft } as TokenRef);
                setDraft("");
              }}
            >
              <input
                className="input"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="e.g.  Paris"
                aria-label="Token to track (a leading space matters)"
                disabled={track.length >= 4}
              />
              <button type="submit" className="btn btn--sm" disabled={!draft || track.length >= 4}>
                Track
              </button>
            </form>
          </div>
          <p className="muted" style={{ fontSize: "var(--t-xs)" }}>
            Type the token as the model would see it: <span className="mono">" Paris"</span> (with a leading space) is a
            different token from <span className="mono">"Paris"</span>. A string that is several tokens is tracked by its
            first one.
          </p>
          {d?.resolved.filter((r) => r.note).map((r) => (
            <p key={r.query} className="banner banner--info" style={{ fontSize: "var(--t-sm)" }}>
              <span>{r.note}</span>
            </p>
          ))}
        </div>
      </section>

      {remote.error && (
        <div className="banner banner--error" role="alert">
          <span>{remote.error}</span>
        </div>
      )}

      {d && (
        <div className={remote.loading ? "stack is-stale" : "stack"}>
          <div className="grid2">
            <section className="card">
              <div className="card__head">
                <h2 className="card__title">Rank at every row</h2>
                <span className="card__hint">1 = the row's top guess · log scale</span>
              </div>
              <div className="card__body">
                <LineChart
                  ariaLabel="rank of each followed token at every row"
                  xLabels={xLabels}
                  series={series((t) => t.rank)}
                  yLabel="rank"
                  log
                  invert
                  yMin={1}
                  yMax={run.anatomy.d_vocab}
                  format={(v) => fmtRank(Math.round(v))}
                />
              </div>
            </section>
            <section className="card">
              <div className="card__head">
                <h2 className="card__title">Probability at every row</h2>
                <span className="card__hint">softmax of the lens logits</span>
              </div>
              <div className="card__body">
                <LineChart
                  ariaLabel="probability of each followed token at every row"
                  xLabels={xLabels}
                  series={series((t) => t.prob)}
                  yLabel="probability"
                  yMin={0}
                  yMax={1}
                  format={(v) => fmtProb(v)}
                />
              </div>
            </section>
          </div>

          {A && B && (
            <section className="card">
              <div className="card__head">
                <h2 className="card__title">
                  Logit difference: <span className="tok">{A.label}</span> − <span className="tok">{B.label}</span>
                </h2>
                <span className="card__hint">above 0 = the row prefers the first · the logit, not the probability, is what layers add to</span>
                <span className="spacer" />
                <select
                  className="select select--sm"
                  aria-label="first token"
                  value={A.key}
                  onChange={(e) => setDiff([e.target.value, B.key])}
                >
                  {lines.map((l) => (
                    <option key={l.key} value={l.key}>
                      {l.label}
                    </option>
                  ))}
                </select>
                <span className="muted">−</span>
                <select
                  className="select select--sm"
                  aria-label="second token"
                  value={B.key}
                  onChange={(e) => setDiff([A.key, e.target.value])}
                >
                  {lines.map((l) => (
                    <option key={l.key} value={l.key}>
                      {l.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="card__body">
                <LineChart
                  ariaLabel="logit difference at every row"
                  xLabels={xLabels}
                  series={[
                    {
                      id: "diff",
                      label: `${A.label} − ${B.label}`,
                      values: rows.map((r) => A.data.logit[r] - B.data.logit[r]),
                      color: "var(--series-1)",
                    },
                  ]}
                  yLabel="logit difference"
                  zeroLine={0}
                  format={(v) => fmtSigned(v, 1)}
                  height={200}
                />
              </div>
            </section>
          )}

          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Top 10 at every row</h2>
              <span className="card__hint">output at the top, embedding at the bottom · followed tokens are marked</span>
            </div>
            <div className="card__body card__body--tight tablewrap">
              <table className="toptable">
                <tbody>
                  {[...rows].reverse().map((r) => (
                    <tr key={r} className={summary.rows[r].kind === "output" ? "toptable__output" : undefined}>
                      <th scope="row">{summary.rows[r].label}</th>
                      {d.top[r].map((t, k) => {
                        const line = lines.find((l) => l.id === t.id);
                        return (
                          <td key={k} title={`#${k + 1} · p = ${fmtProb(t.prob)} · id ${t.id}`}>
                            {line && <span className="legend__swatch" style={{ borderColor: line.color }} />}
                            <span className="tok">{t.label || "∅"}</span> <span className="muted">{fmtProb(t.prob)}</span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
