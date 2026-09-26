import { useEffect, useRef } from "react";
import { getModels } from "../../api/client";
import { TopBar } from "../../components/TopBar";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { useDelayedFlag } from "../../hooks/useDelayedFlag";
import { useStore } from "../../state/store";
import { subscribeToSystemTheme } from "../../theme";
import { PathNav } from "../PathNav";
import { labById, labHref } from "../registry";
import { errText } from "../tokens/hooks";
import { lensApi } from "./api";
import { Examples, Setup } from "./components/Setup";
import { VIEW_TABS } from "./copy";
import { useLens, writeLensUrl } from "./store";
import { AttributionView } from "./views/AttributionView";
import { GridView, LensControls } from "./views/GridView";
import { HoodView } from "./views/HoodView";
import { LayersView } from "./views/LayersView";
import { TrajectoryView } from "./views/TrajectoryView";

const TEXT_DEBOUNCE_MS = 400;

/**
 * Step 4 of the path. One forward pass per text; every view reads the same
 * stored residual stream, so switching views, lenses or positions never runs
 * the model again.
 */
export function LensLab() {
  const lab = labById("logit-lens");
  const s = useLens();
  const themeMode = useStore((st) => st.themeMode);
  const setResolvedTheme = useStore((st) => st.setResolvedTheme);
  const debouncedText = useDebouncedValue(s.text, TEXT_DEBOUNCE_MS);

  useEffect(() => {
    if (themeMode !== "system") return;
    return subscribeToSystemTheme(setResolvedTheme);
  }, [themeMode, setResolvedTheme]);

  useEffect(() => {
    getModels()
      .then((r) => useLens.getState().setModels(r.models))
      .catch((e) => useLens.getState().setModels([], errText(e)));
  }, []);

  useEffect(() => {
    writeLensUrl(s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.view, s.model, s.text, s.lens, s.rowMode, s.metric, s.pos, s.row, s.track, s.target, s.contrast, s.bos]);

  // The forward pass: whenever the model, the (debounced) text or BOS changes.
  const requestId = useRef(0);
  useEffect(() => {
    const id = ++requestId.current;
    const st = useLens.getState();
    if (!debouncedText.trim()) return;
    st.beginRun();
    lensApi
      .run(st.model, debouncedText, st.lens, st.bos)
      .then((r) => id === requestId.current && useLens.getState().setRun(r))
      .catch((e) => id === requestId.current && useLens.getState().failRun(errText(e)));
  }, [s.model, debouncedText, s.bos, s.runNonce]);

  // Another lens on the same run: summaries only, no forward pass.
  useEffect(() => {
    const { run, lens, summaries } = useLens.getState();
    if (!run || summaries[lens] || !run.lenses.includes(lens)) return;
    lensApi
      .view(run.run_id, lens)
      .then((v) => useLens.getState().run?.run_id === v.run_id && useLens.getState().cacheSummary(lens, v))
      .catch(() => useLens.getState().rerun()); // most likely expired: run again
  }, [s.lens, s.run]);

  const run = s.run;
  const summary = run ? s.summaries[s.lens] ?? s.summaries[run.lens] ?? null : null;
  const loading = s.runPhase === "loading";
  const dim = useDelayedFlag(loading, 300) && run !== null;
  const tab = VIEW_TABS.find((t) => t.id === s.view)!;

  return (
    <>
      <TopBar />
      <main className="page">
        <header className="labhead">
          <p className="labhead__step">Step {lab.step}</p>
          <h1 className="labhead__title">{lab.title}</h1>
          <p className="labhead__lede">{lab.summary}</p>
        </header>

        {s.modelsError && (
          <div className="banner banner--error" role="alert">
            <span>Couldn't load the model list: {s.modelsError}</span>
          </div>
        )}

        <Examples />
        <Setup />

        {s.runError && (
          <div className="banner banner--error" role="alert">
            <span>{s.runError}</span>
          </div>
        )}

        {!run || !summary ? (
          <div className="card">
            <div className="empty">
              {loading ? (
                <>
                  <span className="spinner" style={{ display: "inline-block", verticalAlign: "-2px" }} /> running the model
                  and reading every layer… (the first run of a model loads it)
                </>
              ) : !s.text.trim() ? (
                <>Type some text above to watch the prediction form.</>
              ) : (
                <>No run yet.</>
              )}
            </div>
          </div>
        ) : (
          <div className={dim ? "stack is-stale" : "stack"} aria-busy={loading}>
            <div className="tabs" role="tablist" aria-label="Logit lens views">
              {VIEW_TABS.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  role="tab"
                  id={`lens-tab-${t.id}`}
                  aria-selected={s.view === t.id}
                  aria-controls="lens-panel"
                  className="tabs__tab"
                  onClick={() => s.setView(t.id)}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <p className="tabs__question">{tab.question}</p>
            {s.view !== "attribution" && s.view !== "hood" && <LensControls run={run} />}

            <div id="lens-panel" role="tabpanel" aria-labelledby={`lens-tab-${s.view}`}>
              {s.view === "grid" && <GridView run={run} summary={summary} />}
              {s.view === "trajectory" && <TrajectoryView run={run} summary={summary} />}
              {s.view === "attribution" && <AttributionView run={run} summary={summary} />}
              {s.view === "layers" && <LayersView run={run} summary={summary} />}
              {s.view === "hood" && <HoodView run={run} />}
            </div>

            <p className="handoff">
              <a href={labHref("attention", { model: s.model, prompt: s.text })}>
                Which heads move information here? See this text in Attention patterns →
              </a>
              <a href={labHref("tokens", { model: s.model, text: s.text })}>Why these columns? Tokenizer lab →</a>
              <span className="muted">
                {run._meta.device} · {run._meta.duration_ms.toFixed(0)} ms · {run.tokens.length} tokens ·{" "}
                {run.rows.length - 1} rows
              </span>
            </p>
          </div>
        )}

        <PathNav labId={lab.id} />
      </main>
    </>
  );
}
