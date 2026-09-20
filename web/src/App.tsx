import { useCallback, useEffect, useRef, useState } from "react";
import { getModels, getPatterns, runText, tokenize } from "./api/client";
import { ApiError } from "./api/types";
import { decodePatterns } from "./api/patterns";
import { AboutModal } from "./components/AboutModal";
import { CostPanel } from "./components/CostPanel";
import { DirectionToggle } from "./components/DirectionToggle";
import { HeadDetail } from "./components/HeadDetail";
import { HeadGrid } from "./components/HeadGrid";
import { KeyboardHelp } from "./components/KeyboardHelp";
import { LayerStrip } from "./components/LayerStrip";
import { ModelDetailsModal } from "./components/ModelDetailsModal";
import { ModelPicker } from "./components/ModelPicker";
import { PredictionPanel } from "./components/PredictionPanel";
import { PromptBox } from "./components/PromptBox";
import { TokenizerPanel } from "./components/TokenizerPanel";
import { TopBar } from "./components/TopBar";
import { useDebouncedValue } from "./hooks/useDebouncedValue";
import { useDelayedFlag } from "./hooks/useDelayedFlag";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { DIRECTION_COPY } from "./lib/attention";
import { useStore } from "./state/store";
import { writePermalinkToUrl } from "./state/urlSync";
import { subscribeToSystemTheme } from "./theme";

const PROMPT_DEBOUNCE_MS = 300;

type OpenModal = "help" | "about" | "models" | null;

function errText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export default function App() {
  const model = useStore((s) => s.model);
  const prompt = useStore((s) => s.prompt);
  const selectedLayer = useStore((s) => s.selectedLayer);
  const selectedHead = useStore((s) => s.selectedHead);
  const selectedTokenIdx = useStore((s) => s.selectedTokenIdx);
  const direction = useStore((s) => s.direction);
  const models = useStore((s) => s.models);
  const runResult = useStore((s) => s.runResult);
  const runPhase = useStore((s) => s.runPhase);
  const errorMessage = useStore((s) => s.errorMessage);
  const themeMode = useStore((s) => s.themeMode);

  const setModels = useStore((s) => s.setModels);
  const setTokenizeResult = useStore((s) => s.setTokenizeResult);
  const beginRun = useStore((s) => s.beginRun);
  const setRunResult = useStore((s) => s.setRunResult);
  const failRun = useStore((s) => s.failRun);
  const clearRun = useStore((s) => s.clearRun);
  const cachePatterns = useStore((s) => s.cachePatterns);
  const setError = useStore((s) => s.setError);
  const setSelectedLayer = useStore((s) => s.setSelectedLayer);
  const setSelectedHead = useStore((s) => s.setSelectedHead);
  const setSelectedToken = useStore((s) => s.setSelectedToken);
  const setDirection = useStore((s) => s.setDirection);
  const setResolvedTheme = useStore((s) => s.setResolvedTheme);

  const promptRef = useRef<HTMLTextAreaElement>(null);
  const [openModal, setOpenModal] = useState<OpenModal>(null);

  const debouncedPrompt = useDebouncedValue(prompt, PROMPT_DEBOUNCE_MS);

  // The [data-theme] stamp happens inside setThemeMode (see the store); this
  // only has to follow the OS while the mode is "system".
  useEffect(() => {
    if (themeMode !== "system") return;
    return subscribeToSystemTheme(setResolvedTheme);
  }, [themeMode, setResolvedTheme]);

  // --- the model catalogue (permalink state was already seeded from the URL
  // in the store's initialiser, before first render) ---
  useEffect(() => {
    setModels([], null, "loading");
    getModels()
      .then((r) => setModels(r.models, r.budget, "idle"))
      .catch((e) => {
        setModels([], null, "error");
        setError(errText(e));
      });
  }, [setModels, setError]);

  // --- the URL is the single source of truth: write it back on any
  // permalink-relevant state change (docs/PLAN.md Stage 1) ---
  // Safe to run on first render too: it writes back exactly what was just
  // read, which only normalises the URL.
  useEffect(() => {
    writePermalinkToUrl({ model, prompt, selectedLayer, selectedHead, direction });
  }, [model, prompt, selectedLayer, selectedHead, direction]);

  // --- tokenize + run whenever the model or (debounced) prompt changes ---
  const requestIdRef = useRef(0);
  useEffect(() => {
    const requestId = ++requestIdRef.current;
    const fresh = () => requestIdRef.current === requestId;

    if (!debouncedPrompt.trim()) {
      setTokenizeResult(null);
      clearRun();
      setError(null);
      return;
    }

    tokenize(model, debouncedPrompt)
      .then((r) => fresh() && setTokenizeResult(r))
      .catch(() => {
        /* the /run error below is the one worth surfacing; a tokenize failure
           for the same input would only duplicate it */
      });

    // Note: no clearing of runResult here. The previous run stays on screen,
    // dimmed, until this one lands — otherwise the whole page empties out on
    // every debounced keystroke.
    beginRun();
    runText(model, debouncedPrompt)
      .then((r) => {
        if (!fresh()) return;
        setRunResult(r);
        setError(null);
      })
      .catch((e) => {
        if (!fresh()) return;
        failRun();
        setError(errText(e));
      });
  }, [model, debouncedPrompt, setTokenizeResult, beginRun, setRunResult, failRun, clearRun, setError]);

  // --- fetch (and decode) patterns for the selected layer, prefetching its
  // neighbours (docs/PLAN.md: "Prefetch the neighbouring layer") ---
  const inflightRef = useRef(new Set<string>());
  useEffect(() => {
    if (!runResult) return;
    const { run_id: runId, n_layers: nLayers } = runResult;
    const have = useStore.getState().patternsByLayer;

    for (const layer of [selectedLayer, selectedLayer - 1, selectedLayer + 1]) {
      if (layer < 0 || layer >= nLayers || have.has(layer)) continue;
      const key = `${runId}:${layer}`;
      if (inflightRef.current.has(key)) continue; // rapid layer flipping refetched the same layer
      inflightRef.current.add(key);

      getPatterns(runId, [layer])
        .then((buf) => {
          // Ignore if a newer run superseded this one while in flight.
          if (useStore.getState().runResult?.run_id !== runId) return;
          cachePatterns(layer, decodePatterns(buf));
        })
        .catch((e) => {
          if (layer === selectedLayer) setError(errText(e));
        })
        .finally(() => inflightRef.current.delete(key));
    }
  }, [runResult, selectedLayer, cachePatterns, setError]);

  const toggleDirection = useCallback(
    () => setDirection(direction === "dest2src" ? "src2dest" : "dest2src"),
    [direction, setDirection],
  );
  const showHelp = useCallback(() => setOpenModal("help"), []);
  const closeModal = useCallback(() => setOpenModal(null), []);

  useKeyboardShortcuts({
    promptRef,
    nLayers: runResult?.n_layers ?? null,
    nHeads: runResult?.n_heads ?? null,
    selectedLayer,
    selectedHead,
    selectedTokenIdx,
    setSelectedLayer,
    setSelectedHead,
    setSelectedToken,
    toggleDirection,
    onShowHelp: showHelp,
    enabled: openModal === null,
  });

  const loading = runPhase === "loading";
  // Hold the stale treatment back until a run is actually slow — see
  // useDelayedFlag. The top bar's spinner covers the fast case.
  const dim = useDelayedFlag(loading, 400) && runResult !== null;

  return (
    <>
      <TopBar onShowHelp={showHelp} onShowAbout={() => setOpenModal("about")} />

      <main className="page">
        {errorMessage && (
          <div className="banner banner--error" role="alert">
            <span>{errorMessage}</span>
          </div>
        )}

        <div className="setup">
          <ModelPicker onShowDetails={() => setOpenModal("models")} />
          <PromptBox ref={promptRef} />
        </div>

        {!runResult ? (
          <div className="card">
            <div className="empty">
              {loading ? (
                <>
                  <span className="spinner" style={{ display: "inline-block", verticalAlign: "-2px" }} /> running the
                  model…
                </>
              ) : !prompt.trim() ? (
                <>Type a prompt above to see what its attention heads do.</>
              ) : models.length === 0 ? (
                <>Waiting for the model catalogue…</>
              ) : (
                <>No run yet.</>
              )}
            </div>
          </div>
        ) : (
          <div className={dim ? "stack is-stale" : "stack"} aria-busy={loading}>
            <section className="card">
              <div className="card__head">
                <h2 className="card__title">Tokens</h2>
                <span className="card__hint">
                  {runResult.tokens.length} positions
                  {(() => {
                    // Characters actually covered, from the token offsets — the
                    // tokens-per-character ratio is the whole multilingual
                    // fragmentation story in one number.
                    const chars = runResult.tokens.reduce((m, t) => Math.max(m, t.end), 0);
                    return chars > 0 ? (
                      <>
                        {" "}
                        · {chars} chars · {(runResult.tokens.length / chars).toFixed(2)} tokens/char
                      </>
                    ) : null;
                  })()}{" "}
                  · hover to link, click to pin
                </span>
              </div>
              <div className="card__body card__body--tight">
                <TokenizerPanel />
              </div>
            </section>

            <div className="toolbar">
              <span className="toolbar__label">Layer</span>
              <LayerStrip />
              <span className="spacer" />
              <span className="toolbar__label">Read as</span>
              <DirectionToggle />
              <span className="muted" style={{ fontSize: "var(--t-sm)" }}>
                {DIRECTION_COPY[direction].question}
              </span>
            </div>

            <div className={selectedHead !== null ? "workspace" : "workspace workspace--solo"}>
              <section className="card">
                <div className="card__head">
                  <h2 className="card__title">Heads · layer {selectedLayer}</h2>
                  <span className="card__hint">
                    {selectedHead !== null ? `head ${selectedHead} expanded` : "click a head to expand it"}
                  </span>
                </div>
                <div className="card__body">
                  <HeadGrid />
                </div>
              </section>

              {selectedHead !== null && (
                <aside className="workspace__aside">
                  <HeadDetail />
                </aside>
              )}
            </div>

            <div className="grid2">
              <CostPanel />
              <PredictionPanel />
            </div>
          </div>
        )}
      </main>

      {openModal === "help" && <KeyboardHelp onClose={closeModal} />}
      {openModal === "about" && <AboutModal onClose={closeModal} />}
      {openModal === "models" && <ModelDetailsModal onClose={closeModal} />}
    </>
  );
}
