import { useStore } from "../state/store";
import type { ThemeMode } from "../theme";

const NEXT_MODE: Record<ThemeMode, ThemeMode> = { system: "light", light: "dark", dark: "system" };
const MODE_ICON: Record<ThemeMode, string> = { system: "◐", light: "☀", dark: "☾" };
const MODE_LABEL: Record<ThemeMode, string> = { system: "follow system", light: "light", dark: "dark" };

interface Props {
  onShowHelp: () => void;
  onShowAbout: () => void;
}

export function TopBar({ onShowHelp, onShowAbout }: Props) {
  const runPhase = useStore((s) => s.runPhase);
  const runResult = useStore((s) => s.runResult);
  const themeMode = useStore((s) => s.themeMode);
  const setThemeMode = useStore((s) => s.setThemeMode);

  const meta = runResult?._meta;

  return (
    <header className="topbar">
      <div className="topbar__brand">
        attnlab
        <small>attention patterns, live</small>
      </div>

      <div className="topbar__spacer" />

      {/* Honest, always-visible status. The device/latency pair matters here:
          docs/PLAN.md's two-mode rule means an MPS number and a CPU number are
          not comparable, so the view never shows one without the other. */}
      {runPhase === "loading" ? (
        <span className="badge">
          <span className="spinner" aria-hidden="true" /> running
        </span>
      ) : meta ? (
        <span className="badge" title={`TransformerLens ${meta.tl_version} · ${meta.dtype}`}>
          {meta.device} · {meta.duration_ms.toFixed(0)} ms
        </span>
      ) : null}

      <button type="button" className="btn btn--ghost btn--sm" onClick={onShowAbout}>
        How to read this
      </button>
      <button type="button" className="btn btn--ghost btn--sm" onClick={onShowHelp}>
        Shortcuts <kbd>?</kbd>
      </button>
      <button
        type="button"
        className="btn btn--ghost btn--icon"
        onClick={() => setThemeMode(NEXT_MODE[themeMode])}
        aria-label={`Theme: ${MODE_LABEL[themeMode]}. Switch to ${MODE_LABEL[NEXT_MODE[themeMode]]}.`}
        title={`Theme: ${MODE_LABEL[themeMode]}`}
      >
        {MODE_ICON[themeMode]}
      </button>
    </header>
  );
}
