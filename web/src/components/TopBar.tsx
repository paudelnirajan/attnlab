import { LABS, currentLabId } from "../labs/registry";
import { useStore } from "../state/store";
import type { ThemeMode } from "../theme";

const NEXT_MODE: Record<ThemeMode, ThemeMode> = { system: "light", light: "dark", dark: "system" };
const MODE_ICON: Record<ThemeMode, string> = { system: "◐", light: "☀", dark: "☾" };
const MODE_LABEL: Record<ThemeMode, string> = { system: "follow system", light: "light", dark: "dark" };

interface Props {
  /** lab-specific actions; a lab without shortcuts or a reading guide omits them */
  onShowHelp?: () => void;
  onShowAbout?: () => void;
}

export function TopBar({ onShowHelp, onShowAbout }: Props) {
  const here = currentLabId();
  const runPhase = useStore((s) => s.runPhase);
  const runResult = useStore((s) => s.runResult);
  const themeMode = useStore((s) => s.themeMode);
  const setThemeMode = useStore((s) => s.setThemeMode);

  const meta = runResult?._meta;

  return (
    <header className="topbar">
      <a className="topbar__brand" href="/">
        attnlab
        <small>interpretability labs</small>
      </a>

      {/* The lab switcher. Plain links, not client-side routing: each lab
          hydrates its own state from its URL, so a full navigation is both
          simpler and exactly what a copied permalink does anyway. */}
      <nav className="labnav" aria-label="Labs">
        {LABS.filter((l) => l.status === "live").map((l) => (
          <a key={l.id} className="labnav__link" href={l.path} aria-current={here === l.id ? "page" : undefined}>
            <span className="labnav__step">{l.step}</span>
            {l.title}
          </a>
        ))}
      </nav>

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

      {onShowAbout && (
        <button type="button" className="btn btn--ghost btn--sm" onClick={onShowAbout}>
          How to read this
        </button>
      )}
      {onShowHelp && (
        // hidden on narrow screens: a phone has no keyboard to use them with
        <button type="button" className="btn btn--ghost btn--sm topbar__shortcuts" onClick={onShowHelp}>
          Shortcuts <kbd>?</kbd>
        </button>
      )}
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
