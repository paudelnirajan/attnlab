// Theme resolution mirrors styles.css's precedence exactly: an explicit
// [data-theme] stamp on <html> beats the OS prefers-color-scheme setting.
// The canvas renderer can't read CSS custom properties through a class, so
// it needs the *resolved* theme as a value — which is why this exists rather
// than the components just leaning on CSS.

export type Theme = "light" | "dark";
/** "system" follows the OS; the other two are an explicit user override. */
export type ThemeMode = Theme | "system";

const STORAGE_KEY = "attnlab:theme";

export function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function resolveTheme(mode: ThemeMode): Theme {
  return mode === "system" ? systemTheme() : mode;
}

export function readStoredMode(): ThemeMode {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === "light" || raw === "dark" || raw === "system") return raw;
  } catch {
    // Safari in private mode throws on localStorage access; the default is fine.
  }
  return "system";
}

export function persistMode(mode: ThemeMode): void {
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    /* non-fatal — the toggle still works for this page load */
  }
}

/** Stamps (or clears) the [data-theme] attribute the stylesheet keys off. */
export function applyThemeAttr(mode: ThemeMode): void {
  const root = document.documentElement;
  if (mode === "system") delete root.dataset.theme;
  else root.dataset.theme = mode;
}

/** Fires whenever the OS theme flips. Returns an unsubscribe function. */
export function subscribeToSystemTheme(onChange: (theme: Theme) => void): () => void {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const handler = (e: MediaQueryListEvent) => onChange(e.matches ? "dark" : "light");
  mq.addEventListener("change", handler);
  return () => mq.removeEventListener("change", handler);
}
