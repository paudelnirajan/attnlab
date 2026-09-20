import { useEffect } from "react";

interface Options {
  promptRef: React.RefObject<HTMLTextAreaElement | null>;
  nLayers: number | null;
  nHeads: number | null;
  selectedLayer: number;
  selectedHead: number | null;
  selectedTokenIdx: number | null;
  setSelectedLayer: (layer: number) => void;
  setSelectedHead: (head: number | null) => void;
  setSelectedToken: (idx: number | null) => void;
  toggleDirection: () => void;
  onShowHelp: () => void;
  /** false while a modal is open — the dialog owns the keyboard then */
  enabled: boolean;
}

function isTypingTarget(el: Element | null): boolean {
  if (!el) return false;
  if ((el as HTMLElement).isContentEditable) return true;
  const tag = el.tagName;
  return tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT";
}

/** docs/PLAN.md Stage 1: "left/right head, up/down layer, / focus prompt, ?
 * shortcuts", plus `d` for the read direction. Everything except Escape is
 * suppressed while a text field has focus, so typing "?" or "/" into the
 * prompt just types those characters. */
export function useKeyboardShortcuts(opts: Options) {
  const {
    promptRef,
    nLayers,
    nHeads,
    selectedLayer,
    selectedHead,
    selectedTokenIdx,
    setSelectedLayer,
    setSelectedHead,
    setSelectedToken,
    toggleDirection,
    onShowHelp,
    enabled,
  } = opts;

  useEffect(() => {
    if (!enabled) return;

    function onKeyDown(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const typing = isTypingTarget(document.activeElement);

      if (e.key === "Escape") {
        // Unwind one layer of state per press, most transient first.
        if (typing) (document.activeElement as HTMLElement).blur();
        else if (selectedTokenIdx !== null) setSelectedToken(null);
        else if (selectedHead !== null) setSelectedHead(null);
        return;
      }

      if (typing) return;

      switch (e.key) {
        case "/":
          e.preventDefault();
          promptRef.current?.focus();
          return;
        case "?":
          e.preventDefault();
          onShowHelp();
          return;
        case "d":
        case "D":
          e.preventDefault();
          toggleDirection();
          return;
        case "ArrowUp":
        case "ArrowDown": {
          if (nLayers === null) return;
          e.preventDefault();
          const delta = e.key === "ArrowUp" ? -1 : 1;
          setSelectedLayer(Math.max(0, Math.min(nLayers - 1, selectedLayer + delta)));
          return;
        }
        case "ArrowLeft":
        case "ArrowRight": {
          if (nHeads === null) return;
          e.preventDefault();
          if (selectedHead === null) {
            setSelectedHead(e.key === "ArrowRight" ? 0 : nHeads - 1);
          } else {
            const delta = e.key === "ArrowLeft" ? -1 : 1;
            setSelectedHead(Math.max(0, Math.min(nHeads - 1, selectedHead + delta)));
          }
          return;
        }
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    enabled,
    promptRef,
    nLayers,
    nHeads,
    selectedLayer,
    selectedHead,
    selectedTokenIdx,
    setSelectedLayer,
    setSelectedHead,
    setSelectedToken,
    toggleDirection,
    onShowHelp,
  ]);
}
