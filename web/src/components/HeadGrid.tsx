import { useEffect, useRef } from "react";
import type { DecodedPatterns } from "../api/patterns";
import { useLinkedWeights } from "../hooks/useLinkedWeights";
import { renderHeadToCanvas } from "../render/heatmap";
import { useStore } from "../state/store";

interface TileProps {
  head: number;
  decoded: DecodedPatterns;
  anchor: number | null;
  selected: boolean;
  onSelect: () => void;
}

function HeadTile({ head, decoded, anchor, selected, onSelect }: TileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useStore((s) => s.theme);
  const direction = useStore((s) => s.direction);

  useEffect(() => {
    if (!canvasRef.current) return;
    renderHeadToCanvas(canvasRef.current, decoded, 0, head, theme);
  }, [decoded, head, theme]);

  // The linked band is a row when reading destination -> source and a column
  // when reading source -> destination: the same geometry as the crosshair in
  // the detail view, so the two never contradict each other.
  const pct = (n: number) => `${(n / decoded.seq) * 100}%`;
  const thickness = `max(2px, ${(1 / decoded.seq) * 100}%)`;
  // Clipped to the causally reachable extent: a destination row only spans
  // sources 0..dest, a source column only spans destinations dest>=src.
  // Running the band the full width made impossible cells look like data.
  const band =
    anchor === null
      ? null
      : direction === "dest2src"
        ? { top: pct(anchor), height: thickness, left: 0, width: pct(anchor + 1) }
        : { left: pct(anchor), width: thickness, top: pct(anchor), height: pct(decoded.seq - anchor) };

  return (
    <button type="button" className="tile" aria-pressed={selected} onClick={onSelect} title={`Head ${head}`}>
      <canvas ref={canvasRef} className="pixel-canvas tile__canvas" />
      {band && <div className="tile__band" style={band} />}
      <span className="tile__label">H{head}</span>
    </button>
  );
}

/** Every head of the selected layer as a small canvas. Clicking one expands it
 * alongside (docs/PLAN.md Stage 1). */
export function HeadGrid() {
  const runResult = useStore((s) => s.runResult);
  const decoded = useStore((s) => s.patternsByLayer.get(s.selectedLayer));
  const selectedHead = useStore((s) => s.selectedHead);
  const setSelectedHead = useStore((s) => s.setSelectedHead);
  const linked = useLinkedWeights();

  if (!runResult) return null;

  if (!decoded) {
    // Skeletons at the real tile size, so the grid doesn't jump when the
    // patterns land.
    return (
      <div className="headgrid" aria-busy="true">
        {Array.from({ length: runResult.n_heads }, (_, h) => (
          <div key={h} className="skeleton" style={{ aspectRatio: "1" }} />
        ))}
      </div>
    );
  }

  return (
    <div className="headgrid">
      {Array.from({ length: runResult.n_heads }, (_, h) => h).map((head) => (
        <HeadTile
          key={head}
          head={head}
          decoded={decoded}
          anchor={linked?.anchor ?? null}
          selected={head === selectedHead}
          onSelect={() => setSelectedHead(head === selectedHead ? null : head)}
        />
      ))}
    </div>
  );
}
