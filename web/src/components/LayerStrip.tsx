import { useStore } from "../state/store";

export function LayerStrip() {
  const runResult = useStore((s) => s.runResult);
  const selectedLayer = useStore((s) => s.selectedLayer);
  const setSelectedLayer = useStore((s) => s.setSelectedLayer);

  if (!runResult) return null;

  return (
    <div className="pillrow" role="group" aria-label="Layer">
      {Array.from({ length: runResult.n_layers }, (_, i) => i).map((layer) => (
        <button
          key={layer}
          type="button"
          className="pill"
          aria-pressed={layer === selectedLayer}
          onClick={() => setSelectedLayer(layer)}
        >
          L{layer}
        </button>
      ))}
    </div>
  );
}
