import { useStore } from "../state/store";

export function ModelPicker({ onShowDetails }: { onShowDetails: () => void }) {
  const models = useStore((s) => s.models);
  const modelsPhase = useStore((s) => s.modelsPhase);
  const model = useStore((s) => s.model);
  const setModel = useStore((s) => s.setModel);
  const selected = models.find((m) => m.id === model);

  return (
    <div className="field">
      <label className="label" htmlFor="model-picker">
        Model
        {models.length > 0 && (
          <button type="button" className="btn btn--ghost btn--sm" onClick={onShowDetails}>
            details
          </button>
        )}
      </label>

      {modelsPhase === "loading" && models.length === 0 ? (
        <div className="skeleton" style={{ height: "var(--control-h)" }} />
      ) : (
        <select id="model-picker" className="select" value={model} onChange={(e) => setModel(e.target.value)}>
          {/* A permalink can name a model this server doesn't serve. Keep it as
              a visible option rather than letting the <select> silently snap to
              its first entry and show the wrong label for the loaded run. */}
          {selected === undefined && <option value={model}>{model} (not in this server's registry)</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id} disabled={m.tier === "disabled"}>
              {m.label}
              {m.status === "resident" ? " · loaded" : ""}
              {m.tier === "disabled" ? " · unavailable" : ""}
            </option>
          ))}
        </select>
      )}

      {selected && (
        <p className="muted" style={{ fontSize: "var(--t-sm)" }}>
          <span className="mono">
            {selected.n_layers}L × {selected.n_heads}H
          </span>{" "}
          · {selected.blurb}
        </p>
      )}
    </div>
  );
}
