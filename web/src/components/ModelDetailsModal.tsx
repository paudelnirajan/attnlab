import type { ModelInfo } from "../api/types";
import { useStore } from "../state/store";
import { Modal } from "./Modal";

function paramCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${Math.round(n / 1e6)}M`;
  return String(n);
}

function statusBadge(m: ModelInfo) {
  if (m.tier === "disabled") return <span className="badge badge--warn">disabled</span>;
  if (m.status === "resident") return <span className="badge badge--good">in memory</span>;
  if (m.status === "downloading") return <span className="badge">downloading</span>;
  if (m.tier === "lazy") return <span className="badge">loads on first use</span>;
  return <span className="badge">ready</span>;
}

/** The registry, in full. The picker itself only has room for a label, and
 * cramming "disabled — reason" into an <option> string was unreadable; the
 * things that actually decide which model you pick (RAM cost, whether it's
 * resident, why one is off) belong somewhere with room to say them. */
export function ModelDetailsModal({ onClose }: { onClose: () => void }) {
  const models = useStore((s) => s.models);
  const budget = useStore((s) => s.budget);
  const current = useStore((s) => s.model);
  const setModel = useStore((s) => s.setModel);

  return (
    <Modal title="Models" onClose={onClose} wide>
      {budget && (
        <p className="muted" style={{ marginBottom: "var(--s4)" }}>
          Server memory budget: <span className="mono">{budget.used_mb.toFixed(0)}</span> of{" "}
          <span className="mono">{budget.limit_mb.toFixed(0)}</span> MB in use. Loading a model that doesn't fit
          evicts the least recently used one; a model larger than the whole budget is refused outright rather than
          swapped.
        </p>
      )}

      {models.map((m) => (
        <div key={m.id} className="modal__section">
          <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)", flexWrap: "wrap" }}>
            <strong style={{ fontSize: "var(--t-md)" }}>{m.label}</strong>
            {statusBadge(m)}
            {m.id === current && <span className="badge badge--good">selected</span>}
            <span className="spacer" />
            {m.id !== current && m.tier !== "disabled" && (
              <button
                type="button"
                className="btn btn--sm"
                onClick={() => {
                  setModel(m.id);
                  onClose();
                }}
              >
                Use
              </button>
            )}
          </div>
          <p style={{ marginTop: 4 }}>{m.blurb}</p>
          <p className="muted mono" style={{ fontSize: "var(--t-sm)", marginTop: 4 }}>
            {m.n_layers} layers × {m.n_heads} heads · d_model {m.d_model} · {paramCount(m.n_params)} params · ctx{" "}
            {m.max_seq} · ~{m.est_ram_mb} MB RAM · {m.languages.join(", ")}
          </p>
          {m.reason && (
            <p style={{ marginTop: 4, color: "var(--status-serious)", fontSize: "var(--t-sm)" }}>{m.reason}</p>
          )}
        </div>
      ))}
    </Modal>
  );
}
