import { useStore } from "../state/store";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${units[i]}`;
}

function formatFlops(n: number): string {
  if (n < 1e9) return `${(n / 1e6).toFixed(1)} MFLOPs`;
  if (n < 1e12) return `${(n / 1e9).toFixed(2)} GFLOPs`;
  return `${(n / 1e12).toFixed(2)} TFLOPs`;
}

/**
 * Stage 1's cost panel: the exact, deterministic `cost` block /api/run
 * computes from shapes (docs/02-api.md) — reproducible, identical on every
 * machine, never measured. The live "double your prompt -> N MB" slider
 * projection is Stage 1.5.
 *
 * The one measured-ish number is the wire size, and it's reported as what it
 * is: bytes of encoded payload for the layer currently on screen, against the
 * float32 figure for that same layer.
 */
export function CostPanel() {
  const runResult = useStore((s) => s.runResult);
  const decoded = useStore((s) => s.patternsByLayer.get(s.selectedLayer));
  const selectedLayer = useStore((s) => s.selectedLayer);

  if (!runResult) return null;
  const { cost } = runResult;
  const seq = runResult.tokens.length;

  const rows: [string, string, string][] = [
    [
      "Attention for this prompt",
      formatBytes(cost.attention_bytes_f32),
      `${runResult.n_layers} layers × ${runResult.n_heads} heads × ${seq}² × 4 B, as float32`,
    ],
    ["KV cache at this length", formatBytes(cost.kv_cache_bytes), "2 × layers × d_model × seq × 4 B"],
    ["Model weights, resident", formatBytes(cost.weights_bytes), "parameter count × 4 B"],
    ["Forward pass", formatFlops(cost.forward_flops), "2·N·seq + 4·layers·seq²·d_model"],
  ];

  if (decoded) {
    const naiveLayer = decoded.nHeads * seq * seq * 4;
    const ratio = naiveLayer / decoded.byteLength;
    rows.push([
      `Sent to your browser (layer ${selectedLayer})`,
      formatBytes(decoded.byteLength),
      `${ratio.toFixed(0)}× smaller than ${formatBytes(naiveLayer)} of float32 — causal packing + one companded byte per cell, before HTTP compression`,
    ]);
  }

  return (
    <div className="card">
      <div className="card__head">
        <h2 className="card__title">Cost</h2>
        <span className="card__hint">computed from shapes, not measured — identical on any machine</span>
      </div>
      <div className="card__body">
        <dl className="statlist">
          {rows.map(([label, value, note]) => (
            <div key={label} style={{ display: "contents" }}>
              <dt title={note}>
                {label}
                <br />
                <span className="muted" style={{ fontSize: "var(--t-xs)" }}>
                  {note}
                </span>
              </dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <p className="muted" style={{ fontSize: "var(--t-xs)", marginTop: "var(--s3)" }}>
          Attention memory grows with the <em>square</em> of the prompt length. Double the tokens and the first row
          quadruples.
        </p>
      </div>
    </div>
  );
}
