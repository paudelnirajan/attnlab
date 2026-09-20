import { useMemo } from "react";
import { getSequentialLut } from "../render/colormap";
import { useStore } from "../state/store";

/** Probability ticks. Their *positions* are sqrt(p), because the wire format
 * sqrt-compands before quantizing (docs/01-wire-format.md) and the ramp is
 * indexed by the companded byte — so the bar itself shows why small weights
 * still get visible colour separation. */
const TICKS = [0, 0.25, 0.5, 1];

/** The legend for the sequential ramp. A heatmap without one is a picture, not
 * a chart — you can't read a value off it. */
export function Colorbar() {
  const theme = useStore((s) => s.theme);

  const gradient = useMemo(() => {
    const lut = getSequentialLut(theme);
    const stops: string[] = [];
    for (let i = 0; i <= 32; i++) {
      const b = Math.round((i / 32) * 255) * 3;
      stops.push(`rgb(${lut[b]} ${lut[b + 1]} ${lut[b + 2]}) ${((i / 32) * 100).toFixed(1)}%`);
    }
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  }, [theme]);

  return (
    <div className="colorbar">
      <div className="colorbar__ramp" style={{ background: gradient }} aria-hidden="true" />
      <div className="colorbar__ticks">
        {TICKS.map((p) => (
          <span key={p} className="colorbar__tick" style={{ left: `${Math.sqrt(p) * 100}%` }}>
            {p}
          </span>
        ))}
      </div>
      <span className="sr-only">Colour scale for attention weight, from 0 to 1.</span>
    </div>
  );
}
