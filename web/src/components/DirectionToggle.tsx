import { DIRECTION_COPY } from "../lib/attention";
import { useStore, type Direction } from "../state/store";

const OPTIONS: Direction[] = ["dest2src", "src2dest"];

/**
 * The read-direction switch, equivalent to CircuitsVis's paired
 * Destination/Source token columns in ARENA 1.2 — same matrix, read as a row
 * or as a column. See lib/attention.ts for the numeric definition.
 */
export function DirectionToggle() {
  const direction = useStore((s) => s.direction);
  const setDirection = useStore((s) => s.setDirection);

  return (
    <div className="segmented" role="group" aria-label="Attention read direction">
      {OPTIONS.map((d) => {
        const copy = DIRECTION_COPY[d];
        return (
          <button
            key={d}
            type="button"
            className="segmented__opt"
            aria-pressed={direction === d}
            title={`${copy.short} — ${copy.question}`}
            onClick={() => setDirection(d)}
          >
            {copy.short}
          </button>
        );
      })}
    </div>
  );
}
