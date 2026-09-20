import { useEffect, useState } from "react";

/**
 * True only once `value` has stayed true for `delayMs`; false the instant it
 * goes false. Used to hold back the "stale" treatment on a re-run: a fast
 * model answers in well under a second, and flashing the whole page grey on
 * every pause in typing is more distracting than showing nothing at all.
 */
export function useDelayedFlag(value: boolean, delayMs: number): boolean {
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (!value) {
      setOn(false);
      return;
    }
    const t = setTimeout(() => setOn(true), delayMs);
    return () => clearTimeout(t);
  }, [value, delayMs]);
  return on;
}
