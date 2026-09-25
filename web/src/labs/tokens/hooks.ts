import { useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/types";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";

export function errText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export interface Remote<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetch-on-change with the two guards every view here needs: a debounce so a
 * keystroke isn't a request, and a request id so a slow early response can
 * never overwrite a fast later one. The previous data stays in place while the
 * next request is in flight (the view dims it) rather than flashing empty.
 *
 * `key` is what the fetch depends on; `enabled: false` skips fetching.
 */
export function useRemote<T>(key: string, fetcher: () => Promise<T>, { delay = 250, enabled = true } = {}): Remote<T> {
  const debouncedKey = useDebouncedValue(key, delay);
  const [state, setState] = useState<Remote<T>>({ data: null, loading: enabled, error: null });
  const requestId = useRef(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) return;
    const id = ++requestId.current;
    setState((s) => ({ ...s, loading: true }));
    fetcherRef
      .current()
      .then((data) => id === requestId.current && setState({ data, loading: false, error: null }))
      .catch((e) => id === requestId.current && setState((s) => ({ ...s, loading: false, error: errText(e) })));
  }, [debouncedKey, enabled]);

  return state;
}
