import { useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/types";
import { errText } from "../tokens/hooks";
import { useLens } from "./store";

export interface Remote<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetch-on-change against the stored run. Like the tokenizer lab's useRemote
 * (a request id so a slow early answer never overwrites a later one; the old
 * data stays up, dimmed, while the next request runs), plus one rule of its
 * own: the server keeps a run for ~10 minutes, and when it has expired the
 * right move is to run the model again, not to show an error.
 */
export function useRunQuery<T>(key: string | null, fetcher: () => Promise<T>): Remote<T> {
  const [state, setState] = useState<Remote<T>>({ data: null, loading: key !== null, error: null });
  const requestId = useRef(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const rerun = useLens((s) => s.rerun);

  useEffect(() => {
    if (key === null) return;
    const id = ++requestId.current;
    setState((s) => ({ ...s, loading: true }));
    fetcherRef
      .current()
      .then((data) => id === requestId.current && setState({ data, loading: false, error: null }))
      .catch((e) => {
        if (id !== requestId.current) return;
        if (e instanceof ApiError && e.code === "run_not_found") {
          rerun();
          return;
        }
        setState((s) => ({ ...s, loading: false, error: errText(e) }));
      });
  }, [key, rerun]);

  return state;
}
