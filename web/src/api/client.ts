import { ApiError, type ApiErrorBody, type ModelsResponse, type RepeatedSpec, type RunResponse, type TokenizeResponse } from "./types";

/** What a failed response means, as an ApiError. The server always answers
 * with the JSON error shape from docs/02-api.md, but a proxy in front of it
 * doesn't: while the server restarts for a release, Cloudflare answers 502 with
 * an HTML page. Those become an error the UI can show as-is. */
export async function errorFrom(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as ApiErrorBody;
    if (body?.error?.code) return new ApiError(res.status, body);
  } catch {
    /* not JSON */
  }
  const unreachable = res.status === 502 || res.status === 503 || res.status === 504 || res.status === 530;
  return new ApiError(res.status, {
    error: {
      code: unreachable ? "unreachable" : "http_error",
      message: unreachable
        ? "The server is restarting or unreachable. Try again in a few seconds."
        : `The server answered HTTP ${res.status}.`,
      detail: {},
    },
  });
}

export async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) throw await errorFrom(res);
  return res.json() as Promise<T>;
}

export function getModels(): Promise<ModelsResponse> {
  return jsonRequest<ModelsResponse>("/api/models");
}

export function tokenize(model: string, text: string): Promise<TokenizeResponse> {
  return jsonRequest<TokenizeResponse>("/api/tokenize", {
    method: "POST",
    body: JSON.stringify({ model, text }),
  });
}

/** `signal` lets the caller abandon a run it no longer wants (the prompt
 * changed); the server then skips it if it hasn't started yet. */
export function runText(model: string, text: string, topK = 5, signal?: AbortSignal): Promise<RunResponse> {
  return jsonRequest<RunResponse>("/api/run", {
    method: "POST",
    body: JSON.stringify({ model, text, top_k: topK }),
    signal,
  });
}

export function runRepeated(model: string, repeated: RepeatedSpec, topK = 5): Promise<RunResponse> {
  return jsonRequest<RunResponse>("/api/run", {
    method: "POST",
    body: JSON.stringify({ model, repeated, top_k: topK }),
  });
}

/** Fetches raw pattern bytes for the given layers of a run. Decode with
 * decodePatterns() from ./patterns. `layers` is required server-side —
 * see docs/01-wire-format.md for why (never send "everything" by default). */
export async function getPatterns(runId: string, layers: number[]): Promise<ArrayBuffer> {
  const qs = new URLSearchParams({ layers: layers.join(",") });
  const res = await fetch(`/api/run/${runId}/patterns?${qs}`);
  if (!res.ok) throw await errorFrom(res);
  return res.arrayBuffer();
}
