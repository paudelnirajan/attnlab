import { ApiError, type ApiErrorBody, type ModelsResponse, type RepeatedSpec, type RunResponse, type TokenizeResponse } from "./types";

export async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = (await res.json()) as ApiErrorBody;
    throw new ApiError(res.status, body);
  }
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

export function runText(model: string, text: string, topK = 5): Promise<RunResponse> {
  return jsonRequest<RunResponse>("/api/run", {
    method: "POST",
    body: JSON.stringify({ model, text, top_k: topK }),
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
  if (!res.ok) {
    const body = (await res.json()) as ApiErrorBody;
    throw new ApiError(res.status, body);
  }
  return res.arrayBuffer();
}
