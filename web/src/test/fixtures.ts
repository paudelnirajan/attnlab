import { vi } from "vitest";
import type { ModelsResponse, RunResponse, TokenInfo, TokenizeResponse } from "../api/types";

export const SEQ = 4;
export const N_LAYERS = 2;
export const N_HEADS = 8;

/** The attention weight this fixture encodes at (head, dest, src). Weight
 * (src+1)^(head+1), normalised over the causal prefix: rows sum to 1, the
 * strongest source is always the diagonal, and heads differ from one another
 * so a per-head vector is distinguishable from the mean over heads. */
export function expectedWeight(head: number, dest: number, src: number): number {
  if (src > dest) return 0;
  let total = 0;
  for (let s = 0; s <= dest; s++) total += Math.pow(s + 1, head + 1);
  return Math.pow(src + 1, head + 1) / total;
}

/** Builds a real ATNP payload per docs/01-wire-format.md: triangle-packed,
 * uint8, sqrt-companded (flags 0b111). This is the encoder side of the
 * contract, written independently of src/api/patterns.ts so the test exercises
 * the decoder against the spec rather than against itself. */
export function encodeAtnp(layerIds: number[], nHeads = N_HEADS, seq = SEQ): ArrayBuffer {
  const perHead = (seq * (seq + 1)) / 2;
  const headerLen = 16 + 2 * layerIds.length;
  const buf = new ArrayBuffer(headerLen + layerIds.length * nHeads * perHead);
  const dv = new DataView(buf);

  dv.setUint8(0, 0x41); // A
  dv.setUint8(1, 0x54); // T
  dv.setUint8(2, 0x4e); // N
  dv.setUint8(3, 0x50); // P
  dv.setUint8(4, 1); // version
  dv.setUint8(5, 0b111); // triangle | uint8 | sqrt-companded
  dv.setUint16(8, layerIds.length, true);
  dv.setUint16(10, nHeads, true);
  dv.setUint32(12, seq, true);
  layerIds.forEach((id, i) => dv.setUint16(16 + 2 * i, id, true));

  const bytes = new Uint8Array(buf, headerLen);
  let o = 0;
  for (let l = 0; l < layerIds.length; l++) {
    for (let h = 0; h < nHeads; h++) {
      for (let dest = 0; dest < seq; dest++) {
        for (let src = 0; src <= dest; src++) {
          bytes[o++] = Math.round(255 * Math.sqrt(expectedWeight(h, dest, src)));
        }
      }
    }
  }
  return buf;
}

function whole(id: number, str: string, display: string, start: number, end: number, cluster: number): TokenInfo {
  return {
    id, str, display, start, end,
    is_byte_fallback: false,
    cluster, cluster_size: 1, cluster_index: 0, cluster_text: display,
    byte_hex: null,
  };
}

export const TOKENS: TokenInfo[] = [
  whole(464, "The", "The", 0, 3, 0),
  whole(2068, " quick", "·quick", 3, 9, 1),
  whole(7586, " brown", "·brown", 9, 15, 2),
  whole(21831, " fox", "·fox", 15, 19, 3),
];

/** Real GPT-2 output for Devanagari: each character costs two tokens, neither
 * of which decodes to a character on its own. Taken from an actual tokenizer
 * run, not invented — see docs/03-decisions.md D12. */
export const FRAGMENTED_TOKENS: TokenInfo[] = [
  { id: 11976, str: "\ufffd", display: "म", start: 0, end: 1, is_byte_fallback: true,
    cluster: 0, cluster_size: 2, cluster_index: 0, cluster_text: "म", byte_hex: "e0a4" },
  { id: 106, str: "\ufffd", display: "⋯", start: 0, end: 1, is_byte_fallback: true,
    cluster: 0, cluster_size: 2, cluster_index: 1, cluster_text: "म", byte_hex: "ae" },
  { id: 48077, str: "ा", display: "ा", start: 1, end: 2, is_byte_fallback: false,
    cluster: 1, cluster_size: 1, cluster_index: 0, cluster_text: "ा", byte_hex: "e0a4be" },
  { id: 11976, str: "\ufffd", display: "ल", start: 2, end: 3, is_byte_fallback: true,
    cluster: 2, cluster_size: 2, cluster_index: 0, cluster_text: "ल", byte_hex: "e0a4" },
  { id: 110, str: "\ufffd", display: "⋯", start: 2, end: 3, is_byte_fallback: true,
    cluster: 2, cluster_size: 2, cluster_index: 1, cluster_text: "ल", byte_hex: "b2" },
];

const META = { op: "forward+cache", duration_ms: 42.5, tl_version: "2.18.0", device: "cpu", dtype: "torch.float32" };

export const MODELS: ModelsResponse = {
  models: [
    {
      id: "attn-only-2l-demo",
      label: "Attn-Only 2L (ARENA 1.2)",
      n_layers: N_LAYERS,
      n_heads: N_HEADS,
      d_model: 128,
      n_params: 3_000_000,
      max_seq: 512,
      languages: ["en"],
      tier: "baked",
      status: "resident",
      est_ram_mb: 757,
      blurb: "Two attention layers, no MLPs.",
    },
    {
      id: "gpt2-small",
      label: "GPT-2 Small",
      n_layers: 12,
      n_heads: 12,
      d_model: 768,
      n_params: 124_000_000,
      max_seq: 1024,
      languages: ["en"],
      tier: "baked",
      status: "available",
      est_ram_mb: 1671,
      blurb: "The classic.",
    },
  ],
  budget: { limit_mb: 6144, used_mb: 757 },
};

export function tokenizeResponse(): TokenizeResponse {
  return { tokens: TOKENS, n_tokens: TOKENS.length, max_seq: 512, _meta: { ...META, op: "tokenize" } };
}

export function runResponse(runId = "run-1", tokens: TokenInfo[] = TOKENS): RunResponse {
  return {
    run_id: runId,
    tokens,
    n_layers: N_LAYERS,
    n_heads: N_HEADS,
    loss_per_token: tokens.slice(1).map((_, i) => 3.2 - i * 0.5),
    top_logits: tokens.map((_, i) => [
      { id: 1000 + i, str: " next", logit: 12.4, prob: 0.4 },
      { id: 21831, str: " fox", logit: 10.1, prob: 0.25 },
    ]),
    cost: {
      attention_bytes_f32: N_LAYERS * N_HEADS * SEQ * SEQ * 4,
      kv_cache_bytes: 2 * N_LAYERS * 128 * SEQ * 4,
      weights_bytes: 3_000_000 * 4,
      forward_flops: 123_456_789,
    },
    expires_at: "2026-01-01T00:00:00Z",
    _meta: META,
  };
}

export interface FetchMock {
  /** resolve/reject the next POST /api/run manually, to test loading states */
  gateRun: (gate: null | Promise<void>) => void;
  failRunWith: (body: { code: string; message: string } | null) => void;
  /** serve a different token set, e.g. a byte-fragmented non-Latin script */
  useTokens: (tokens: TokenInfo[]) => void;
  calls: string[];
}

function jsonRes(body: unknown) {
  return { ok: true, status: 200, json: async () => body, arrayBuffer: async () => new ArrayBuffer(0) };
}

export function installFetchMock(): FetchMock {
  let gate: Promise<void> | null = null;
  let runFailure: { code: string; message: string } | null = null;
  let tokens: TokenInfo[] = TOKENS;
  const calls: string[] = [];
  let runCount = 0;

  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : String(input);
    calls.push(`${init?.method ?? "GET"} ${url}`);

    if (url.startsWith("/api/models")) return jsonRes(MODELS);
    if (url.startsWith("/api/tokenize")) return jsonRes({ ...tokenizeResponse(), tokens, n_tokens: tokens.length });

    if (url.startsWith("/api/run/")) {
      const layers = new URL(url, "http://t").searchParams.get("layers") ?? "0";
      const ids = layers.split(",").map(Number);
      const buf = encodeAtnp(ids);
      return { ok: true, status: 200, json: async () => ({}), arrayBuffer: async () => buf };
    }

    if (url.startsWith("/api/run")) {
      if (gate) await gate;
      if (runFailure) {
        return {
          ok: false,
          status: 422,
          json: async () => ({ error: { ...runFailure, detail: {} } }),
          arrayBuffer: async () => new ArrayBuffer(0),
        };
      }
      return jsonRes(runResponse(`run-${++runCount}`, tokens));
    }

    throw new Error(`unmocked fetch: ${url}`);
  });

  return {
    gateRun: (g) => {
      gate = g;
    },
    failRunWith: (b) => {
      runFailure = b;
    },
    useTokens: (t) => {
      tokens = t;
    },
    calls,
  };
}
