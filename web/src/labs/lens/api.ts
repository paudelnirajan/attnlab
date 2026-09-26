// Types and fetchers for the logit lens endpoints (docs/02-api.md § Logit lens).
// Mirrors src/attnlab/lens.py and api/lens_routes.py exactly.

import { jsonRequest } from "../../api/client";
import type { ApiMeta, TokenInfo } from "../../api/types";

export type LensName = "ln_final" | "plain";

export interface LensRow {
  /** the TransformerLens hook this row reads, e.g. "blocks.3.hook_resid_mid" */
  id: string;
  label: string;
  /** -1 for the embedding; n_layers for the output row */
  layer: number;
  kind: "embed" | "attn" | "mlp" | "output";
  /** the stream between two blocks: what "blocks" mode shows */
  block_end: boolean;
  /** false for a parallel block's "after attention", which the model never forms */
  real: boolean;
}

export interface LabelEntry {
  id: number;
  /** honest label: ⋯ marks a cut character, ऄ–ऽ⋯ a character not yet chosen */
  label: string;
}

/** Every array is [row][position], with rows in `rows` order (the output row last). */
export interface LensCells {
  /** indices into `strings`, [row][pos][k] */
  top: number[][][];
  top_p: number[][][];
  /** null at the last position: nothing follows it */
  p_next: (number | null)[][];
  rank_next: (number | null)[][];
  p_final: number[][];
  rank_final: number[][];
  /** nats */
  entropy: number[][];
  /** KL(output ‖ this row), nats */
  kl: number[][];
  /** residual norm; null on the output row */
  norm: (number | null)[][];
}

export interface LensSummary {
  rows: LensRow[];
  input_labels: string[];
  next_labels: (string | null)[];
  /** index into `strings` of the model's own top-1 at each position */
  final_top: number[];
  strings: LabelEntry[];
  cells: LensCells;
}

export interface Check {
  id: string;
  label: string;
  ok: boolean;
  value: number | null;
  detail: string;
}

export interface Anatomy {
  d_model: number;
  d_vocab: number;
  n_layers: number;
  n_heads: number;
  d_head: number;
  n_rows: number;
  seq: number;
  attn_only: boolean;
  parallel_attn_mlp: boolean;
  positional: string;
  /** after TransformerLens processing: "LNPre" once folded */
  normalization: string | null;
  raw: {
    norm: "LN" | "RMS" | null;
    folded: boolean;
    tied: boolean | null;
    w_min: number | null;
    w_max: number | null;
    w_mean: number | null;
    smallest_w: { dim: number; w: number }[];
    bias_prior: string[];
  } | null;
  outlier_dims: { dim: number; share: number; w: number }[];
}

export interface LensRunResponse extends LensSummary {
  run_id: string;
  model: string;
  lens: LensName;
  lenses: LensName[];
  prepend_bos: boolean;
  tokens: TokenInfo[];
  checks: Check[];
  anatomy: Anatomy;
  stored_mb: number;
  _meta: ApiMeta;
}

export interface LayerCurves {
  run_id: string;
  lenses: Partial<
    Record<
      LensName,
      {
        agree_final: number[];
        agree_next: (number | null)[];
        ce_next: (number | null)[];
        kl_final: number[];
        entropy: number[];
        p_next: (number | null)[];
      }
    >
  >;
  norm_mean: number[];
  norm_max: number[];
  norm_mean_excl_first: number[] | null;
  _meta: ApiMeta;
}

export interface Tracked {
  id: number;
  label: string;
  /** one value per row, the output row last */
  logit: number[];
  prob: number[];
  rank: number[];
}

export interface PositionDetail {
  run_id: string;
  pos: number;
  lens: LensName;
  /** [row][k] */
  top: { id: number; label: string; prob: number }[][];
  tracked: Tracked[];
  resolved: { query: string; id: number; note: string }[];
  _meta: ApiMeta;
}

export interface Component {
  id: string;
  label: string;
  kind: "embed" | "pos" | "head" | "attn_bias" | "mlp";
  layer: number;
  head: number | null;
  /** contribution to the target logit (or the target − contrast difference) */
  value: number;
  /** what this component pushes up/down over the whole vocabulary; absent for undecoded heads */
  top_up?: { label: string; value: number }[];
  top_down?: { label: string; value: number }[];
}

export interface Attribution {
  run_id: string;
  pos: number;
  target: LabelEntry;
  contrast: LabelEntry | null;
  components: Component[];
  bias: number;
  total: number;
  actual: number;
  error: number;
  scale: number;
  decoded_heads: "all" | number;
  notes: string[];
  _meta: ApiMeta;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const lensApi = {
  run: (model: string, text: string, lens: LensName, prependBos: boolean) =>
    jsonRequest<LensRunResponse>("/api/lens/run", post({ model, text, lens, prepend_bos: prependBos })),
  view: (runId: string, lens: LensName) =>
    jsonRequest<LensSummary & { run_id: string; lens: LensName }>("/api/lens/view", post({ run_id: runId, lens })),
  layers: (runId: string) => jsonRequest<LayerCurves>("/api/lens/layers", post({ run_id: runId })),
  position: (runId: string, pos: number, lens: LensName, k: number, track: string[], trackIds: number[]) =>
    jsonRequest<PositionDetail>(
      "/api/lens/position",
      post({ run_id: runId, pos, lens, k, track, track_ids: trackIds }),
    ),
  attribution: (
    runId: string,
    pos: number,
    target: { id?: number; str?: string },
    contrast: { id?: number; str?: string } | null,
  ) =>
    jsonRequest<Attribution>(
      "/api/lens/attribution",
      post({
        run_id: runId,
        pos,
        target: target.id ?? null,
        target_str: target.str ?? null,
        contrast: contrast?.id ?? null,
        contrast_str: contrast?.str ?? null,
      }),
    ),
};
