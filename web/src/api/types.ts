// Mirrors the JSON shapes in docs/02-api.md exactly. Kept as one file so
// a backend shape change is one diff to review against that doc.

export interface TokenInfo {
  id: number;
  /** raw decoded token — U+FFFD for a byte fragment, so never render this directly */
  str: string;
  /** render-safe: visible whitespace, and cluster-aware for fragments (see docs/02-api.md) */
  display: string;
  /** character offsets into the prompt; fragments of one character share its span */
  start: number;
  end: number;
  is_byte_fallback: boolean;
  /** a run of consecutive tokens covering one indivisible piece of source text */
  cluster: number;
  cluster_size: number;
  cluster_index: number;
  cluster_text: string;
  /** raw UTF-8 bytes this token contributes, lowercase hex; null for special tokens */
  byte_hex: string | null;
}

export type ModelStatus = "resident" | "available" | "downloading" | "disabled";
export type ModelTier = "baked" | "lazy" | "disabled";

export interface ModelInfo {
  id: string;
  label: string;
  n_layers: number;
  n_heads: number;
  d_model: number;
  n_params: number;
  max_seq: number;
  languages: string[];
  tier: ModelTier;
  status: ModelStatus;
  est_ram_mb: number;
  blurb: string;
  reason?: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
  budget: { limit_mb: number; used_mb: number };
}

export interface ApiMeta {
  op: string;
  duration_ms: number;
  tl_version: string;
  device: string;
  dtype: string;
  model?: string;
  seq?: number;
  bytes_out?: number;
  // present only when the server runs with MI_DEBUG_METRICS=1
  rss_delta_mb?: number;
  peak_rss_mb?: number;
  cgroup_current_mb?: number | null;
  cgroup_peak_mb?: number | null;
  cpu_throttled_us?: number | null;
  threads?: number;
}

export interface TokenizeResponse {
  tokens: TokenInfo[];
  n_tokens: number;
  max_seq: number;
  _meta: ApiMeta;
}

export interface TopLogit {
  id: number;
  str: string;
  logit: number;
  prob: number;
}

export interface RunCost {
  attention_bytes_f32: number;
  kv_cache_bytes: number;
  weights_bytes: number;
  forward_flops: number;
}

export interface RunResponse {
  run_id: string;
  tokens: TokenInfo[];
  n_layers: number;
  n_heads: number;
  loss_per_token: number[];
  top_logits: TopLogit[][];
  cost: RunCost;
  expires_at: string;
  _meta: ApiMeta;
}

export interface RepeatedSpec {
  length: number;
  seed: number;
  prepend_bos?: boolean;
}

export interface ApiErrorBody {
  error: { code: string; message: string; detail: Record<string, unknown> };
}

export class ApiError extends Error {
  code: string;
  detail: Record<string, unknown>;
  status: number;

  constructor(status: number, body: ApiErrorBody) {
    super(body.error.message);
    this.status = status;
    this.code = body.error.code;
    this.detail = body.error.detail;
  }
}
