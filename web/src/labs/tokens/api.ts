// Types and fetchers for the Tokenizer lab endpoints (docs/02-api.md §
// Tokenizer lab). Mirrors src/attnlab/toklab.py's return shapes exactly.

import { jsonRequest } from "../../api/client";

export type Algorithm = "byte-bpe" | "sp-bpe" | "wordpiece" | "unigram";

export interface TokenizerInfo {
  id: string;
  label: string;
  hf_name: string;
  algorithm: Algorithm;
  year: number;
  source: "official" | "port";
  /** attnlab model ids that read exactly this vocabulary */
  models: string[];
  blurb: string;
  loaded: boolean;
}

export type TokenKind = "piece" | "byte" | "special" | "unk" | "added";

export interface LabToken {
  index: number;
  id: number;
  /** the raw vocabulary string: Ġ for a space in GPT-2, ▁ in SentencePiece, ## in WordPiece */
  piece: string;
  /** render-safe label; a fragment's first token shows its character, the rest ⋯ */
  display: string;
  /** what this token spells on its own, or null when it's half a character */
  text: string | null;
  byte_hex: string | null;
  /** character offsets into the input text */
  start: number;
  end: number;
  kind: TokenKind;
  /** rank of the BPE merge that created this piece; null for base symbols */
  rank: number | null;
  /** Unigram log-probability of this piece */
  score: number | null;
  cluster: number;
  cluster_size: number;
  cluster_index: number;
  cluster_text: string;
}

export interface LabStats {
  n_chars: number;
  n_graphemes: number;
  n_bytes: number;
  n_words: number;
  n_tokens: number;
  n_special: number;
  n_inserted: number;
  n_fragment_tokens: number;
  n_byte_tokens: number;
  n_unk: number;
  /** every byte of the text is accounted for by exactly the tokens' own bytes */
  lossless: boolean;
  /** decode(encode(text)) === text */
  roundtrip: boolean;
}

export interface Pretoken {
  raw: string;
  display: string;
  start: number;
  end: number;
}

export interface Pipeline {
  normalizers: string[];
  pre_tokenizers: string[];
  split_patterns: string[];
  model: string;
  byte_fallback: boolean;
  n_merges: number;
  decoders: string[];
  normalized: string;
  normalized_changed: boolean;
  pretokens: Pretoken[];
  pretokens_truncated: boolean;
}

export interface AnalyzeResult {
  tokenizer: string;
  tokens: LabToken[];
  stats: LabStats;
  decoded: string;
  pipeline: Pipeline;
}

export interface TraceStep {
  rank: number | null;
  left: string;
  right: string;
  merged: string;
  /** indices in `symbols` that this step produced */
  at: number[];
  symbols: string[];
  /** WordPiece only: how many shorter candidates were tried and rejected */
  tried?: number;
}

export interface TraceWord {
  raw: string;
  display: string;
  initial: string[];
  steps: TraceStep[];
  final: string[];
  final_ids: (number | null)[];
  actual: string[];
  scores: (number | null)[] | null;
  verified: boolean;
  notes: string[];
}

export interface TraceResult {
  tokenizer: string;
  algorithm: Algorithm;
  n_merges: number;
  words: TraceWord[];
  truncated: boolean;
}

export interface TextStats {
  n_chars: number;
  n_graphemes: number;
  n_bytes: number;
  n_words: number;
}

export interface CountResult {
  texts: TextStats[];
  results: { tokenizer: string; counts: number[] }[];
}

export interface VocabRow {
  id: number;
  piece: string;
  text: string | null;
  byte_hex: string | null;
  n_bytes: number | null;
  kind: TokenKind;
  script: string;
  display: string;
  rank: number | null;
}

export interface VocabSummary {
  tokenizer: string;
  size: number;
  n_rows: number;
  n_merges: number;
  kinds: Partial<Record<TokenKind, number>>;
  by_script: { script: string; count: number }[];
  longest: VocabRow[];
  special: VocabRow[];
}

export interface VocabSearchResult {
  tokenizer: string;
  query: string;
  total: number;
  results: VocabRow[];
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const toklab = {
  tokenizers: () => jsonRequest<{ tokenizers: TokenizerInfo[] }>("/api/tokenizers"),
  analyze: (tokenizers: string[], text: string, addSpecialTokens: boolean) =>
    jsonRequest<{ results: AnalyzeResult[] }>(
      "/api/toklab/analyze",
      post({ tokenizers, text, add_special_tokens: addSpecialTokens }),
    ),
  trace: (tokenizer: string, text: string) => jsonRequest<TraceResult>("/api/toklab/trace", post({ tokenizer, text })),
  count: (tokenizers: string[], texts: string[]) =>
    jsonRequest<CountResult>("/api/toklab/count", post({ tokenizers, texts })),
  vocab: (tokenizer: string) =>
    jsonRequest<VocabSummary>(`/api/toklab/vocab?${new URLSearchParams({ tokenizer })}`),
  vocabSearch: (tokenizer: string, q: string, script?: string) => {
    const qs = new URLSearchParams({ tokenizer, q });
    if (script) qs.set("script", script);
    return jsonRequest<VocabSearchResult>(`/api/toklab/vocab/search?${qs}`);
  },
};
