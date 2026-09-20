# API reference

FastAPI backend. All JSON bodies; one binary endpoint (see
[`01-wire-format.md`](01-wire-format.md)).

Base path `/api`. Stage in which each endpoint lands is marked.

---

## Conventions

### `_meta` on every JSON response

Every JSON response carries a `_meta` object built from the Stage 0a instrumentation record. It feeds
the public cost panel and the `?debug=1` overlay from one source.

```json
{
  "_meta": {
    "op": "forward+cache",
    "model": "gpt2-small",
    "seq": 256,
    "duration_ms": 312.4,
    "bytes_out": 803423,
    "tl_version": "2.18.0",
    "device": "cpu",
    "dtype": "float32"
  }
}
```

`tl_version`, `device` and `dtype` are always present — they are what lets a user debug a mismatch
against their own notebook. Process metrics (`rss_*`, `cgroup_*`, `cpu_throttled_us`, `threads`) are
included **only** when the server runs with `MI_DEBUG_METRICS=1`.

### Errors

```json
{"error": {"code": "seq_too_long", "message": "...", "detail": {"max_seq": 512, "got": 900}}}
```

| code | HTTP | when |
|---|---|---|
| `unknown_model` | 404 | not in `models.yaml` |
| `model_disabled` | 403 | `tier: disabled` — message says why |
| `seq_too_long` | 422 | exceeds the model's `max_seq` |
| `run_not_found` | 404 | run id unknown or TTL-expired |
| `budget_exceeded` | 503 | model cannot be loaded within `MI_RAM_BUDGET_GB` |
| `busy` | 503 | queue full / request timed out |

`budget_exceeded` is a deliberate, visible failure. The LRU **refuses** rather than swapping, so the
ceiling is observable instead of manifesting as mysterious slowness.

### Limits

- `max_seq` per model, enforced server-side; never trust the client
- one forward pass at a time (`asyncio.Semaphore(1)`), torch runs in a thread executor
- request timeout; per-IP rate limit added in Stage 4

---

## `GET /api/models` — Stage 0b

Catalog from `models.yaml` plus live residency.

```json
{
  "models": [
    {
      "id": "attn-only-2l-demo",
      "label": "Attn-Only 2L (ARENA 1.2)",
      "n_layers": 2, "n_heads": 8, "d_model": 512,
      "n_params": 3145728,
      "max_seq": 512,
      "languages": ["en"],
      "tier": "baked",
      "status": "resident",
      "est_ram_mb": 60,
      "blurb": "Two attention layers, no MLPs. The ARENA 1.2 workhorse."
    }
  ],
  "budget": {"limit_mb": 6144, "used_mb": 60}
}
```

`status`: `resident` | `available` | `downloading` | `disabled`.
For `disabled`, a `reason` field explains why (too large, gated, needs GPU) — the picker shows it
rather than silently greying the row out.

---

## `POST /api/tokenize` — Stage 0b

No forward pass. Must stay under 50 ms so the UI can call it on every keystroke.

```json
{"model": "gpt2-small", "text": "The cat sat on the mat"}
```

```json
{
  "tokens": [
    {"id": 464, "str": "The", "display": "The", "start": 0, "end": 3,
     "is_byte_fallback": false, "cluster": 1, "cluster_size": 1, "cluster_index": 0,
     "cluster_text": "The", "byte_hex": "546865"}
  ],
  "n_tokens": 6,
  "max_seq": 512,
  "_meta": {}
}
```

- `str` — raw decoded token. For a byte fragment this is U+FFFD and is **not** renderable; use
  `display`.
- `display` — render-safe. Whitespace is made visible (`·` space, `⏎` newline, `→` tab). For a
  fragment, the first token of a cluster shows the character(s) the cluster covers and the rest
  show `⋯`, so a strip of fragments never renders as a row of `�`.
- `start` / `end` — **character** offsets into the input, for text-to-heatmap linking. Taken from
  the fast tokenizer's offset mapping when it reproduces TransformerLens's id sequence exactly, and
  from a cursor search otherwise. Several tokens of one character all report that character's span.
- `cluster` / `cluster_size` / `cluster_index` / `cluster_text` — a **cluster** is a run of
  consecutive tokens whose spans overlap, i.e. that together cover one indivisible piece of source
  text. `cluster_size == 1` is the normal case. `cluster_size > 1` means the tokenizer had no merge
  for this character and spent several tokens on it; `cluster_text` is what those tokens jointly
  spell. See `docs/03-decisions.md` D12.
- `byte_hex` — the raw UTF-8 bytes this one token contributes, lowercase hex. `null` for special
  tokens and for tokenizer families we can't decompose. This is the only fully truthful thing that
  can be said about a fragment that isn't a character on its own.
- `is_byte_fallback` — true when the token is an incomplete UTF-8 fragment *or* belongs to a
  multi-token cluster. **This drives the multilingual fragmentation story in Stage 3 — do not drop
  it.**

---

## `POST /api/run` — Stage 0b

Runs the forward pass with `names_filter` restricted to attention patterns, caches the result
server-side (TTL ~10 min), and returns everything *except* the patterns.

```json
{"model": "gpt2-small", "text": "...", "top_k": 5}
```

Or, for the Stage 2 induction probe, in place of `text`:

```json
{"model": "gpt2-small", "repeated": {"length": 25, "seed": 42, "prepend_bos": true}}
```

Response:

```json
{
  "run_id": "r_9f3a2c",
  "tokens": [],
  "n_layers": 12, "n_heads": 12,
  "loss_per_token": [3.21, 2.05],
  "top_logits": [[{"id": 262, "str": " the", "logit": 12.4, "prob": 0.31}]],
  "cost": {
    "attention_bytes_f32": 37748736,
    "kv_cache_bytes": 12582912,
    "weights_bytes": 497000000,
    "forward_flops": 63000000000
  },
  "expires_at": "2026-09-19T12:45:00Z",
  "_meta": {}
}
```

The `cost` block is **computed from shapes, not measured** — exact, reproducible, identical on every
machine. It is what the public cost panel renders.

Seeded `repeated` generation means a permalink reproduces the exact same random sequence.

---

## `GET /api/run/{run_id}/patterns?layers=0,3` — Stage 0b

**Binary.** See [`01-wire-format.md`](01-wire-format.md).

- `layers` — comma-separated, required. Omitting it is an error, not "send everything"; that default
  would be a footgun worth 151 MB.
- Returns `404 run_not_found` once the TTL expires — the client re-runs.
- `Content-Type: application/octet-stream`, gzip via middleware.

---

## `GET /api/health` — Stage 0b

```json
{
  "ok": true,
  "tl_version": "2.18.0",
  "device": "cpu",
  "budget": {"limit_mb": 6144, "used_mb": 1540},
  "resident_models": ["gpt2-small"],
  "queue_depth": 0
}
```

---

## `POST /api/head-scores` — Stage 2

All four score matrices from **one** forward pass over a repeated sequence.

```json
{"model": "gpt2-small", "repeated": {"length": 25, "seed": 42}}
```

```json
{
  "run_id": "r_4b81de",
  "seq_len": 25,
  "scores": {
    "induction":      [[0.01, 0.03]],
    "previous_token": [[0.87, 0.02]],
    "duplicate_token":[[0.00, 0.01]],
    "current_token":  [[0.12, 0.44]]
  },
  "top_heads": {"induction": [{"layer": 1, "head": 4, "score": 0.82}]},
  "_meta": {}
}
```

Each matrix is `n_layers x n_heads`. Definitions:

| score | diagonal offset |
|---|---|
| induction | `seq_len - 1` |
| previous_token | `1` |
| duplicate_token | `seq_len` |
| current_token | `0` |

Computed as the mean of the attention values on that diagonal.

> **These numbers are the tool's credibility.** `tests/test_golden.py` pins them against the ARENA
> 1.2 notebook's own output for `attn-only-2l-demo`. If a user's Colab disagrees with the site, they
> will conclude the site is broken — and be right.

---

## `POST /api/ablate` — Stage 2

```json
{
  "model": "gpt2-small",
  "text": "...",
  "ablations": [{"layer": 1, "head": 4, "mode": "zero"}],
  "target_position": 12
}
```

`mode`: `zero` | `mean`. Mean ablation uses the per-position mean over the current batch; state the
baseline used in the response so the number is interpretable.

```json
{
  "baseline_loss": 3.21,
  "ablated_loss": 4.87,
  "delta_loss": 1.66,
  "delta_logit": -2.4,
  "mean_baseline": "batch",
  "_meta": {}
}
```

Ablation state belongs in the permalink URL.

---

## `POST /api/translate` — Stage 3, deferred

Not in v1. Stage 3 ships **curated parallel prompts** served as static JSON; live translation is a
v1.1 decision gated on Stage 0a's RAM headroom. When it lands:

```json
{"text": "...", "target_lang": "ne_NP", "source_lang": "auto"}
```

---

## Endpoint summary

| Endpoint | Stage | Response |
|---|---|---|
| `GET /api/models` | 0b | JSON |
| `POST /api/tokenize` | 0b | JSON, < 50 ms |
| `POST /api/run` | 0b | JSON |
| `GET /api/run/{id}/patterns` | 0b | **binary** |
| `GET /api/health` | 0b | JSON |
| `POST /api/head-scores` | 2 | JSON |
| `POST /api/ablate` | 2 | JSON |
| `POST /api/translate` | 3+ | JSON, deferred |
