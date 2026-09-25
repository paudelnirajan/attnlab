# attnlab

An interactive, hosted playground for exploring attention patterns in
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) models —
type a prompt, pick a model, and see what every attention head does. Inspired by
[ARENA 3.0 chapter 1.2](https://arena-chapter1-transformer-interp.streamlit.app/)
("Intro to Mech Interp"): no more open-Colab-wait-edit-cell-rerun loop.

**Status:** Stages 0a–1 complete (measurement harness, backend API, frontend
viewer). The induction lab (head scoring, ablation) from Stage 2 is not built
yet — see `docs/PLAN.md` for the full roadmap.

## What it does today

attnlab is a set of **labs**, one per question, meant to be walked in order. Each one links to the
next with the same text (`docs/03-decisions.md` D13). Open `http://127.0.0.1:5173/` for the path.

| Step | Lab | URL |
|---|---|---|
| 1 | **Tokenizer lab**: what does the model actually read? | `/tokens` |
| 2 | **Attention patterns**: where does each token look? | `/attention` |
| 3–5 | Induction heads, logit lens, ablation | planned |

### Tokenizer lab

It runs on tokenizers alone, with no model loaded, across 9 tokenizers: GPT-2, GPT-NeoX/Pythia,
BLOOM, Qwen 2.5, Llama 2, GPT-4's cl100k, GPT-4o's o200k, BERT and XLM-R. Its five views:

- **Inspect**
  - Token chips shown as text, ids, bytes or raw vocabulary strings.
  - Hover a token for its bytes, the merge that made it, and where it came from in your text.
  - Four lengths of the same text: characters you see, code points, bytes, tokens.
  - The full normalize → pre-tokenize → model → ids pipeline.
  - A gallery of 16 quirks, including leading spaces, digits, NFC/NFD, homoglyphs, glitch tokens and
    special-token injection.
- **Compare**: one text through up to six tokenizers, stacked.
- **Languages**: the same FLORES+ sentences in 32 languages, with each tokenizer's premium over
  English. Under GPT-2, Burmese costs ×16.
- **BPE step-through**: replays the real merges one at a time, checked against the real tokenizer.
- **Vocabulary**: search, and how many tokens each script got. GPT-2 has one Devanagari token.

### Attention patterns

- Type a prompt, pick a model, see per-layer/per-head attention heatmaps render
  live, backed by a real `HookedTransformer` forward pass
- Toggle between reading a head as "where does this token look *from*"
  (destination → source, a softmax row) vs. "where does attention *land*"
  (source → destination, a raw column — does **not** sum to 1, this is
  deliberate; see `docs/03-decisions.md` D9)
- Hover a token to link it across the token strip and the heatmap; click to pin
- Non-Latin scripts (Devanagari, etc.) render as their actual characters
  instead of `�` boxes, with byte-fragment provenance on hover — see
  `docs/03-decisions.md` D12
- A cost panel showing the exact, deterministic memory/FLOP cost of the
  current prompt — computed from tensor shapes, not measured, so it's
  identical on every machine
- Permalinks: model + prompt + layer + head + direction all live in the URL
- Every attention matrix is served over a custom compact binary format
  (`docs/01-wire-format.md`) — triangle-packed, sqrt-companded, uint8 — instead
  of raw float32, because naive attention for `gpt2-small` at 512 tokens is
  151 MB and nobody's browser needs that

## Architecture

```
Browser (React + Vite + TS, canvas rendering)
   │  fetch, same-origin via Vite's dev proxy
   ▼
FastAPI backend (single uvicorn worker, one global semaphore)
   │
   ├─ zoo.py        LRU model cache, enforced RAM budget, refuses rather than OOMs
   ├─ inference.py  tokenize (via model.to_tokens) + run_with_cache (names_filter'd
   │                to attention patterns only) + per-token loss + top-k logits
   ├─ patterns.py   compress: triangle-pack → sqrt-compand → uint8 → ATNP binary
   └─ api/          routes, request/response schemas, run cache (10 min TTL)
   │
   ▼
HookedTransformer (TransformerLens, pinned 2.18.0 — matches ARENA's pin so
                    your numbers reproduce the notebook's exactly)
```

Read `docs/00-overview.md`-equivalent content in `docs/PLAN.md` for the full
design rationale, and `docs/03-decisions.md` for a running log of decisions
and bugs found (with root causes) along the way.

## Prerequisites

| Tool | Version used here | Check |
|---|---|---|
| Python | 3.11.x (`.python-version` pins `3.11`) | `python3 --version` |
| [uv](https://docs.astral.sh/uv/) | 0.10+ | `uv --version` |
| Node.js | 20+ (tested on 26) | `node --version` |
| npm | ships with Node | `npm --version` |
| Docker | only needed for Mode B (see below) | `docker --version` |

No GPU required — every baked-in model runs a forward pass on 2 CPU cores in
well under a second.

## Quickstart

**1. Install dependencies (first time only):**

```bash
# backend — uv creates and manages the virtualenv for you
uv sync

# frontend
cd web && npm install && cd ..
```

**2. Start the backend** (terminal 1, from the repo root):

```bash
make dev
```

This runs `uvicorn attnlab.api.app:app --reload --port 8000` with
`MI_DEVICE=mps` (Apple Silicon acceleration for local iteration — see
"Two modes" below). `--reload` restarts automatically on any source change.

**3. Start the frontend** (terminal 2):

```bash
cd web
npm run dev
```

**4. Open the app:** [http://localhost:5173](http://localhost:5173)

Vite proxies every `/api/*` request to the backend on port 8000
(`web/vite.config.ts`), so the browser only ever talks to one origin — no CORS
configuration needed.

**Sanity check the backend directly, if you want:**

```bash
curl http://localhost:8000/api/health
# {"ok": true, "device": "mps", "budget": {...}, "resident_models": [...], ...}
```

## Stopping it

`Ctrl+C` in each terminal. No daemons, no background processes, nothing else
to clean up.

## Two modes: native vs. constrained container

Docker Desktop on macOS runs Linux VMs, so Apple Silicon's MPS acceleration is
**not available inside a container** — you can't have fast MPS and enforced
memory limits in the same process. This project develops in two modes,
switched entirely by environment variable, and only trusts numbers from the
constrained one.

| | Mode A — `make dev` | Mode B — `make dev-docker` |
|---|---|---|
| Purpose | fast local iteration | **ground truth** — the numbers you'd actually get in a small, resource-capped deployment |
| Device | `mps` | `cpu`, forced |
| Threads | all cores | 2 (`--cpus=2`, and `settings.py` forces torch to actually respect the container's CPU quota) |
| Memory | none | `--memory=6g --memory-swap=6g` |
| Frontend | still `npm run dev` in terminal 2 | still `npm run dev` in terminal 2 — the frontend isn't containerized yet |

```bash
make dev-docker
```

builds `docker/Dockerfile.api` and runs it with the limits above, `--rm` so it
self-removes on `Ctrl+C`. Model weights land in a named Docker volume
(`attnlab-hf-cache`) shared with the benchmark target, so they're downloaded
once, not once per run.

```bash
make clean-docker-cache   # force a re-download if you ever need to
```

## Running the tests

```bash
make test                  # Python: pytest, from the repo root
cd web && npm run test     # frontend: vitest (jsdom)
cd web && npm run shots    # Playwright visual + geometry regression
                            # (needs the dev servers running — catches
                            # rendering bugs jsdom structurally cannot,
                            # see docs/03-decisions.md D10)
cd web && npm run typecheck
```

## Which models are available

Defined in `src/attnlab/models.yaml` — adding a model is a data change, never
a code change.

| Model | Params | Layers × Heads | Tier | Notes |
|---|---|---|---|---|
| Attn-Only 2L | 54M | 2 × 8 | baked | ARENA 1.2's own induction-head demo model — no MLPs |
| GPT-2 Small | 163M | 12 × 12 | baked | the ARENA default |
| Pythia 160M | 162M | 12 × 12 | baked | EleutherAI, trained on the Pile |
| GPT-2 Medium | 406M | 24 × 16 | lazy | fetched from the Hub on first use |
| BLOOM 560M | 818M | 24 × 16 | lazy | 46-language multilingual probe; **RAM figure not yet verified in Mode B** — see the comment above its entry in `models.yaml` |

"baked" models are meant to be pre-downloaded into a deploy image for instant
cold starts; locally, both tiers download on first use into the same HF cache.

## Project layout

```
src/attnlab/
├── settings.py     env-driven config, the Mode A/B switch
├── zoo.py          LRU model cache with an enforced RAM budget
├── inference.py    tokenization + forward pass + activation cache extraction
├── patterns.py     the ATNP binary compression format
├── registry.py     loads models.yaml
├── models.yaml     the model catalog — data, not code
├── instrument.py   measurement harness (Stage 0a)
├── bench.py        benchmark driver
└── api/
    ├── app.py, routes.py, schemas.py, state.py, errors.py, meta.py

web/src/
├── api/            fetch client + binary (ATNP) decoder
├── render/         canvas heatmap renderer
├── state/          zustand store, URL-as-source-of-truth
├── components/     React components
└── lib/            direction semantics, token display helpers

docs/
├── PLAN.md              full build plan, stage-by-stage
├── 01-wire-format.md    the ATNP binary contract
├── 02-api.md            endpoint reference
└── 03-decisions.md      running decision + bug log
```

## Further reading

- `docs/PLAN.md` — the full build plan: locked decisions, stage gates, and
  what's deliberately deferred
- `docs/01-wire-format.md` — the binary attention-pattern wire format
- `docs/02-api.md` — every endpoint's request/response shape
- `docs/03-decisions.md` — why things are built the way they are, plus a log
  of real bugs found (with root causes) and how the test suite was extended
  to catch them permanently
