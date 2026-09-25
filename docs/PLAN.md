# attnlab — Build Plan

An interactive, hosted playground for exploring attention patterns and induction heads
in TransformerLens models.

**Companion docs**
- [`01-wire-format.md`](01-wire-format.md) — binary attention wire format (fix before any UI code)
- [`02-api.md`](02-api.md) — endpoint reference
- [`03-decisions.md`](03-decisions.md) — locked decisions, rationale, open items
- [`../FEASIBILITY.md`](../FEASIBILITY.md) — background research, landscape, cost analysis

---

## Context

**The problem.** Working through ARENA 3.0 chapter 1.2 ("Intro to Mech Interp"), there is no hosted
tool where you can type a prompt, pick a model, and *see* what the attention heads do. Every existing
option falls short: CircuitsVis is a library not an app; Neuroscope won't take your own prompt;
BertViz is notebook-bound; Transformer Explainer teaches architecture on a fixed model with no
induction-head tooling. The workflow today is "open Colab, wait, edit a cell, re-run" — a bad loop
for building intuition.

**The outcome we want.** A public web app where you type a prompt, pick a model, and immediately see
tokenization, per-head attention heatmaps, induction scores, and the effect of ablating a head — with
a resource/cost panel that makes the quadratic blow-up of attention viscerally obvious. Fast enough
to feel like play.

**Why it's tractable.** The models that matter pedagogically are tiny. `attn-only-2l-demo`,
`gpt2-small`, and `pythia-160m` all run a cached forward pass on 2 CPU cores in well under a second.
**No GPU is required.** The real engineering problem is data volume, not ML: attention is
`layers x heads x seq x seq`, which is 151 MB of float32 for `gpt2-small` at 512 tokens. Solvable —
but only if it is designed in from the first commit.

---

## Two-mode development

Docker Desktop on macOS runs Linux VMs — **MPS is not available inside a container.** So MPS speed
and enforced resource limits cannot coexist. Two modes, one codebase, switched by env var only.

| | Mode A — native | Mode B — container |
|---|---|---|
| Purpose | fast iteration | **truth** |
| Command | `make dev` | `make dev-docker` / `make bench-docker` |
| `MI_DEVICE` | `mps` | `cpu` |
| `MI_THREADS` | 12 | 2 |
| `MI_RAM_BUDGET_GB` | 12 | 6 |
| Limits | none | `--memory=16g --memory-swap=16g --cpus=2` |

**Rule:** Mode A is for feeling productive. Mode B is for anything you quote, screenshot, commit as a
benchmark, or make a decision on.

### Two gotchas that silently corrupt measurements

1. **`--cpus=2` uses CFS quota, not affinity.** `os.cpu_count()` inside the container still returns
   12, so torch spawns 12 OMP threads that thrash against 2 cores of quota. **Must** set
   `torch.set_num_threads()` and `OMP_NUM_THREADS` explicitly.
2. **`psutil.virtual_memory()` reads `/proc/meminfo`** = the VM's memory, not the cgroup limit. It
   reports plenty of headroom right up until the OOM killer fires. Read cgroup v2 files instead.

### Architecture

Build the container `linux/arm64` (native). Memory figures transfer across architectures; latency
under `--platform linux/amd64` Rosetta emulation is fiction. Multi-arch is a Stage 4 concern only.

---

## Target repo layout

```
mi/
├── FEASIBILITY.md
├── README.md
├── Makefile
├── pyproject.toml              uv, py3.11, pinned deps
├── .python-version
├── docker/
│   ├── Dockerfile.bench        stage 0a
│   ├── Dockerfile.api          stages 0b+
│   └── Dockerfile              stage 4, multi-stage, bakes models + built frontend
├── docs/
│   ├── PLAN.md                 this file
│   ├── 01-wire-format.md       binary contract
│   ├── 02-api.md               endpoint reference
│   ├── 03-decisions.md         decisions + open items
│   └── 04-benchmarks.md        generated from bench results
├── src/attnlab/
│   ├── settings.py             env-driven config, the two-mode switch
│   ├── instrument.py           measurement context manager + cgroup probes
│   ├── registry.py             model registry loader
│   ├── models.yaml             model catalog (data, not code)
│   ├── zoo.py                  load + LRU cache with enforced byte budget
│   ├── patterns.py             extract -> compand -> quantize -> pack -> encode
│   ├── analysis.py             induction / prev-token / dup-token scores, ablation
│   ├── translate.py            stage 3
│   ├── bench.py                stage 0a driver
│   └── api/
│       ├── app.py              FastAPI app, queue, lifespan
│       ├── routes.py
│       └── schemas.py          pydantic models
├── web/
│   ├── package.json            vite + react + ts
│   └── src/
│       ├── api/                fetch client + binary decoder
│       ├── render/             canvas heatmap renderer
│       ├── state/              URL-as-source-of-truth store
│       └── components/
├── tests/
│   ├── test_patterns.py        round-trip, packing, error bounds
│   ├── test_zoo.py             budget enforcement, eviction
│   └── test_golden.py          numbers match ARENA notebook
└── bench_results/
    ├── results.json
    └── RESULTS.md
```

---

## Contracts to fix early

Three things are expensive to change later. Everything else is cheap to refactor.

1. **Binary wire format** → [`01-wire-format.md`](01-wire-format.md)
2. **Model registry schema** → below
3. **Instrumentation record** → below

### Model registry schema (`src/attnlab/models.yaml`)

```yaml
- id: attn-only-2l-demo
  label: "Attn-Only 2L (ARENA 1.2)"
  tl_name: attn-only-2l-demo
  tier: baked              # baked | lazy | disabled
  max_seq: 512
  languages: [en]
  est_ram_mb: 60
  blurb: "Two attention layers, no MLPs. The ARENA 1.2 workhorse."
```

Adding a model must be a **data** change, never a code change. `tier` drives both the Dockerfile's
pre-download list and the greyed-out entries in the model picker.

### Instrumentation record

One shape, three consumers: the benchmark table, the `_meta` block on every API response, and the
debug overlay.

```json
{"op":"forward+cache","model":"gpt2-small","seq":256,
 "duration_ms":312.4,"rss_delta_mb":41.2,"peak_rss_mb":1180.5,
 "cgroup_current_mb":1402.1,"cgroup_peak_mb":1533.0,
 "cpu_throttled_us":0,"threads":2,"bytes_out":803423}
```

---

## Stage 0a — Measurement harness

**Goal:** replace every estimate in `FEASIBILITY.md` §5 with a measured number, and build intuition
for the resource envelope before any product code exists.
**Effort:** 1–2 h of work + download time.

**Build:** `settings.py`, `instrument.py`, `patterns.py`, `registry.py` + `models.yaml`, `bench.py`,
`docker/Dockerfile.bench`, `Makefile`.

### `instrument.py` specifics

- `with measure("forward+cache", model=..., seq=...) as m:` → populates the record above
- background sampler thread at 20 ms for peak memory
- cgroup v2 reads: `memory.current`, `memory.max`, `memory.peak` (kernel >= 5.19 — probe for it),
  `cpu.stat` → `usage_usec` **and `throttled_usec`** (the only clear CPU-starvation signal)
- returns `None` cleanly for cgroup fields when running natively on macOS
- `resource.getrusage().ru_maxrss` is **bytes on macOS, kilobytes on Linux** — normalize, or every
  Mac number is 1024x wrong

### Benchmark matrix

5 models x 4 sequence lengths, run in both modes.

| Model | TL name | ~download |
|---|---|---|
| Attn-Only 2L | `attn-only-2l-demo` | ~50 MB |
| GPT-2 small | `gpt2-small` | ~500 MB |
| Pythia 160m | `pythia-160m` | ~650 MB |
| GPT-2 medium | `gpt2-medium` | ~1.4 GB |
| BLOOM 560m | `bloom-560m` | ~2.2 GB |

Sequence lengths **64 / 128 / 256 / 512**. First-run download ~**4.8 GB**, into a named Docker volume
so it happens once.

Per cell, measure:
- cold load time + RSS delta
- forward with `names_filter` pattern-only
- forward with full cache (once per model, to quantify the saving)
- encode sizes at f32 / f16 / u8 / u8+sqrt+triangle / +gzip
- peak memory throughout

**Deliverable:** `bench_results/RESULTS.md`.

### Questions it settles

1. Real RSS per model → how many fit in 6 GB → final `tier` values in `models.yaml`
2. Whether `names_filter` saves what FEASIBILITY §5 claims
3. Whether the wire-format size math holds
4. Honest p50 latency at 2 threads, and whether CPU throttling kicks in
5. Warm-disk load time → the cold-start number
6. Mode A ÷ Mode B ratio → how much to discount native measurements

### Verify

`make bench-docker` completes without OOM; `RESULTS.md` has no empty cells; sqrt-u8 round-trip
meets the error bounds in [`01-wire-format.md`](01-wire-format.md#acceptance-criteria-teststest_patternspy)
(relative error <= 4% for p >= 0.01, absolute error <= 0.004 everywhere).

### Options

| Option | Verdict |
|---|---|
| 5 models as listed | **Now** |
| Drop `bloom-560m` to halve the download | Later — but its multilingual RAM cost is a Stage 3 blocker, better to know now |
| Add bf16 comparison | **Later.** TL weight processing is numerically sensitive, and x86 without AMX is slower in bf16. Revisit only on GPU. |
| Add `gpt2-large` / `gpt2-xl` | **Later**, once the budget picture is clear |
| Skip 0a, go straight to the API | **No.** This is the stage that de-risks everything else. |

---

## Stage 0b — Backend skeleton

**Goal:** a running API serving real patterns over the wire format.
**Effort:** ~1 day.

**Build:** `zoo.py` (LRU + enforced byte budget — *refuses* to exceed rather than swapping, so the
ceiling is visible), `api/app.py`, `api/routes.py`, `api/schemas.py`, `docker/Dockerfile.api`.

Endpoints → [`02-api.md`](02-api.md).

**Concurrency:** single uvicorn worker, `asyncio.Semaphore(1)` around forward passes, torch call in a
thread executor so the event loop stays responsive. Hard `max_seq` enforced server-side per model.
Request timeout.

### Options

| Option | Verdict |
|---|---|
| Single worker + semaphore | **Now.** Correct at this scale, trivially understood. |
| Multi-worker with shared model memory | **Never at this size** — each worker gets its own copy; this *multiplies* RAM. |
| Separate inference process + IPC | Later, only if a real queueing problem appears |
| Redis for run cache | **Later.** In-process dict with TTL is right for one worker. |
| Streaming responses | Later |

### Verify

`curl` each endpoint in the constrained container; load 3 models in sequence and confirm the LRU
evicts instead of OOMing; `/api/tokenize` p50 < 50 ms.

---

## Stage 1 — Tier-1 viewer

**Goal:** the core loop — type a prompt, see the heads. Already better than anything hosted today.
**Effort:** ~1 week.

### Components

- **Prompt box** — debounced, token counter, over-limit warning
- **Model picker** — from `/api/models`, greys out disabled tiers *with the reason*
- **Tokenizer panel** — token chips with visible whitespace (`·` for space, `⏎` for newline), IDs on
  hover, byte-fallback indicator. Clicking a chip sets the selected position globally.
- **Layer strip → head grid** — all heads of a layer as small canvases; click to expand one
- **Head detail** — full heatmap, axis labels, hover crosshair, top-attended-tokens list
- **Hover linking** — hovering a destination token highlights its source distribution both on the
  heatmap and inline in the text. *This is the feature that makes it feel alive.*
- **Keyboard nav** — `left/right` head, `up/down` layer, `/` focus prompt, `?` shortcuts
- **Permalink** — model + prompt + layer + head in the URL; the URL is the single source of truth
- **Cost panel v1** — deterministic numbers only (see Stage 1.5)

### Rendering

One `<canvas>` per head, `putImageData` from the decoded `Uint8Array`,
`imageSmoothingEnabled = false` for nearest-neighbour upscale. A 512x512 heatmap is 262k cells — DOM
or SVG will not survive this.

| Rendering option | Verdict |
|---|---|
| Canvas 2D + `putImageData` | **Now.** Simple, fast enough to 512. |
| WebGL / regl | **Later**, only if contexts > 1024 are wanted |
| SVG | **Never** |
| CircuitsVis components | **Evaluate, probably fork.** Right primitives, but we need hover-linking and the cost panel it doesn't have. Read its source before writing ours. |

| State option | Verdict |
|---|---|
| URL + small store (zustand) | **Now** |
| Redux / heavy state lib | Never at this size |
| Server-side sessions | Never — permalinks must be stateless |

### Verify

Type a prompt, hover a token, see linked highlighting; copy the URL into a fresh tab and get the
identical view; run in Mode B and confirm it still feels responsive.

> **Containerize here, not at Stage 4.** Deploy something ugly but real at the end of Stage 1 so
> shipping becomes a habit rather than a cliff.

---

## Stage 1.5 — Cost panel

The pedagogical differentiator. **Two renderers, one instrumentation source — keep them strictly
separate.**

### Public: deterministic, computed from shapes

Exact, reproducible, identical on every machine:

- `12 layers x 12 heads x 256^2 x 4 B = 37.7 MB` of attention for this prompt
- bytes actually sent to your browser after sqrt+u8+triangle+gzip, with the compression ratio
- KV cache size at this context length: `2 x L x d_model x S x 4 B`
- resident model weights
- forward FLOPs ~= `2*N*S` + attention `4*L*S^2*d_model`
- **a live "double your prompt → 151 MB" projection next to the seq-length slider**

That last item is the lesson. Dragging a slider and watching a number rise quadratically teaches the
quadratic in a way reading `O(n^2)` does not. Nothing else in interp tooling does this.

### Dev-only, behind `?debug=1`

Live process metrics: CPU %, RSS vs budget, `throttled_usec`, thread count, queue depth. Invaluable
to you; teaches a learner nothing generalizable ("1.4 GB RSS" is a fact about who else is on the
box), and leaks capacity information to anyone wanting to knock the public instance over.

> **Scope warning.** This is the most seductive scope creep in the project — fun to build, adjacent
> to real work. The instrumentation already exists from 0a. Build the *public* panel only after the
> viewer works. Half a day then; a week-long detour if started early.

---

## Stage 2 — Induction lab

**Goal:** the differentiator. Maps 1:1 onto ARENA 1.2's exercises.
**Effort:** ~1 week.

- **Repeated-sequence generator** — random tokens repeated twice, configurable length and **seed**
  (seeded, so permalinks reproduce)
- **Head score matrices** — layer x head grids, one per score; click a cell to jump to that head:
  - induction score — mean of the diagonal at offset `seq_len - 1`
  - previous-token score — diagonal offset 1
  - duplicate-token score
  - current-token score
- **Per-token loss curve** over the repeated sequence — the "loss drops in the second half" plot that
  makes induction click
- **Ablation** — zero- or mean-ablate any head or set of heads via hooks; live delta-loss and
  delta-logit on the selected token. Ablation state goes in the URL.

New endpoints: `POST /api/head-scores`, `POST /api/ablate` → [`02-api.md`](02-api.md).

| Option | Verdict |
|---|---|
| Zero + mean ablation | **Now** |
| Resample ablation / activation patching | **Later** — a big feature in its own right |
| Attribution patching | Later |
| Score the full layer x head grid in one pass | **Now** — it's one forward pass; per-head would be wasteful |

### Verify — `tests/test_golden.py`

On `attn-only-2l-demo`, the induction heads the app identifies must match the ones the ARENA 1.2
notebook finds, and the scores must agree to ~3 decimal places. **Run the notebook yourself and
hardcode its output as the fixture.**

This test is what makes the tool trustworthy to its audience. Without it, a user who gets different
numbers than their Colab will assume the tool is broken — and they'd be right to.

---

## Stage 3 — Multilingual

**Goal:** the thing that makes it more than an ARENA companion.
**Effort:** ~3–4 days.

- Multilingual models in the picker: `bloom-560m` (46 languages), `Qwen2.5-0.5B`, with `pythia-160m`
  as the English control
- **Curated parallel prompts** — the same semantic content pre-translated into ~10 languages,
  human-verified
- **Comparison view** — same content, different languages, side by side: token counts, fragmentation
  ratio (tokens per character), induction scores

**The finding users should discover for themselves:** GPT-2's byte-level BPE shreds Devanagari, Thai,
Amharic and similar scripts into per-byte tokens. A word becomes 8 tokens of meaningless byte
fragments, and induction behaviour visibly degrades because there is no stable token to induct on.
That is a real, publishable-flavoured observation a learner can find in five minutes with this tool
and essentially cannot find any other way.

| Translation option | Verdict |
|---|---|
| **Curated parallel prompt sets** | **Now.** Zero RAM, zero deps, zero latency — and *better pedagogy*, since MT errors would confound exactly what users are inspecting. |
| Self-hosted `nllb-200-distilled-600M` | **Later (v1.1).** 200 languages, no API key, offline. Costs 2.4 GB of RAM budget and 2–5 s/sentence on 2 vCPU. Add only if 0a shows headroom. |
| `m2m100_418M` | Later — lighter fallback if NLLB doesn't fit |
| Hosted API (LibreTranslate / DeepL / Google) | **Later or never.** Zero RAM, but adds a key, a rate limit, and an external dependency that can take the site down. |

### Verify

Same prompt in English vs Devanagari shows the expected fragmentation difference; comparison view
renders both.

> **Update 2026-09-25.** The tokenization half of this stage has shipped early, as the **Tokenizer
> lab** (`/tokens`, step 1 of the path; D13, D14). It covers the curated parallel prompts (FLORES+, 32
> languages), the side-by-side comparison, and the fragmentation ratio, across 9 tokenizers including
> BLOOM's and Qwen's. What remains here is the *model* side: loading a multilingual model into the
> attention lab and comparing induction scores across languages.

---

## Stage 4 — Ship

**Effort:** ~3–4 days.

- Multi-stage `Dockerfile`: build frontend → python deps → `snapshot_download` the `tier: baked`
  models into the image (~3 GB) so cold start has no first-user penalty
- Rate limiting per IP, request timeout, `max_seq` cap, Cloudflare in front
- Deploy, then **load test** — concurrency and cold starts are the one thing local development
  cannot de-risk
- README, a 2-minute demo video, an ARENA Slack / LessWrong post

### Decide the deploy target here, informed by real numbers

| Option | Cost | Notes |
|---|---|---|
| **Modal** (`@modal.asgi_app()`) + Cloudflare Pages | $30/mo free credits, likely $0 | Scale-to-zero, ~1–2 s cold start, no 48 h sleep. **Current front-runner.** |
| HF PRO + Docker Space | $9/mo | 2 vCPU / 16 GB, sleeps after 48 h. Worse infra, **much better distribution** — being listed on huggingface.co/spaces reaches exactly this audience. |
| Google Cloud Run | likely free tier | Scale-to-zero, up to 32 GB |
| Fly.io / Railway / Render | $5–20/mo | Render's free tier spins down aggressively |

Also: **apply for an HF community GPU grant** once it's live and good. This is the kind of project
they fund.

### Verify

Deployed URL works from another machine; load test 10 concurrent users without OOM; cold start
measured and stated honestly.

---

## Later / never

| Feature | Verdict |
|---|---|
| Logit lens / direct logit attribution | **Later** — natural Stage 5, high value for the same audience |
| OV/QK circuits, eigenvalue copying score | **Later** — the back half of ARENA 1.2, strong follow-up |
| Composition scores (Q/K/V) between heads | Later, pairs with the above |
| Two-model comparison view | Later |
| Neuron activations | Later — different product surface, big scope |
| Attention head *search* ("find heads that do X") | Later — the most interesting long-term idea here |
| SAE / feature visualization | **Not this project** |
| Models > 3 GB | Needs a GPU tier — different cost structure |
| Gated models (Llama, Gemma, Mistral) | **Not in v1** — needs `HF_TOKEN` + per-user license acceptance, a whole category of support burden |
| User accounts, saved sessions | **Never.** Permalinks do this job statelessly. |
| In-browser transformers.js backend | **Later, maybe never.** Zero server cost and infinite scale, but no TL weight processing (so numbers won't match ARENA) and hooks are hard. Keep the API boundary clean so it stays possible. |
| TransformerLens 3.x `TransformerBridge` | **Later**, behind a flag, as a second loader backend once v1 is stable |

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Attention payload size (151 MB naive) | High | Wire format fixed before any UI code; per-layer fetch; validated in 0a |
| Memory blowup under concurrency | High | Single worker, semaphore, RAM-budgeted LRU that refuses rather than swaps, `names_filter`, `max_seq` cap |
| Numbers don't match users' Colab output | High | Pin TL 2.18.0, fp32, golden test against notebook output, show version + flags in the footer |
| Mac-vs-prod performance gap (~4–8x) | Medium | Two-mode discipline; only Mode B numbers are quotable |
| Scope creep — "just do everything" | **High** | Stage gates. Tier 1 alone beats everything hosted today. Cost panel held until after the viewer works. |
| Abuse / accidental DoS | Medium | Token cap, per-IP rate limit, timeout, Cloudflare |
| Cold starts / Space sleeping | Medium | Bake tier-1 models; honest "waking up..." state; Modal avoids it |
| `numpy<2` ceiling from TL 2.18 | Low | Check every new dep against it |
| TL 2.x eventually unmaintained (4.0.0b2 exists) | Low now | Pinned, vendored if needed; Bridge backend is the escape hatch |

---

## Execution order

| Stage | Effort | Gate |
|---|---|---|
| 0a Measurement harness | 1–2 h | `RESULTS.md` exists and the numbers are believed |
| 0b Backend skeleton | ~1 day | Patterns served over the wire format |
| 1 Tier-1 viewer | ~1 week | The core loop feels good |
| — containerize + ugly deploy | ~0.5 day | A real URL exists |
| 1.5 Cost panel | ~0.5 day | — |
| 2 Induction lab | ~1 week | Golden test passes |
| 3 Multilingual | ~3–4 days | — |
| 4 Ship properly | ~3–4 days | Public, load-tested, announced |

**~3–4 weeks to a public v1.**

---

## Immediate next step

Build Stage 0a: `settings.py`, `instrument.py`, `patterns.py`, `registry.py` + `models.yaml`,
`bench.py`, `docker/Dockerfile.bench`, `Makefile`. Then `make bench-docker` and react to real numbers.
