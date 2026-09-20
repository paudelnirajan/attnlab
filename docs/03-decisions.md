# Decisions

What was decided, why, what was rejected, and what is still open.

Format is loosely ADR. When you revisit one, edit it in place and add a dated note at the bottom of
that entry rather than starting a new document.

---

## D1 — Build to a 16 GB / 2 vCPU envelope; defer the deploy target

**Status:** locked · **Revisit:** Stage 4

Design and benchmark against the tightest realistic target (Hugging Face free CPU Basic: 2 vCPU,
16 GB RAM, 50 GB non-persistent disk). Anything that fits there fits everywhere else.

**Why defer the host.** The choice depends on numbers we don't have yet — real RSS per model, real
cold-start time, real concurrency behaviour. Stage 0a produces them. Choosing now would be guessing.

**Consequence.** Every stage is developed in two modes (see `PLAN.md` § Two-mode development), and no
number is quotable unless it came from the constrained container.

**Candidates, revisited at Stage 4:** Modal + Cloudflare Pages (front-runner: $30/mo free credits,
scale-to-zero, no 48 h sleep) · HF PRO Docker Space ($9/mo, worse infra but much better distribution
to this exact audience) · Google Cloud Run · Fly.io.

---

## D2 — Optimize v1 for ARENA 1.2 learners

**Status:** locked · **Revisit:** after v1 ships

Defaults match the notebook (`attn-only-2l-demo`, `gpt2-small`). Terms explained inline. The
induction lab is the centerpiece, not an add-on.

**Why.** It is a real, reachable, currently-underserved audience with a shared vocabulary and a
curriculum the tool can map onto exercise by exercise. "Researchers" is a vaguer target that costs
more (bigger models, GPU tier) and is harder to win. "General ML-curious public" puts us in direct
competition with Transformer Explainer, which is polished and well-funded — a crowded space rather
than a gap.

Researchers are a natural Stage 5+ expansion; nothing here forecloses it.

---

## D3 — Legacy `HookedTransformer`, `transformer-lens==2.18.0`

**Status:** locked · **Revisit:** after v1 stable

**Evidence gathered 2026-09-19:**

| Fact | Source |
|---|---|
| ARENA 3.0 pins `transformer_lens>=2.16.1,<3.0.0` | `ARENA_3.0/requirements.txt` |
| Last 2.x release is **2.18.0** | PyPI release index |
| TL **3.9.0** (2026-09-12) makes `TransformerBridge` the recommended interface and marks `HookedTransformer.from_pretrained` **deprecated** | PyPI / TL docs |
| TL **4.0.0b2** exists (2026-09-02) | PyPI |
| `attn-only-2l-demo` is a valid alias in 2.18.0's model table | `loading_from_pretrained.py` @ v2.18.0 |

**Why 2.x.** The audience (D2) will compare our numbers against their own Colab output. Matching
ARENA's pin exactly is the cheapest way to guarantee they agree. TransformerBridge's 15,000-model zoo
is impressive and irrelevant to v1 — we ship five models.

**Cost of this choice.** We sit on a deprecated API while 3.x and 4.x move on. Mitigations: the pin is
exact; `tests/test_golden.py` catches any drift; a `TransformerBridge` loader backend behind a flag is
the escape hatch if 2.x becomes unmaintainable.

**Derived pins:**

| Package | Constraint | Source of constraint |
|---|---|---|
| Python | **3.11** | you have 3.11.14; torch wheels on 3.14 are unreliable |
| `transformer-lens` | `==2.18.0` | this decision |
| `torch` | `>=2.6` | TL 2.18 requires |
| `transformers` | `>=4.57` | TL 2.18 requires |
| `numpy` | `>=1.24,<2` | TL 2.18 requires on py3.9–3.11 |

> **The `numpy<2` ceiling is a live constraint.** Check every new dependency against it before adding.

---

## D4 — Custom React + Vite + TypeScript, canvas rendering

**Status:** locked · **Revisit:** no

**Why not Gradio.** Four things the product needs are awkward or impossible in Gradio: hover-linked
highlighting between text and heatmap, keyboard navigation, stateless permalinks, and the cost panel.
Those are the difference between a demo someone tries once and a tool they keep open in a tab.

**What it costs.** It rules out Hugging Face's free ZeroGPU tier, which is Gradio-SDK-only. That tier
was the only zero-cost HF path (a ZeroGPU Space that never calls `@spaces.GPU` runs CPU-unmetered
indefinitely). Losing it means either $9/mo for HF PRO or a non-HF host — see D1.

**Canvas, not SVG or DOM.** A 512x512 heatmap is 262,144 cells. DOM will not survive it. Plan:
`<canvas>` + `putImageData` + `imageSmoothingEnabled = false`. WebGL only if contexts beyond 1024
are ever wanted.

**CircuitsVis** has the right primitives and is what ARENA uses in-notebook. Read its source during
Stage 1 before deciding fork vs. reuse vs. write fresh — but assume we need hover-linking it doesn't
provide.

---

## D5 — sqrt-companded uint8, causal-triangle packed

**Status:** locked · **Revisit:** no · **Spec:** [`01-wire-format.md`](01-wire-format.md)

Derived from D4 plus the payload arithmetic. Linear uint8 quantizes every attention probability below
~0.002 to zero, which is invisible on a linear heatmap and destroys the log-scale view. `u = round(255*sqrt(p))`
fixes it for the same one byte: smallest representable value drops from 3.9e-3 to 1.5e-5.

Full rationale, error bounds, rejected alternatives and a reference decoder are in the spec.

---

## D6 — Single worker, semaphore-serialized forward passes

**Status:** locked · **Revisit:** Stage 4 load test

One uvicorn worker, `asyncio.Semaphore(1)` around forward passes, torch in a thread executor.

**Why not multiple workers.** Each worker would hold its own copy of the model weights. At a 6 GB
budget that *multiplies* the binding constraint. Multi-worker is the standard answer to a problem we
don't have (CPU-bound throughput) and makes worse the one we do have (memory).

**The LRU refuses rather than swaps.** Exceeding `MI_RAM_BUDGET_GB` returns `503 budget_exceeded`.
A visible, debuggable failure beats mysterious slowness or an OOM kill.

---

## D7 — Curated parallel prompts now; live translation deferred

**Status:** locked for v1 · **Revisit:** v1.1, gated on Stage 0a RAM headroom

Stage 3 ships human-verified parallel prompt sets (same content, ~10 languages) as static JSON.

**Why, beyond cost.** Machine-translation errors would confound exactly what users are inspecting. If
someone sees odd attention behaviour on a Nepali prompt, "is this the model or is this a bad
translation?" is a question the tool cannot answer — so it shouldn't create it. Curated prompts are
zero RAM, zero latency, zero dependencies, *and* better pedagogy.

`nllb-200-distilled-600M` (200 languages, self-hosted, no API key) is the v1.1 candidate at a cost of
~2.4 GB of the RAM budget and 2–5 s/sentence on 2 vCPU. Hosted translation APIs are rejected: they add
a key, a rate limit, and an external dependency that can take the site down.

---

## Open items

| # | Item | Decide by | Notes |
|---|---|---|---|
| 1 | **Project name** — `attnlab` is a placeholder | before Stage 1 | After Stage 1 it's in the URL |
| 2 | Deploy target | Stage 4 | D1; decided on real numbers |
| 3 | Live translation | v1.1 | D7; gated on 0a headroom |
| 4 | CircuitsVis: fork, reuse, or write fresh | Stage 1 | Read its source first |
| 5 | **Golden fixture** — run ARENA 1.2 yourself and capture its induction scores | Stage 2 | Blocks `tests/test_golden.py`; nobody else can produce this |
| 6 | Final `tier` values in `models.yaml` | end of Stage 0a | Depends on measured RSS |

---

## Decision log

| Date | Change |
|---|---|
| 2026-09-19 | D1–D7 recorded. Version evidence for D3 gathered from PyPI and the ARENA 3.0 repo. |
| 2026-09-19 | Quantization acceptance criterion corrected: the first draft used the *linear* uint8 error bound ("< 0.002 for p > 0.01"), which sqrt companding fails at large `p` by design. Correct bounds are in [`01-wire-format.md`](01-wire-format.md). |

---

## D8 — LRU budget accounting uses analytic model size, not measured RSS delta

**Status:** locked · **Discovered:** Stage 0a benchmarking, 2026-09-19

**Finding.** Benchmarking multiple models sequentially in one process (native run, 5 models,
`bench.py`) produced RSS deltas that were implausibly low for some models — `pythia-160m` showed
57 MB (expected ~640 MB), `bloom-560m` showed 763 MB (expected ~2.2 GB). Root cause: macOS/Python
memory allocators reuse freed pages from a previously-unloaded model rather than returning them to
the OS, so `rss_after - rss_before` for model *N* can be measured against a baseline still holding
freed-but-unreturned memory from model *N-1*. `gc.collect()` does not fix this — it frees Python
references, not the OS-level page mapping.

**Why this is not just a benchmark artifact.** The exact same allocator behavior will occur in the
real backend: Stage 0b's `zoo.py` LRU cache holds multiple models in one long-running process and
evicts/reloads within it. If eviction/admission decisions were based on empirically measured RSS
deltas, the *same* underestimation could let the cache admit a model that actually exceeds
`MI_RAM_BUDGET_GB`, defeating the entire point of D6's "refuse rather than swap" design.

**Decision.** `zoo.py`'s budget accounting must use each model's **analytic size**
(`sum(p.numel() for p in model.parameters()) * 4` bytes, since dtype is always fp32 per D3) as the
authoritative cost, not a measured RSS delta. Analytic size has no allocator-history dependence —
it is exactly the parameter count times bytes-per-parameter, computable before or after load. Add a
fixed per-model overhead margin (activations, tokenizer, buffers — estimate ~10-15% from the 0a
data once `RESULTS.md` is generated) on top of the analytic figure for the actual admission check.

Measured RSS (native or containerized) remains valuable for a *different* purpose: validating that
the analytic-size-plus-margin estimate is in the right ballpark, and for the absolute
`cgroup_current_mb` / `cgroup_max_mb` reads that Stage 0b's `/api/health` and the debug overlay use
for real-time headroom — those read the OS's actual current state, not a delta, so they don't share
this problem.

**Consequence for `models.yaml`.** `est_ram_mb` values should be (and once 0a's Docker run
completes, will be) set from `analytic_fp32_mb` plus margin, not from `net_rss_delta_mb`.

**Amendment, same day — analytic size alone is not sufficient either.** Even in a clean, isolated
subprocess with no prior allocation history, measured resident RSS after `from_pretrained` came in
at **2.1x–4.8x the analytic weight size**, and — critically — `peak_rss_mb` measured *during* the
load equals `rss_after_mb` exactly, for every model tested. That means the excess is not a transient
spike that frees back down; it is a permanent floor. The pattern fit across all 5 models (native/MPS,
Stage 0a exploratory run):

```
resident_rss_mb  ≈  1.9 * analytic_fp32_mb  +  ~700 MB fixed
```

Likely mechanism: `HookedTransformer.from_pretrained` holds both the raw downloaded weights and the
TransformerLens-processed copies (folded LayerNorm, centered writing weights) simultaneously at some
point, and PyTorch's own allocator caches that high-water mark rather than returning it to the OS —
the same allocator-retention behavior D8 identified *across* sequential loads, but it turns out to
also apply *within* a single model's load.

**Practical consequence.** Analytic size (`params * 4 bytes`) remains useful as an
allocator-independent floor for sanity-checking measurements, but **it is not what to budget
against** — it under-counts real footprint by roughly 2-5x. `zoo.py`'s admission check should use
`measured_resident_mb` from an isolated benchmark (or the fitted formula above as a fallback for
models not yet benchmarked), not raw analytic size.

**Concretely, for the 6 GB Mode-B budget (D1):** the three `tier: baked` models
(`attn-only-2l-demo` + `gpt2-small` + `pythia-160m`) sum to ~4.7 GB of the 6 GB budget on this
(native, MPS) measurement — tighter than `FEASIBILITY.md`'s original back-of-envelope estimate.
**Not final** — per the two-mode discipline, only a Mode B (Docker, CPU, cgroup-limited) measurement
decides `models.yaml`'s shipped `tier` values and `est_ram_mb`. This number is flagged here because
it changes the shape of the decision (baked-model selection may need to drop to two models, or the
budget may need to grow) — confirm or correct it with the Docker run before treating it as settled.

---

## D1 amendment — local Mode B ceiling is 6 GB, not 16 GB, by the user's choice

**Status:** locked · **Discovered:** Stage 0a, first real Docker benchmark run, 2026-09-19

**Finding.** `bloom-560m` was SIGKILL'd (exit -9, the kernel OOM killer) mid-run under
`--memory=16g`. Root cause was not the container flag — `docker info --format '{{.MemTotal}}'`
showed Docker Desktop's Linux VM on this machine has a fixed **7.7 GB total RAM** allocation,
a systemwide setting independent of any per-container `--memory` flag. Requesting `--memory=16g`
inside a VM that only has 7.7 GB does nothing useful; the real ceiling was 7.7 GB the whole time.

**Decision, explicit and final:** the user chose not to raise Docker Desktop's memory allocation
("if RAM, no way") — correctly identifying that this is a systemwide tradeoff against the rest of
the Mac (24 GB total), not a project-scoped setting. **This is not revisited without the user
raising it again.**

**Practical resolution.** The local Mode B benchmark now targets **6 GB** (`--memory=6g` in the
Makefile), which fits inside the real 7.7 GB VM with ~1.5 GB headroom for the VM/kernel itself. This
is not a downgrade of D1's intent: the actual product question was always "how many models fit in a
realistic operational budget" (6 GB, matching `MI_RAM_BUDGET_GB`), not the raw 16 GB container spec
figure — HF free CPU Basic's *hardware* is 16 GB, but a real deployment needs headroom for the OS,
the web process, and request-time activations well below that ceiling anyway. Testing at 6 GB
locally is a reasonable, slightly conservative proxy for that.

**Consequence for `bloom-560m` specifically.** Its measured resident weight size alone
(~6.2 GB, from the earlier isolated native run) exceeds this machine's local 6 GB test ceiling
before any request activity. This does **not** mean `bloom-560m` is disqualified from production —
that is a Stage 4 decision made against the real deploy target's actual RAM, which may be more than
6 GB. It does mean **`bloom-560m` cannot be locally verified on this machine** and must be confirmed
on a bigger box (a cloud CI runner, or the real Stage 4 host) before its `tier: lazy` status in
`models.yaml` is trusted. Until then, treat `Qwen2.5-0.5B` (already noted as an alternative in
`docs/PLAN.md` Stage 3) as the safer default multilingual candidate for local development.

**Process note, independent of the memory finding:** the first version of `run_isolated_matrix`
let one model's subprocess failure (a crash, an OOM-kill) abort the entire matrix, discarding every
result gathered before it. Fixed: a failed subprocess is now recorded as `{"failed": true,
"exit_code": ...}` and the run continues to the next model. A partial benchmark that reports what it
could measure is strictly more useful than losing everything to the last model tested.

---

## D9 — Hover-linking is directional: destination→source and source→destination

**Status:** locked · **Decided:** Stage 1 UI pass, 2026-09-19

CircuitsVis's `attention_patterns`, the component ARENA 1.2 uses, shows paired **Destination** and
**Source** token columns: hovering one colours the other. That is not two features, it is one matrix
read two ways, and Stage 1 originally shipped only one of them.

| Direction | Matrix slice | Question it answers | Sums to 1? |
|---|---|---|---|
| `dest2src` (default) | row `A[anchor][*]` | "at this token, what does the model look back at?" | **yes** — it's a softmax |
| `src2dest` | column `A[*][anchor]` | "which later tokens look back at this one?" | **no** — attention *received* |

**Why a toggle rather than two token columns.** Two columns doubles the vertical cost of the token
strip and, on a 512-token prompt, produces two long wrapped blocks the eye has to reconcile. One
strip plus an explicit switch says the same thing in half the space, and makes the asymmetry
impossible to miss instead of implicit in the layout.

**The caveat is load-bearing, so the UI states it.** A column is not a distribution. Measured on
`attn-only-2l-demo` with a 12-token prompt, layer 0 head 0: the column for position 0 totals **9.03**
— the familiar attention-sink on the first token. A user who assumes both directions normalise will
read that as a bug. Every surface that shows `src2dest` therefore carries the "does NOT sum to 1"
note, and `lib/attention.ts` is the only module that knows what a direction means numerically, so
the token strip, the head tiles, the crosshair and the top-k list cannot drift apart.

The direction is permalink state (`?dir=src2dest`, omitted at the default) and has a keyboard
binding (`d`).

**Rejected:** normalising the column so both directions sum to 1. It would make the two views look
symmetric and would be a lie about what attention is.

---

## D10 — The frontend gets a headless test suite, because this project cannot rely on looking at it

**Status:** locked · **Decided:** Stage 1 UI pass, 2026-09-19

Stage 1 was built and verified without a browser available. Type-checking, a clean production build
and curl-level API checks all passed while the interactive behaviour — hover linking, keyboard
navigation, permalink round-tripping, modal focus handling — remained entirely unverified. That is
the wrong shape of risk for the feature the plan itself calls "the one that makes it feel alive".

**Decision.** `vitest` + `jsdom` + Testing Library, 47 tests, run by `npm test`. It covers the pure
logic (wire-format decode against an *independently written* encoder in `src/test/fixtures.ts`,
direction maths, permalink parsing) and the actual mounted application (hover linking tints the
right tokens, the causal mask dims the right ones and flips when the direction flips, arrow keys
move layers without closing the open head, Escape unwinds one step at a time, modals restore focus
and suspend the global shortcuts, a slow run keeps the previous view on screen).

**Why an independent encoder in the fixtures.** Testing the decoder against our own encoder proves
only that the two agree. `fixtures.ts` builds ATNP buffers straight from `docs/01-wire-format.md`,
so the test fails if either implementation drifts from the spec. The real Python encoder is then
checked separately against a live server (rows summing to 1, nothing above the diagonal, header
dimensions, byte length).

**Not covered, and honestly so:** anything that needs a real renderer. jsdom has no canvas backend,
so the heatmap's pixels go to a stub — the render loop executes and is exercised for bounds and the
mask branch, but nobody has confirmed the image *looks* right. Colour choice, layout at real
viewport sizes and dark-mode appearance still need human eyes.

---

## D11 — Styling moves out of the components into a token layer

**Status:** locked · **Decided:** Stage 1 UI pass, 2026-09-19

Stage 1 shipped with every visual decision inline in JSX. With ~15 components that meant re-deciding
"is this 4px or 6px, 12px or 13px?" in each one, and the result read as assembled rather than
designed.

`styles.css` now carries the spacing, radius, type, elevation and control-height scales plus a small
component layer (`.card`, `.btn`, `.segmented`, `.chip`, `.pill`, `.modal`, `.statlist`, `.ranked`,
`.heat*`). Components carry class names and use an inline style only for genuinely data-driven
values — an attention tint, a percentage offset for a crosshair. The dataviz palette block is
unchanged and still the validated reference instance.

Two consequences worth naming:

- **A theme toggle became possible.** The stylesheet already declared dark values under both
  `prefers-color-scheme` and `[data-theme="dark"]`; nothing set the attribute. It now does, the
  resolved theme lives in the store because the canvas renderer needs it as *data* (CSS can't reach
  into `putImageData`), and every canvas re-renders on a theme change.
- **The causal-masked half of the matrix is painted in the page background**, not in the colour
  ramp's minimum. Those cells aren't "attention of zero", they're structurally impossible, and
  rendering them as ramp-minimum made half of every plot a solid block that looked like data.

---

## D10 amendment — a real browser, driven from the repo

**Status:** locked · **Added:** 2026-09-19, same day as D10

D10 closed by naming what the jsdom suite structurally could not reach: "anything that needs a real
renderer... nobody has confirmed the image *looks* right." That gap is now closed by Playwright,
committed as `web/scripts/screenshots.mjs` and run with `npm run shots`. It drives headless Chromium
against the live dev servers, captures one screenshot per state (overview, hover link, head detail,
tooltip, flipped direction, modal, dark mode, 680px, 390px), and exits non-zero on any console
error, page error or failed request.

**It paid for itself on the first run, with two bugs nothing else had caught:**

1. **The causal mask rendered light in dark mode.** `setThemeMode` updated the store synchronously,
   and React runs child effects before parent effects — so every canvas repainted, reading
   `--heat-mask` off `<html>`, *before* the parent effect stamped `[data-theme]`. It got the light
   value, and a memo keyed by theme then cached that wrong value under `"dark"` permanently. Fixed
   by making the attribute stamp part of the state transition (in the store action and at module
   init) rather than an effect that follows it, and by deleting the memo — one custom-property read
   is nothing next to the pixel loop it precedes. **A unit test could not have caught this:** it
   needs a real cascade, real effect ordering and a real painted canvas simultaneously.
2. **The linked row/column band painted across causally masked cells**, tinting impossible cells so
   they read as data. Now clipped to the reachable extent (`0..dest` for a row, `src..seq-1` for a
   column) and drawn as an outline rather than a translucent wash — a fill over the exact row you
   are trying to read shifts every colour in it, which defeats the colour scale.

**Division of labour, to keep both honest.** `npm test` (jsdom, 47 tests) owns behaviour and runs in
milliseconds with no servers. `npm run shots` owns appearance and layout, needs both servers up, and
produces artefacts a human still has to look at. Neither replaces the other; the screenshots are
evidence for a reviewer, not an assertion.

**Also verified by it:** zero horizontal overflow at 390px, the permalink updating correctly through
a full interaction sequence, and a clean console across every state.

**Third bug, found by a user and then pinned here (2026-09-19).** With a 57-token prompt and a head
expanded, the linked band was drawn *below* the matrix. `.heat__yticks` laid its labels out as
`seq` grid rows, and in an auto-height grid a `1fr` row cannot shrink below its own line-height — so
57 rows of ~11px forced the grid row to ~630px while the canvas stayed pinned to ~450px by
`aspect-ratio: 1`. Every overlay in `.heat__plot` positions itself as a percentage of that box, so
the band and the crosshair both landed past the bottom of the image. Under roughly 40 tokens the
ticks fit inside the canvas height and nothing looked wrong, which is why every screenshot taken so
far (10-token prompts) missed it.

Fixed by positioning the y ticks absolutely, so they contribute no height and the row is sized
purely by the canvas, and by moving the canvas frame from `border` to `outline` so the overlay box
and the drawing area are identical to the pixel. The screenshot script now **asserts** it: it opens
a 60-word prompt with a head expanded and fails if `.heat__plot` drifts more than 1px from
`.heat__canvas`, or if the band is drawn outside the canvas. jsdom cannot catch this class of bug at
all — it has no layout engine, so every box is 0x0 there.

---

## D12 — Byte fragments render as the character they belong to, never as U+FFFD

**Status:** locked · **Decided:** 2026-09-19, prompted by a user typing Nepali into the app

**The report.** A Devanagari prompt produced a strip of `�` boxes — 167 unreadable chips — with no
indication of what any of them were.

**The mechanism.** GPT-2 is byte-level BPE, and its merges were learned on a corpus with essentially
no Devanagari. A character like `म` is three UTF-8 bytes (`e0 a4 ae`); GPT-2 has a token for the
common two-byte prefix `e0 a4` and another for `ae`, so one character costs **two tokens, neither of
which is a character**. `tokenizer.decode([id])` on half a UTF-8 sequence returns U+FFFD, and that
is what reached the UI. Measured: 29 characters of Nepali → 47 tokens (1.62 tok/char); 19 characters
of English → 4 tokens (0.21 tok/char). Roughly **8× worse per character.**

This is not a bug in the app. It is the Stage 3 finding (`PLAN.md`: "GPT-2's byte-level BPE shreds
Devanagari... a word becomes 8 tokens of meaningless byte fragments") arriving early, via a user.
The job is therefore to *show* it clearly, not to hide it.

**What was rejected.**

| Option | Why not |
|---|---|
| Show the token id instead | Truthful but inert. `11976` tells a learner nothing about their text. |
| Show the raw BPE piece (`à¤`) | What the HF tokenizer prints internally. Still gibberish, and now gibberish that looks like it might be meaningful. |
| Show only the hex bytes on the chip | Accurate and unreadable at 167 chips; buries the text completely. |
| Repeat the character on every fragment | Two chips both reading `म` implies two `म`s. Actively misleading. |
| Merge the fragments into one chip | **Wrong.** Each fragment is its own position in the attention matrix. Merging them would misalign the token strip against the heatmap — the exact class of bug the strip is built to avoid. |

**What was built.** The fast tokenizer's offset mapping gives the *character span* each token covers,
and fragments of one character all report that character's span. Consecutive tokens with overlapping
spans are grouped into a **cluster** server-side (`cluster`, `cluster_size`, `cluster_index`,
`cluster_text` on every `TokenInfo`, plus `byte_hex`). The UI then:

- shows the character on the cluster's first chip and `⋯` on the rest — so two tokens for one
  character read as *one character that cost two tokens*, not as two characters;
- draws the cluster as one joined, underlined unit while keeping **one button per token**, because
  each is still its own attention position and must stay independently hoverable;
- puts the truthful detail in the tooltip: `byte 2 of 2 of "म" · bytes 0xae`;
- reports **tokens per character** in the panel header, which is the whole story as one number;
- explains the situation inline once fragments exceed 20% of the sequence, including the part that
  matters for interpretation: *much of what the attention heads are doing here is reassembling UTF-8,
  not modelling language.*

**Guard on the offsets.** `model.to_tokens` remains the single source of truth for the sequence. The
offset mapping is borrowed from the fast tokenizer only after checking it reproduces TL's exact id
sequence (allowing for a prepended BOS); otherwise the old cursor-search heuristic is used. Two
tokenization paths must never be able to silently disagree.

**Open.** This makes the fragmentation legible; it does not make GPT-2 good at Nepali. The genuine
fix for a user who wants to study their own language is a tokenizer with merges for that script —
`Qwen2.5-0.5B` per Stage 3, still unverified under Mode B (see the D1 amendment). Until such a model
is in `models.yaml` and verified, the notice deliberately does not name one.
