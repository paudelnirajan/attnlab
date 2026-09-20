# Attention / Induction-Head Playground — Feasibility Report

**Date:** 2026-09-19
**Question:** Can we build a public, hosted web app where anyone types a prompt, picks a
HookedTransformer model, and explores attention patterns, tokenization, induction heads, and
head-level interventions — including across languages?

---

## 1. Verdict

**Yes, and it's smaller than it looks.** The core of it is a weekend prototype and a ~3–4 week
polished v1 for one person.

Three findings that shape everything below:

1. **You almost certainly don't need a GPU.** Every model that matters pedagogically for ARENA 1.2
   (`attn-only-2l-demo`, `gpt2-small`, `pythia-*`, `gpt2-medium`) runs a forward pass with cached
   activations on 2 CPU cores in well under a second for typical prompt lengths. GPU only becomes
   necessary if you want 7B+ models, which is a different product.
2. **The hard engineering problem is not ML, it's data volume.** Attention patterns are
   `n_layers × n_heads × seq × seq` floats. For `gpt2-small` at 512 tokens that's 151 MB as float32
   JSON-ish payloads. This is entirely solvable (binary + uint8 + per-layer fetch) but it must be
   designed in from day one, not bolted on.
3. **You do not download all models to the VM.** Disk is cheap and lazy; **RAM is the binding
   constraint.** A tiered scheme — a few models baked into the image, the rest pulled on demand into
   an LRU cache — handles this.

The one genuine cost surprise: **Hugging Face now requires a paid plan (PRO, $9/mo) to create a
Gradio or Docker Space.** Free personal accounts only get 2 ZeroGPU *Gradio* Spaces. Details and
workarounds in §8.

---

## 2. Does this already exist? (mostly no — your read is right)

I searched for hosted tools. The closest things and why each doesn't cover it:

| Tool | Hosted? | Arbitrary prompt? | Why it's not this |
|---|---|---|---|
| **CircuitsVis** (TransformerLensOrg) | No — a library | n/a | This is the React/Python component ARENA uses. Components, not an app. **We should reuse or fork it.** |
| **Transformer Explainer** (Poloclub, CHI '26) | Yes | Yes | Teaches *architecture* (Q/K/V, softmax, positional encoding) on GPT-2 small in-browser. No induction heads, no ablation, no model choice. |
| **BertViz** | No — notebook | Yes | Head/model/neuron views, encoder-centric, Jupyter-bound. |
| **Neuroscope** (Nanda) | Yes | **No** | Max-activating *dataset* examples for neurons. You can't type your own text. |
| **LLM Transparency Tool** (Meta) | Self-host | Yes | Contribution graphs, heavy to run, not a teaching playground. |
| **AttentionViz** (Harvard/Google) | Yes | Limited | Query/key embedding space across a fixed corpus. Different question entirely. |
| **AttentionApp** (PROPOR '26) | Yes | Yes | Portuguese-specific, linguistic analysis focus. |

**The gap:** no hosted tool lets you type a prompt, pick from the TransformerLens model zoo, and do
*mech-interp-specific* things — induction scores per head, prev-token/duplicate-token head
detection, ablation with a live logit-diff readout, repeated-random-token sequences, OV/QK circuit
inspection. That's a real, defensible niche, and it maps 1:1 onto a curriculum thousands of people
work through.

*Caveat: this is from a search pass, not exhaustive. Worth 30 minutes of your own spot-checking
before you commit.*

---

## 3. Feature spec

### Tier 1 — the thing that must work (this is the whole value prop)
- **Prompt box + model picker.** Model list driven by a config file, not hardcoded.
- **Tokenizer panel.** Tokens rendered as chips with visible whitespace/BPE boundaries, token IDs,
  byte fallbacks, token count. Click a token → it becomes the selected query/key position everywhere
  else.
- **Attention grid.** All heads of a layer as small heatmaps; click to expand one head full-size.
  Hover a destination token → highlight which source tokens it attends to, both on the heatmap and
  inline in the text.
- **Layer/head navigation** with keyboard shortcuts (this is what makes it feel like a playground
  rather than a demo).
- **Permalinks.** Model + prompt + layer + head encoded in the URL. Essential for teaching and for
  people sharing findings.

### Tier 2 — the induction-head lab (the differentiator, maps to ARENA 1.2)
- **Repeated-sequence generator.** Random tokens repeated twice, configurable length/seed — the
  canonical induction-head probe.
- **Head score table.** Layer × head grid colored by:
  - induction score (mean of the `seq_len − 1` diagonal),
  - previous-token score (diagonal offset 1),
  - duplicate-token score,
  - current-token score.
  Click a cell → jump to that head's pattern.
- **Per-token loss curve** over the repeated sequence — the "loss drops in the second half" plot that
  makes induction click.
- **Ablation.** Zero- or mean-ablate any head (or set of heads) via hooks; show Δ loss and Δ logit
  on the selected token, live.

### Tier 3 — multilingual
- **In-page translation** of the prompt into N languages (§7).
- **Side-by-side comparison:** same semantic content, different languages → compare token counts,
  tokenizer fragmentation, and induction scores. The tokenizer story alone is a great lesson: GPT-2's
  byte-level BPE shreds Devanagari, Thai, Amharic etc. into per-byte tokens, which visibly wrecks
  induction behavior. That's a finding users can *discover* in the tool.
- Needs genuinely multilingual models in the picker: `bloom-560m` (46 languages),
  `Qwen2.5-0.5B`, `pythia` as the English control.

### Tier 4 — stretch, only after v1 ships
- Logit lens / direct logit attribution.
- OV and QK circuit inspection (eigenvalue copying score, full circuit matrices).
- Composition scores (Q-, K-, V-composition between heads) — the back half of ARENA 1.2.
- Neuron activation views.
- Attention-head *comparison* mode (two models, same prompt).

---

## 4. Architecture

### Recommended: thin Python backend + custom React frontend

```
┌──────────────────────────────┐        ┌────────────────────────────────┐
│  React + Vite + TS           │        │  FastAPI (single process)      │
│                              │        │                                │
│  • canvas heatmap renderer   │ ─HTTP→ │  • TransformerLens             │
│  • token chips / tokenizer   │ ←bin── │  • LRU model cache (RAM-budget)│
│  • head score grid           │        │  • request queue (concurrency 1│
│  • URL state / permalinks    │        │    –2 per model)               │
└──────────────────────────────┘        │  • uint8 quantize + gzip       │
   static hosting (free)                │  • translation service         │
                                        └────────────────────────────────┘
                                             container w/ baked models
```

**API surface** (small on purpose):

| Endpoint | Returns |
|---|---|
| `GET /models` | catalog: name, n_layers, n_heads, d_model, size, languages, status (baked/lazy) |
| `POST /tokenize` | tokens, ids, offsets — **fast path, no model forward needed** |
| `POST /run` | run id, tokens, logits/top-k per position, loss per token |
| `GET /run/{id}/patterns?layers=0,1` | **binary** uint8 attention, per layer |
| `POST /head-scores` | induction / prev-token / dup-token / current-token matrices |
| `POST /ablate` | Δloss, Δlogits under a hook spec |
| `POST /translate` | translated text + detected source language |

**Key decisions:**

- **Binary, not JSON, for patterns.** Quantize to uint8 (attention probs are in [0,1] — 1/255
  resolution is far below what a heatmap can display), pack only the causal lower triangle, gzip,
  return as `application/octet-stream`. Frontend decodes into a `Uint8Array` and writes straight to
  canvas `ImageData`.
- **Per-layer, on demand.** Never ship all layers at once. Prefetch the neighbouring layer.
- **Canvas, not SVG/DOM.** A 512×512 heatmap is 262k cells — DOM will die. One `<canvas>` per head,
  `putImageData`, nearest-neighbour upscale. If you later need 2048-token contexts, move to WebGL.
- **`names_filter` on `run_with_cache`.** Cache *only* `hook_pattern` unless the request needs more.
  This is the single biggest memory win and easy to forget.
- **Cap `n_ctx` per model** (e.g. 512 for small, 256 for larger). Enforce server-side.

### Alternative A: Gradio + CircuitsVis
Fastest to a demo — possibly one day. But Gradio fights you on custom interaction (hover-linked
token highlighting, keyboard nav, permalinks), and it's the difference between "a demo" and "a tool
researchers keep open in a tab." **Good for a throwaway proof-of-life, not for the real thing.**
Relevant exception: it's the *only* way onto HF's free tier (§8).

### Alternative B: fully in-browser with transformers.js
Run GPT-2 as ONNX via WebGPU/WASM client-side. **Zero server cost, infinite scale, no cold starts.**
Real downsides: no TransformerLens weight processing (folded LayerNorm, centered weights — the thing
that makes TL numbers match ARENA's), hooks/ablation are hard, and you'd be limited to models you've
exported to ONNX with `output_attentions`.

**Worth considering as a hybrid:** browser handles the Tier-1 viewer for `gpt2-small` (instant, free,
works offline), server handles Tier-2/3 TransformerLens analyses. That's more work, so don't do it
for v1 — but design the API boundary so it stays possible.

---

## 5. The numbers that determine the design

### Attention payload size — `gpt2-small` (12 layers × 12 heads = 144 head-matrices)

| seq len | values | float32 | float16 | **uint8** | uint8 + causal-triangle |
|---:|---:|---:|---:|---:|---:|
| 64 | 590 K | 2.4 MB | 1.2 MB | **0.6 MB** | 0.3 MB |
| 128 | 2.4 M | 9.4 MB | 4.7 MB | **2.4 MB** | 1.2 MB |
| 256 | 9.4 M | 37.7 MB | 18.9 MB | **9.4 MB** | 4.8 MB |
| 512 | 37.7 M | 151 MB | 75 MB | **37.7 MB** | 19 MB |

Per **single layer** at seq=256: 786 K values → **0.79 MB** uint8, ~0.4 MB triangle-packed, and
gzip typically takes another 40–60% off because attention is sparse. That's the design: **fetch one
layer at a time and it's always sub-megabyte.** JSON-encoding any of this would be 5–7× worse and is
a non-starter.

### Model memory (full params incl. embeddings)

| Model | Params | float32 | Notes |
|---|---:|---:|---|
| `attn-only-2l-demo` | ~3 M + emb | ~50 MB | ARENA 1.2's workhorse |
| `pythia-14m` | 14 M | 56 MB | |
| `gpt2-small` | 124 M | 497 MB | the default |
| `pythia-160m` | 162 M | 648 MB | |
| `gemma-3-270m` | 268 M | 1.1 GB | **gated** — needs `HF_TOKEN` + license |
| `gpt2-medium` | 355 M | 1.4 GB | |
| `pythia-410m` | 405 M | 1.6 GB | |
| `Qwen2.5-0.5B` | 494 M | 2.0 GB | multilingual |
| `bloom-560m` | 559 M | 2.2 GB | 46 languages |
| `gpt2-large` | 774 M | 3.1 GB | |
| `gpt2-xl` | 1.56 B | 6.2 GB | probably too big for a 16 GB shared box |

On a 16 GB box you can comfortably keep ~4–6 GB of models resident. That's `attn-only-2l` +
`gpt2-small` + `gpt2-medium` + one multilingual 0.5B, with headroom for activations and the web
process.

> **bfloat16 caveat:** halving memory with `dtype=torch.bfloat16` is tempting, but TransformerLens's
> weight processing (LayerNorm folding, weight centering) is numerically sensitive, and x86 CPUs
> without AMX are *slower* in bf16 than fp32. **Use fp32 on CPU.** Revisit only on GPU.

### Latency (rough estimates — benchmark before trusting)
On 2 vCPU, fp32, forward pass with pattern-only caching:
- `gpt2-small` @ 128 tokens: ~0.15–0.4 s
- `gpt2-small` @ 512 tokens: ~1–2 s
- `gpt2-large` @ 128 tokens: ~1–2 s

Fine for an interactive tool if you show a spinner and debounce typing. Not fine if 20 people hit it
at once — hence the request queue (§9).

---

## 6. "Do we need to download all the models to our VM?"

**No.** Three tiers:

1. **Baked into the container image at build time** (~2–3 GB): `attn-only-2l-demo`, `gpt2-small`,
   `pythia-160m`, one multilingual model. These are instant on cold start — no first-user penalty.
   Do this by running a `huggingface_hub.snapshot_download` in the Dockerfile.
2. **Lazy, on first request**: everything else. Pulled from the Hub into `HF_HOME` on disk, with a
   "downloading model, ~30s" state in the UI. Free Spaces give you 50 GB of (non-persistent) disk,
   which is plenty — the download just repeats after a container restart unless you add persistent
   storage ($5/mo for 20 GB on HF).
3. **Disabled by default**: anything over ~3 GB. Show it in the picker greyed out with "requires GPU
   tier."

**RAM, not disk, is the real constraint.** Implement an LRU cache keyed by model name with a
*byte budget* (e.g. 6 GB), evicting least-recently-used models and calling `gc.collect()` +
`torch.cuda.empty_cache()` on eviction. Never let the model count grow unbounded.

Also worth knowing: **gated models** (Llama, Gemma, Mistral) require an `HF_TOKEN` in the
environment *and* accepted licenses on the account. Keep v1 to ungated models to avoid that whole
category of support burden.

---

## 7. Translation

Three options, in order of my preference:

1. **Self-hosted `facebook/nllb-200-distilled-600M`** (~2.4 GB fp32, 200 languages). No API key, no
   rate limits, no external dependency, works offline, and it covers low-resource languages that
   commercial APIs handle poorly. Cost: 2.4 GB of your RAM budget and ~2–5 s per sentence on 2 vCPU.
   `m2m100_418M` is the lighter alternative.
2. **A hosted translation API** (LibreTranslate, MyMemory, DeepL/Google paid). Zero RAM, adds a key
   to manage, a rate limit to hit, and a dependency that can break your site.
3. **Skip auto-translation in v1**, ship curated parallel prompt sets instead — the same sentence
   pre-translated into 10 languages, verified for quality. Honestly this may be *better* pedagogy
   (machine-translation errors would confound what users are looking at), and it costs nothing.

**Recommendation: ship v1 with curated parallel prompts + a free-text box. Add NLLB in v1.1** once
you know whether the RAM budget can spare it.

---

## 8. Deployment and cost

### The HF Spaces catch
Per the current Hub docs: *"Gradio and Docker Spaces run on compute and require a paid plan to
create: PRO for personal accounts."* Static Spaces are free for everyone, and free personal accounts
in good standing can host **2 Gradio Spaces on ZeroGPU**.

So the HF free path is narrow but real: **a ZeroGPU Gradio Space that never calls `@spaces.GPU`**.
The CPU portion of a ZeroGPU Space is unmetered — the 5 min/day quota only burns inside decorated
functions. A CPU-only TransformerLens app would run free indefinitely. The cost is that you're
locked to Gradio (no custom React frontend), since ZeroGPU is Gradio-SDK-only.

### Options

| Option | Custom frontend? | Cost | Cold start | Notes |
|---|---|---|---|---|
| **HF ZeroGPU Gradio Space** | ✗ Gradio only | **$0** | sleeps after 48 h | Free, but caps you at Alternative A |
| **HF PRO + Docker Space (CPU Basic)** | ✓ | **$9/mo** | sleeps after 48 h, wakes on request | 2 vCPU / 16 GB / 50 GB disk; +$0.03/hr to upgrade to 8 vCPU / 32 GB (~$22/mo if always-on) |
| **Modal** (`@modal.asgi_app()`) | ✓ | **$30/mo free credits**, likely $0 in practice | ~1–2 s, sub-second with cached image | Scale-to-zero, pay only while serving. Best DX for this shape of app. |
| **Google Cloud Run** | ✓ | free tier likely covers it | a few s | Scale-to-zero, up to 32 GB RAM, generous free quota |
| **Fly.io / Railway / Render** | ✓ | ~$5–20/mo | varies | Render free tier spins down aggressively |
| **Static frontend split** | ✓ | $0 for the frontend | — | Cloudflare Pages / Vercel / HF Static Space, hitting any backend above |

### My recommendation
**Static React frontend on Cloudflare Pages (free) + FastAPI backend on Modal ($30/mo credits,
scale-to-zero).** This gives you the custom UI you need, costs nothing at research-community traffic
levels, and has no 48-hour sleep problem. Fall back to HF PRO + Docker Space if you'd rather have
everything in the HF ecosystem for discoverability — being listed on huggingface.co/spaces is real
distribution for this audience, and $9/mo is not much.

You should also **apply for an HF community GPU grant** once the Space is live and good. This is
exactly the kind of project they fund.

---

## 9. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **TransformerLens 3.x API churn.** v3.9.0 (Sept 2026) makes `TransformerBridge` the recommended interface and marks `HookedTransformer.from_pretrained` **deprecated**. ARENA 1.2 teaches the legacy API. | High | **Pin an exact version.** Decide deliberately: match ARENA (legacy `HookedTransformer`, familiar to users) or go `TransformerBridge` (15,000+ models, 140+ architecture families — a much bigger zoo). I lean legacy for v1 fidelity, with Bridge behind a flag. |
| **Memory blowup under concurrency.** Several users × several models × cached activations = OOM. | High | Single worker, `asyncio` queue with concurrency 1–2, RAM-budgeted LRU model cache, hard `n_ctx` cap, `names_filter` on caching. |
| **Payload size.** Naive JSON of full attention = 150 MB responses. | High | uint8 + triangle packing + gzip + per-layer fetch. Designed in from day one (§5). |
| **Abuse / accidental DoS.** Someone pastes a 10k-token document, or a crawler hammers `/run`. | Medium | Token cap, rate limit per IP, request timeout, Cloudflare in front. |
| **Cold starts / sleeping.** | Medium | Bake tier-1 models into the image; show an honest "waking up…" state; Modal avoids the 48 h sleep entirely. |
| **Gated models** need `HF_TOKEN` + license acceptance. | Low | Exclude from v1. |
| **Numerical mismatch with ARENA notebooks.** Users will compare your numbers to their Colab output. | Medium | fp32, same TL version, same `from_pretrained` flags. Display the exact version + flags in a footer so discrepancies are debuggable. |
| **Scope creep.** "Just do everything" is the failure mode here. | High | Ship Tier 1 alone first. It's already better than anything hosted today. |

---

## 10. Build plan

| Phase | Scope | Effort |
|---|---|---|
| **0. Proof of life** | FastAPI + `gpt2-small` + one endpoint returning uint8 patterns; React page with one canvas heatmap. Prove the pipeline end to end, locally. | ~1 day |
| **1. Tier 1 viewer** | Model picker, tokenizer panel, layer/head grid, hover-linked highlighting, keyboard nav, permalinks, LRU model cache. | ~1 week |
| **2. Induction lab** | Repeated-sequence generator, four head-score matrices, per-token loss curve, ablation via hooks. | ~1 week |
| **3. Multilingual** | Multilingual models in the zoo, curated parallel prompts, tokenizer-comparison view, (optionally) NLLB translation. | ~3–4 days |
| **4. Ship** | Dockerfile with baked models, deploy, rate limiting, load test, docs, a 2-minute demo video. | ~3–4 days |

**~3–4 weeks of focused solo work to a public v1.** Phase 0 alone will tell you 80% of what you need
to know about whether the numbers in §5 hold on real hardware — do that before committing to the
rest.

---

## 11. Decisions I need from you

1. **Legacy `HookedTransformer` or TransformerLens 3.x `TransformerBridge`?** Fidelity to ARENA vs.
   a 15,000-model zoo. (I lean: legacy for v1.)
2. **Custom React frontend or Gradio?** Custom costs ~$9/mo or a Modal account but is the difference
   between a demo and a tool. (I lean: custom, on Modal + Cloudflare Pages.)
3. **Translation in v1, or curated parallel prompts?** (I lean: curated prompts first.)
4. **Who is this for, precisely** — ARENA learners following the notebook, or researchers doing
   exploratory work? It changes the default model, the default prompt, and how much hand-holding the
   UI does. (These aren't mutually exclusive, but v1 should pick one to optimize for.)

---

## Sources

- [TransformerLens on PyPI](https://pypi.org/project/transformer-lens/) — v3.9.0, Sept 2026; TransformerBridge; `HookedTransformer.from_pretrained` deprecated
- [TransformerLens model properties table](https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html)
- [TransformerLens on GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis)
- [ARENA Chapter 1.2 — Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/)
- [HF Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview) — paid-plan requirement, hardware table, sleep behavior
- [HF Spaces ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu) — quotas, Gradio-only constraint
- [HF Pricing](https://huggingface.co/pricing)
- [Modal pricing](https://modal.com/pricing) — $30/mo free credits, sub-second cached cold starts
- [Transformer Explainer (CHI 2026)](https://dl.acm.org/doi/10.1145/3772318.3791725)
- [AttentionApp (PROPOR 2026)](https://aclanthology.org/2026.propor-2.6/)
