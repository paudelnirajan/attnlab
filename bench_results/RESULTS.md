# Benchmark results

device=`cpu` dtype=`torch.float32` threads=`2` ram_budget_gb=`6.0` isolation=`one subprocess per model`

framework baseline RSS (torch + `cpu` backend init, paid once): **527 MB** — subtracted from each model's load to get `net_rss_delta_mb`.

**Every row below pays one-time Python import overhead** for `transformers` / `tokenizers` / `huggingface_hub` (their C extensions, vocab/config parsing machinery), since each model ran in its own fresh subprocess (see D8 in docs/03-decisions.md) — that overhead has nothing to do with any individual model's actual size. It is a closer match to Stage 4's real cold-start number (a fresh worker process loading its first model) than to the marginal cost of adding a 2nd/3rd model to an already-warm server, which is what Stage 0b's LRU eviction budget actually needs — use `analytic fp32 MB` for that, not this column.

**`net rss delta` vs. `analytic fp32`:** these can diverge significantly for models loaded after a larger one was freed in the same process — macOS/Python allocators reuse freed pages rather than returning them to the OS, so the empirical delta can *understate* a model's true cost. `analytic fp32` (`params * 4 bytes`) has no such ambiguity. **Stage 0b's LRU budget should account against the analytic number, not the empirical RSS delta** — see docs/03-decisions.md.

| model | load ms | net rss delta MB | analytic fp32 MB | seq | pattern-only ms | full-cache ms | cache saving | per-layer f32 KB | per-layer u8+tri KB | per-layer u8+tri+gz KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn-only-2l-demo \* | 6016 | 757 | 207 | 65 | 51.2 | 22.8 | 90% | 132.0 | 16.8 | 11.8 |
| attn-only-2l-demo \* | 6016 | 757 | 207 | 129 | 37.3 | 37.8 | 83% | 520.0 | 65.5 | 38.0 |
| attn-only-2l-demo \* | 6016 | 757 | 207 | 257 | 72.6 | 69.4 | 75% | 2064.0 | 259.0 | 114.0 |
| attn-only-2l-demo \* | 6016 | 757 | 207 | 513 | 141.2 | 142.4 | 67% | 8224.0 | 1030.0 | 325.7 |
| gpt2-small \* | 2768 | 1671 | 622 | 65 | 121.2 | 126.2 | 95% | 198.0 | 25.2 | 17.8 |
| gpt2-small \* | 2768 | 1671 | 622 | 129 | 184.6 | 176.6 | 91% | 780.0 | 98.3 | 57.5 |
| gpt2-small \* | 2768 | 1671 | 622 | 257 | 371.2 | 380.1 | 85% | 3096.0 | 388.5 | 175.6 |
| gpt2-small \* | 2768 | 1671 | 622 | 513 | 711.0 | 809.2 | 77% | 12336.0 | 1545.0 | 484.7 |
| pythia-160m \* | 2279 | 1641 | 619 | 65 | 106.6 | 107.7 | 95% | 198.0 | 25.2 | 19.3 |
| pythia-160m \* | 2279 | 1641 | 619 | 129 | 168.6 | 172.1 | 92% | 780.0 | 98.3 | 61.3 |
| pythia-160m \* | 2279 | 1641 | 619 | 257 | 321.5 | 334.9 | 86% | 3096.0 | 388.5 | 166.6 |
| pythia-160m \* | 2279 | 1641 | 619 | 513 | 678.9 | 745.8 | 78% | 12336.0 | 1545.0 | 347.9 |
| gpt2-medium \* | 3905 | 3782 | 1550 | 65 | 338.6 | 324.5 | 95% | 264.1 | 33.5 | 26.3 |
| gpt2-medium \* | 3905 | 3782 | 1550 | 129 | 522.0 | 632.7 | 91% | 1040.1 | 131.0 | 92.9 |
| gpt2-medium \* | 3905 | 3782 | 1550 | 257 | 1013.8 | 1196.6 | 85% | 4128.1 | 518.0 | 309.1 |
| gpt2-medium \* | 3905 | 3782 | 1550 | 513 | 2135.0 | 2420.2 | 77% | 16448.1 | 2060.0 | 894.7 |
