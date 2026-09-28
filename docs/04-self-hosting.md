# How serving works, and why

The site runs on a MacBook Pro (M1 Pro, 16 GB RAM, 256 GB SSD) at home, behind your own domain. This
document explains what happens to a request, where memory goes, what stops the Mac from being
overwhelmed, and the options that were weighed at each step. The decisions behind it are D16–D21 in
[`03-decisions.md`](03-decisions.md).

- Setting up the server: [`05-server-setup.md`](05-server-setup.md)
- Shipping new versions: [`06-releasing.md`](06-releasing.md)

---

## 1. The path of a request

```
 visitor's browser
      │  https://yourdomain.com/api/run
      ▼
 Cloudflare edge ─────────── TLS, DDoS protection, caching of /assets/*, a rate-limit rule
      │  (outbound tunnel the Mac opened; no open port at home)
      ▼
 cloudflared  (launchd: com.attnlab.tunnel)
      │  http://127.0.0.1:8000
      ▼
 uvicorn, 1 worker  (launchd: com.attnlab.api)        ← --limit-concurrency 100
      │
      ├─ guards.Headers     security headers on every response
      ├─ guards.BodyLimit   > 256 KB body              → 413
      ├─ guards.RateLimit   > 60 POSTs burst, 1/s      → 429 + Retry-After
      ├─ gzip               responses over 1 KB
      ▼
 route (routes.py / lens_routes.py / toklab_routes.py)
      │  text > 10,000 chars                            → 422, before any tokenizing
      ▼
 the model slot (AppState.run_serialized)
      │  8 already waiting                              → 503 busy + Retry-After
      │  client gave up while waiting                   → skipped, never computed
      │  memory guard                                   → free caches / evict / 503
      ▼
 torch in a thread, one at a time                       → 30 s limit
      │
      ▼
 result stored (bounded run store) → JSON / binary back up the same path
```

Anything that isn't `/api/*` (the pages and their JavaScript) is the built frontend, served by the
same process (`app.py`, `mount_frontend`). Its files under `/assets/` have a content hash in their
names and are sent `Cache-Control: immutable`, so after the first visitor Cloudflare's edge serves
them and the Mac never sees those requests. `index.html` is `no-cache`, so a release reaches everyone
on their next page load.

**Why one process for both** (D19). The frontend and the API it talks to come from the same
release, always. With the frontend hosted separately (Cloudflare Pages, say), every release has a
window where a new frontend calls an old API, or the reverse, and you'd have to design every change
to survive that. Here it can't happen.

---

## 2. Where user data lives, and when it's cleared

There are no accounts and no database, and nothing about users is written to disk except the access
log (IP address, time, request line; rotated at 20 MB, 5 files kept).

| What | Where | Bound | Cleared when |
|---|---|---|---|
| Loaded models | API process memory | `MI_RAM_BUDGET_GB` = 7 GB, including load peaks | least recently used evicted when needed; restart |
| Attention runs | API process memory, encoded | `MI_RUN_BUDGET_MB` = 512 MB | 10 min, or oldest first when full; restart |
| Logit-lens runs | API process memory | `MI_LENS_BUDGET_MB` = 512 MB and 4 runs | same |
| Model files | `~/.cache/huggingface` on the server | ~4 GB for the current zoo | only when you delete them |
| Logs | `~/attnlab/shared/logs` | 20 MB × 6 per log | rotated |
| View state | the URL (permalinks) | — | it's the visitor's |
| Theme | the visitor's browser `localStorage` | bytes | — |

A run that's been dropped (expired, evicted, or lost in a restart) comes back as `404 run_not_found`,
and both labs answer that by running the same input again. Visitors don't see an error.

---

## 3. Memory: how 16 GB is shared

macOS doesn't kill a process that uses too much memory the way Linux does. It compresses memory and
then **swaps to the SSD**. The machine gets slow for everyone, and heavy swapping wears the SSD. So
the limits have to be the app's own, and they have to leave room for macOS.

| Who | Allowance |
|---|---|
| macOS, the logged-in session, cloudflared | ~4–5 GB |
| Models (`MI_RAM_BUDGET_GB`) | 7 GB |
| Stored runs | 0.5 + 0.5 GB |
| Python + torch + tokenizers, forward-pass temporaries | ~1.5 GB |
| **API process ceiling (`MI_MEMORY_LIMIT_GB`)** | **10.5 GB** |
| Watchdog restart threshold | 12.5 GB |

### What each model costs (measured, `make measure`)

| model | stays resident | peak while loading | 512-token forward (M4 Pro) | stored run, 512 tokens |
|---|---:|---:|---:|---:|
| attn-only-2l-demo | 0.21 GB | 0.42 GB | 0.06 s | 2 MB |
| gpt2-small | 0.64 GB | 1.2 GB | 0.28 s | 18 MB |
| pythia-160m | 0.68 GB | 1.3 GB | 0.33 s | 18 MB |
| gpt2-medium | 1.6 GB | 2.4 GB | 0.72 s | 48 MB |
| Qwen3-0.6B | 3.1 GB | 5.4 GB | 1.0 s | 56 MB |

`models.yaml` holds these plus 10%. Expect the M1 Pro to be roughly 1.5–2× slower. Memory is the
same on any Apple Silicon Mac.

### Three things that were learned by measuring

**1. The peak is while loading, not while running** (D17). TransformerLens holds the raw
checkpoint and the processed weights at the same time, then drops the raw copy. So loading Qwen
needs 5.4 GB for a moment even though it keeps 3.1 GB. The zoo reserves `est_ram_mb +
load_extra_mb` before it loads anything, evicting other models first if it has to. A model whose
peak can't fit at all is shown as disabled with the reason.

**2. macOS kept freed memory counted against the process** (D17). After a load, "Malloc Large
(empty)", memory already freed, was 2.8 GB of Qwen's 6.4 GB footprint. The allocator keeps large
freed blocks for reuse. `MallocLargeCache=0` turns that off: 3.4 GB. It must be in the environment
before Python starts, which `deploy/run-api.sh` guarantees. The price is a slower gpt2-small
forward pass (0.13 → 0.25 s), because large tensors get fresh pages each time.

**3. Storing float32 attention was the real leak risk** (D20). A gpt2-small run at 512 tokens is
151 MB as float32. Runs were kept for 10 minutes with no size limit, so 20 visitors could pin 3 GB.
Runs are now encoded to the wire format (u8, causal triangle) *inside the forward pass*, one layer at
a time: 18 MB, and the full float32 tensor never exists.

### The layers of protection

From the cheapest to the bluntest:

1. **The zoo's budget.** Models load only if their peak fits, evicting the least recently used.
2. **Bounded run stores.** Oldest dropped first.
3. **The memory guard**, before every piece of model work. If the process footprint is over
   10.5 GB it drops all stored runs, then evicts models one by one. If it's *still* over with
   nothing left to free, the memory isn't anything the app can give back (a leak), so it refuses
   the request and exits cleanly for launchd to restart it.
4. **The watchdog**, a separate process every minute. It restarts the API if its footprint passes
   12.5 GB anyway, or if it stops answering for 3 minutes.
5. **launchd's `KeepAlive`** restarts the API whenever it exits, at most once per 10 s.

"Footprint" here is macOS's *physical footprint*, the number in Activity Monitor's Memory column
(`memory.py`). Unlike RSS, it includes compressed and swapped-out memory the process still owns.

---

## 4. Load: not cooking the Mac

**One forward pass at a time** (D6). There is a single worker and a single model slot, so
however many people arrive, the Mac does one model computation at a time on 6 threads, leaving 2 of
the M1 Pro's 8 performance cores for macOS. More workers would each hold their own copy of every
model, and memory is the constraint.

**So capacity is simple arithmetic.** At ~0.3–0.5 s per gpt2-small run on the M1 Pro, the server
does ~2–3 runs a second. A visitor typing triggers a run after each 300 ms pause, so one busy
visitor might send one run a second. Roughly: a handful of people typing at once is fine, and more
than that means queueing.

**What happens when there's more demand than that:**

- **The queue is capped at 8.** The 9th request gets `503 busy` with `Retry-After: 5` immediately,
  instead of joining a queue that grows without end. Measured: 20 simultaneous Qwen requests gave 9
  served and 11 turned away, all within 7.4 s.
- **Stale work is skipped.** When a visitor keeps typing, the frontend aborts the previous run. When
  that request's turn comes, the server sees the client is gone and skips it (5 of 6 in the test).
- **Rate limit per visitor.** 60 POSTs at once, then 1 a second, keyed by the address Cloudflare
  reports (`CF-Connecting-IP`). That header is only trusted because the server is reachable through
  the tunnel alone (`MI_TRUST_PROXY=1`).
- **30 s per request.** A Python thread can't be killed, so a timed-out request fails for the
  visitor, but the model slot stays taken until the computation actually finishes. Releasing it
  early would let two forward passes run at once.
- **Tokenizer-lab work** runs outside the model slot (tokenizing shouldn't queue behind a forward
  pass), bounded separately at 2 at a time.

**Heat and battery.** Six threads running constantly will warm an M1 Pro but not hurt it. Keep it
on a hard surface with the vents clear. A laptop plugged in 24/7 should cap its charge at 80%
([`05-server-setup.md`](05-server-setup.md) § 2).

**Bandwidth.** A home connection's *upload* is usually the narrowest pipe. Attention patterns are
~1.5 MB per layer at 512 tokens, sent one layer at a time, and gzip roughly thirds them. Static
files come from Cloudflare's cache, not your connection.

---

## 5. The knobs

Every limit is an environment variable in `deploy/server.env`. Override any of them on the server in
`~/attnlab/shared/local.env`, then `launchctl kickstart -k gui/$(id -u)/com.attnlab.api`.

| If you see… | Consider |
|---|---|
| Lots of `busy` in `/api/health` counters | more visitors than one Mac serves. Raise `MI_MAX_QUEUE` a little (longer waits), or accept it |
| `rate_limited` from normal use | raise `MI_RATE_BURST` |
| Qwen evicting the small models constantly | raise `MI_RAM_BUDGET_GB` to 8 (and `MI_MEMORY_LIMIT_GB` to 11.5), or make Qwen the only big model you keep |
| Swap in use (`deploy/status.sh`) | lower `MI_RAM_BUDGET_GB`, close other apps on the server |
| `memory_guard` counter rising | something is using more than expected; check `app.log` for which model |
| The Mac hot or loud | lower `MI_THREADS` to 4 (slower runs, cooler machine) |

---

## 6. Options considered

| Question | Chosen | Alternatives, and why not (for now) |
|---|---|---|
| CPU or GPU (MPS)? | CPU | MPS measured slower for these model sizes, and TransformerLens warns it can be silently wrong (D16) |
| How to reach the Mac | Cloudflare Tunnel | Port forwarding exposes your IP and an open port, and breaks behind CGNAT. Tailscale Funnel works too but gives less control over caching and rate limits |
| Where the frontend lives | Same process | Cloudflare Pages: faster edge, but two deploys that can disagree (D19) |
| How to run it | launchd LaunchAgents | Docker Desktop costs a fixed slice of the 16 GB for its VM; `pm2`/`supervisord` are extra installs doing what launchd already does |
| Zero-downtime deploys | ~10 s restart | Two processes side by side would need two copies of every model |
| Model downloads | at deploy, offline serving | Lazy downloads at request time: slow first requests, disk growth, Hub outages breaking loads (D18) |
| Bigger models | no (D21) | They don't fit 16 GB with headroom. If you ever want them: a cloud GPU for those models only (Modal, a Hugging Face GPU Space), leaving this Mac for the small ones |

**When to outgrow this setup.** When the `busy` counter climbs every day, or you want models that
don't fit. The cheapest next step is a second machine running the same release, with
cloudflared on both connected to the same tunnel. One thing would have to change first: a stored run
lives in the process that made it, so a visitor's follow-up requests (`/run/{id}/patterns`, the lens
views) must reach the same machine. That means session affinity at Cloudflare (a paid Load Balancer
feature) or putting the machine's name in the run id and routing on it.
