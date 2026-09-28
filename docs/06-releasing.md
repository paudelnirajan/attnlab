# Releasing, rolling back, and adding features

How a change gets from your dev machine to the site, and what every new feature has to do so it
can't hurt the server.

---

## The loop

```
 dev machine                                   GitHub                    server
 ───────────                                   ──────                    ──────
 branch, change, test locally
   make dev / npm run dev
 add a line to CHANGELOG.md "Unreleased"
 merge to main ───────────────────────────────► CI runs (tests, build)
 make release V=0.3.0
   tests again, bump version, tag
 git push origin main v0.3.0 ─────────────────► CI on the tag
                                                                         deploy.sh v0.3.0
                                                                           build · fetch · test
                                                                           switch · verify
                                                                           (auto-rollback)
```

1. **Work on a branch.** `make dev` plus `cd web && npm run dev` locally. `make serve` runs the
   production configuration (`deploy/server.env`, built frontend, one origin) when you want to see
   what the server will do.
2. **Write the changelog entry as you go**, under `## Unreleased` in `CHANGELOG.md`: what a visitor
   or you would notice, not the commit list.
3. **Merge to main.** GitHub Actions (`.github/workflows/ci.yml`) runs the backend tests on Apple
   Silicon, offline like the server, plus the frontend typecheck, tests and build.
4. **Release:** `make release V=0.3.0`. It refuses unless you're on a clean `main` with a non-empty
   Unreleased section. Then it runs the tests, sets the version in `pyproject.toml` and
   `web/package.json`, turns Unreleased into `## 0.3.0 — <date>`, commits, and tags `v0.3.0`.
   Nothing is pushed yet.
5. **Push:** `git push origin main v0.3.0`.
6. **Deploy:** `ssh attnlab@<server>.local '~/attnlab/repo/deploy/deploy.sh v0.3.0'`.
7. **Check:** `https://yourdomain.com/api/version` shows the new version, and the site works.

### What `deploy.sh` does, and when it stops

| Step | While… | If it fails |
|---|---|---|
| export the tag into `releases/v0.3.0-<sha>/` | old release serving | stops; nothing changed |
| `uv sync --frozen`, `npm ci && npm run build` | old release serving | stops; nothing changed |
| `fetch_models.py` (only new files download) | old release serving | stops; nothing changed |
| the whole test suite, offline, 2 threads | old release serving | stops; nothing changed |
| point `current` at it, restart the API | **~10 s of 502s** | — |
| wait up to 3 min for `/api/health` to show the new revision, ready | | points `current` back, restarts, exits 1 |

Every stop is logged to `~/attnlab/shared/logs/deploy.log`. Deploying the same tag again reuses the
built release directory, so it's quick.

### Rolling back

`~/attnlab/repo/deploy/rollback.sh` switches to the previous release and restarts. No build, no
tests (that release already passed them), about 10 seconds. Run it again to go forward. To go back
further, `deploy.sh v0.2.0` redeploys any older tag.

### A fix that can't wait

Branch from the tag if main has moved on: `git switch -c hotfix v0.3.0`, fix, then tag `v0.3.1`
from that branch by hand (`git tag -a v0.3.1 -m v0.3.1 && git push origin v0.3.1`) and deploy it.
Merge the branch back into main afterwards. `release.sh` only releases from main on purpose, so
hotfixes are the one manual case.

---

## Versions

Semantic versioning, while under 1.0:

- **0.x.0**: a new lab, a new model, a new feature, or anything that changes the API
- **0.x.y**: fixes, copy changes, performance, dependency updates
- **1.0.0**: when you decide it's stable and say so publicly

The version lives in `pyproject.toml` (the API reports it) and `web/package.json`. `release.sh`
keeps them equal. The deployed commit is the *revision*, shown next to the version in `/api/health`.

### Open tabs after a deploy

The frontend and API always ship together, but a visitor who had the page open before a deploy
keeps running the old JavaScript until they reload, and it now talks to the new API. So within a
release:

- **add** fields and endpoints freely;
- don't rename or remove anything the previous release's frontend uses. Remove it one release
  later, once no old tabs can be left.

Stored runs don't survive a restart, but that's already handled: the old tab gets `run_not_found`
and runs again.

---

## Adding a feature without hurting the server

Every limit in [`04-self-hosting.md`](04-self-hosting.md) exists because something could otherwise
grow without bound. A new feature has to keep that true. The checklist:

### A new endpoint or lab

- [ ] **Model work goes through the model slot**: `await _serialized(request, fn, ...)`
      (`api/routes.py`). Never call the model from a route directly. That's what keeps one forward
      pass at a time, the queue cap, the memory guard, the timeout and the abandoned-request skip
      in force.
- [ ] **CPU-heavy work that isn't the model** (like the tokenizer lab) goes behind its own
      semaphore, as `_tok_work` in `api/toklab_routes.py` does.
- [ ] **Bound every input**: text through `check_text()`, lists with `Field(max_length=…)`, numbers
      with `ge`/`le`. Ask "what does the biggest request cost?"
- [ ] **Anything kept between requests goes in a `RunStore`** with a byte budget (`api/state.py`),
      never a plain dict. Give the budget a setting in `settings.py` and a value in
      `deploy/server.env`.
- [ ] **Measure it**: the transient memory of the biggest allowed request, on the biggest model.
      `memory.footprint_mb()` before and after is enough.
- [ ] **Tests** next to the existing ones, and a line in `docs/02-api.md`.
- [ ] For a new lab page: a route in `web/src/main.tsx`, an entry in `web/src/labs/registry.ts`.
      Handle `run_not_found` by re-running (see `labs/lens/hooks.ts`), and abort requests the
      visitor has superseded (see the run effect in `App.tsx`).

### A new model

1. Measure it: `make measure MODEL=<TransformerLens name>`. It prints the `models.yaml` fields.
2. **Check that it fits**: `est_ram_mb + load_extra_mb` must be well under 7 GB (the server's
   `MI_RAM_BUDGET_GB`), which in practice means under ~1B parameters. Remember it evicts other models
   while it loads. The zoo refuses a model that can't fit, but the rule (D21) is to not add one.
3. Add the entry to `src/attnlab/models.yaml`, `tier: lazy` unless it should load at startup.
4. Try it in every lab locally, and check the logit lens's anchor checks all pass for it.
5. Release as usual. The deploy's `fetch_models.py` downloads it before the switch.

### A new dependency

- Check it against the `numpy<2` ceiling (D3) and install it with `uv add`, so `uv.lock` records it.
- Anything that downloads at import or first use (models, datasets) must go through
  `scripts/fetch_models.py`, because the server runs offline (D18).

### A change to a cap

Change the value in `deploy/server.env`, say why in the commit, and release it like code. To try a
value on the server first, put it in `~/attnlab/shared/local.env` and restart the API. That file
overrides `server.env` and isn't in git, so move the value into `server.env` once you keep it.
