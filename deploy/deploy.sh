#!/bin/bash
# Deploy a tagged release on the server. Run as the server user, on the server
# (or over SSH: `ssh attnlab@server '~/attnlab/repo/deploy/deploy.sh v0.2.0'`).
#
#   deploy.sh v0.2.0        a tag (the normal case)
#   deploy.sh main          a branch or commit, for trying something out
#
# Everything slow or risky happens while the old release keeps serving:
#
#   1. export the commit into releases/<ref>-<sha>/   (a plain directory)
#   2. install Python deps (uv, from uv.lock exactly) and build the frontend
#   3. download any model files it needs that aren't on disk yet
#   4. run the test suite against it. Failure stops here; nothing changed.
#   5. point `current` at it and restart the API        <- ~10 s of downtime
#   6. wait for /api/health to report the new revision, ready
#   7. if it doesn't within 3 minutes, point `current` back and restart again
#
# Rolling back later is deploy/rollback.sh. Old releases beyond the newest 3
# are deleted.
set -euo pipefail
. "$(dirname "$0")/common.sh"

REF="${1:-}"
[ -n "$REF" ] || die "usage: deploy.sh <tag|branch|commit>"
[ -d "$REPO/.git" ] || die "$REPO is not a git clone; run deploy/install.sh first"
command -v uv >/dev/null || die "uv not found (docs/05-server-setup.md § 3)"
command -v npm >/dev/null || die "node/npm not found (docs/05-server-setup.md § 3)"

mkdir -p "$RELEASES" "$LOGS"
exec > >(tee -a "$LOGS/deploy.log") 2>&1
log "=== deploy $REF ==="

git -C "$REPO" fetch --tags --prune --force origin
SHA="$(git -C "$REPO" rev-parse --verify --short "origin/$REF^{commit}" 2>/dev/null || git -C "$REPO" rev-parse --verify --short "$REF^{commit}")" \
  || die "unknown ref: $REF"
NAME="$(printf '%s' "$REF" | tr '/' '-')-$SHA"
REL="$RELEASES/$NAME"
log "commit $SHA -> $REL"

if [ ! -f "$REL/.built" ]; then
  rm -rf "$REL"
  mkdir -p "$REL"
  git -C "$REPO" archive "$SHA" | tar -x -C "$REL"
  echo "$SHA" > "$REL/REVISION"

  log "python dependencies"
  (cd "$REL" && uv sync --frozen)

  log "frontend build"
  (cd "$REL/web" && npm ci --no-audit --no-fund && npm run build)

  touch "$REL/.built"
fi

log "model files"
load_env "$REL"
(cd "$REL" && HF_HUB_OFFLINE=0 .venv/bin/python scripts/fetch_models.py)

log "tests"
# A clean environment, not server.env: the tests make their own settings
# (server.env's rate limit would trip them). Two threads, to stay out of
# the live server's way.
(cd "$REL" && env -i HOME="$HOME" PATH="$PATH" HF_HUB_OFFLINE=1 MallocLargeCache=0 \
  MI_THREADS=2 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m pytest -q -x -p no:cacheprovider) || die "tests failed; still serving the old release"

PREV="$(readlink "$CURRENT" 2>/dev/null || true)"
switch_to() {
  ln -sfn "$1" "$CURRENT.new"
  mv -fh "$CURRENT.new" "$CURRENT"   # atomic replace of the symlink
}

wait_ready() {
  local want="$1" i body
  for i in $(seq 1 90); do
    body="$(api_health 2>/dev/null || true)"
    if printf '%s' "$body" | grep -q "\"revision\":\"$want\"" && printf '%s' "$body" | grep -q '"ready":true'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

log "switching to $NAME (was ${PREV:-nothing})"
[ -n "$PREV" ] && [ "$PREV" != "$REL" ] && echo "$PREV" > "$PREVIOUS_FILE"
switch_to "$REL"
if launchctl print "$DOMAIN/$LABEL_API" >/dev/null 2>&1; then
  launchctl kickstart -k "$DOMAIN/$LABEL_API"
else
  log "the API service isn't installed yet; run deploy/install.sh"
  exit 0
fi

if wait_ready "$(cat "$REL/REVISION")"; then
  log "live: $NAME"
else
  log "new release did not become healthy; see $LOGS/app.log"
  if [ -n "$PREV" ] && [ "$PREV" != "$REL" ]; then
    switch_to "$PREV"
    launchctl kickstart -k "$DOMAIN/$LABEL_API"
    wait_ready "$(cat "$PREV/REVISION")" && log "rolled back to $(basename "$PREV")"
  fi
  exit 1
fi

# Keep the newest 3 releases, plus whatever current and previous point at.
keep_prev="$(cat "$PREVIOUS_FILE" 2>/dev/null || true)"
ls -1dt "$RELEASES"/*/ 2>/dev/null | sed 's:/$::' | tail -n +4 | while read -r old; do
  [ "$old" = "$REL" ] || [ "$old" = "$keep_prev" ] || { log "pruning $(basename "$old")"; rm -rf "$old"; }
done
log "=== done ==="
