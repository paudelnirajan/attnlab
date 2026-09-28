#!/bin/bash
# Go back to the release that was live before the last deploy. Instant: no
# build, no tests, since that release already ran. Run it again to go forward.
set -euo pipefail
. "$(dirname "$0")/common.sh"

PREV="$(cat "$PREVIOUS_FILE" 2>/dev/null || true)"
[ -n "$PREV" ] && [ -d "$PREV" ] || die "no previous release recorded"
CUR="$(readlink "$CURRENT")"
log "rolling back: $(basename "$CUR") -> $(basename "$PREV")"
ln -sfn "$PREV" "$CURRENT.new" && mv -fh "$CURRENT.new" "$CURRENT"
echo "$CUR" > "$PREVIOUS_FILE"
launchctl kickstart -k "$DOMAIN/$LABEL_API"
for _ in $(seq 1 60); do
  if api_health 2>/dev/null | grep -q '"ready":true'; then log "live: $(basename "$PREV")"; exit 0; fi
  sleep 2
done
die "not healthy after rollback; see $LOGS/app.log"
