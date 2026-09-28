#!/bin/bash
# Runs every minute (com.attnlab.watchdog). The last line of defence, outside
# the API process, for the cases the app's own guards can't cover:
#
#   - API stuck or dead for 3 checks in a row  -> restart it
#   - API footprint over WATCHDOG_KILL_GB       -> restart it (the in-app guard
#                                                  should never let this happen)
#   - disk low, swap high                       -> log it, report failure
#   - launchd's stdout/stderr logs over 20 MB   -> rotate (5 kept)
#
# If HEALTHCHECK_URL is set (a free healthchecks.io check, in local.env), it
# is pinged every run: success when all is well, /fail otherwise. When the Mac
# itself is off or offline, the pings stop, and healthchecks.io emails you.
. "$(dirname "$0")/common.sh"
set +e
load_env "$(readlink "$CURRENT" 2>/dev/null || echo "$REPO")" 2>/dev/null
mkdir -p "$LOGS"
exec >>"$LOGS/watchdog.log" 2>&1
STATE="$SHARED/watchdog.failures"
problems=()

body="$(api_health 2>/dev/null)"
if [ -z "$body" ]; then
  n=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "$STATE"
  problems+=("API not answering ($n)")
  if [ "$n" -ge 3 ]; then
    log "API unhealthy for $n checks; restarting"
    launchctl kickstart -k "$DOMAIN/$LABEL_API"
    echo 0 > "$STATE"
  fi
else
  echo 0 > "$STATE"
  fp="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["memory"]["footprint_mb"] or 0)' 2>/dev/null || echo 0)"
  limit=$(python3 -c "print(int(float('${WATCHDOG_KILL_GB:-12.5}') * 1024))")
  if [ "${fp%.*}" -gt "$limit" ]; then
    log "API footprint ${fp} MB > ${limit} MB; restarting"
    problems+=("footprint $fp MB")
    launchctl kickstart -k "$DOMAIN/$LABEL_API"
  fi
fi

free_gb=$(df -g "$HOME" | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt "${WATCHDOG_MIN_FREE_DISK_GB:-15}" ]; then
  problems+=("disk ${free_gb} GB free")
fi
swap_mb=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
if [ "$swap_mb" -gt $(( ${WATCHDOG_MAX_SWAP_GB:-2} * 1024 )) ]; then
  problems+=("swap ${swap_mb} MB used")
fi

for f in "$LOGS"/*.out.log "$LOGS"/*.err.log "$LOGS/watchdog.log" "$LOGS/deploy.log"; do
  [ -f "$f" ] || continue
  if [ "$(stat -f %z "$f")" -gt 20971520 ]; then
    for i in 4 3 2 1; do [ -f "$f.$i" ] && mv -f "$f.$i" "$f.$((i + 1))"; done
    cp "$f" "$f.1" && : > "$f"   # copy-truncate: launchd keeps its file open
  fi
done

if [ ${#problems[@]} -gt 0 ]; then
  log "problems: ${problems[*]}"
  [ -n "${HEALTHCHECK_URL:-}" ] && curl -fsS -m 10 --data-raw "${problems[*]}" "$HEALTHCHECK_URL/fail" >/dev/null
else
  [ -n "${HEALTHCHECK_URL:-}" ] && curl -fsS -m 10 "$HEALTHCHECK_URL" >/dev/null
fi
exit 0
