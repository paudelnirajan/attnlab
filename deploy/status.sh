#!/bin/bash
# One screen: what's live, whether it's healthy, memory, disk, services.
. "$(dirname "$0")/common.sh"
set +e
load_env "$(readlink "$CURRENT" 2>/dev/null || echo "$REPO")" 2>/dev/null
echo "live release:     $(basename "$(readlink "$CURRENT" 2>/dev/null)" 2>/dev/null)"
echo "previous release: $(basename "$(cat "$PREVIOUS_FILE" 2>/dev/null)" 2>/dev/null)"
echo
for l in $LABEL_API $LABEL_TUNNEL $LABEL_WATCHDOG $LABEL_AWAKE; do
  if launchctl print "$DOMAIN/$l" >/dev/null 2>&1; then
    printf '%-24s %s\n' "$l" "$(launchctl print "$DOMAIN/$l" | awk -F'= ' '/^\tstate/ {print $2; exit}')"
  else
    printf '%-24s %s\n' "$l" "not installed"
  fi
done
echo
api_health | python3 -m json.tool 2>/dev/null || echo "API not answering on ${API_HOST:-127.0.0.1}:${API_PORT:-8000}"
echo
echo "disk: $(df -h "$HOME" | awk 'NR==2 {print $4 " free of " $2}')"
echo "swap: $(sysctl -n vm.swapusage)"
echo "memory pressure: $(memory_pressure -Q 2>/dev/null | tail -1)"
