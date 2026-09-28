# Shared by the deploy scripts. Sourced, not run.
set -euo pipefail

ATTNLAB_HOME="${ATTNLAB_HOME:-$HOME/attnlab}"
REPO="$ATTNLAB_HOME/repo"
RELEASES="$ATTNLAB_HOME/releases"
SHARED="$ATTNLAB_HOME/shared"
LOGS="$SHARED/logs"
CURRENT="$ATTNLAB_HOME/current"
PREVIOUS_FILE="$ATTNLAB_HOME/previous"
LABEL_API="com.attnlab.api"
LABEL_TUNNEL="com.attnlab.tunnel"
LABEL_WATCHDOG="com.attnlab.watchdog"
LABEL_AWAKE="com.attnlab.awake"
DOMAIN="gui/$(id -u)"

# uv, fnm/node and cloudflared are installed per user, without admin.
export PATH="$HOME/.local/bin:$ATTNLAB_HOME/bin:$HOME/.local/share/fnm:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
if command -v fnm >/dev/null 2>&1; then eval "$(fnm env --shell bash)"; fi

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# Load server.env from a release, then local overrides. `set -a` exports
# every assignment; the files are plain KEY=value with # comments.
load_env() {
  local release="$1"
  set -a
  # shellcheck disable=SC1090,SC1091
  . "$release/deploy/server.env"
  [ -f "$SHARED/local.env" ] && . "$SHARED/local.env"
  set +a
}

api_health() {
  curl -fsS --max-time 5 "http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}/api/health"
}
