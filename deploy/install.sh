#!/bin/bash
# One-time setup on the server, as the server's standard (non-admin) user.
# docs/05-server-setup.md walks through everything before and after this.
#
#   deploy/install.sh <git-url> [first-ref]
#
# Creates ~/attnlab/{repo,releases,shared/logs,bin}, clones the repo, writes
# the launchd services into ~/Library/LaunchAgents and loads them:
#
#   com.attnlab.api       the server (deploy/run-api.sh); restarted if it exits
#   com.attnlab.tunnel    cloudflared; only once ~/.cloudflared/config.yml exists
#   com.attnlab.watchdog  deploy/watchdog.sh every 60 s
#   com.attnlab.awake     caffeinate: no system sleep while on power
#
# Safe to re-run: it rewrites the service files and reloads them.
set -euo pipefail
. "$(dirname "$0")/common.sh"

URL="${1:-}"
FIRST_REF="${2:-}"

[ "$(uname -s)" = Darwin ] || die "this is for macOS"
if id -Gn | tr ' ' '\n' | grep -qx admin; then
  log "warning: $(id -un) is an admin account. The plan is a standard user (docs/05-server-setup.md § 1)."
fi
for tool in git curl uv npm; do command -v "$tool" >/dev/null || die "$tool not found (docs/05-server-setup.md § 3)"; done

mkdir -p "$RELEASES" "$LOGS" "$ATTNLAB_HOME/bin" "$HOME/Library/LaunchAgents"
if [ ! -d "$REPO/.git" ]; then
  [ -n "$URL" ] || die "usage: install.sh <git-url> [first-ref]"
  git clone "$URL" "$REPO"
fi
if [ ! -f "$SHARED/local.env" ]; then
  cat > "$SHARED/local.env" <<'ENV'
# Server-only settings and secrets, read after deploy/server.env. Not in git.
# HEALTHCHECK_URL=https://hc-ping.com/your-check-uuid
ENV
  chmod 600 "$SHARED/local.env"
fi

plist() {  # label, then the <dict> body
  local file="$HOME/Library/LaunchAgents/$1.plist"
  cat > "$file" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$1</string>
$2
</dict>
</plist>
XML
  plutil -lint "$file" >/dev/null
  launchctl bootout "$DOMAIN/$1" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$file"
  log "loaded $1"
}

# The API runs whatever `current` points at, so a deploy only swaps the link.
plist "$LABEL_API" "  <key>ProgramArguments</key><array><string>$CURRENT/deploy/run-api.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Standard</string>
  <key>SoftResourceLimits</key><dict><key>NumberOfFiles</key><integer>4096</integer></dict>
  <key>StandardOutPath</key><string>$LOGS/api.out.log</string>
  <key>StandardErrorPath</key><string>$LOGS/api.err.log</string>"

plist "$LABEL_WATCHDOG" "  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$CURRENT/deploy/watchdog.sh</string></array>
  <key>StartInterval</key><integer>60</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$LOGS/watchdog.out.log</string>
  <key>StandardErrorPath</key><string>$LOGS/watchdog.err.log</string>"

# -i idle sleep, -m disk sleep, -s system sleep (the last only on AC power).
plist "$LABEL_AWAKE" "  <key>ProgramArguments</key><array><string>/usr/bin/caffeinate</string><string>-ims</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>"

if [ -f "$HOME/.cloudflared/config.yml" ]; then
  CF="$(command -v cloudflared || echo "$ATTNLAB_HOME/bin/cloudflared")"
  [ -x "$CF" ] || die "cloudflared not found (docs/05-server-setup.md § 5)"
  plist "$LABEL_TUNNEL" "  <key>ProgramArguments</key><array><string>$CF</string><string>tunnel</string><string>--config</string><string>$HOME/.cloudflared/config.yml</string><string>run</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOGS/tunnel.out.log</string>
  <key>StandardErrorPath</key><string>$LOGS/tunnel.err.log</string>"
else
  log "no ~/.cloudflared/config.yml yet: skipping the tunnel service (re-run install.sh after § 5)"
fi

if [ -n "$FIRST_REF" ]; then
  "$REPO/deploy/deploy.sh" "$FIRST_REF"
elif [ ! -e "$CURRENT" ]; then
  log "nothing deployed yet: run $REPO/deploy/deploy.sh <tag>"
fi
