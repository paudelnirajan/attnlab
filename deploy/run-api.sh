#!/bin/bash
# What launchd runs (com.attnlab.api). Starts the release `current` points at,
# with deploy/server.env and ~/attnlab/shared/local.env loaded. launchd
# restarts it if it exits.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd -P)"
. "$here/deploy/common.sh"
load_env "$here"
cd "$here"
exec .venv/bin/uvicorn attnlab.api.app:app \
  --host "$API_HOST" --port "$API_PORT" \
  --workers 1 \
  --limit-concurrency 100 \
  --timeout-keep-alive 5 \
  --timeout-graceful-shutdown 10 \
  --proxy-headers --forwarded-allow-ips 127.0.0.1 \
  --no-server-header \
  --log-config deploy/logging.json
