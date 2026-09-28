#!/bin/bash
# Cut a release on the dev machine: bump the version, record it in
# CHANGELOG.md, commit, tag. It does NOT push; it prints the commands to push
# and to deploy, so nothing leaves your machine until you say so.
#
#   scripts/release.sh 0.2.0
#
# Semantic versioning, while under 1.0 (docs/06-releasing.md):
#   0.x.0  a new lab or feature, or anything that changes the API
#   0.x.y  fixes and small changes
set -euo pipefail
cd "$(dirname "$0")/.."

V="${1:-}"
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: scripts/release.sh X.Y.Z" >&2; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "release from main" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "commit or stash your changes first" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/v$V" >/dev/null && { echo "v$V already exists" >&2; exit 1; }
grep -q '^## Unreleased' CHANGELOG.md || { echo "CHANGELOG.md needs an '## Unreleased' section" >&2; exit 1; }
if [ -z "$(awk '/^## Unreleased/{f=1;next} /^## /{f=0} f && NF' CHANGELOG.md)" ]; then
  echo "the Unreleased section of CHANGELOG.md is empty: write what changed first" >&2; exit 1
fi

echo "checks: backend tests, frontend typecheck + tests"
uv run pytest -q -x
(cd web && npm run typecheck && npm test --silent)

sed -i '' -E "s/^version = \"[0-9.]+\"/version = \"$V\"/" pyproject.toml
(cd web && npm version "$V" --no-git-tag-version --allow-same-version >/dev/null)
uv lock -q
sed -i '' "s/^## Unreleased$/## Unreleased\n\n## $V — $(date +%Y-%m-%d)/" CHANGELOG.md

git add pyproject.toml uv.lock web/package.json web/package-lock.json CHANGELOG.md
git commit -qm "release: v$V"
git tag -a "v$V" -m "v$V"
echo
echo "tagged v$V. To publish and deploy:"
echo "  git push origin main v$V"
echo "  ssh <server> '~/attnlab/repo/deploy/deploy.sh v$V'"
