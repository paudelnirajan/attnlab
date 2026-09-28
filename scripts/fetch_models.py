"""
Download everything the server will need, so it can run with HF_HUB_OFFLINE=1
(D18). deploy/deploy.sh runs this before switching to a new release.

    uv run python scripts/fetch_models.py

Models too big for this server's MI_RAM_BUDGET_GB are skipped, never fetched.

Only the files needed to load each model are fetched (hub.py), and every
tokenizer the Tokenizer lab lists. Files already on disk are not downloaded
again, so re-running it is cheap. Nothing is loaded into memory: it's safe to
run next to the live server.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import constants

from attnlab import hub
from attnlab.toklab import load_tokenizer_registry
from attnlab.zoo import ModelZoo


def _gb(n: int) -> str:
    return f"{n / 2**30:.1f} GB"


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    cache = Path(constants.HF_HUB_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    min_free = float(os.environ.get("FETCH_MIN_FREE_GB", "20"))
    print(f"HF cache: {cache} ({_gb(free)} free on that disk)")
    if free < min_free * 2**30:
        print(f"refusing: under {min_free:.0f} GB free. Clear space first (docs/05-server-setup.md § 8).")
        return 1

    zoo = ModelZoo()
    failed: list[str] = []
    repos: dict[str, bool] = {}  # repo -> tokenizer_only
    for spec in zoo.list_specs():
        if zoo.status(spec.id) == "disabled":
            print(f"  skip {spec.id}: {zoo.disabled_reason(spec.id)}")
            continue
        try:
            weights, tokenizer = hub.repos_for(spec.tl_name)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {spec.id}: cannot resolve its repos: {e}")
            failed.append(spec.id)
            continue
        repos[weights] = False
        if tokenizer:
            repos.setdefault(tokenizer, True)
    for tok in load_tokenizer_registry():
        repos.setdefault(tok.hf_name, True)

    for repo, tokenizer_only in repos.items():
        kind = "tokenizer" if tokenizer_only else "model"
        try:
            path = hub.fetch(repo, tokenizer_only=tokenizer_only)
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"  ok   {kind:9s} {repo} ({_gb(size)})")
        except Exception as e:  # noqa: BLE001 - report every failure, then exit non-zero
            print(f"  FAIL {kind:9s} {repo}: {e}")
            failed.append(repo)

    used = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    print(f"HF cache now {_gb(used)}; {_gb(shutil.disk_usage(cache).free)} free")
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
