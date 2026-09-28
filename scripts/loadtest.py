"""
A small load test: N simulated visitors typing into the attention lab at once.
Run it against the real server before telling anyone about it, and again after
any change to the caps.

    uv run python scripts/loadtest.py https://yourdomain.com --users 10 --seconds 60
    uv run python scripts/loadtest.py http://127.0.0.1:8000 --model qwen3-0.6b

Each visitor sends a run about once a second (as the frontend does while
someone types), then fetches one layer of patterns, like the page does. Watch
`deploy/status.sh` or Activity Monitor on the server while it runs: memory
pressure should stay green and swap near zero. Expect some `503 busy` and
`429` once demand passes what one machine serves; that is the server
protecting itself, not failing.
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics
import threading
import time

import httpx

WORDS = "the cat sat on a mat while induction heads copy tokens they have seen before in context".split()


def visitor(base: str, model: str, until: float, stats: dict, lock: threading.Lock) -> None:
    rng = random.Random()
    with httpx.Client(base_url=base, timeout=60) as c:
        while time.time() < until:
            text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(20, 120)))
            t = time.time()
            try:
                r = c.post("/api/run", json={"model": model, "text": text})
                code = r.status_code
                if code == 200:
                    c.get(f"/api/run/{r.json()['run_id']}/patterns", params={"layers": "0"})
            except httpx.HTTPError as e:
                code = type(e).__name__
            with lock:
                stats["codes"][code] += 1
                if code == 200:
                    stats["latency"].append(time.time() - t)
            time.sleep(max(0.0, 1.0 - (time.time() - t)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="e.g. https://yourdomain.com or http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--model", default="gpt2-small")
    args = ap.parse_args()

    stats = {"codes": collections.Counter(), "latency": []}
    lock = threading.Lock()
    until = time.time() + args.seconds
    threads = [
        threading.Thread(target=visitor, args=(args.base, args.model, until, stats, lock)) for _ in range(args.users)
    ]
    print(f"{args.users} visitors on {args.model} for {args.seconds}s against {args.base} ...")
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lat = sorted(stats["latency"])
    print("responses:", dict(stats["codes"]))
    if lat:
        print(
            f"successful runs: {len(lat)} ({len(lat) / args.seconds:.1f}/s), latency median "
            f"{statistics.median(lat):.2f}s, p95 {lat[int(0.95 * (len(lat) - 1))]:.2f}s"
        )
    health = httpx.get(f"{args.base}/api/health", timeout=10).json()
    print("server:", {k: health[k] for k in ("resident_models", "queue_depth", "memory", "counters")})


if __name__ == "__main__":
    main()
