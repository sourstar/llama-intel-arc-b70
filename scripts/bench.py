#!/usr/bin/env python3
"""Throughput benchmark. Appends one row per run to baselines/history.csv.

The point is comparability across image bumps and flag changes, so the run
metadata is recorded alongside the numbers. Nothing here is meaningful as an
absolute figure; it is meaningful as a delta against the previous row.

Usage:
    bench.py --url http://localhost:30436 --note "b10630 oneDNN on"
    bench.py --url http://localhost:30436 --concurrency 4 --passes 5
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from queue import Queue

HISTORY = "baselines/history.csv"
FIELDS = [
    "timestamp", "note", "image", "concurrency", "passes", "chunks",
    "total_tokens", "wall_seconds", "tokens_per_second",
    "p50_ms", "p95_ms", "max_ms", "errors",
]


def embed(url, text, timeout=600):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings",
        data=json.dumps({"input": text, "model": "embed"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body.get("usage", {}).get("prompt_tokens")


def default_image():
    """The image is the load-bearing column in history.csv: a row that does not
    name it cannot be compared against the next bump, which is the only thing
    the file is for. `--image` wins, then $IMAGE, then config.env — so a bare
    `bench.py --note ...`, which is how the README says to run it, still
    records what it measured.

    config.env is resolved by sourcing it, not by reading it. Its values are
    shell expressions (`IMAGE="${IMAGE:-...}"`, so the environment can win), and
    a text parse recorded the expression itself as the image name. Asking the
    shell is also the only way this cannot drift from what the scripts do.
    """
    from_env = os.environ.get("IMAGE")
    if from_env:
        return from_env
    try:
        out = subprocess.run(
            ["bash", "-c", 'source config.env >/dev/null 2>&1; printf %s "$IMAGE"'],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def wait_ready(url, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30436")
    ap.add_argument("--fixtures", default="fixtures/chunks.json")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="match -np on the server; higher just queues")
    ap.add_argument("--passes", type=int, default=3,
                    help="how many times to send the full fixture set")
    ap.add_argument("--note", default="")
    ap.add_argument("--image", default=default_image())
    ap.add_argument("--warmup", type=int, default=None,
                    help="requests to discard before timing; default is one "
                         "full pass over the fixture pool")
    ap.add_argument("--history", default=HISTORY,
                    help="CSV to append the row to. soak.sh points this at its "
                         "own file: a 30-minute soak is ~30 rounds, and 30 rows "
                         "of 'soak round N' in the baseline history buries the "
                         "one-row-per-run comparison it exists for.")
    args = ap.parse_args()

    if not wait_ready(args.url):
        print(f"{args.url} not healthy", file=sys.stderr)
        return 1

    with open(args.fixtures, encoding="utf-8") as f:
        chunks = json.load(f)

    # Warm up on exactly the work about to be measured, for a full pass.
    #
    # This default was 2 requests, and 2 is not close. Measured on B70/b10630,
    # fresh container, 24-chunk pool at concurrency 2:
    #
    #     warmup=0    first bench 2095 tok/s, next 3237
    #     warmup=24   first bench 3115 tok/s, next 3135
    #     warmup=72   first bench 3127 tok/s, next 3134
    #
    # A cold first bench reads ~35% low, which is far larger than any flag
    # difference this harness is trying to detect — and because every
    # configuration in a sweep gets a fresh container, EVERY row was cold. One
    # full pass converges; three passes add nothing.
    warmup = args.warmup if args.warmup is not None else len(chunks)
    for i in range(warmup):
        embed(args.url, chunks[i % len(chunks)]["text"])

    work = Queue()
    for _ in range(args.passes):
        for c in chunks:
            work.put(c)

    latencies, tokens, errors = [], [], []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                c = work.get_nowait()
            except Exception:
                return
            t0 = time.time()
            try:
                ntok = embed(args.url, c["text"])
                dt = (time.time() - t0) * 1000
                with lock:
                    latencies.append(dt)
                    tokens.append(ntok if ntok is not None else c["actual_tokens"])
            except Exception as e:
                with lock:
                    errors.append(str(e))
            finally:
                work.task_done()

    t_start = time.time()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_start

    if not latencies:
        print("all requests failed", file=sys.stderr)
        for e in errors[:3]:
            print(f"  {e}", file=sys.stderr)
        return 1

    total_tokens = sum(tokens)
    ordered = sorted(latencies)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": args.note,
        "image": args.image,
        "concurrency": args.concurrency,
        "passes": args.passes,
        "chunks": len(latencies),
        "total_tokens": total_tokens,
        "wall_seconds": round(wall, 2),
        "tokens_per_second": round(total_tokens / wall, 1),
        "p50_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1], 1),
        "max_ms": round(ordered[-1], 1),
        "errors": len(errors),
    }

    os.makedirs(os.path.dirname(args.history) or ".", exist_ok=True)
    exists = os.path.exists(args.history)
    with open(args.history, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

    width = max(len(k) for k in row)
    for k, v in row.items():
        print(f"  {k:<{width}}  {v}")
    print()
    print(f"appended to {args.history}")

    if errors:
        print(f"\n{len(errors)} request(s) failed — results are not comparable",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
