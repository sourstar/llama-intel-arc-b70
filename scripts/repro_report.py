#!/usr/bin/env python3
"""Pool many captures and report reproducibility with its sample size attached.

Takes the directory a `repro_trial.sh` run wrote and answers three questions
that a single capture cannot:

  1. Over ALL trials, how many chunks returned exactly one stored vector?
  2. Does the answer depend on the request pattern or the concurrency?
  3. When a chunk gave more than one answer, is it the prompt-cache path
     rather than drift?

Question 3 matters because the two look identical in a distinct-count and are
completely different problems. A cache hit returns ~0.015s and a stable
alternative vector; drift returns a full-prefill time and a different vector
every run. The timing test only works at concurrency 1 — above it queue time
swamps the difference — so that split is reported from concurrency-1 responses
only.

Usage:
    repro_report.py baselines/repro-<stamp>
"""

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage_reproducibility import digest, l2_normalize, to_f16_bytes  # noqa: E402

# Anything faster than this did not run a prefill. The gap is wide — prefills
# of production-length chunks are 0.15s and up, cache hits are ~0.015s.
#
# BUT this test is only valid at concurrency 1. Above it, wall-clock time
# includes queue time, so a cache hit sitting behind three other requests
# measures ~0.3s and reads as a prefill. An earlier version of this script
# classified on time regardless of concurrency and concluded "drift in the
# compute path" for 24 chunks that were simply cache hits waiting in a queue.
CACHE_HIT_SECONDS = 0.05


def load(run):
    caps = []
    for p in sorted(glob.glob(os.path.join(run, "*.json"))):
        try:
            c = json.load(open(p))
        except (OSError, ValueError):
            continue
        if "records" in c:
            caps.append(c)
    return caps


def main():
    if len(sys.argv) < 2:
        print("usage: repro_report.py <repro-run-dir>", file=sys.stderr)
        return 2
    run = sys.argv[1]
    caps = load(run)
    if not caps:
        print(f"no captures under {run}", file=sys.stderr)
        return 1

    # chunk -> stored hash -> list of response times
    overall = defaultdict(lambda: defaultdict(list))
    # (pattern, concurrency) -> chunk -> set of hashes
    by_cell = defaultdict(lambda: defaultdict(set))
    # prefill-only view: the path production actually uses
    prefill = defaultdict(set)

    requests = 0
    for c in caps:
        key = (c.get("pattern", "?"), c.get("concurrency", 1))
        for r in c["records"]:
            h = digest(to_f16_bytes(l2_normalize(r["embedding"])))
            overall[r["id"]][h].append(r["seconds"])
            by_cell[key][r["id"]].add(h)
            if key[1] == 1 and r["seconds"] >= CACHE_HIT_SECONDS:
                prefill[r["id"]].add(h)
            requests += 1

    chunks = sorted(overall)
    print(f"{len(caps)} captures, {requests} requests, {len(chunks)} chunks")
    print()

    stable = [c for c in chunks if len(overall[c]) == 1]
    prefill_stable = [c for c in chunks if len(prefill.get(c, set())) <= 1]
    print(f"all responses pooled:      {len(stable)}/{len(chunks)} chunks gave one stored vector")
    print(f"full prefills at concurrency 1: {len(prefill_stable)}/{len(chunks)} chunks gave one stored vector")
    print("  (only concurrency-1 responses can be classified by time; above that,")
    print("   queue time makes a cache hit look like a prefill)")
    print()

    print("by pattern and concurrency (chunks with a single stored vector):")
    for key in sorted(by_cell):
        cell = by_cell[key]
        ok = sum(1 for c in cell if len(cell[c]) == 1)
        print(f"  pattern={key[0]:<9} conc={key[1]:<3} {ok}/{len(cell)}")
    print()

    multi = [c for c in chunks if len(overall[c]) > 1]
    if not multi:
        print("every chunk returned exactly one stored vector in every trial.")
        return 0

    print(f"{len(multi)} chunk(s) returned more than one stored vector:")
    print(f"  {'chunk':8}{'values':>8}   breakdown")
    for c in multi:
        vals = overall[c]
        parts = []
        for h, times in sorted(vals.items(), key=lambda kv: -len(kv[1])):
            med = statistics.median(times)
            # "fast"/"slow" rather than "cache"/"prefill": these times are
            # pooled across concurrencies, where queue time makes the label
            # unreliable. The concurrency-1 view above is the one to trust.
            path = "fast" if med < CACHE_HIT_SECONDS else "slow"
            parts.append(f"{h[:8]} n={len(times)} {path} med={med:.3f}s")
        print(f"  {c:8}{len(vals):>8}   " + "; ".join(parts))

    print()
    cache_explained = [
        c for c in multi
        if len(prefill.get(c, set())) <= 1
    ]
    if len(cache_explained) == len(multi):
        print("Every one of them is explained by the prompt-cache path: restricted to")
        print("full-prefill responses, each chunk has a single value. That is the path")
        print("production takes, since every chunk in a corpus is a distinct prompt.")
    else:
        rest = [c for c in multi if c not in cache_explained]
        print(f"{len(rest)} chunk(s) differ even among full-prefill responses: {rest}")
        print("That is not the cache. It is drift in the compute path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
