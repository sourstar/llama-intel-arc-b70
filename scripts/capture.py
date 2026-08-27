#!/usr/bin/env python3
"""Embed a fixture pool against one endpoint and save the vectors.

Usage:
    capture.py --url http://127.0.0.1:30437 --out /tmp/prod.json
    capture.py --url ... --out /tmp/p.json --repeat 10 --pattern shuffle --seed 7
    capture.py --url ... --out /tmp/p.json --concurrency 8

REQUEST ORDER IS PART OF THE EXPERIMENT, not an implementation detail. Two
things here depend on it:

  * llama.cpp keeps each slot's prompt cached, and a cache hit returns a
    measurably different vector than a full prefill — cosine 0.999965, max
    component 1.26e-3, which is ~40x the kernel-level drift this harness was
    built to find. Whether a request hits depends on what the slot last held.

  * with more than one slot, tokens from concurrent requests share a ubatch,
    so a vector depends on what else was in flight.

A single fixed order therefore measures one arbitrary path through both effects
and can report a configuration as perfectly reproducible when it is not. Every
run records its pattern and seed so a result can be reproduced exactly, and so
that a claim can be stated as "over N orderings" rather than "in the one order
we happened to use".

Patterns:
    cycle     fixture order, every repeat. The original behaviour. Guarantees
              no chunk follows itself, so every request is a full prefill.
    shuffle   independently shuffled per repeat from --seed. Adjacent
              duplicates across a repeat boundary are broken up, so this too
              is all full prefills — it varies everything else.
    adjacent  every chunk requested twice in a row. Adversarial: maximises
              cache hits, so roughly half the responses take the cache path.
    reverse   reverse fixture order, every repeat.
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def embed(url, text, timeout=600):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings",
        data=json.dumps({"input": text, "model": "embed"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body["data"][0]["embedding"], body.get("usage", {}).get("prompt_tokens")


def wait_ready(url, timeout=300):
    """Poll /health until the server finishes loading.

    Cold start is ~26s on the B70 (SYCL kernel JIT plus weight upload), so
    anything that assumes immediate readiness will race it.
    """
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


def build_order(chunks, pattern, repeat, seed):
    """Return a list of (rep, chunk) in the order they will be sent."""
    rng = random.Random(seed)
    order = []
    last = None
    for rep in range(repeat):
        if pattern == "cycle":
            seq = list(chunks)
        elif pattern == "reverse":
            seq = list(reversed(chunks))
        elif pattern == "adjacent":
            seq = [c for c in chunks for _ in (0, 1)]
        elif pattern == "shuffle":
            seq = list(chunks)
            rng.shuffle(seq)
            # A shuffle can put the same chunk at the end of one repeat and the
            # start of the next, which would serve it from cache and quietly
            # mix a second code path into a run meant to be all prefills.
            if last is not None and len(seq) > 1 and seq[0]["id"] == last:
                seq[0], seq[1] = seq[1], seq[0]
        else:
            raise ValueError(pattern)
        order.extend((rep, c) for c in seq)
        last = seq[-1]["id"]
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--fixtures", default="fixtures/chunks.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=1,
                    help="embed each chunk N times; use 2 to test determinism")
    ap.add_argument("--pattern", default="cycle",
                    choices=("cycle", "shuffle", "adjacent", "reverse"),
                    help="request order; see the module docstring")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --pattern shuffle; recorded in the capture")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight at once; >1 tests whether a "
                         "vector depends on what else was being embedded")
    ap.add_argument("--label", default="", help="free-text note stored in the capture")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not wait_ready(args.url):
        print(f"{args.url} never became healthy", file=sys.stderr)
        return 1

    with open(args.fixtures, encoding="utf-8") as f:
        chunks = json.load(f)

    order = build_order(chunks, args.pattern, args.repeat, args.seed)

    records = []
    lock = threading.Lock()

    def one(rep, c):
        t0 = time.time()
        vec, ntok = embed(args.url, c["text"])
        dt = time.time() - t0
        rec = {
            "id": c["id"],
            "rep": rep,
            "tokens": ntok if ntok is not None else c["actual_tokens"],
            "seconds": round(dt, 4),
            "dim": len(vec),
            "embedding": vec,
        }
        with lock:
            records.append(rec)
            if not args.quiet:
                print(f"  {c['id']} rep{rep}: {len(vec)}d in {dt:.3f}s")

    # Between waves, evict the slot's prompt cache with a throwaway request.
    #
    # Draining the wave is not enough. Completion order is not submission
    # order under concurrency, so the chunk the server finished last is not
    # the one the shuffle guard looked at, and it can be re-requested first in
    # the next wave — served from cache, on the other code path. Measured: two
    # of five shuffled runs at concurrency 4 produced exactly one unstable
    # chunk each, and in both cases that chunk sat within four positions of a
    # wave boundary. A unique prompt in between makes it impossible.
    def flush_cache(tag):
        try:
            embed(args.url, f"cache flush {tag} {time.time()}")
        except Exception:
            pass

    if args.concurrency <= 1:
        for rep, c in order:
            one(rep, c)
    else:
        # One wave per repeat, so every request in a wave really is in flight
        # alongside the others rather than trickling.
        for rep in range(args.repeat):
            wave = [c for r, c in order if r == rep]
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(lambda c: one(rep, c), wave))
            if args.pattern != "adjacent":
                flush_cache(rep)

    records.sort(key=lambda r: (r["rep"], r["id"]))

    with open(args.out, "w") as f:
        json.dump({
            "url": args.url,
            "label": args.label,
            "fixtures": args.fixtures,
            "pattern": args.pattern,
            "seed": args.seed,
            "repeat": args.repeat,
            "concurrency": args.concurrency,
            "sent_order": [c["id"] for _, c in order],
            "records": records,
        }, f)
    if not args.quiet:
        print(f"wrote {len(records)} vectors to {args.out} "
              f"(pattern={args.pattern} seed={args.seed} conc={args.concurrency})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
