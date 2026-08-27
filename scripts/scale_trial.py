#!/usr/bin/env python3
"""Embed a large pool repeatedly and report the corpus-scale reproducibility bound.

Each chunk is embedded once per pass, in separate full passes over the whole
pool, never back to back, so every response is a full prefill and none is
served from the slot prompt cache. The stored form — L2-normalised, f16 — is hashed and
compared.

The result is reported as a confidence bound, because "0 mismatches" on its own
means nothing without the sample size. With n independent chunks and zero
mismatches, the one-sided 95% upper bound on the per-chunk mismatch rate is
1 - 0.05**(1/n), which is the exact Clopper-Pearson bound and close to the
familiar 3/n rule of three.

That bound is then projected onto the real corpus, because a rate that sounds
tiny is not tiny multiplied by 1.37 million chunks. This is the number that
says whether hash verification is safe, and it is usually sobering.

Usage:
    scale_trial.py --url http://127.0.0.1:30437 --fixtures fixtures/scale.json
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import urllib.request

CORPUS_CHUNKS = 1_373_515  # gutenberg, measured 2026-08-27


def embed(url, text, timeout=600):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings",
        data=json.dumps({"input": text, "model": "embed"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return (body["data"][0]["embedding"],
            body.get("usage", {}).get("prompt_tokens"))


def wait_ready(url, timeout=300):
    """Poll /health until the model has loaded.

    Cold load is ~26s on this card. Without this, a caller that starts a
    container and immediately runs a trial gets connection errors and reports
    them as "the trial did not complete" — which is what correctness.sh did.
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


def stored_hash(v):
    n = math.sqrt(sum(x * x for x in v))
    if n:
        v = [x / n for x in v]
    b = b"".join(struct.pack("<e", x) for x in v)
    return hashlib.sha256(b).hexdigest()


def sweep(url, chunks, label):
    out, toks = {}, {}
    t0 = time.time()
    for i, c in enumerate(chunks, 1):
        vec, n = embed(url, c["text"])
        out[c["id"]] = stored_hash(vec)
        toks[c["id"]] = n
        if i % 100 == 0 or i == len(chunks):
            el = time.time() - t0
            rate = sum(v or 0 for v in toks.values()) / el if el else 0
            print(f"\r  {label}: {i}/{len(chunks)}  {rate:.0f} tok/s", end="", flush=True)
    print()
    return out, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--fixtures", default="fixtures/scale.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--passes", type=int, default=2,
                    help="full passes over the pool; each chunk gets one "
                         "observation per pass, and no chunk ever follows "
                         "itself, so every response is a full prefill")
    ap.add_argument("--corpus-chunks", type=int, default=CORPUS_CHUNKS)
    args = ap.parse_args()

    if not wait_ready(args.url):
        print(f"{args.url} never became healthy", file=sys.stderr)
        return 2

    with open(args.fixtures, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"{len(chunks)} distinct chunks from {len(chunks)} distinct documents")
    print()

    # N full passes rather than N calls per chunk: a chunk must never follow
    # itself, or the second response comes from the prompt cache and is a
    # different code path. Only hashes are kept, so passes are cheap in memory
    # however large the pool.
    seen = {}
    toks = {}
    for p in range(args.passes):
        h, t = sweep(args.url, chunks, f"pass {p + 1}")
        for k, v in h.items():
            seen.setdefault(k, set()).add(v)
        toks.update({k: v for k, v in t.items() if v})

    mismatches = sorted(k for k, v in seen.items() if len(v) > 1)
    n = len(seen)
    print()
    print(f"chunks compared: {n} over {args.passes} passes "
          f"({n * args.passes} embeddings)")
    print(f"mismatches:      {len(mismatches)}")

    tv = [t for t in toks.values() if t]
    if tv:
        tv.sort()
        print(f"token lengths:   min {tv[0]}  p50 {tv[len(tv)//2]}  "
              f"p95 {tv[int(len(tv)*0.95)]}  max {tv[-1]}")

    print()
    # The unit of evidence is a COMPARISON, not a chunk. k passes over a chunk
    # give k-1 independent sequential comparisons, all of which had to match.
    # Counting chunks instead throws away every pass after the second — with
    # 6 passes over 1100 chunks that is 5500 comparisons reported as 1100.
    comparisons = n * max(args.passes - 1, 1)
    if not mismatches:
        bound = 1 - 0.05 ** (1.0 / comparisons)
        print(f"zero mismatches over {n} distinct chunks x {args.passes} passes")
        print(f"= {comparisons:,} independent comparisons.")
        print(f"95% upper bound on the per-comparison mismatch rate: {bound:.3e}")
        print(f"projected onto {args.corpus_chunks:,} corpus chunks, that permits")
        print(f"up to {bound * args.corpus_chunks:,.0f} mismatching vectors.")
        print()
        print("Read that as the limit of what sampling can prove, not as a")
        print("defect count. Driving it to zero by sampling would need a sample")
        print("the size of the corpus — which is just the verification pass.")
    else:
        rate = len(mismatches) / comparisons
        print(f"per-comparison mismatch rate: {rate:.3e}")
        print(f"projected onto {args.corpus_chunks:,} corpus chunks: "
              f"{rate * args.corpus_chunks:,.0f} mismatching vectors.")
        print(f"first few: {mismatches[:10]}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"n": n, "passes": args.passes,
                       "mismatches": mismatches,
                       "hashes": {k: sorted(v) for k, v in seen.items()},
                       "tokens": toks}, f)
        print(f"\nwrote {args.out}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
