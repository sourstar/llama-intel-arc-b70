#!/usr/bin/env python3
"""Compare captured embeddings. Exits non-zero on failure, so it gates cleanly.

Two modes:

  compare.py a.json b.json --min-cosine 0.999
      Cross-config check. Same chunks embedded under production settings and
      under a conservative reference. Divergence means the production path is
      corrupting output — ggml-org/llama.cpp#26206.

  compare.py --determinism a.json
      Same-config check on a --repeat 2 capture. Repeated embeddings of
      identical input should land on the same vector. Drift means something in
      the SYCL path is non-deterministic, which makes every other result
      suspect.

      Bit-exact equality is the wrong bar here, and was tried: on B70 / SYCL
      every chunk of 1024 tokens or more comes back bit-identical, while the
      512-token ones drift by up to 5e-6 — the oneDNN SDPA path below the
      oneMKL FA floor is not bit-reproducible. Requiring `==` fails the gate
      every run and buries the failure that would matter. The threshold is
      separate from, and much tighter than, the cross-config one, which has
      to absorb a real change of batch geometry.
"""

import argparse
import json
import math
import sys

# Cross-config comparison spans two ubatch sizes, so it tolerates more.
DEFAULT_CROSS_COSINE = 0.999
# Same config, same input: anything below this is not float noise.
DEFAULT_DET_COSINE = 0.9999


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load(path):
    with open(path) as f:
        return json.load(f)


def check_cross(path_a, path_b, min_cosine):
    a, b = load(path_a), load(path_b)
    idx_b = {(r["id"], r["rep"]): r for r in b["records"]}

    worst, failures = 1.0, []
    print(f"{'chunk':8} {'rep':>4} {'tokens':>7} {'cosine':>10}")
    for ra in a["records"]:
        rb = idx_b.get((ra["id"], ra["rep"]))
        if rb is None:
            print(f"missing {ra['id']} rep{ra['rep']} in {path_b}", file=sys.stderr)
            failures.append((ra["id"], None))
            continue
        if ra["dim"] != rb["dim"]:
            print(f"dimension mismatch on {ra['id']}: {ra['dim']} vs {rb['dim']}",
                  file=sys.stderr)
            failures.append((ra["id"], None))
            continue
        c = cosine(ra["embedding"], rb["embedding"])
        worst = min(worst, c)
        flag = "" if c >= min_cosine else "  <-- FAIL"
        print(f"{ra['id']:8} {ra['rep']:>4} {ra['tokens']:>7} {c:>10.6f}{flag}")
        if c < min_cosine:
            failures.append((ra["id"], c))

    print(f"\nworst cosine: {worst:.6f}  (threshold {min_cosine})")
    if failures:
        print(f"FAIL: {len(failures)} chunk(s) diverged.")
        print("The production ubatch path is producing different vectors than the")
        print("reference. Do not ingest with these settings — lower -ub and re-run.")
        return 1
    print("PASS")
    return 0


def check_determinism(path, min_cosine):
    cap = load(path)
    by_id = {}
    for r in cap["records"]:
        by_id.setdefault(r["id"], []).append(r)

    worst, failures = 1.0, []
    for cid, reps in sorted(by_id.items()):
        if len(reps) < 2:
            continue
        base = reps[0]["embedding"]
        for other in reps[1:]:
            if other["embedding"] == base:
                print(f"{cid:8} rep{other['rep']}: identical")
                continue
            c = cosine(base, other["embedding"])
            worst = min(worst, c)
            if c >= min_cosine:
                print(f"{cid:8} rep{other['rep']}: cosine={c:.9f}")
            else:
                print(f"{cid:8} rep{other['rep']}: DIFFERS  cosine={c:.9f}  <-- FAIL")
                failures.append(cid)

    print(f"\nworst cosine: {worst:.9f}  (threshold {min_cosine})")
    if failures:
        print(f"FAIL: {len(failures)} chunk(s) non-deterministic.")
        return 1
    print("PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-cosine", type=float, default=None,
                    help=f"default {DEFAULT_CROSS_COSINE} cross-config, "
                         f"{DEFAULT_DET_COSINE} for --determinism")
    ap.add_argument("--determinism", action="store_true")
    args = ap.parse_args()

    if args.determinism:
        threshold = args.min_cosine
        if threshold is None:
            threshold = DEFAULT_DET_COSINE
        return check_determinism(args.files[0], threshold)
    if len(args.files) != 2:
        print("cross-config mode needs exactly two capture files", file=sys.stderr)
        return 2
    threshold = args.min_cosine
    if threshold is None:
        threshold = DEFAULT_CROSS_COSINE
    return check_cross(args.files[0], args.files[1], threshold)


if __name__ == "__main__":
    sys.exit(main())
