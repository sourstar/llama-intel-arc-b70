#!/usr/bin/env python3
"""Profile how reproducible each chunk is, over many repeats.

`compare.py --determinism` answers pass/fail on two repeats. That is the right
shape for a gate and the wrong shape for understanding: with two samples a
chunk that drifts one run in three looks identical two thirds of the time, and
a single "identical" tells you nothing about whether longer chunks are safe or
merely luckier.

This takes a capture made with a large `--repeat` and reports, per chunk, how
many DISTINCT vectors came back and how far apart the extremes were. Run it
when the question is "which lengths are reproducible and how often", not
"did this build break".

Usage:
    capture.py --url ... --out /tmp/p.json --repeat 10
    determinism_profile.py /tmp/p.json
    determinism_profile.py /tmp/p.json --json      # machine readable
"""

import argparse
import json
import math
import sys
from collections import Counter


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def profile(cap):
    by_id = {}
    for r in cap["records"]:
        by_id.setdefault(r["id"], []).append(r)

    out = []
    for cid, reps in sorted(by_id.items()):
        vecs = [tuple(r["embedding"]) for r in reps]
        counts = Counter(vecs)
        distinct = len(counts)
        # Fraction of repeats that agreed with the most common result. A chunk
        # that is genuinely deterministic gives 1.0; one that drifts every time
        # gives 1/n.
        modal = counts.most_common(1)[0][1] / len(vecs)

        worst_cos, worst_abs = 1.0, 0.0
        if distinct > 1:
            uniq = list(counts)
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    worst_cos = min(worst_cos, cosine(uniq[i], uniq[j]))
                    worst_abs = max(worst_abs, max_abs_diff(uniq[i], uniq[j]))

        out.append({
            "id": cid,
            "category": reps[0].get("category", ""),
            "tokens": reps[0].get("tokens"),
            "repeats": len(vecs),
            "distinct": distinct,
            "modal_fraction": round(modal, 3),
            "worst_cosine": worst_cos,
            "worst_abs_component": worst_abs,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--fixtures", default="fixtures/chunks.json",
                    help="read token lengths from here when the capture omits them")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with open(args.capture) as f:
        cap = json.load(f)
    rows = profile(cap)

    # Captures carry the server's token count; fall back to the fixture file.
    try:
        with open(args.fixtures, encoding="utf-8") as f:
            fx = {c["id"]: c for c in json.load(f)}
        for r in rows:
            if not r["tokens"]:
                r["tokens"] = fx.get(r["id"], {}).get("actual_tokens")
            if not r["category"]:
                r["category"] = fx.get(r["id"], {}).get("category", "")
    except OSError:
        pass

    if args.json:
        print(json.dumps(rows, indent=1))
        return 0

    rows.sort(key=lambda r: (r["tokens"] or 0))
    print(f"{'chunk':7}{'tokens':>8}{'category':>12}{'repeats':>9}"
          f"{'distinct':>10}{'modal':>8}{'worst cosine':>15}{'worst |dx|':>13}")
    for r in rows:
        wc = "-" if r["distinct"] == 1 else f"{r['worst_cosine']:.9f}"
        wa = "-" if r["distinct"] == 1 else f"{r['worst_abs_component']:.3e}"
        print(f"{r['id']:7}{r['tokens'] or 0:>8}{r['category']:>12}{r['repeats']:>9}"
              f"{r['distinct']:>10}{r['modal_fraction']:>8.2f}{wc:>15}{wa:>13}")

    reproducible = [r for r in rows if r["distinct"] == 1]
    print()
    print(f"{len(reproducible)}/{len(rows)} chunks returned one vector every time")
    if reproducible:
        lens = [r["tokens"] or 0 for r in reproducible]
        print(f"   reproducible lengths: {sorted(lens)}")
    drifting = [r for r in rows if r["distinct"] > 1]
    if drifting:
        lens = [r["tokens"] or 0 for r in drifting]
        print(f"   drifting lengths:     {sorted(lens)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
