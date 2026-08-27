#!/usr/bin/env python3
"""Is a per-dimension translator from one configuration's vectors to another's
implementable?

The question is whether B[i] = f_i(A[i]) — whether the new configuration's
value in dimension i can be recovered from the old configuration's value in
that same dimension, via some per-dimension table or curve. If it can, a corpus
could be converted instead of re-embedded.

Two things have to hold for that to even be well posed, and this script tests
both:

  1. A must be a FUNCTION of the input. For 62% of token lengths the old
     configuration is not reproducible, so there is no single A to translate
     from. Only the bands where both configurations are deterministic can be
     asked about at all — hence the restriction to 900-1099 tokens, where both
     are stable and where production chunks actually sit.

  2. D = B - A must be determined by A, per dimension. If D is instead
     uncorrelated noise, no table exists: the same A[i] occurs with many
     different B[i], and a lookup keyed on (dimension, value) is one-to-many.

Reports, per dimension: the linear fit D ~ a*A + b with its R², and the
residual spread. An R² near zero across dimensions means the relationship is
not there.

Usage:
    translator_probe.py old.json new.json
"""

import json
import sys

import numpy as np


def load(path):
    cap = json.load(open(path))
    recs = {}
    for r in cap["records"]:
        recs.setdefault(r["id"], r["embedding"])
    return recs


def main():
    if len(sys.argv) < 3:
        print("usage: translator_probe.py old.json new.json", file=sys.stderr)
        return 2
    a_recs, b_recs = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(set(a_recs) & set(b_recs))
    if len(ids) < 8:
        print(f"only {len(ids)} chunks in common; need more", file=sys.stderr)
        return 1

    A = np.array([a_recs[i] for i in ids], dtype=np.float64)
    B = np.array([b_recs[i] for i in ids], dtype=np.float64)
    D = B - A
    n, dim = A.shape
    print(f"{n} chunks x {dim} dimensions")
    print()

    # Scale of the difference against the scale of the values.
    print("magnitude")
    print(f"  |A| component        : mean {np.abs(A).mean():.4e}  max {np.abs(A).max():.4e}")
    print(f"  |B - A| component    : mean {np.abs(D).mean():.4e}  max {np.abs(D).max():.4e}")
    rel = np.abs(D) / np.maximum(np.abs(A), 1e-12)
    print(f"  relative |D|/|A|     : median {np.median(rel):.3e}  p99 {np.percentile(rel, 99):.3e}")
    print(f"  identical components : {(D == 0).sum():,} of {D.size:,} "
          f"({100.0 * (D == 0).sum() / D.size:.1f}%)")
    print()

    # Is the difference a systematic per-dimension offset or scale?
    print("is D a per-dimension constant, or proportional to A?")
    # R^2 of D ~ a*A + b, computed per dimension across chunks.
    r2 = np.zeros(dim)
    for j in range(dim):
        x, y = A[:, j], D[:, j]
        if x.std() == 0 or y.std() == 0:
            r2[j] = 0.0
            continue
        c = np.corrcoef(x, y)[0, 1]
        r2[j] = 0.0 if np.isnan(c) else c * c
    print(f"  R^2 of D ~ a*A + b   : median {np.median(r2):.4f}  "
          f"p95 {np.percentile(r2, 95):.4f}  max {r2.max():.4f}")
    # A per-dimension constant offset would show as std(D)/|mean(D)| near zero.
    md, sd = D.mean(axis=0), D.std(axis=0)
    ratio = sd / np.maximum(np.abs(md), 1e-300)
    print(f"  std(D)/|mean(D)|     : median {np.median(ratio):.2f}  "
          f"(near 0 = constant offset, >>1 = noise)")
    print()

    # The decisive test: same A[i], same B[i]? Take the closest pairs of
    # chunks within each dimension and see whether their D agrees.
    print("decisive test: do near-equal A values give near-equal B values?")
    rng = np.random.default_rng(0)
    cols = rng.choice(dim, size=min(400, dim), replace=False)
    worst = 0.0
    ratios = []
    for j in cols:
        x, y = A[:, j], B[:, j]
        o = np.argsort(x)
        xs, ys = x[o], y[o]
        dx = np.diff(xs)
        dy = np.abs(np.diff(ys))
        k = np.argmin(dx)
        if dx[k] <= 0:
            continue
        # If B were a function of A, dy should shrink with dx. Compare the
        # B-gap between the two closest A values against the typical |D|.
        ratios.append(dy[k] / max(np.abs(D[:, j]).mean(), 1e-300))
        worst = max(worst, dy[k])
    if ratios:
        rr = np.array(ratios)
        print(f"  for the two chunks closest in A[i], the gap in B[i] is")
        print(f"  {np.median(rr):.1f}x the typical |D| for that dimension (median over "
              f"{len(rr)} dimensions)")
        print(f"  a genuine function would give a ratio well below 1")
    print()

    print("verdict")
    if np.median(r2) < 0.05 and np.median(ratio) > 3:
        print("  No per-dimension translator. D is not explained by A: the fit is")
        print("  flat and the difference behaves as noise rather than as a")
        print("  transformation. The same A[i] maps to many different B[i].")
    else:
        print("  There is structure worth looking at — see the numbers above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
