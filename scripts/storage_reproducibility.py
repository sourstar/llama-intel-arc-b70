#!/usr/bin/env python3
"""Does the encoder's run-to-run drift survive the storage pipeline?

`determinism_profile.py` asks whether the SERVER returns the same f32 vector
twice. That is the wrong question if what gets hashed is not the f32 vector.
The RAG pipeline L2-normalises client-side and stores f16 (the collection's
space record: `"normalization": "l2-client", "dtype": "f16"`), and f16 carries
about 11 bits of mantissa. A difference of 1e-6 relative may simply not exist
by the time anything is written down.

So this asks the question that decides whether embeddings can be verified by
hash: after normalisation and downcast, how many DISTINCT stored vectors does a
chunk produce across repeats?

    raw f32        what the server returned
    l2 + f16       what actually lands in LanceDB, and what a hash would cover

Usage:
    capture.py --url ... --out /tmp/p.json --repeat 10
    storage_reproducibility.py /tmp/p.json
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter


def l2_normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return list(v)
    return [x / n for x in v]


def to_f16_bytes(v):
    """IEEE-754 half precision, the storage format. `struct` does this exactly,
    so it needs no numpy and matches whatever writes the Lance file."""
    return b"".join(struct.pack("<e", x) for x in v)


def digest(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--fixtures", default="fixtures/chunks.json")
    ap.add_argument("--hashes", action="store_true",
                    help="emit chunk_id -> stored-vector sha256 as JSON, "
                         "for comparing one configuration against another")
    args = ap.parse_args()

    with open(args.capture) as f:
        cap = json.load(f)

    try:
        with open(args.fixtures, encoding="utf-8") as f:
            fx = {c["id"]: c for c in json.load(f)}
    except OSError:
        fx = {}

    by_id = {}
    for r in cap["records"]:
        by_id.setdefault(r["id"], []).append(r)

    if args.hashes:
        # One hash per chunk, or null where the configuration is not even
        # self-consistent. Comparing these across configurations answers the
        # question that matters for a peer-to-peer corpus: would another node
        # store the same bytes?
        out = {}
        for cid, reps in sorted(by_id.items()):
            stored = {to_f16_bytes(l2_normalize(r["embedding"])) for r in reps}
            out[cid] = digest(next(iter(stored))) if len(stored) == 1 else None
        print(json.dumps(out))
        return 0

    print(f"{'chunk':7}{'tokens':>8}{'repeats':>9}"
          f"{'distinct f32':>14}{'distinct l2+f16':>17}{'stored sha256':>16}")
    total_raw_drift = total_stored_drift = 0
    for cid, reps in sorted(by_id.items(), key=lambda kv: fx.get(kv[0], {}).get("actual_tokens", 0)):
        raw = Counter(tuple(r["embedding"]) for r in reps)
        stored = Counter(to_f16_bytes(l2_normalize(r["embedding"])) for r in reps)
        tok = reps[0].get("tokens") or fx.get(cid, {}).get("actual_tokens", 0)
        if len(raw) > 1:
            total_raw_drift += 1
        if len(stored) > 1:
            total_stored_drift += 1
        mark = "" if len(stored) == 1 else "   <-- HASH UNSTABLE"
        one = digest(next(iter(stored)))
        print(f"{cid:7}{tok:>8}{len(reps):>9}"
              f"{len(raw):>14}{len(stored):>17}{one:>16}{mark}")

    n = len(by_id)
    print()
    print(f"raw f32:    {n - total_raw_drift}/{n} chunks reproducible")
    print(f"l2 + f16:   {n - total_stored_drift}/{n} chunks reproducible")
    if total_stored_drift == 0 and total_raw_drift > 0:
        print()
        print("The drift does not survive the storage pipeline: every chunk hashes")
        print("to one value even though the server returned several. Hash-based")
        print("verification of STORED vectors is safe on this configuration.")
    elif total_stored_drift:
        print()
        print(f"{total_stored_drift} chunk(s) hash to more than one value. Stored")
        print("vectors are NOT hash-stable here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
