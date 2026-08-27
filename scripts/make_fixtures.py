#!/usr/bin/env python3
"""Cut fixture chunks at exact token lengths, from a categorised source pool.

Chunk length is what triggers #26206, and length in TOKENS is what selects the
kernel, so fixtures are sized in tokens rather than characters. Lengths come
from the server's own tokenizer, not a local copy, so they match exactly what
inference will see.

Two profiles, because consistency and performance want opposite pools:

  --profile gate   Straddles every boundary that changes the dispatch or the
                   batching. Deliberately unrepresentative — it is looking for
                   the length where the maths breaks, not for a typical day.

  --profile bench  The production chunk-length distribution, so throughput
                   numbers mean something. Measured from the gutenberg
                   manifest on 2026-08-27 over 12,784 embedded documents and
                   1,373,515 chunks: mean 926 tokens, p99 of per-document mean
                   1000. Nothing in production is longer than ~1000 tokens.
                   Benchmarking 7900-token chunks measures a case that never
                   occurs.

Usage:
    make_fixtures.py --profile gate  --pool /root/fixture-src --out fixtures/chunks.json
    make_fixtures.py --profile bench --pool /root/fixture-src --out fixtures/bench.json

Writes a list of {id, category, source, target_tokens, actual_tokens, text}.
Commit the result. The gate is only meaningful if every run embeds identical
input.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# Every length here is a boundary or one side of one:
#   8      below the 32-query-token minimum for oneDNN SDPA
#   32     exactly at it
#   511/512/513, 1023/1024/1025   the oneMKL FA floor is K->ne[1] >= 1024
#   2047/2048/2049                a -ub the harness actually tests
#   7900   just under the ctx=8192 truncation policy
GATE_LENGTHS = [8, 32, 64, 256, 511, 512, 513, 1023, 1024, 1025,
                2047, 2048, 2049, 4096, 6144, 7900]

# The production distribution, sampled at the percentiles measured on the
# manifest. Mean of this list is 899 against a corpus mean of 926 — close
# enough that a throughput number transfers, and every length is one that
# really occurs.
BENCH_LENGTHS = ([421, 751]
                 + [830] * 4
                 + [901] * 6
                 + [955] * 6
                 + [986] * 4
                 + [995, 1000])

# Category mix. The gate spreads content types evenly across the length range;
# the bench pool follows the corpus, which is ~30% English, ~60% other European
# languages and ~3% Chinese. PubMed is not in this collection, so it stays out
# of the throughput pool but stays in the gate — a tokenizer that fragments
# differently is exactly what a consistency check should cover.
GATE_MIX = ["prose-en", "prose-eu", "prose-cjk", "sci"]
BENCH_MIX = (["prose-en"] * 4) + (["prose-eu"] * 5) + ["prose-cjk"]

PROFILES = {
    "gate": (GATE_LENGTHS, GATE_MIX),
    "bench": (BENCH_LENGTHS, BENCH_MIX),
}


def parse_lengths(spec):
    """`400-1100`, `400-1100:5`, or `8,32,512`.

    A length sweep is a different kind of evidence from a random sample. Every
    failure mode found on this card selects on SHAPE — the GEMM's M is the
    token count, and which kernel and reduction order that picks is a function
    of M. A random sample of 5000 documents bounds the failure rate at 6e-4,
    which against 1.37M corpus chunks still permits ~800 bad vectors. Covering
    every M in the production range instead turns "we sampled and saw none"
    into "there is no length at which it happens", which is the claim that
    actually matters.
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            rng, _, step = part.partition(":")
            lo, _, hi = rng.partition("-")
            out.extend(range(int(lo), int(hi) + 1, int(step) if step else 1))
        else:
            out.append(int(part))
    return out

def post(url, path, payload, timeout=300):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(url, text):
    return post(url, "/tokenize", {"content": text})["tokens"]


def detokenize(url, tokens):
    return post(url, "/detokenize", {"tokens": tokens})["content"]


def cut(url, text, start, n_tokens):
    """Return exactly n_tokens' worth of text starting near character `start`.

    Round-tripping through the tokenizer beats binary-searching on character
    count: it is three requests instead of twenty, and it is exact rather than
    convergent. It still needs a verify pass, because detokenising a token
    prefix and re-tokenising it does not always give the same count back —
    BPE merges across the cut, and leading whitespace is handled differently
    at a string boundary than mid-string.
    """
    # Characters per token varies from ~1 (Chinese) to ~6 (English prose), so
    # widen until the window really holds enough tokens.
    mult = 8
    while True:
        window = text[start:start + n_tokens * mult]
        toks = tokenize(url, window)
        if len(toks) >= n_tokens + 8 or start + n_tokens * mult >= len(text):
            break
        mult *= 2

    take = n_tokens
    body = detokenize(url, toks[:take])
    for _ in range(6):
        got = len(tokenize(url, body))
        if got == n_tokens:
            return body, got
        take = max(1, min(take + (n_tokens - got), len(toks)))
        body = detokenize(url, toks[:take])
    return body, len(tokenize(url, body))


def load_category(pool, category):
    """Concatenate a category's files in sorted order, for determinism."""
    d = os.path.join(pool, category)
    if not os.path.isdir(d):
        return None
    parts = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                parts.append(f.read())
    return "\n\n".join(parts) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30436")
    ap.add_argument("--pool", default="/root/fixture-src",
                    help="directory of category subdirectories "
                         "(see build_source_pool.py)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="gate")
    ap.add_argument("--lengths", default="",
                    help="explicit token lengths, e.g. 400-1100 or 400-1100:5; "
                         "overrides --profile")
    ap.add_argument("--out")
    args = ap.parse_args()

    out = args.out or f"fixtures/{'chunks' if args.profile == 'gate' else 'bench'}.json"
    lengths, mix = PROFILES[args.profile]
    if args.lengths:
        lengths = parse_lengths(args.lengths)
        mix = GATE_MIX

    try:
        tokenize(args.url, "probe")
    except urllib.error.URLError as e:
        print(f"cannot reach {args.url}: {e}", file=sys.stderr)
        print("start the server first — fixtures need its tokenizer", file=sys.stderr)
        return 1

    texts, offsets = {}, {}
    for category in sorted(set(mix)):
        t = load_category(args.pool, category)
        if t is None:
            print(f"no source for category {category} under {args.pool}",
                  file=sys.stderr)
            return 2
        texts[category] = t
        offsets[category] = 0

    chunks = []
    for i, target in enumerate(lengths):
        category = mix[i % len(mix)]
        text = texts[category]
        if offsets[category] >= len(text):
            print(f"category {category} exhausted at chunk {i}", file=sys.stderr)
            break
        body, actual = cut(args.url, text, offsets[category], target)
        offsets[category] += max(len(body), 1)
        chunks.append({
            "id": f"c{i:02d}",
            "category": category,
            "target_tokens": target,
            "actual_tokens": actual,
            "text": body,
        })
        flag = "" if actual == target else f"  (target {target})"
        print(f"  c{i:02d} {category:<10} {actual:>5} tokens{flag}")

    if not chunks:
        print("no chunks produced", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    total = sum(c["actual_tokens"] for c in chunks)
    print(f"wrote {len(chunks)} chunks ({total:,} tokens) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
