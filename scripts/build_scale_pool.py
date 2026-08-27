#!/usr/bin/env python3
"""Build a large pool of distinct chunks, one per document, for a scale trial.

The gate and bench pools are 16 and 24 chunks. Repeating those many times
bounds how often a GIVEN chunk drifts; it says nothing about how many chunks
in a 1.37-million-chunk corpus would. Every failure mode found on this card so
far has been input-dependent — a function of token count, and of whichever
kernel that shape selects — so the sample that matters is distinct inputs, not
repeats of the same input.

So: one chunk from each of N different documents, sampled to match the corpus
language mix, cut at a production-like length.

    build_scale_pool.py --count 5000 --out fixtures/scale.json

Chunks are cut by character window rather than by exact token count, because
5000 exact cuts would mean 20,000 tokenizer round-trips and the point here is
input diversity, not landing on a boundary. The window is sized per language:
Chinese runs near one token per character where European prose runs near four,
so a fixed character count would produce 900-token chunks in one language and
3600-token chunks in another.
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys

MANIFEST = "/mnt/ssd4/apps/louise-rag/data/manifest.db"
CORPUS = "/mnt/ssd4/data/gutenberg"

# Characters per chunk, by language, aiming at roughly 900 tokens — the corpus
# mean is 926. Rough is fine and is itself realistic: production chunk lengths
# vary too.
CHARS = {"zh": 900, "ja": 900, "ko": 900, "el": 2200, "ru": 2200}
CHARS_DEFAULT = 3600

START = re.compile(r"\*\*\* *START OF TH")
END = re.compile(r"\*\*\* *END OF TH")


def strip_pg(text):
    lines = text.splitlines(keepends=True)
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if START.search(line):
            start = i + 1
            break
    for i in range(start, len(lines)):
        if END.search(lines[i]):
            end = i
            break
    return "".join(lines[start:end])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default="fixtures/scale.json")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.manifest}?mode=ro", uri=True)
    rows = list(db.execute(
        "select lang, native_id from documents where chunks > 0 and tokens > 0"))
    if not rows:
        print("no documents with chunk data in the manifest", file=sys.stderr)
        return 1

    by_lang = {}
    for lang, nid in rows:
        by_lang.setdefault(lang or "und", []).append(nid)

    total = sum(len(v) for v in by_lang.values())
    rng = random.Random(args.seed)

    # Proportional allocation, then take documents at random within a language.
    # Sampling the whole corpus rather than the embedded subset: the embedded
    # subset is whatever the pipeline has reached so far, which is not the
    # distribution the corpus will end up with.
    picked = []
    for lang, ids in sorted(by_lang.items()):
        want = round(args.count * len(ids) / total)
        if want <= 0:
            continue
        ids = sorted(ids)
        rng.shuffle(ids)
        for nid in ids[:want]:
            picked.append((lang, nid))
    rng.shuffle(picked)

    chunks, missing, short = [], 0, 0
    for lang, nid in picked:
        path = os.path.join(args.corpus, str(nid), f"pg{nid}.txt")
        if not os.path.isfile(path):
            missing += 1
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                body = strip_pg(f.read())
        except OSError:
            missing += 1
            continue
        width = CHARS.get(lang, CHARS_DEFAULT)
        # A quarter of the way in, past front matter and tables of contents.
        offset = len(body) // 4
        window = body[offset:offset + width]
        if len(window) < width // 2:
            short += 1
            continue
        chunks.append({
            "id": f"pg{nid}",
            "category": lang,
            "target_tokens": 0,
            "actual_tokens": 0,
            "text": window,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(chunks, f, ensure_ascii=False)

    langs = {}
    for c in chunks:
        langs[c["category"]] = langs.get(c["category"], 0) + 1
    print(f"wrote {len(chunks)} chunks from {len(chunks)} distinct documents to {args.out}")
    print(f"  skipped: {missing} with no plain-text file, {short} too short")
    print("  language mix: " + ", ".join(
        f"{k} {v}" for k, v in sorted(langs.items(), key=lambda kv: -kv[1])[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
