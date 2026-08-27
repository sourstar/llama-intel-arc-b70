#!/usr/bin/env python3
"""Assemble the raw text the fixture builder draws from, by category.

The fixtures have to be reproducible — the gate is only meaningful if every
run embeds byte-identical input — so the material they are cut from cannot be
"some directory the operator pointed at once". This script names the exact
documents, and why each one is in the pool.

The categories exist because tokenization, not just length, changes what the
encoder does. A 1000-token Chinese chunk is a very different span of text from
a 1000-token English one, and PubMed abstracts are dense with numerals,
chemical names and acronyms that fragment differently again.

Language mix follows the corpus. Measured on the gutenberg manifest
(2026-08-27, 12,784 embedded documents): en 3853, fr 2559, fi 1462, de 1224,
nl 716, it 653, pt 539, es 476, zh 366, el 215. Roughly 40% of the corpus is
not English, so a pool of pure English prose would not represent it.

Usage:
    build_source_pool.py --out /root/fixture-src
    build_source_pool.py --out /root/fixture-src --gutenberg-root /mnt/ssd4/data/gutenberg
"""

import argparse
import gzip
import os
import re
import sys

# Project Gutenberg ids, chosen for size (all >200k tokens, so a fixture can
# be cut anywhere) and for spread across the corpus's real language mix.
GUTENBERG = {
    "prose-en": [
        (11, "Alice's Adventures in Wonderland"),
        (1342, "Pride and Prejudice"),
        (2701, "Moby Dick"),
        (84, "Frankenstein"),
    ],
    "prose-eu": [
        (2650, "A la recherche du temps perdu (fr)"),
        (2404, "German prose (de)"),
        (2000, "Don Quijote (es)"),
        (33396, "Greek prose (el)"),
    ],
    # Chinese is the strongest tokenizer stress in the corpus: no whitespace
    # word boundaries, so a token covers far fewer characters.
    "prose-cjk": [
        (23825, "Chinese prose (zh)"),
        (23841, "Chinese prose (zh)"),
    ],
}

# PG wraps every plain-text ebook in a licence header and footer. Both are
# boilerplate rather than prose, so a fixture cut from offset 0 would be
# measuring the licence — and stripping it also keeps the trademark notice out
# of a committed fixture file.
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


def build_gutenberg(root, out):
    written = 0
    for category, books in GUTENBERG.items():
        d = os.path.join(out, category)
        os.makedirs(d, exist_ok=True)
        for pgid, label in books:
            src = os.path.join(root, str(pgid), f"pg{pgid}.txt")
            if not os.path.isfile(src):
                print(f"  MISSING {src} ({label})", file=sys.stderr)
                continue
            with open(src, encoding="utf-8", errors="replace") as f:
                body = strip_pg(f.read())
            dst = os.path.join(d, f"pg{pgid}.txt")
            with open(dst, "w", encoding="utf-8") as f:
                f.write(body)
            print(f"  {category}/pg{pgid}.txt  {len(body):>9,} chars  {label}")
            written += 1
    return written


# PubMed ships as gzipped XML. Pulling the text out with a regex rather than
# an XML parser is deliberate: the baseline files are large, we want a couple
# of megabytes off the front, and abstract bodies carry inline markup that a
# parser would hand back as fragments anyway.
ABSTRACT = re.compile(r"<AbstractText[^>]*>(.*?)</AbstractText>", re.S)
TAG = re.compile(r"<[^>]+>")
ENTITY = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&apos;": "'"}


def build_pubmed(root, out, target_chars=2_000_000):
    d = os.path.join(out, "sci")
    os.makedirs(d, exist_ok=True)
    name = "pubmed26n0001.xml.gz"
    src = os.path.join(root, name)
    if not os.path.isfile(src):
        print(f"  MISSING {src}", file=sys.stderr)
        return 0

    parts, total = [], 0
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
        buf = f.read(64 * 1024 * 1024)
    for m in ABSTRACT.finditer(buf):
        body = TAG.sub("", m.group(1))
        for k, v in ENTITY.items():
            body = body.replace(k, v)
        body = body.strip()
        if len(body) < 200:
            continue
        parts.append(body)
        total += len(body) + 2
        if total >= target_chars:
            break

    dst = os.path.join(d, "abstracts.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    print(f"  sci/abstracts.txt  {total:>9,} chars  {len(parts):,} abstracts from {name}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/fixture-src")
    ap.add_argument("--gutenberg-root", default="/mnt/ssd4/data/gutenberg")
    ap.add_argument("--pubmed-root", default="/mnt/tank/louise-corpora/pubmed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"building source pool in {args.out}")
    n = build_gutenberg(args.gutenberg_root, args.out)
    n += build_pubmed(args.pubmed_root, args.out)
    print(f"{n} source file(s)")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
