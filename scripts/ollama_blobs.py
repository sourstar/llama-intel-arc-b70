#!/usr/bin/env python3
"""Resolve ollama model names to the GGUF blobs on disk.

The ollama app on this node already holds several of the models
jeffgrover/b70-setup benchmarked, so a stack comparison needs no downloads —
but ollama stores them content-addressed, as `blobs/sha256-<digest>` with the
name only in a manifest. This maps one to the other.

Usage:
    ollama_blobs.py                 # list everything, human readable
    ollama_blobs.py --quiet         # name=path pairs for stack_compare.sh
    ollama_blobs.py --match qwen3.8 # only models whose name contains this
"""

import argparse
import json
import os
import sys

DEFAULT_ROOT = "/mnt/ssd5/apps/ollama_data/models"

# Models jeffgrover/b70-setup published prompt-processing figures for, so a
# local copy is directly comparable. The number is his, in tokens/second.
PUBLISHED_PP = {
    "qwen3.8:27b": 850.6,          # his Q4_K_S, 836.5-864.7 across 512-2048
    "qwen3.6:35b-a3b-q4_K_M": 220.0,
}


def manifests(root):
    base = os.path.join(root, "manifests")
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, base)
            parts = rel.replace("\\", "/").split("/")
            # registry/namespace/name/tag
            if len(parts) < 4:
                continue
            name = f"{parts[-2]}:{parts[-1]}"
            yield name, path


def model_blob(root, manifest_path):
    try:
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None, 0
    for layer in m.get("layers", []):
        if "model" in layer.get("mediaType", ""):
            digest = layer["digest"].replace(":", "-")
            return os.path.join(root, "blobs", digest), layer.get("size", 0)
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--match", default="")
    ap.add_argument("--quiet", action="store_true",
                    help="emit name=path pairs only, for stack_compare.sh")
    args = ap.parse_args()

    found = []
    for name, mpath in sorted(manifests(args.root)):
        if args.match and args.match not in name:
            continue
        blob, size = model_blob(args.root, mpath)
        if blob and os.path.exists(blob):
            found.append((name, blob, size))

    if args.quiet:
        # Only the models with a published figure to compare against; anything
        # else would be a number with nothing to compare it to.
        for name, blob, _size in found:
            if name in PUBLISHED_PP:
                print(f"{name}={blob}")
        return 0

    if not found:
        print(f"no ollama models found under {args.root}", file=sys.stderr)
        return 1

    print(f"{'model':<34}{'size':>9}  {'published pp t/s':>17}  blob")
    for name, blob, size in found:
        pub = PUBLISHED_PP.get(name)
        pub_s = f"{pub:.1f}" if pub else "-"
        print(f"{name:<34}{size / 1e9:>8.1f}G  {pub_s:>17}  {blob}")
    print()
    print("'published pp t/s' is jeffgrover/b70-setup's prompt-processing figure")
    print("for the same model on a source-built llama.cpp. Only those models are")
    print("worth running through scripts/stack_compare.sh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
