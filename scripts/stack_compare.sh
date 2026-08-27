#!/usr/bin/env bash
# Compare THIS stack against a published one, using a model both have measured.
#
#   ./scripts/stack_compare.sh
#   MODELS="/blobs/sha256-abc..." ./scripts/stack_compare.sh
#
# The question this answers is not "which model is fast" — it is whether the
# prebuilt GHCR image is as fast as a tuned source build. jeffgrover/b70-setup
# builds llama.cpp from source with GGML_SYCL_GRAPH=ON and GGML_SYCL_DNN=ON
# against oneAPI 2026.1; we run a published container whose build flags are not
# visible from the outside. Running one of HIS models on OUR stack and
# comparing prompt-processing throughput isolates the stack, because the model
# and quantization are held fixed.
#
# Prompt processing is the only comparable half. Embedding is pure prefill, so
# `pp` transfers directly and `tg` (generation) says nothing about this
# workload.
#
# His published figures, Arc Pro B70, source build:
#   Qwen3.8-27B Q4_K_S       836.5-864.7 t/s prompt (512-2048 tokens)
#   Qwen3.6-35B-A3B UD-Q4_K_S    220.0 t/s prompt
#   Gemma-4 E4B Q4_K_M          1710.0 t/s prompt
#
# Models come from the ollama app's blob store, which already holds several of
# them — no download needed. Quantization may differ slightly from his (ollama
# ships Q4_K_M where he used Q4_K_S); that is a real caveat and is why the
# comparison is directional rather than exact.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

OLLAMA_BLOBS="${OLLAMA_BLOBS:-/mnt/ssd5/apps/ollama_data/models/blobs}"
PROMPTS="${PROMPTS:-512,2048}"
REPS="${REPS:-3}"

# name:blob-digest — resolve with scripts/ollama_blobs.py
MODELS="${MODELS:-}"
if [ -z "$MODELS" ]; then
  MODELS="$(python3 scripts/ollama_blobs.py --quiet 2>/dev/null)"
fi
if [ -z "$MODELS" ]; then
  echo "no models resolved; run scripts/ollama_blobs.py to see what is available" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="baselines/stack-compare-${STAMP}.log"
mkdir -p baselines

echo "stack comparison ${STAMP}" | tee "$OUT"
echo "image:   $IMAGE" | tee -a "$OUT"
echo "prompts: $PROMPTS   repetitions: $REPS" | tee -a "$OUT"
echo | tee -a "$OUT"

for entry in $MODELS; do
  name="${entry%%=*}"
  blob="${entry#*=}"
  if [ ! -f "$blob" ]; then
    echo "== $name — blob missing at $blob, skipping" | tee -a "$OUT"
    continue
  fi
  echo "== $name" | tee -a "$OUT"
  echo "   $blob" | tee -a "$OUT"
  # -n 0 disables the generation benchmark: only prefill is comparable to an
  # embedding workload, and generation would double the runtime for nothing.
  docker run --rm --device /dev/dri \
    -v "$(dirname "$blob"):/blobs:ro" \
    --entrypoint /app/llama \
    "$IMAGE" bench \
    -m "/blobs/$(basename "$blob")" \
    -p "$PROMPTS" -n 0 -ngl 999 -fa 1 -r "$REPS" 2>&1 \
    | grep -vE "^(load_backend|ggml_sycl|build:|$)" | tee -a "$OUT"
  echo | tee -a "$OUT"
done

echo "== done ==" | tee -a "$OUT"
echo "compare the pp columns against the published figures in the header of" | tee -a "$OUT"
echo "this script and in the README's upstream-reference section." | tee -a "$OUT"
echo
echo "$OUT"
