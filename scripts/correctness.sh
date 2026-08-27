#!/usr/bin/env bash
# Correctness gate. Run after every image bump or inference-flag change.
#
# Starts two throwaway containers — production settings and a conservative
# reference — embeds identical fixtures through both, and compares. Also
# checks that repeated embeddings of the same input are bit-identical.
#
# The production app on PROD_URL is left alone; this uses its own ports.
#
#   ./scripts/correctness.sh
#
# Exit 0 = safe to ingest. Non-zero = do not ingest.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

TMP="$(mktemp -d)"
CID_A=""; CID_B=""; CID_C=""

cleanup() {
  [ -n "$CID_A" ] && docker rm -f "$CID_A" >/dev/null 2>&1
  [ -n "$CID_B" ] && docker rm -f "$CID_B" >/dev/null 2>&1
  [ -n "$CID_C" ] && docker rm -f "$CID_C" >/dev/null 2>&1
  rm -rf "$TMP"
}
trap cleanup EXIT

if [ ! -f "$FIXTURES" ]; then
  echo "no fixtures at $FIXTURES — run scripts/make_fixtures.py first" >&2
  exit 2
fi

start() {
  # $1 name  $2 port  $3 ctx  $4 np  $5 ubatch
  # An interrupted earlier run leaves the name taken; docker run would fail on
  # the collision rather than on anything real.
  docker rm -f "$1" >/dev/null 2>&1
  # EXTRA_ENV from config.env is passed through. The gate has to test the
  # environment the app actually deploys with — a setting that changes kernel
  # selection changes what "correct" means, so a gate that ignored it would be
  # checking a configuration nobody runs.
  local env_args=()
  local kv
  for kv in $EXTRA_ENV; do
    [ -z "$kv" ] && continue
    env_args+=(-e "$kv")
  done
  docker run -d --name "$1" --device /dev/dri \
    -v "${MODELS_DIR}:/models" \
    -p "$2:8080" \
    "${env_args[@]}" \
    "$IMAGE" \
    -m "$MODEL" --embeddings --pooling last -ngl 999 \
    -c "$3" -np "$4" -b "$5" -ub "$5" -fa on \
    --host 0.0.0.0 --port 8080
}

echo "== starting production-settings container (ub=$UBATCH) =="
CID_A="$(start lem-gate-a "$PORT_A" "$CTX" "$NP" "$UBATCH")" || exit 1

echo "== capturing (also testing determinism via --repeat 2) =="
python3 scripts/capture.py --url "http://127.0.0.1:${PORT_A}" \
  --fixtures "$FIXTURES" --out "$TMP/prod.json" --repeat 2 \
  --label "ub=$UBATCH image=$IMAGE" || exit 1

docker rm -f "$CID_A" >/dev/null; CID_A=""

echo
echo "== starting reference container (ub=$REF_UBATCH) =="
CID_B="$(start lem-gate-b "$PORT_B" "$REF_CTX" "$REF_NP" "$REF_UBATCH")" || exit 1

python3 scripts/capture.py --url "http://127.0.0.1:${PORT_B}" \
  --fixtures "$FIXTURES" --out "$TMP/ref.json" --repeat 2 \
  --label "ub=$REF_UBATCH image=$IMAGE" || exit 1

docker rm -f "$CID_B" >/dev/null; CID_B=""

rc=0

echo
echo "== determinism: production settings =="
python3 scripts/compare.py --determinism "$TMP/prod.json" \
  --min-cosine "$DET_MIN_COSINE" || rc=1

echo
echo "== cross-config: ub=$UBATCH vs ub=$REF_UBATCH =="
python3 scripts/compare.py "$TMP/prod.json" "$TMP/ref.json" \
  --min-cosine "$MIN_COSINE" || rc=1


# ---- bit exactness -------------------------------------------------------
# A compliance parameter for evaluating a new release, alongside the cosine
# bars above. Cosine says a build computes the same FUNCTION; this says it
# computes it the same WAY. A release that quietly changes kernel selection
# passes the cosine bar and fails here, which is the earliest possible signal
# that the arithmetic moved.
#
# Checked on its own container so the passes are full prefills over a pool
# that never repeats a chunk back to back — a chunk served from the slot
# prompt cache returns a different vector and would fail this for a reason
# that has nothing to do with the build.
echo
echo "== bit exactness: ${BIT_EXACT_PASSES} passes, stored (L2 + f16) vectors =="
CID_C="$(start lem-gate-c "$PORT_A" "$CTX" "$NP" "$UBATCH")" || exit 1
# Not piped: scale_trial exits non-zero when it finds mismatches, and under
# `pipefail` that made a real failure report itself as "did not complete".
python3 scripts/scale_trial.py --url "http://127.0.0.1:${PORT_A}" \
  --fixtures "$FIXTURES" --passes "$BIT_EXACT_PASSES" \
  --out "$TMP/bitexact.json" > "$TMP/bitexact.log" 2>&1
grep -E "chunks compared|^mismatches" "$TMP/bitexact.log" || true
if [ ! -s "$TMP/bitexact.json" ]; then
  echo "FAIL: bit-exactness trial did not complete — see $TMP/bitexact.log"
  sed -n '$p' "$TMP/bitexact.log"
  rc=1
elif python3 -c "
import json, sys
sys.exit(1 if json.load(open('$TMP/bitexact.json'))['mismatches'] else 0)
"; then
  echo "PASS"
else
  echo "FAIL: stored vectors are not bit-exact across passes."
  echo "The build is not reproducing its own output byte for byte."
  rc=1
fi
docker rm -f "$CID_C" >/dev/null 2>&1; CID_C=""

# Keep captures on failure so they can be inspected.
if [ "$rc" -ne 0 ]; then
  mkdir -p baselines/failures
  ts="$(date +%Y%m%d-%H%M%S)"
  cp "$TMP/prod.json" "baselines/failures/${ts}-prod.json"
  cp "$TMP/ref.json"  "baselines/failures/${ts}-ref.json"
  [ -s "$TMP/bitexact.log" ] && cp "$TMP/bitexact.log" "baselines/failures/${ts}-bitexact.log"
  echo
  echo "captures saved to baselines/failures/${ts}-*.json"
fi

exit "$rc"
