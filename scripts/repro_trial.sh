#!/usr/bin/env bash
# Is this configuration reproducible? Answered over many orderings, not one.
#
#   ./scripts/repro_trial.sh
#   TRIALS=10 CONCURRENCIES="1 4" ./scripts/repro_trial.sh
#   EXTRA_ENV_ARGS="" ./scripts/repro_trial.sh        # oneDNN left enabled
#
# Every reproducibility claim in KNOWN-ISSUES before this script existed was
# measured with `capture.py` in fixture order, the same order every time. That
# holds request ordering fixed, and ordering is exactly what decides whether a
# request is served from the slot prompt cache — so a fixed order measures one
# arbitrary path and can call a configuration reproducible when it is not.
#
# This runs the pool under several patterns and several seeds, at more than one
# concurrency, and pools the results: a chunk counts as reproducible only if it
# returned the same stored vector in EVERY trial, and the report says how many
# trials that is over.
#
# `adjacent` is included deliberately even though it is not how anything embeds
# in production: it forces cache hits, so it is the pattern that shows the two
# code paths clearly instead of leaving them to chance.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

TRIALS="${TRIALS:-5}"
PATTERNS="${PATTERNS:-cycle shuffle reverse adjacent}"
CONCURRENCIES="${CONCURRENCIES:-1 4}"
REPEAT="${REPEAT:-3}"
POOL="${POOL:-$BENCH_FIXTURES}"
EXTRA_ENV_ARGS="${EXTRA_ENV_ARGS:--e GGML_SYCL_ENABLE_DNN=0}"
SRV_FLAGS="${SRV_FLAGS:--c 8192 -np 1 -b 8192 -ub 8192}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="baselines/repro-${STAMP}"
NAME="lem-repro"
mkdir -p "$RUN"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT

echo "reproducibility trial ${STAMP}"
echo "server:   $SRV_FLAGS   env: ${EXTRA_ENV_ARGS:-none}"
echo "pool:     $POOL"
echo "patterns: $PATTERNS      concurrencies: $CONCURRENCIES"
echo "trials:   $TRIALS seeds x $REPEAT repeats each"
echo

docker rm -f "$NAME" >/dev/null 2>&1
# shellcheck disable=SC2086
docker run -d --name "$NAME" --device /dev/dri $EXTRA_ENV_ARGS \
  -v "${MODELS_DIR}:/models" -p "${PORT_A}:8080" \
  "$IMAGE" \
  -m "$MODEL" --embeddings --pooling last -ngl 999 -fa on -t 6 --metrics \
  --host 0.0.0.0 --port 8080 $SRV_FLAGS >/dev/null 2>&1

n=0
while [ "$n" -lt 90 ]; do
  curl -fsS --max-time 3 "http://127.0.0.1:${PORT_A}/health" >/dev/null 2>&1 && break
  n=$((n + 1)); sleep 2
done
if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT_A}/health" >/dev/null 2>&1; then
  docker logs "$NAME" > "${RUN}/crash.log" 2>&1
  echo "container did not start — see ${RUN}/crash.log" >&2
  exit 1
fi

# Warm first: a cold container is ~35% slow and the timings are used below to
# tell a prefill from a cache hit.
python3 scripts/capture.py --url "http://127.0.0.1:${PORT_A}" \
  --fixtures "$POOL" --out /dev/null --repeat 1 --quiet >/dev/null 2>&1 || true

count=0
for pattern in $PATTERNS; do
  for conc in $CONCURRENCIES; do
    for seed in $(seq 1 "$TRIALS"); do
      out="${RUN}/${pattern}-c${conc}-s${seed}.json"
      python3 scripts/capture.py --url "http://127.0.0.1:${PORT_A}" \
        --fixtures "$POOL" --out "$out" --repeat "$REPEAT" \
        --pattern "$pattern" --seed "$seed" --concurrency "$conc" \
        --label "${pattern} c${conc} s${seed}" --quiet >/dev/null 2>&1
      count=$((count + 1))
      printf '\r  %d captures' "$count"
    done
  done
done
echo
docker rm -f "$NAME" >/dev/null 2>&1
echo

python3 scripts/repro_report.py "$RUN"
echo
echo "captures under $RUN/"
