#!/usr/bin/env bash
# Map throughput and latency against CLIENT concurrency for one server config.
#
#   ./scripts/concurrency.sh                     # deployed settings
#   NP=2 CTX=16384 UBATCH=2048 ./scripts/concurrency.sh
#   CONCURRENCIES="1 2 4 8 16" ./scripts/concurrency.sh
#
# Why this is separate from matrix.sh: the matrix benches each configuration at
# concurrency == its own slot count, which is the natural operating point but
# conflates two variables. A 2-slot server measured at concurrency 2 has less
# queuing than a 4-slot server at concurrency 4, so its better tail latency is
# partly an artefact of the comparison rather than a property of the config.
# Here the server is held fixed and only the client changes, which is the
# question that actually matters: how hard should the ingest pipeline drive it.
#
# Consistency is checked once, up front, against the same fixed reference the
# matrix uses. Concurrency cannot change the arithmetic — but it is cheap to
# confirm, and a run that skipped the check would be the one exception in a
# harness whose whole rule is that it never skips it.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="baselines/concurrency-${STAMP}"
CSV="${RUN}/results.csv"
NAME="lem-conc"

for f in "$FIXTURES" "$BENCH_FIXTURES"; do
  if [ ! -f "$f" ]; then
    echo "missing fixture pool $f — run scripts/make_fixtures.py first" >&2
    exit 2
  fi
done

mkdir -p "$RUN"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT

echo "concurrency sweep ${STAMP}"
echo "server: ctx=$CTX np=$NP ub=$UBATCH  (${CONCURRENCIES})"
echo

docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri \
  -v "${MODELS_DIR}:/models" -p "${PORT_A}:8080" \
  "$IMAGE" \
  -m "$MODEL" --embeddings --pooling last -ngl 999 \
  -c "$CTX" -np "$NP" -b "$UBATCH" -ub "$UBATCH" -fa on --metrics \
  --host 0.0.0.0 --port 8080 >/dev/null 2>&1

n=0
while [ "$n" -lt 150 ]; do
  if ! docker ps --filter "name=${NAME}" --format '{{.Names}}' | grep -q "$NAME"; then
    docker logs "$NAME" > "${RUN}/crash.log" 2>&1
    echo "container exited during load — see ${RUN}/crash.log" >&2
    exit 1
  fi
  curl -fsS --max-time 3 "http://127.0.0.1:${PORT_A}/health" >/dev/null 2>&1 && break
  n=$((n + 1)); sleep 2
done

echo "== consistency =="
python3 scripts/capture.py --url "http://127.0.0.1:${PORT_A}" \
  --fixtures "$FIXTURES" --out "${RUN}/capture.json" --repeat 2 \
  --label "ctx=$CTX np=$NP ub=$UBATCH" >"${RUN}/capture.log" 2>&1 || {
    echo "capture failed — see ${RUN}/capture.log" >&2; exit 1; }
if ! python3 scripts/compare.py --determinism "${RUN}/capture.json" \
     --min-cosine "$DET_MIN_COSINE" >"${RUN}/determinism.log" 2>&1; then
  echo "DETERMINISM FAILED — not benchmarking. See ${RUN}/determinism.log" >&2
  exit 1
fi
echo "   ok — $(grep -o 'worst cosine: [0-9.]*' "${RUN}/determinism.log" | tail -1)"
echo

printf 'concurrency,tokens_per_second,wall_seconds,p50_ms,p95_ms,max_ms,errors\n' > "$CSV"
printf '%-12s %10s %9s %9s %9s %10s %7s\n' conc tok/s wall_s p50_ms p95_ms max_ms errors

for c in $CONCURRENCIES; do
  bench_csv="${RUN}/c${c}.csv"
  if ! python3 scripts/bench.py --url "http://127.0.0.1:${PORT_A}" \
        --fixtures "$BENCH_FIXTURES" --concurrency "$c" --passes 3 \
        --note "ctx=$CTX np=$NP ub=$UBATCH conc=$c" --image "$IMAGE" \
        --history "$bench_csv" >"${RUN}/c${c}.log" 2>&1; then
    printf '%-12s %10s\n' "$c" "FAILED"
    printf '%s,,,,,,\n' "$c" >> "$CSV"
    continue
  fi
  vals="$(python3 -c "
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))[-1]
print(r['tokens_per_second'], r['wall_seconds'], r['p50_ms'],
      r['p95_ms'], r['max_ms'], r['errors'])
" "$bench_csv")"
  read -r tps wall p50 p95 mx errs <<<"$vals"
  printf '%-12s %10s %9s %9s %9s %10s %7s\n' "$c" "$tps" "$wall" "$p50" "$p95" "$mx" "$errs"
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$c" "$tps" "$wall" "$p50" "$p95" "$mx" "$errs" >> "$CSV"
done

echo
echo "results in $CSV"
