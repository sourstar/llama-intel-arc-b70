#!/usr/bin/env bash
# Sweep a set of inference configurations: consistency first, throughput only
# if consistency passed.
#
#   ./scripts/matrix.sh                    # uses matrix.conf
#   ./scripts/matrix.sh my-configs.conf
#
# For every configuration:
#   1. start a throwaway container with those flags
#   2. embed the GATE pool twice — determinism, then cross-config against a
#      single fixed reference capture taken once at the start
#   3. ONLY if both checks pass, embed the BENCH pool and record throughput
#
# Step 3 is conditional on purpose. A configuration that produces different
# vectors is not a slow-or-fast question, it is a broken one, and the number
# would only invite someone to adopt it. A failed row carries its reason and
# an empty tokens_per_second.
#
# The reference is captured once rather than per configuration: it is a fixed
# yardstick, and re-deriving it each time would both cost a container start
# and let the yardstick drift.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

CONF="${1:-matrix.conf}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="baselines/matrix-${STAMP}"
CSV="${RUN}/results.csv"
NAME="lem-mx"

if [ ! -f "$CONF" ]; then
  echo "no config file at $CONF" >&2
  exit 2
fi
for f in "$FIXTURES" "$BENCH_FIXTURES"; do
  if [ ! -f "$f" ]; then
    echo "missing fixture pool $f — run scripts/make_fixtures.py first" >&2
    exit 2
  fi
done

mkdir -p "$RUN"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT

# $1 name  $2 ctx  $3 np  $4 ubatch  $5 fa  $6 extra flags  $7 extra env
#
# -fa is its own column rather than something EXTRA can set, because passing
# it twice would leave which one wins up to the arg parser, and a config that
# silently ran the opposite of what its name says is worse than no data.
start() {
  local env_args=()
  local kv
  for kv in $7; do
    [ "$kv" = "-" ] && continue
    env_args+=(-e "$kv")
  done
  docker rm -f "$NAME" >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --device /dev/dri \
    -v "${MODELS_DIR}:/models" -p "${PORT_A}:8080" \
    "${env_args[@]}" \
    "$IMAGE" \
    -m "$MODEL" --embeddings --pooling last -ngl 999 \
    -c "$2" -np "$3" -b "$4" -ub "$4" -fa "$5" --metrics \
    --host 0.0.0.0 --port 8080 $6 >/dev/null 2>&1
}

# A container that dies during load would otherwise sit in capture.py's
# 300s health wait. Fail in seconds instead, and keep the log: a bad flag
# combination reports itself on stderr, which the TrueNAS pane never shows.
wait_or_die() {
  local n=0
  while [ "$n" -lt 150 ]; do
    if ! docker ps --filter "name=${NAME}" --format '{{.Names}}' | grep -q "$NAME"; then
      docker logs "$NAME" > "${RUN}/$1.crash.log" 2>&1
      return 1
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:${PORT_A}/health" >/dev/null 2>&1; then
      return 0
    fi
    n=$((n + 1)); sleep 2
  done
  docker logs "$NAME" > "${RUN}/$1.timeout.log" 2>&1
  return 1
}

echo "matrix run $STAMP"
echo "image:     $IMAGE"
echo "gate pool: $FIXTURES"
echo "bench pool:$BENCH_FIXTURES"
echo "results:   $CSV"
echo

# ---- the fixed reference -----------------------------------------------
REF="${RUN}/reference.json"
echo "== reference capture (ctx=$REF_CTX np=$REF_NP ub=$REF_UBATCH) =="
start ref "$REF_CTX" "$REF_NP" "$REF_UBATCH" on "" "-"
if ! wait_or_die reference; then
  echo "reference container failed to start — see ${RUN}/reference.*.log" >&2
  exit 1
fi
if ! python3 scripts/capture.py --url "http://127.0.0.1:${PORT_A}" \
      --fixtures "$FIXTURES" --out "$REF" --repeat 2 \
      --label "reference ctx=$REF_CTX np=$REF_NP ub=$REF_UBATCH" >"${RUN}/reference.log" 2>&1; then
  echo "reference capture failed — see ${RUN}/reference.log" >&2
  exit 1
fi
docker rm -f "$NAME" >/dev/null 2>&1
echo "   ok"
echo

printf 'name,ctx,np,ubatch,fa,extra,consistency,worst_det_cosine,worst_cross_cosine,tokens_per_second,wall_seconds,p50_ms,p95_ms,max_ms,errors,note\n' > "$CSV"

worst_of() { grep -o 'worst cosine: [0-9.]*' "$1" | tail -1 | awk '{print $3}'; }

# ---- the sweep ----------------------------------------------------------
while IFS= read -r line; do
  case "$line" in ''|\#*) continue ;; esac
  read -r cname cctx cnp cub cfa cextra <<<"$line"
  [ -z "${cname:-}" ] && continue
  cextra="${cextra:--}"
  [ "$cextra" = "-" ] && cextra=""

  echo "== $cname  (ctx=$cctx np=$cnp ub=$cub fa=$cfa ${cextra:-no extra flags}) =="
  cenv=""
  case "$cextra" in
    env:*) cenv="${cextra#env:}"; cextra="" ;;
  esac

  if ! start "$cname" "$cctx" "$cnp" "$cub" "$cfa" "$cextra" "${cenv:--}" || ! wait_or_die "$cname"; then
    echo "   DID NOT START — see ${RUN}/${cname}.*.log"
    printf '%s,%s,%s,%s,%s,%s,did-not-start,,,,,,,,,container exited during load\n' \
      "$cname" "$cctx" "$cnp" "$cub" "$cfa" "${cextra:-${cenv:+env:$cenv}}" >> "$CSV"
    continue
  fi

  cap="${RUN}/${cname}.json"
  if ! python3 scripts/capture.py --url "http://127.0.0.1:${PORT_A}" \
        --fixtures "$FIXTURES" --out "$cap" --repeat 2 \
        --label "$cname" >"${RUN}/${cname}.capture.log" 2>&1; then
    echo "   CAPTURE FAILED — see ${RUN}/${cname}.capture.log"
    printf '%s,%s,%s,%s,%s,%s,capture-failed,,,,,,,,,embedding requests failed\n' \
      "$cname" "$cctx" "$cnp" "$cub" "$cfa" "${cextra:-${cenv:+env:$cenv}}" >> "$CSV"
    docker rm -f "$NAME" >/dev/null 2>&1
    continue
  fi

  det_log="${RUN}/${cname}.determinism.log"
  cross_log="${RUN}/${cname}.cross.log"
  ok=1
  python3 scripts/compare.py --determinism "$cap" \
    --min-cosine "$DET_MIN_COSINE" >"$det_log" 2>&1 || ok=0
  python3 scripts/compare.py "$cap" "$REF" \
    --min-cosine "$MIN_COSINE" >"$cross_log" 2>&1 || ok=0
  det_worst="$(worst_of "$det_log")"
  cross_worst="$(worst_of "$cross_log")"
  echo "   determinism worst=${det_worst:-?}   vs reference worst=${cross_worst:-?}"

  if [ "$ok" -ne 1 ]; then
    echo "   CONSISTENCY FAILED — not benchmarking it"
    printf '%s,%s,%s,%s,%s,%s,FAIL,%s,%s,,,,,,,vectors differ; see %s.{determinism,cross}.log\n' \
      "$cname" "$cctx" "$cnp" "$cub" "$cfa" "${cextra:-${cenv:+env:$cenv}}" \
      "$det_worst" "$cross_worst" "$cname" >> "$CSV"
    docker rm -f "$NAME" >/dev/null 2>&1
    continue
  fi

  # Consistency passed. Now, and only now, is a throughput number worth having.
  bench_csv="${RUN}/${cname}.bench.csv"
  if ! python3 scripts/bench.py --url "http://127.0.0.1:${PORT_A}" \
        --fixtures "$BENCH_FIXTURES" --concurrency "$cnp" --passes 3 \
        --note "$cname" --image "$IMAGE" --history "$bench_csv" \
        >"${RUN}/${cname}.bench.log" 2>&1; then
    echo "   BENCH FAILED — see ${RUN}/${cname}.bench.log"
    printf '%s,%s,%s,%s,%s,%s,pass,%s,%s,,,,,,,bench failed after consistency passed\n' \
      "$cname" "$cctx" "$cnp" "$cub" "$cfa" "${cextra:-${cenv:+env:$cenv}}" \
      "$det_worst" "$cross_worst" >> "$CSV"
    docker rm -f "$NAME" >/dev/null 2>&1
    continue
  fi

  # Read the bench row by COLUMN NAME. Counting fields from the end was wrong
  # by one and silently reported wall_seconds as tokens/s; and it would break
  # again the moment a note contained a comma, which csv quotes and awk does
  # not understand.
  vals="$(python3 -c "
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))[-1]
print(r['tokens_per_second'], r['wall_seconds'], r['p50_ms'],
      r['p95_ms'], r['max_ms'], r['errors'])
" "$bench_csv")"
  read -r tps wall p50 p95 mx errs <<<"$vals"
  echo "   PASS — ${tps} tok/s  p50=${p50}ms p95=${p95}ms max=${mx}ms"
  printf '%s,%s,%s,%s,%s,%s,pass,%s,%s,%s,%s,%s,%s,%s,%s,\n' \
    "$cname" "$cctx" "$cnp" "$cub" "$cfa" "${cextra:-${cenv:+env:$cenv}}" \
    "$det_worst" "$cross_worst" "$tps" "$wall" "$p50" "$p95" "$mx" "$errs" >> "$CSV"

  docker rm -f "$NAME" >/dev/null 2>&1
  echo
done < "$CONF"

echo
echo "== matrix complete =="
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
echo
echo "full logs and captures under $RUN/"
