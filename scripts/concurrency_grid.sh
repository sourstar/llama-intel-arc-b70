#!/usr/bin/env bash
# Server slots x client concurrency, the whole grid.
#
#   ./scripts/concurrency_grid.sh
#   SERVERS="2:16384 4:32768" CONCURRENCIES="1 2 4 8" ./scripts/concurrency_grid.sh
#
# Each entry of SERVERS is np:ctx. Per-slot context is ctx/np and is held at
# 8192 throughout, because gutenberg.toml declares `ctx = 8192` and a server
# that cannot hold a policy-legal chunk is not a candidate however fast it
# looks — scripts/concurrency.sh checks consistency before benchmarking, so
# such a config fails there rather than producing a misleading number.
#
# -ub is held constant across the grid so that concurrency is the only thing
# that varies. It was measured not to matter much (2383-2433 tok/s across
# ub 1024..8192), and 2048 is the value that fits at np=8.
#
# NOTE ON THE SHEBANG: this exists as a script rather than a shell one-liner
# because the TrueNAS root shell is zsh, where an unquoted `$var` is NOT
# word-split. `set -- $spec` yields one argument there, and the first attempt
# at this grid silently launched `-np "1 8192"` with a default context. Under
# a bash shebang the split behaves; the `${spec%%:*}` form below works in
# either shell regardless.

set -uo pipefail
cd "$(dirname "$0")/.."

SERVERS="${SERVERS:-1:8192 2:16384 4:32768 8:65536}"
export CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"
export UBATCH="${UBATCH:-2048}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="baselines/grid-${STAMP}.csv"
mkdir -p baselines
printf 'np,ctx,ctx_per_slot,concurrency,tokens_per_second,wall_seconds,p50_ms,p95_ms,max_ms,errors\n' > "$OUT"

echo "concurrency grid ${STAMP}"
echo "servers:      $SERVERS"
echo "client conc:  $CONCURRENCIES"
echo "ubatch held:  $UBATCH"
echo "collecting:   $OUT"
echo

for spec in $SERVERS; do
  np="${spec%%:*}"
  ctx="${spec##*:}"
  echo "################ server: -np $np -c $ctx  ($((ctx / np)) tokens/slot) ################"
  if ! NP="$np" CTX="$ctx" ./scripts/concurrency.sh; then
    echo "   server np=$np ctx=$ctx did not produce results"
    echo
    continue
  fi
  # concurrency.sh writes baselines/concurrency-<stamp>/results.csv; fold the
  # newest one into the grid, tagged with the server config that produced it.
  latest="$(ls -1d baselines/concurrency-* 2>/dev/null | tail -1)"
  if [ -n "$latest" ] && [ -f "$latest/results.csv" ]; then
    tail -n +2 "$latest/results.csv" | while IFS= read -r line; do
      printf '%s,%s,%s,%s\n' "$np" "$ctx" "$((ctx / np))" "$line" >> "$OUT"
    done
  fi
  echo
done

echo "== grid complete =="
column -s, -t "$OUT" 2>/dev/null || cat "$OUT"
echo
echo "$OUT"
