#!/usr/bin/env bash
# Soak test. Sustained embedding load while watching the kernel log.
#
# #25692 (GPU hang under sustained load) and #24946 (gt-c0 power pinning) do
# not appear at startup — only under continuous work. This is the test that
# stands in for a real ingest.
#
#   ./scripts/soak.sh            # uses SOAK_SECONDS from config.env
#   SOAK_SECONDS=300 ./scripts/soak.sh
#
# Runs against PROD_URL, i.e. the real app, deliberately: the point is to
# exercise the deployed configuration.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

DURATION="${SOAK_SECONDS}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="baselines/soak-${STAMP}.log"
# Round throughput is worth keeping — a rising p95 across rounds is the drift
# signal this test looks for — but it does not belong in the baseline history,
# which is one row per deliberate run.
HIST="baselines/soak-${STAMP}.csv"
mkdir -p baselines

# dmesg is a ring buffer. Counting matching lines before and after and taking
# the difference is unsound: under pressure the buffer drops old entries, so
# four engine resets during the run plus four older lines rolling off shows as
# a delta of zero — a PASS that hides exactly what this test exists to catch.
# Observed on the first 1800s run: the reset count went 84 -> 80 with nothing
# happening at all. Compare content instead of totals.
DMESG_BEFORE="$(mktemp)"
trap 'rm -f "$DMESG_BEFORE"' EXIT
dmesg --time-format=iso > "$DMESG_BEFORE" 2>/dev/null

if [ ! -s "$DMESG_BEFORE" ]; then
  echo "warning: cannot read dmesg — driver events will not be detected" >&2
fi

new_lines() {
  # -F -x -v: keep only lines that are not byte-for-byte in the snapshot.
  # Timestamps make every line unique, so this is exactly the new entries.
  dmesg --time-format=iso 2>/dev/null | grep -Fxv -f "$DMESG_BEFORE" 2>/dev/null
}
count_new() {
  new_lines | grep -ci "$1" || true
}

echo "soak: ${DURATION}s against ${PROD_URL}" | tee "$LOG"
echo "watching dmesg for new engine resets and gp faults" | tee -a "$LOG"
echo | tee -a "$LOG"

if ! curl -fsS "${PROD_URL}/health" >/dev/null 2>&1; then
  echo "server not healthy at ${PROD_URL}" | tee -a "$LOG" >&2
  exit 1
fi

END=$((SECONDS + DURATION))
ROUND=0
FAILED=0

while [ "$SECONDS" -lt "$END" ]; do
  ROUND=$((ROUND + 1))
  REMAIN=$((END - SECONDS))
  echo "-- round $ROUND (${REMAIN}s remaining)" | tee -a "$LOG"

  if ! python3 scripts/bench.py --url "$PROD_URL" \
        --fixtures "$FIXTURES" --concurrency "$NP" --passes 2 \
        --warmup 0 --note "soak round $ROUND" --image "$IMAGE" \
        --history "$HIST" \
        >>"$LOG" 2>&1; then
    echo "   bench failed on round $ROUND" | tee -a "$LOG"
    FAILED=$((FAILED + 1))
  fi

  # A hung GPU often leaves /health answering 200 while requests never
  # return, so check both.
  if ! curl -fsS --max-time 10 "${PROD_URL}/health" >/dev/null 2>&1; then
    echo "   health check failed — server is down" | tee -a "$LOG"
    break
  fi
done

RESETS="$(count_new "Engine reset")"
FAULTS="$(count_new "general protection fault")"

{
  echo
  echo "== soak complete after $ROUND round(s) =="
  echo "new engine resets: $RESETS"
  echo "new gp faults:     $FAULTS"
  echo "failed rounds: $FAILED"
} | tee -a "$LOG"

echo
echo "throughput per round is in $LOG and $HIST"
echo "a rising p95 across rounds means thermal or memory drift, not a hang"

if [ "$RESETS" -gt 0 ] || [ "$FAULTS" -gt 0 ] || [ "$FAILED" -gt 0 ]; then
  echo
  echo "FAIL: driver events or request failures during soak."
  echo "If engine resets appeared, try GGML_SYCL_FA_ONEDNN_MAX_KV=8192 before"
  echo "disabling oneDNN FA outright — see README."
  exit 1
fi

echo "PASS"
