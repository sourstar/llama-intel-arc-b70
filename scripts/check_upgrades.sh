#!/usr/bin/env bash
# Which newer SYCL images actually exist?
#
# llama.cpp images are published for only a subset of builds, and the tags are
# build numbers rather than semver — so neither the releases page nor the tag
# list answers "what can I upgrade to". This probes GHCR directly.
#
#   ./scripts/check_upgrades.sh            # scans 200 builds past current
#   ./scripts/check_upgrades.sh 500        # wider scan
#
# Anything it prints still has to pass scripts/correctness.sh before it goes
# into production.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

RANGE="${1:-200}"
VARIANT="server-intel"

CURRENT="$(printf '%s' "$IMAGE" | sed -n "s/.*${VARIANT}-b\([0-9]\+\).*/\1/p")"
if [ -z "$CURRENT" ]; then
  echo "could not parse a build number from IMAGE=$IMAGE" >&2
  echo "expected something like ghcr.io/ggml-org/llama.cpp:${VARIANT}-b10630" >&2
  exit 2
fi

echo "current: b${CURRENT}"
echo "scanning b$((CURRENT + 1)) .. b$((CURRENT + RANGE)) for published ${VARIANT} images"
echo

TOKEN="$(curl -s "https://ghcr.io/token?scope=repository:ggml-org/llama.cpp:pull&service=ghcr.io" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")"
if [ -z "$TOKEN" ]; then
  echo "could not get a GHCR pull token" >&2
  exit 1
fi

ACCEPT="application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json"
FOUND=0

for n in $(seq $((CURRENT + 1)) $((CURRENT + RANGE))); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -I \
    -H "Authorization: Bearer $TOKEN" -H "Accept: $ACCEPT" \
    "https://ghcr.io/v2/ggml-org/llama.cpp/manifests/${VARIANT}-b${n}")"
  if [ "$code" = "200" ]; then
    echo "  ghcr.io/ggml-org/llama.cpp:${VARIANT}-b${n}"
    FOUND=$((FOUND + 1))
  fi
done

echo
if [ "$FOUND" -eq 0 ]; then
  echo "nothing newer published in that range"
  exit 0
fi

echo "$FOUND newer image(s)."
echo
echo "To evaluate one:"
echo "  1. edit IMAGE in config.env"
echo "  2. ./scripts/correctness.sh          # must pass before anything else"
echo "  3. python3 scripts/bench.py --note 'b<NEW> eval'"
echo "  4. compare against the previous row in baselines/history.csv"
echo "  5. ./scripts/soak.sh"
echo
echo "Check the SYCL commits in the range before upgrading:"
echo "  git log --oneline b${CURRENT}..b<NEW> -i --grep='sycl|onednn|intel'"
