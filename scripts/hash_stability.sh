#!/usr/bin/env bash
# Would another node store the same bytes?
#
#   ./scripts/hash_stability.sh
#   CONFIGS="a:-np 1 -c 8192|b:-np 4 -c 32768" ./scripts/hash_stability.sh
#
# The Louise project treats hash-verifiable embeddings as a requirement, which
# is a much stronger property than "the server returns the same vector twice".
# It means the STORED bytes — L2-normalised, downcast to f16 — must be
# identical across whatever varies between two nodes holding the same corpus.
#
# So this runs a set of configurations, computes the sha256 of each fixture
# chunk's stored vector under each, and reports which configurations agree.
#
# What it can vary: server flags, environment, image. What it cannot vary from
# one machine: the GPU, the driver, and the backend. Those are the ones most
# likely to break cross-peer identity, and they need a second node to test.
#
# Every configuration is captured with --repeat 2 so that a configuration which
# is not even self-consistent is reported as such (hash `null`) rather than
# being silently compared on whichever result came back first.

set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source config.env

# name:flags, pipe-separated. Defaults probe the axes that can differ between
# two nodes running the same image: slot count, context, ubatch, and a plain
# restart.
CONFIGS="${CONFIGS:-restart-1:-np 1 -c 8192 -b 8192 -ub 8192|restart-2:-np 1 -c 8192 -b 8192 -ub 8192|slots-4:-np 4 -c 32768 -b 8192 -ub 8192|ubatch-2048:-np 1 -c 8192 -b 2048 -ub 2048|ctx-32768:-np 1 -c 32768 -b 8192 -ub 8192}"
EXTRA_ENV_ARGS="${EXTRA_ENV_ARGS:--e GGML_SYCL_ENABLE_DNN=0}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="baselines/hash-stability-${STAMP}"
NAME="lem-hash"
mkdir -p "$RUN"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT

echo "hash stability ${STAMP}"
echo "image:  $IMAGE"
echo "env:    ${EXTRA_ENV_ARGS:-none}"
echo

names=()
IFS='|' read -r -a entries <<< "$CONFIGS"
for entry in "${entries[@]}"; do
  name="${entry%%:*}"
  flags="${entry#*:}"
  names+=("$name")
  echo "== $name  ($flags)"

  docker rm -f "$NAME" >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --device /dev/dri $EXTRA_ENV_ARGS \
    -v "${MODELS_DIR}:/models" -p "${PORT_A}:8080" \
    "$IMAGE" \
    -m "$MODEL" --embeddings --pooling last -ngl 999 -fa on --metrics \
    --host 0.0.0.0 --port 8080 $flags >/dev/null 2>&1

  n=0
  while [ "$n" -lt 90 ]; do
    curl -fsS --max-time 3 "http://127.0.0.1:${PORT_A}/health" >/dev/null 2>&1 && break
    n=$((n + 1)); sleep 2
  done
  if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT_A}/health" >/dev/null 2>&1; then
    docker logs "$NAME" > "${RUN}/${name}.crash.log" 2>&1
    echo "   did not start — see ${RUN}/${name}.crash.log"
    echo '{}' > "${RUN}/${name}.hashes.json"
    continue
  fi

  python3 scripts/capture.py --url "http://127.0.0.1:${PORT_A}" \
    --fixtures "$FIXTURES" --out "${RUN}/${name}.json" --repeat 2 \
    --label "$name" >"${RUN}/${name}.capture.log" 2>&1
  python3 scripts/storage_reproducibility.py "${RUN}/${name}.json" --hashes \
    > "${RUN}/${name}.hashes.json"
  echo "   captured"
done

docker rm -f "$NAME" >/dev/null 2>&1

echo
python3 - "$RUN" "${names[@]}" <<'PYEOF'
import json, os, sys

run, names = sys.argv[1], sys.argv[2:]
hashes = {}
for n in names:
    p = os.path.join(run, f"{n}.hashes.json")
    try:
        hashes[n] = json.load(open(p))
    except (OSError, ValueError):
        hashes[n] = {}

chunks = sorted({c for h in hashes.values() for c in h})
base = names[0]

w = max(len(n) for n in names) + 2
print("stored-vector hash agreement, against", base)
print()
print(f"{'chunk':8}" + "".join(f"{n:<{w}}" for n in names))
for c in chunks:
    row = f"{c:8}"
    ref = hashes[base].get(c)
    for n in names:
        v = hashes[n].get(c)
        if v is None:
            cell = "unstable"
        elif v == ref:
            cell = "same"
        else:
            cell = "DIFFERS"
        row += f"{cell:<{w}}"
    print(row)

print()
for n in names[1:]:
    diff = [c for c in chunks
            if hashes[n].get(c) is not None
            and hashes[base].get(c) is not None
            and hashes[n][c] != hashes[base][c]]
    unstable = [c for c in chunks if hashes[n].get(c) is None]
    if not diff and not unstable:
        print(f"{n}: identical to {base} on every chunk")
    else:
        print(f"{n}: {len(diff)} differing, {len(unstable)} self-inconsistent")
PYEOF
echo
echo "captures and hashes under $RUN/"
