# llama-embed — Qwen3-Embedding-8B on Intel Arc Pro B70

Embedding server for the RAG ingest pipeline. Runs as a TrueNAS 26 custom app
(Docker Compose), SYCL backend, single Arc Pro B70.

- Config: [`embed-xpu-qwen3-8b.yaml`](./embed-xpu-qwen3-8b.yaml)
- Known issues, with the measurement behind each: [`KNOWN-ISSUES.md`](./KNOWN-ISSUES.md)
- Upstream reference, check before any version bump: [jeffgrover/b70-setup](https://github.com/jeffgrover/b70-setup)

## Image selection

llama.cpp container images are not tagged with semver. GHCR publishes
`<variant>[-backend]-b<NUM>` only, so there is no `:v0.3.0` image, and bare
build-number tags like `:b10642` do not exist either.

Release tags map to build numbers like this:

| release | build |
|---|---|
| v0.1.1 | b10470 |
| v0.1.2 | b10485 |
| v0.2.0 | b10566 |
| v0.3.0 | b10621 |

Images are only pushed for a subset of builds. No image exists for b10621, so
the nearest published SYCL image containing v0.3.0 is **b10630** (v0.3.0 plus
nine builds). Nearby published intel images: b10603, b10615, b10630.

To check whether a given build has an image:

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:ggml-org/llama.cpp:pull&service=ghcr.io" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -o /dev/null -w "%{http_code}\n" -I \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.oci.image.index.v1+json" \
  "https://ghcr.io/v2/ggml-org/llama.cpp/manifests/server-intel-b10630"
```

## Flag notes

**`-np` is the flag that matters most, and the deployed value is wrong.**
Measured 2026-08-27 on the production chunk distribution, warm, per-slot
context held at 8192, zero errors anywhere in the grid:

| server | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| **`-np 1`** | 3133 | **3241** | **3239** | **3236** | **3234** | **3232** |
| `-np 2` | 3154 | 3238 | 2397 | 2419 | 2421 | 2407 |
| `-np 4` (deployed) | 3155 | 3237 | 2396 | 1953 | 1934 | 1932 |
| `-np 8` | 3152 | 3235 | 2483 | 2224 | 2015 | 2011 |

tokens/second, client concurrency across the top.

The B70 wants **two** ~900-token prefills in flight and no more. Give the
server enough slots to run four and aggregate throughput drops by a quarter;
give it enough for eight and it drops by 40%, with worst-case latency going
from 1.8s to 33s. Nothing errors and nothing logs — it is silently a third
slower.

`-np 1` cannot be pushed into that regime, because it serialises: it holds
~3232 tok/s whether the client sends 2 requests or 32, and its worst case stays
proportional to queue depth rather than collapsing. For a throughput-bound
ingest that is the right shape. The cost is that a query embedding queues
behind ingest work — about 600ms at concurrency 2, which is fine, but worth
remembering if this endpoint ever fronts interactive search under load.

See KNOWN-ISSUES D4. Reproduce with `scripts/concurrency_grid.sh`.

**Warm up before believing any number.** A fresh container runs ~35% slow for
its first ~20,000 tokens while SYCL compiles kernels for the batch shapes. It
does not decay — five minutes of idle keeps the fast state — but it does mean
a benchmark taken immediately after `docker run` reads far low, and that a
restart between ingests costs real throughput. KNOWN-ISSUES C7.

**`GGML_SYCL_ENABLE_DNN=0` is required for hash-verifiable embeddings, and costs 1.7%.** Short
prefills — below ~1024 tokens, which is nearly every production chunk — come
back slightly different on every run with oneDNN enabled: four of the gate
chunks produced ten distinct vectors in ten runs. The divergence is tiny
(cosine ≥ 0.9999974) but it is real, and it means a re-embedded chunk can only
be verified by cosine with a tolerance, never by hash.

It is oneDNN's GEMM, not attention — `-fa off` changes nothing, and neither
does `-np 1`. Disabling oneDNN makes all 16 gate chunks bit-identical over 10
runs, produces the same vectors (worst cosine 0.999999029 against oneDNN-on),
and costs 3241 → 3186 tok/s (n=3 each). KNOWN-ISSUES C3.

Do not confuse it with `GGML_SYCL_FA_ONEDNN=0` below, which is a different flag
disabling only the fused SDPA.

**Do not set `GGML_SYCL_FA_ONEDNN=0`.** The prefill dispatch in `fattn.cpp` is
oneDNN SDPA first (needs 32+ query tokens), then oneMKL FA, then TILE/VEC. Both
of the first two are XMX-accelerated, but MKL additionally requires
`K->ne[1] >= 1024`. With oneDNN disabled, chunks shorter than ~1024 tokens fall
through to the non-XMX kernels. Embedding is pure prefill, so the FA dispatch
is on the hot path for every request. (Not measured directly: `-fa off`, which
is the blunter version of the same idea, costs 8x and fails most requests —
KNOWN-ISSUES D5.)

If long sequences trigger `xe` engine resets under sustained load, cap the fused
path with `GGML_SYCL_FA_ONEDNN_MAX_KV=8192` rather than disabling oneDNN
outright. That is the upstream escape hatch for exactly this.

**Never quantize the KV cache.** No `-ctk` / `-ctv`. FA with quantized KV hangs
the GPU on B70 under sustained server load (ggml-org/llama.cpp#25692); f16 KV
with FA is stable.

**`-c 32768` is sized deliberately.** Four slots of 8192 = max chunk length.
Qwen3-8B is 36 layers x 1024 KV dim, so f16 KV costs ~144 KiB/token: 32768
reserves ~4.6 GiB where 65536 would reserve ~9.2 GiB for no benefit.

**`-ub` does NOT have to be >= max chunk length.** This was the original
claim here, on the reasoning that the server clamps `n_batch` to `n_ubatch`
for pooled embeddings (`server.cpp:146`). The clamp is real; the conclusion
is not. It holds for BERT-style encoders, where the whole sequence has to sit
in one ubatch because attention is bidirectional. Qwen3-Embedding is a causal
decoder with last-token pooling, so a prompt split across ubatches still sees
everything before it through the KV cache.

Measured 2026-08-27 on b10630, 7900-token chunks, `-ub 8192` vs `-ub 2048`:
**cosine 1.000000** — bit-identical vectors, not merely close. The split is
free, and `-ub 8192` costs throughput (see Measured on this card).

**Production chunks are ~1000 tokens, not 8000.** From the gutenberg
manifest, 2026-08-27: 12,784 embedded documents, 1,373,515 chunks, mean 926
tokens per chunk, p99 of per-document mean 1000. The chunker targets ~1000
and the `ctx=8192` truncation policy never binds. Sizing `-c` and `-ub` for
8192-token chunks reserves KV and compute buffers for work that does not
arrive. Any change to `-c`/`-np`/`-ub` should be argued against that number,
and re-derived if the collection's chunker version changes.

## Recommended configuration (measured 2026-08-27, NOT yet applied)

`embed-xpu-qwen3-8b.yaml` still holds the deployed configuration. This is what
the measurements say it should be, kept separate so that file always describes
what is actually running:

```diff
     command: >-
       -m /models/Qwen3-Embedding-8B-Q4_K_M.gguf --embeddings --pooling last -ngl
-      999 -c 32768 -np 4 -b 8192 -ub 8192 -fa on -t 6 --metrics --host 0.0.0.0
+      999 -c 8192 -np 1 -b 8192 -ub 8192 -fa on -t 6 --metrics --host 0.0.0.0
       --port 8080
+    environment:
+      GGML_SYCL_ENABLE_DNN: "0"
```

Against the deployed configuration, both warm, both consistency-gated:

| | tok/s | p50 | p95 | max | stored-hash stable |
|---|---|---|---|---|---|
| as deployed, `-c 32768 -np 4` @ conc 4 | 2548 | 1362 | 1574 | 1585 | no — 11/16 gate chunks |
| **recommended @ conc 2** | **3186** | **571** | **606** | **610** | **yes — 24/24** |

**+25% throughput, p95 2.6x better, and every stored vector reproducible.**
It also returns ~3.5 GiB of VRAM, since one slot of 8192 replaces four.

The recommended configuration across client concurrency, on the bench pool
(24 chunks at production length), hash stability checked over three repeats at
each level:

| client concurrency | tok/s | p50 ms | p95 ms | max ms | errors | hash stable |
|---|---|---|---|---|---|---|
| 1 | 3109 | 294 | 312 | 315 | 0 | 24/24 |
| **2** | **3186** | 571 | 606 | 610 | 0 | **24/24** |
| 4 | 3174 | 1144 | 1212 | 1215 | 0 | 24/24 |
| 8 | 3173 | 2278 | 2395 | 2408 | 0 | 24/24 |

Throughput plateaus at concurrency 2 and stays flat; beyond that only latency
grows, linearly, because requests queue against a single slot. **Reproducibility
does not constrain the client at all** — one slot cannot mix requests into a
shared batch, so nothing about the result depends on load. Drive it at 2 for
the throughput; anything higher is harmless but pointless.

Each change, and why:

- **`-np 4` → `-np 1`.** Required twice over. For reproducibility: above one
  slot, concurrent requests share a ubatch, so a stored vector depends on what
  else was in flight — measured at `-np 4`, 0 of 16 chunks stable under 8-way
  load against 16 of 16 serially. For throughput: `-np` exists to amortise
  weight reads across parallel *decode* steps, and an embedding server never
  decodes; a 900-token prefill already amortises them. Four slots cost 25%.
- **`-c 32768` → `-c 8192`.** A consequence of the `-np` change, not an
  independent decision. Per-slot context must stay at 8192 to hold a
  policy-legal chunk (`gutenberg.toml` declares `ctx = 8192`); with one slot
  that is 8192 total. Do not lower per-slot context below 8192.
- **`GGML_SYCL_ENABLE_DNN=0`.** Costs 1.7% and makes every chunk length
  bit-reproducible. Required: hash-verifiable embeddings are a Louise project
  requirement, and without this flag five of sixteen gate chunks hash to a
  different stored vector on repeat — the L2 + f16 downcast does not absorb
  the drift. KNOWN-ISSUES C3 and F.

**`-ub 8192` is unchanged, and the reason is stronger than throughput.** It is
identical in speed to 1024 or 2048, but at 8192 it equals the truncation policy
ceiling, so no chunk is ever sliced — which takes `-ub` out of the vector's
identity entirely. At a smaller `-ub`, chunks longer than it slice, and slicing
changes the bytes.

Before applying, and after: `./scripts/correctness.sh`, then
`python3 scripts/bench.py --note "..."` so `baselines/history.csv` carries a row
on each side of the change. For the reproducibility half,
`./scripts/hash_stability.sh`.

## Environment gotchas

These caused a multi-hour restart loop. Neither is a llama.cpp problem.

**`start_period: 180s` is load-bearing.** The image ships its own
`HEALTHCHECK` with no start period. Cold load takes ~30s, during which
`/health` does not answer, so the built-in check fails from its first probe and
the app is torn down before it finishes initialising. The symptom is a log that
stops at `llama threadpool init` and starts over.

**`SYCL_CACHE_PERSISTENT` crashes this image.** Tested four ways — nested mount,
separate mount, `apps`-owned dir, `root`-owned dir, empty cache — and it dies
~4s in, at JIT, every time. Not used. It would only have saved ~30s of startup.

**GPU passthrough is just `devices: - /dev/dri`.** The image has no `USER`
directive so it runs as root; no `group_add` needed. The
`resources: reservations: devices:` block is NVIDIA-only and does nothing here.
If the B70 is listed under System Settings > Advanced > Isolated GPU Devices,
apps cannot claim it.

Host prerequisite: `arc-b70.service` (GuC firmware bind-mount, `probe_display=0`)
must be re-applied after TrueNAS version updates.

## Debugging

The TrueNAS app log pane shows stdout only and swallows stderr, which hides
every actual error. Reproduce by hand instead:

```bash
midclt call -j app.stop embed-gpu-qwen3-8b
docker run -t --rm --device /dev/dri \
  -v /mnt/ssd5/apps/llama-embed:/models \
  ghcr.io/ggml-org/llama.cpp:server-intel-b10630 \
  -m /models/Qwen3-Embedding-8B-Q4_K_M.gguf --embeddings --pooling last \
  -ngl 999 -c 32768 -np 4 -b 8192 -ub 8192 -fa on --port 8080 2>&1 | cat
```

`-t` forces line buffering. Do not pipe to `tail` — a crash discards the block
buffer and you see nothing. Wait for `model loaded`; ~30s cold is normal, so a
pause at `threadpool init` is not yet a failure.

Check the GPU is visible at all:

```bash
docker run --rm --device /dev/dri --entrypoint sycl-ls \
  ghcr.io/ggml-org/llama.cpp:server-intel-b10630
```

A `general protection fault ... in libc.so.6` in `dmesg` followed ~3s later by
`xe ... Engine reset: engine_class=ccs` is the process dying first and the
driver reclaiming afterwards — not a GPU hang. Read the order.

## Upstream reference: jeffgrover/b70-setup

<https://github.com/jeffgrover/b70-setup> is the most thorough public record of
this card under llama.cpp that we know of, kept current by someone running it
daily. **Check it before any image or driver bump** — it is the cheapest way to
find out that a version is broken.

Read it knowing where it differs from us, because most of it does not transfer
directly:

| | b70-setup | here |
|---|---|---|
| llama.cpp | **built from source**, IntelLLVM 2026.1.1 | **prebuilt** `ghcr.io/ggml-org/llama.cpp:server-intel-b<NUM>` |
| build flags | `-DGGML_SYCL_DNN=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_HOST_MEM_FALLBACK=ON`, `GGML_SYCL_DEVICE_ARCH` left unset | whatever upstream ships; not visible from the image |
| Intel stack | oneAPI 2026.1, oneDNN 2026.0, Level-Zero 26.05.37020.3 | Level-Zero 1.15.38308 in the container, `xe` on TrueNAS 26 |
| OS | Ubuntu 26.04, kernel 7.0.0 | TrueNAS 26.0.0-BETA.2, kernel 6.18.23 |
| workload | chat completion — the repo documents **no embedding models** | embeddings, i.e. pure prefill |
| models | 27B–35B chat models, ~24 GB | Qwen3-Embedding-8B Q4_K_M, ~4.7 GB |

**The comparable half is his prompt-processing figures**, since embedding is
prefill and nothing else. Generation rates say nothing about this workload.

His numbers, for reference:

| model | prompt t/s | generation t/s |
|---|---|---|
| Gemma-4 E4B Q4_K_M | 1710 | 76.5 |
| Qwen3.8-27B Q4_K_S | 836.5–864.7 (at 512–2048 tokens) | 25.9 |
| GLM-4.7-Flash Q4_K_M | 496.2 | 20.4 |
| Nemotron-3.5-Lightning Q4_K_S | 342.4 | 56.5 |
| Gemma-4 31B QAT Q4_0 | 283.8 | 10.9 |
| Qwen3.6-35B-A3B UD-Q4_K_S | 220.0 | 55.0 |

He puts the practical ceiling at "approximately 75% of the B70's GDDR6
bandwidth ceiling".

Two of his observations that apply here regardless of the workload difference:

- **"The `xe` driver doesn't always release Level-Zero allocations promptly."**
  Independently visible here — see KNOWN-ISSUES C6.
- **Vulkan measured ~9 t/s** against SYCL's hundreds, because Vulkan does not
  take the XMX path. Do not be tempted by a Vulkan build.

Two of his build flags have no visible equivalent in the prebuilt image and are
worth settling if throughput ever needs to go further: `GGML_SYCL_GRAPH` and
`GGML_SYCL_DNN`. If the published image does not enable them, a source build
becomes a real option rather than a chore — which is why this repo benchmarks
one of his models on our stack, so the comparison is stack-to-stack rather than
model-to-model.

## Open upstream issues affecting this setup

- **#26206** — garbled output on 5000+ token prompts, B70/Xe2. No fix.
  Directly relevant at `-ub 8192`. **Validate before any large ingest.**
- **#25692** — GPU hang with FA + quantized KV under sustained load. Avoided by
  using f16 KV.
- **#27038** — high CPU from the SYCL host-pinned memory path on large
  allocations. Mitigation is `GGML_SYCL_ENABLE_HOST_PINNED_MEM=0`; not applied,
  since the symptom has not appeared.
- **#24946** — continuous batching pins the GPU at gt-c0 on Battlemage,
  preventing idle power savings. Related to fans staying high after a run.
  `--no-cont-batching` would fix it at a throughput cost. **Do not "stop the
  container between ingests"**, which this file used to recommend: a fresh
  container runs ~35% slow for its first ~20,000 tokens while SYCL JITs the
  batch shapes, and the fast state survives at least five minutes of idle
  anyway (KNOWN-ISSUES C7). Paying that on every pause buys nothing.

Not applicable: #25455 (MUL_MAT_ID, MoE only — this model is dense), #27063
(A770/Alchemist), #25203 (Q8_0 reorder — this model is Q4_K_M).

## Correctness check before a large ingest

#26206 has no fix, and bad vectors written to LanceDB will not announce
themselves — they surface much later as poor retrieval.

```bash
./scripts/correctness.sh
```

That is the automated form of what this section used to describe by hand
(embed an 8k chunk at `-ub 8192`, restart at `-ub 2048`, cosine-compare). It
does the same comparison across a 16-chunk pool spanning four content types
and every dispatch boundary, and it also checks that repeated embeddings of
identical input agree. Exit 0 means safe to ingest.

Measured result as of 2026-08-27: **#26206 does not reproduce on this card at
b10630** — cosine 1.000000 on the 7900-token chunks between `-ub 8192` and
`-ub 2048`. That is a measurement, not a guarantee: re-run it on every image
bump, which is the whole point of the script.

## Postmortem: the first deployment

The initial bring-up took most of a night. Almost none of it was llama.cpp's
fault, and none of the fixes that eventually worked came from the theories that
looked most plausible at the time. Recorded here so the same hours are not
spent twice.

### The actual root cause

A restart loop that looked like a GPU crash. It was the healthcheck.

The image ships its own `HEALTHCHECK` with no `start_period`. Cold load takes
~30s, during which `/health` does not answer, so the check failed from its
first probe and the app was torn down mid-initialisation. Each killed instance
left the B70 with work queued, which the driver then reclaimed — producing
kernel messages that looked exactly like a hardware fault.

Fix: `start_period: 180s`. One line.

### What the misleading evidence looked like

**`dmesg` showed `general protection fault` then `Engine reset: engine_class=ccs`.**
Read the order and the timestamps. The fault came first, the engine reset ~3s
later. That is the process dying and the driver cleaning up afterwards. A real
GPU hang (#25692) is the opposite: the GPU wedges and the process survives,
often still answering `/health` with 200 while every request hangs forever.
Ordering distinguishes them; the presence of an engine reset does not.

**The log always stopped at `llama threadpool init`.** This is not where it
crashed and not a hang — it is simply the last line before ~30s of SYCL kernel
JIT and weight upload. A pause there is normal. Do not conclude anything until
either `model loaded` appears or the process exits.

**No SYCL device table in the log.** Absence of a log line only means something
if you have confirmed that build prints it at that verbosity. This one does
not. Reasoning from a missing line that was never expected produced a confident,
wrong diagnosis (that backend init was failing).

### Diagnostic mistakes worth not repeating

**`2>&1 | tail -40` hid everything.** Piping switches stdout to block
buffering; a crash discards the buffer unflushed, so the output is lost
entirely. The pipe intended to capture the error is what destroyed it. Use
`docker run -t` (forces line buffering) and pipe to `cat` or `tee`, never
`tail`, on a process expected to crash.

**The TrueNAS app log pane shows stdout only.** Every real error went to stderr
and was never displayed. Any crash investigation has to be reproduced with a
manual `docker run`, not read out of the app UI.

**Several changes were applied at once.** Speculative environment variables
(`GGML_SYCL_ENABLE_HOST_PINNED_MEM=0`) went in alongside real fixes, so when it
failed there was no way to attribute it. The bisection only worked once changes
were reintroduced one at a time, each validated by manual `docker run` before
touching the deployed YAML.

**Fixes were proposed for problems never observed.** `HOST_PINNED_MEM=0`
targeted #27038, whose symptom (high CPU) had not appeared. It added a variable
to the search space and nothing else. Only mitigate bugs whose symptoms are
present.

### Dead ends, so they are not re-walked

`SYCL_CACHE_PERSISTENT` crashes this image at JIT, ~4s in, reliably. Tested
with the cache directory nested inside the models mount and outside it, owned
by `apps` and by `root`, with a populated cache and an empty one. All six
combinations fail identically. The theories tried and disproved: permissions
(`drwxrwx---` with a root container), overlapping bind mounts, and a poisoned
cache entry left by an earlier crash. The cache directory does get written —
3.1M and a hash-named subdirectory appear — so the runtime reaches it and dies
anyway. Not worth further time: the payoff was ~30s of startup latency.

`GGML_SYCL_FA_ONEDNN=0` was in the original config and is a throughput
regression, not a fix. See Flag notes above.

Bare build-number image tags (`:b10642`) do not exist. See Image selection.

### Shell notes

Under `zsh` on TrueNAS, `$BASE` in a command position is not word-split the way
`bash` does it, so a variable holding a full command fails with
`no such file or directory: docker run --rm ...`. Use `${=BASE}` or write the
command out. Interactive comments are also off by default, so pasted `#` lines
run as commands.

### The rule that would have saved the night

Reproduce with `docker run -t` before changing the deployed YAML, and change
one thing per iteration. Both failures that mattered — the healthcheck grace
period and an unwritable cache path — were environmental, not inference
configuration. When the app restart-loops but a manual run works, the
difference between those two contexts is the bug.

## Project layout

```
config.env               shared settings — change IMAGE here, nothing else.
                         Every value is ${NAME:-default}, so the environment
                         wins: UBATCH=2048 ./scripts/correctness.sh works.
matrix.conf              the configurations scripts/matrix.sh sweeps
embed-xpu-qwen3-8b.yaml  the deployed compose file
KNOWN-ISSUES.md          what is wrong, what was fixed, and the measurement
                         behind each claim

fixtures/chunks.json     GATE pool — 16 chunks straddling every dispatch and
                         batching boundary (8/32, 511/512/513, 1023/1024/1025,
                         2047/2048/2049, 4096, 6144, 7900). Deliberately
                         unrepresentative: it looks for the length where the
                         maths breaks. Commit it.
fixtures/bench.json      BENCH pool — 24 chunks at the production length
                         distribution (mean 899 against a corpus mean of 926).
                         Throughput numbers come from this one. Commit it.

baselines/history.csv    benchmark results, one row per deliberate run
baselines/matrix-*/      one sweep: results.csv, plus every capture and log
baselines/soak-*.{log,csv}  one soak

scripts/
  build_source_pool.py   assemble the raw text fixtures are cut from, by
                         category, naming the exact documents
  make_fixtures.py       cut chunks at exact token lengths (--profile
                         gate|bench)
  capture.py             embed a pool against one endpoint
  compare.py             cross-config cosine, and determinism (pass/fail)
  determinism_profile.py how reproducible each length is over many repeats —
                         the question two repeats cannot answer
  storage_reproducibility.py  does the drift survive L2 + f16, i.e. does it
                         reach the bytes that get hashed
  hash_stability.sh      would another node store the same bytes: hashes the
                         stored vectors across flags, restarts and images
  repro_trial.sh         reproducibility over many orderings, seeds and
                         concurrencies — not one fixed order
  repro_report.py        pools those captures into one verdict
  build_scale_pool.py    one chunk from each of N documents, to the corpus
                         language mix
  scale_trial.py         large-sample reproducibility with a confidence bound
  correctness.sh         the gate: production settings vs a conservative
                         reference, pass/fail
  matrix.sh              sweep many configurations — consistency first,
                         throughput only for the ones that pass
  bench.py               throughput, one row appended per run
  soak.sh                sustained load + kernel log watch
  concurrency.sh         one server config across client concurrencies
  concurrency_grid.sh    server slots x client concurrency, the whole grid
  stack_compare.sh       our stack vs a published one, same model
  ollama_blobs.py        resolve ollama model names to GGUF blobs on disk
  check_upgrades.sh      which newer SYCL images actually exist
```

## The two pools

Consistency and performance want opposite fixtures, so there are two.

The **gate pool** exists to find the length at which the kernels stop agreeing.
Every length in it is a boundary or one side of one, and it spans four content
types — English prose, other European languages, Chinese, and PubMed abstracts
— because tokenization, not just length, changes what the encoder does.

The **bench pool** exists to produce a number that transfers to production. It
follows the measured chunk-length distribution of the corpus (mean 926 tokens,
nothing above ~1000) and the corpus language mix, which is ~40% non-English.
Benchmarking the gate pool would time 7900-token chunks that production never
produces.

Rebuild both after any change to the chunker, and re-commit:

```bash
python3 scripts/build_source_pool.py --out /root/fixture-src
python3 scripts/make_fixtures.py --profile gate  --pool /root/fixture-src
python3 scripts/make_fixtures.py --profile bench --pool /root/fixture-src
```

## First run

Fixtures need the server's tokenizer, so the app must be up. Build the source
pool, then cut both fixture pools from it:

```bash
python3 scripts/build_source_pool.py --out /root/fixture-src
python3 scripts/make_fixtures.py --profile gate  --pool /root/fixture-src
python3 scripts/make_fixtures.py --profile bench --pool /root/fixture-src
git add fixtures/ && git commit -m "fixtures"
```

Commit both. The gate means nothing unless every run embeds identical input.

Then establish the baseline:

```bash
./scripts/correctness.sh                              # must pass first
python3 scripts/bench.py --note "b10630 baseline"
./scripts/soak.sh
```

## Before every ingest

```bash
./scripts/correctness.sh
```

Exit 0 means three things passed:

1. **determinism** — repeated embeddings of identical input agree to within
   float noise;
2. **cross-config** — production settings and a conservative reference produce
   the same vectors (this is the #26206 check);
3. **bit exactness** — the stored (L2 + f16) vectors are byte-identical across
   passes.

The third is a compliance parameter for evaluating a new release, not just a
property worth knowing. Cosine says a build computes the same FUNCTION; bit
exactness says it computes it the same WAY, so a release that quietly changes
kernel selection passes (1) and (2) and fails (3). Non-zero means stop.

## Before every image bump

```bash
./scripts/check_upgrades.sh          # which builds actually have images
# edit IMAGE in config.env, or override it for one run:
IMAGE=ghcr.io/ggml-org/llama.cpp:server-intel-b10644 ./scripts/correctness.sh
IMAGE=ghcr.io/ggml-org/llama.cpp:server-intel-b10644 ./scripts/hash_stability.sh
IMAGE=ghcr.io/ggml-org/llama.cpp:server-intel-b10644 ./scripts/matrix.sh
```

**A release is compliant when it passes both bars: cosine AND bit exactness.**
`correctness.sh` checks them on the candidate alone; `hash_stability.sh` checks
the candidate against the current build, which is the one that says whether the
new release computes the same bytes as the old. b10630 -> b10644 was
byte-identical on all 16 gate chunks, so a bump need not move the arithmetic —
which is exactly why a bump that does move it is worth catching.

`matrix.sh` is the thorough form and the one to prefer for a new image: it
sweeps every configuration in `matrix.conf`, and for each one checks
determinism and agreement with the reference **before** measuring throughput.
A configuration that produces different vectors never gets a performance
number — a fast garbage producer is not a tradeoff, and a number beside it
only invites someone to adopt it.

Then `./scripts/soak.sh` against whatever you deploy.

Compare against the previous rows. A tokens/s regression with no SYCL commits
in the range usually means thermal state or a cold cache, so re-run before
concluding.

## Measuring a single flag change

Every value in `config.env` reads from the environment first, so a one-off
does not need an edit:

```bash
UBATCH=2048 ./scripts/correctness.sh
UBATCH=2048 python3 scripts/bench.py --note "ub 2048"
```

Consistency first, then throughput — in that order, always. `matrix.sh`
enforces it; by hand you have to remember.

