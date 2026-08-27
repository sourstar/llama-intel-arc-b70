# Known issues

Everything here was observed on this machine — TrueNAS 26.0.0-BETA.2, Ryzen 5
7600, one Intel Arc Pro B70 (Battlemage G21, `xe` driver, 32,656 MiB), running
`ghcr.io/ggml-org/llama.cpp:server-intel-b10630` — unless it says otherwise.
Claims carry the measurement that produced them, so a later run can contradict
them properly rather than by assertion.

First systematic pass: 2026-08-27. Before that date the harness in this repo
had never been executed.

---

## A. Harness defects (found 2026-08-27, all fixed)

The scripts were written from the deployment postmortem and committed without
being run. The items below are what running them found.

**A1. `PROD_URL` pointed at `localhost`.** TrueNAS 26 has no IPv4 entry for
`localhost` — `getent hosts localhost` returns `::1` only — and the IPv6
docker-proxy resets the connection rather than refusing it, so the failure
reads as a server problem. Nothing could reach the endpoint. Fixed: the
literal `127.0.0.1` in `config.env` and in both `capture.py` call sites.

**A2. The determinism check measured the prompt cache, not determinism.**
`capture.py` embedded each chunk `--repeat 2` back to back. llama.cpp keeps
each slot's prompt cached, so the second request returned **in 0.02s against
4.43s for the first** on a 7900-token chunk — no prefill ran at all. What the
check then reported as non-determinism was the cached-pooling path diverging
from the full-prefill path (~5e-5), which is not what it set out to measure.
Fixed: repeats are the outer loop, so the whole fixture set cycles through the
slots in between and the repeat is genuinely recomputed. Verified — rep1
timings now match rep0.

**A3. The determinism check demanded bit-exact equality.** `==` on the vector.
On this stack that fails on most runs (see C3), so the gate could never pass
reliably and a real failure would have been indistinguishable from the standing
one. Fixed: a tolerance, set from measurement rather than taste
(`DET_MIN_COSINE`, derivation in C3).

**A4. `soak.sh` polluted the baseline history.** It called `bench.py` once per
round with no way to redirect the output, so a 1800s soak appended **73 rows**
of `soak round N` to `baselines/history.csv` — the file whose entire purpose is
one deliberate row per run, compared against the previous one. Fixed:
`bench.py --history PATH`, and soak rounds go to `baselines/soak-<stamp>.csv`.

**A5. Documented environment overrides did not work.** The `soak.sh` header
documents `SOAK_SECONDS=300 ./scripts/soak.sh`. It ran for 1800s, because
`source config.env` executes *after* the shell applies the prefix and every
assignment in the file was unconditional. Fixed: every value in `config.env` is
`${NAME:-default}`, which also makes `scripts/matrix.sh` and the concurrency
grid possible without rewriting the file between runs.

**A6. The soak driver-event check could not detect driver events.** It counted
`dmesg` lines matching "Engine reset" before and after and compared the totals.
dmesg is a ring buffer: on the first full run the count went **84 → 80 while
nothing happened**, because old lines rolled off. Four resets during a run plus
four lines rolling off would have reported a delta of zero and passed. Fixed:
snapshot dmesg first, then count only lines that are not in the snapshot.

**A7. An interrupted gate poisoned the next one.** `correctness.sh` starts
containers by fixed name; if a previous run died before its cleanup trap, the
next `docker run` failed on the name collision rather than on anything real.
Fixed: `docker rm -f` before each start.

**A8. Benchmark rows did not record the image they measured.** `bench.py` took
`--image` from `$IMAGE`, which nothing exports — so the command the README
gives, `python3 scripts/bench.py --note "..."`, wrote an empty image column.
That column is the only thing making rows comparable across an image bump,
which is what the file exists for. Fixed: `--image`, then `$IMAGE`, then
config.env — resolved by *sourcing* config.env, since after A5 its values are
shell expressions and a text parse recorded the literal `${IMAGE:-...}`.

**A9. Fixture lengths were approximate, and the pool was English-only.** The
builder binary-searched on character count (~20 tokenizer requests per chunk)
and returned its best effort when it did not converge; the source was four
English novels. Rewritten as a tokenize → slice → detokenize → verify
round-trip, which is exact and takes 3–5 requests, over a categorised source
pool (see D3 and `scripts/build_source_pool.py`).

**A10. The README named an app that does not exist.** `midclt call -j app.stop
llama-embed`. The TrueNAS app is `embed-gpu-qwen3-8b`; the container is
`ix-embed-gpu-qwen3-8b-llama-embed-1`. Fixed.

---

## B. Environment

**B1. `localhost` is IPv6-only on this host.** See A1. Any script, healthcheck
or client written against `localhost` on TrueNAS 26 will fail in a way that
looks like the server. Use `127.0.0.1`.

**B2. A fresh clone on Windows would have shipped CRLF scripts.** The client
has `core.autocrlf=true` and `.gitattributes` said only `* text=auto`, so a
clone delivers CRLF working-tree files. On the NAS that breaks two ways: the
shebang becomes `/usr/bin/env bash\r` ("bad interpreter"), and `source
config.env` puts a trailing `\r` inside every variable, so the image tag and
the URL are silently wrong. The checkout in hand was already LF — this is
prevention, not a repair. `.gitattributes` now pins `*.sh`, `*.py`, `*.yaml`,
`*.json` and `config.env` to `eol=lf`.

**B3. There is no Python on the Windows client.** Every script here runs on the
NAS. The repo is the source of truth; the NAS holds a working copy under
`/root/llama-intel-arc-b70`.

**B4. The TrueNAS root shell is zsh, which does not word-split.** Driving these
scripts over `ssh root@host '...'` runs the command under zsh, where an
unquoted `$var` expands to ONE word. A loop written as `set -- $spec` launched
a server with `-np "1 8192"` and a default context, and nothing complained —
the container started, answered health checks, and would have produced a
plausible-looking number for a configuration that was never requested. Anything
that has to word-split belongs in a script with a `#!/usr/bin/env bash`
shebang, not in an ssh one-liner. `${spec%%:*}` works in either shell.

---

## C. Hardware and driver behaviour (measured)

**C1. VRAM.** The card reports 32,656 MiB. The deployed server (`-c 32768
-np 4 -ub 8192`) is resident at **13.8 GiB vram0 + 416 MiB gtt**, leaving
~18.4 GiB free — enough to run a second full-size instance alongside it, which
is what makes the correctness gate and the sweeps possible without stopping
production.

**C2. Cold load is ~26s** to `model loaded`, effectively all of it after
`llama threadpool init`. This is why `start_period: 180s` is load-bearing.

**C3. Short prefills are not bit-reproducible. The cause is oneDNN's GEMM, and
there is a one-variable fix that costs 1.7%.**

Profiled with 10 repeats per chunk (`scripts/determinism_profile.py`), cache
evicted between repeats, requests serial:

| tokens | 9 | 33 | 65 | 257 | 512 | 513 | 514 | 1024 and above |
|---|---|---|---|---|---|---|---|---|
| distinct vectors in 10 runs | 1 | 2 | **10** | **10** | 1 | **10** | **10** | 1, at all nine lengths tested |

Four chunks returned a **different vector on every single run**. The magnitude
is tiny — worst pairwise cosine 0.9999974, worst single component 1.9e-4 — but
"tiny" is not "absent", and two repeats (what the gate uses) is far too few to
see it: a chunk that drifts every run still looked "identical" in early gate
runs whenever both samples happened to land the same way.

Two repeats answers pass/fail. Ten answers *what is going on*. Use
`determinism_profile.py` for the second question.

Note the 33-token row above, which shows 2 rather than 10. That one is a
different effect entirely and was mistaken for a mild case of this one for most
of a day: a chunk served from the slot prompt cache returns a different vector
than a full prefill, by a much larger margin, and there are exactly two stable
answers rather than a drift. See F.

### Isolation

Three runs of the *unmodified* baseline first, because the drifting set is not
itself stable and an intervention has to beat that noise to mean anything:

| unmodified baseline, `-np 1` | chunks that drifted |
|---|---|
| run 1 | 257, 513, 514 |
| run 2 | 33, 65, 257, 513, 514 |
| run 3 | 65, 257, 513, 514 |

**257, 513 and 514 drift in every run. 33 and 65 are marginal and flip.** Any
claim about a fix has to move the first group.

| change | chunks that drifted |
|---|---|
| `-np 1` | 5 — slot count is not the cause |
| `--no-cont-batching` | 3 — **within baseline noise, no real effect** |
| `ONEDNN_DETERMINISTIC=1` | 3 — likewise, and see below: it is not an env var |
| `-fa off` | 5 — **attention is not the cause** |
| `GGML_SYCL_ENABLE_OPT=0` | 4 — no effect |
| `GGML_SYCL_ENABLE_FUSION=0` | 5, including a 1024-token chunk — no effect |
| `GGML_SYCL_ENABLE_ESIMD=0` | 5 — no effect |
| **`GGML_SYCL_ENABLE_DNN=0`** | **0 — every chunk bit-identical, 10/10** |

An earlier version of this entry blamed the oneMKL FA floor
(`K->ne[1] >= 1024`), on the reasoning that shorter prefills fall through to
oneDNN SDPA. That explanation cannot be right, and the evidence against it was
already in this file: `-fa off` shows the identical pattern, and with flash
attention disabled neither FA path runs at all.

### What oneDNN is doing here, and why the length matters

`ONEDNN_VERBOSE=1` on this image says exactly what is dispatched. The container
carries **oneDNN v3.9.1**, GPU engine Level Zero, driver 1.15.38308, and the
only primitive llama.cpp sends it is `matmul`:

```
onednn_verbose,v1,primitive,exec,gpu,matmul,jit:gemm:any,undef,
  src:f16::blocked:abc::f0 wei:f16::blocked:cab::f0 dst:f32::blocked:cab::f0,
  attr-scratchpad:user attr-fpmath:f16,,1x4096x4096:1x4096x513,...
```

The trailing `513` in the problem descriptor is the token count — M. So
`GGML_SYCL_ENABLE_DNN` decides whether the model's matmuls go through oneDNN's
JIT GEMM or through ggml's own SYCL kernels. It is a different switch from
`GGML_SYCL_FA_ONEDNN`, which only governs the fused SDPA.

The implementation string is `jit:gemm:any` for M=513 and M=2049 alike, so
oneDNN's own dispatch is not choosing a different primitive — whatever changes
is inside the JIT'd kernel, which the verbose log does not expose.

The likely mechanism, stated as inference rather than observation: a GEMM with
small M has little parallelism along M, so to fill the GPU the kernel splits
the K dimension across work-groups and combines partial sums. Floating-point
addition is not associative, so if that combination uses atomics or any order
that is not fixed, the result varies between launches. With large M there is
enough parallelism without splitting K and the accumulation order is fixed.
That matches oneDNN's own documentation, which names "atomic operations" as
the non-deterministic construct, and it explains why the boundary sits around
1024 tokens and is ragged rather than sharp — it is a library heuristic over
problem shape, not a threshold anyone declared.

### Bug or feature? Feature, with a gap in llama.cpp

oneDNN documents this as intended behaviour. Its deterministic mode is off by
default, and off means the library "permits ... non-deterministic constructs
resulting in non-identical run-to-run outputs". It can be turned on — but
**only through the API**, `dnnl::primitive_attr::set_deterministic` (C++) or
`dnnl_primitive_attr_set_deterministic` (C), and the docs warn it "might impact
the performance of Convolution, Matmul, and normalization primitives,
especially on some GPU devices".

There is no environment variable for it. `ONEDNN_DETERMINISTIC=1` was tested
here and did nothing beyond baseline noise, exactly as the documentation
implies.

So: not a bug in oneDNN, not a bug in the driver, and not specific to this
card. The gap is in **llama.cpp**, which routes matmul through oneDNN without
ever setting that attribute and documents `GGML_SYCL_ENABLE_DNN` without
mentioning what enabling it costs in reproducibility. The ideal fix upstream is
a flag that sets `set_deterministic(true)` — keeping oneDNN's kernels and
paying only the deterministic-mode cost. What we have instead is the
all-or-nothing `GGML_SYCL_ENABLE_DNN=0`, which happens to be cheap here.

### The fix

Set `GGML_SYCL_ENABLE_DNN=0` in the container environment.

| | oneDNN on | oneDNN off |
|---|---|---|
| chunks reproducible over 10 runs | 11–13 of 16, varying | **16/16** |
| throughput, `-np 1` at concurrency 4, n=3 | 3240.9 ± 1.2 | 3185.9 ± 1.7 |
| | | **1.7% slower** |

The vectors do not change: worst cosine between the two modes across all 16
gate chunks is **0.999999029**. Same embedding space, fixed accumulation order.
It holds at `-np 4` as well, so it does not depend on the `-np 1` recommendation
in D4.

### What is written down, upstream

oneDNN's side is documented (see above). llama.cpp's side is not:

- llama.cpp documents non-determinism only generally: `--parallel > 1` gives
  higher throughput but "the results will not be deterministic due to
  differences in rounding error".
- [#7052](https://github.com/ggml-org/llama.cpp/issues/7052) reports
  multi-slot non-determinism but is **closed as `bug-unconfirmed` / `stale`**
  with no root-cause analysis, and concerns sampling rather than this.
- `docs/backend/SYCL.md` lists `GGML_SYCL_ENABLE_DNN` ("Runs computations
  through oneDNN; default 1") and says nothing about reproducibility.

So the knob is documented and the reason to reach for it is not. Worth
reporting upstream: a single-variable isolation, a cheap workaround, and a
better fix available through oneDNN's own deterministic attribute.

### Consequence for the RAG pipeline

With oneDNN on, **re-embedding a chunk does not reproduce the same vector
byte-for-byte**, and since production chunks average 926 tokens (D1)
essentially all of them are in the affected range. Verification of re-embedded
vectors then has to compare by cosine with a tolerance — never by hash — with
~1e-5 as the floor to allow for. `DET_MIN_COSINE` is set to 0.9999: two orders
of magnitude above that noise, two below corruption of the #26206 kind, which
lands far under 0.99.

With `GGML_SYCL_ENABLE_DNN=0` that constraint goes away and re-embedding can be
verified by hash, which is both cheaper and stronger. 1.7% of throughput is a
reasonable price for it.


**C4. #26206 does not reproduce on this card at b10630.** The bug is garbled
output on 5000+ token prompts on B70/Xe2, and it is the reason this repo
exists. Embedding the same 7900-token chunks at `-ub 8192` and at `-ub 2048`
gives **cosine 1.000000** — bit-identical, not merely close. Worst cosine
across all 16 gate chunks: 0.999846, on the very short ones. This is not a fix
upstream and it is not a guarantee; it is a measurement to be repeated on every
image bump, which is what `scripts/correctness.sh` is for.

**C5. Sustained load is stable.** 1800s, 73 rounds, continuous 4-way concurrent
embedding: **0 engine resets, 0 GP faults, 0 failed requests.** Throughput held
~2225 tok/s for three quarters of the run and eased ~3.5% in the last quarter
(2148 tok/s), with p95 rising 11.6s → 12.1s. That is thermal drift, not the
#25692 hang, which presents as `/health` still answering 200 while every
request blocks forever.

**C6. Starting a test container evicts the production model from VRAM.**
During the config sweep the deployed server's `drm-total-vram0` fell from
**13.8 GiB to 1.27 GiB** while it stayed up and healthy. The `xe` driver
migrates the idle allocations out rather than failing the new allocation.

It recovers by itself and the cost is small: the first request after eviction
took **0.49s**, and a full benchmark immediately afterwards gave **2384.0
tok/s** against **2377.5** for a freshly started container — statistically
identical. So this is a thing to know, not a thing to fear. It does mean the
first production request after running the harness pays a re-upload, and it
matches jeffgrover/b70-setup's note that "the `xe` driver doesn't always
release Level-Zero allocations promptly".

**C7. The first ~20,000 tokens after a container start run ~35% slow.** Not a
clock ramp and not thermal — a one-time cost, almost certainly SYCL kernel JIT
for the batch shapes, cached in-process for the life of the container.
Measured on a fresh container, 24-chunk pool at concurrency 2:

| | tok/s |
|---|---|
| first benchmark, cold | 2045–2315 |
| after one warm pass (24 requests) | 3115 |
| after three warm passes (72 requests) | 3127 |
| warm, then idle 30s | 3156 |
| warm, then idle 120s | 3155 |
| warm, then idle **300s** | **3150** |

Idling does not lose it. One full pass over the pool reaches steady state and
three passes add nothing.

Two consequences, and the second contradicts advice this README used to give:

- **Any benchmark must warm up on a full pass first.** This was found the hard
  way: `bench.py` defaulted to a 2-request warmup, every configuration in a
  sweep got a fresh container, and so every row of the first two sweeps was a
  cold number reading ~35% low — far larger than any flag difference the sweep
  was trying to resolve. `--warmup` now defaults to one full pass.
- **Do not stop the container between ingests.** The suggestion under #24946
  was to stop it to let the card drop to an idle power state. That costs this
  penalty on every restart, and the measurement above shows the fast state
  survives at least five minutes of idle anyway, so a paused pipeline keeps it
  for free.

---

**C8. The prebuilt image is as fast as a tuned source build.** This was the one
open question with a real cost attached — if the GHCR container were leaving
throughput behind, the answer would have been to build llama.cpp ourselves.
It is not.

Running Qwen3.8-27B (already on this node in the ollama app's blob store, so no
download) through `llama bench` on our stack, against
jeffgrover/b70-setup's published figures for the same model on a source build
with `GGML_SYCL_GRAPH=ON` and `GGML_SYCL_DNN=ON`:

| | pp512 t/s | pp2048 t/s | quantization |
|---|---|---|---|
| **ours** — GHCR `server-intel-b10630` | **841.4 ± 7.9** | **873.2 ± 0.9** | Q4_K_**M** |
| his — source build, oneAPI 2026.1 | 836.5 | 864.7 | Q4_K_**S** |

We match him, marginally ahead, on the *heavier* quantization. **Do not build
from source for performance.** Reproduce with `scripts/stack_compare.sh`.

(`qwen3.6:35b-a3b` failed to load in the same run — 23.9 GB against a resident
production model. Not pursued; the 27B result is decisive on its own.)

---

## D. The deployed configuration

**D1. It is sized for chunks that do not exist.** From the gutenberg manifest,
2026-08-27. `tokens` and `chunks` are recorded at scan time, not only after
embedding, so this covers **77,291 of the 79,282 documents in the pool** — the
remaining 1,991 have no chunk data, 1,497 of them carrying errors:

| statistic | tokens |
|---|---|
| corpus-wide mean chunk | **926** |
| documents whose mean chunk exceeds 1000 | **0** |
| largest exact chunk (from the 327 single-chunk documents) | **997** |
| highest per-document mean anywhere | **1000.0** (`sv/pg14050`, 30 chunks / 30,000 tokens) |

The cap holds in every language — zh 989.8 mean, ja 984.3, el 989.9, te 994.1,
fi 958.4 — so it is not an English-corpus artefact.

**But `-c` is not sized against what the chunker emits; it is sized against
what the policy allows.** `gutenberg.toml` declares `ctx = 8192`, so a chunk of
up to 8192 tokens is legal and the server must be able to hold one. Per-slot
context (`-c ÷ -np`) is a capacity limit and belongs at 8192. That headroom is
policy, not waste.

**D2. `-ub` is not a capacity limit, and only matters if it is below the chunk
size.** The README asserted `-ub` must be >= the longest chunk, from the real
clamp of `n_batch` to `n_ubatch` for pooled embeddings (`server.cpp:146`). That
constraint belongs to BERT-style bidirectional encoders. Qwen3-Embedding is a
causal decoder with last-token pooling, so a prompt split across ubatches still
sees everything before it through the KV cache — C4 measures the result as
bit-identical at 7901 tokens.

An earlier draft of this file claimed `-ub 2048` was **+32% throughput**. That
was measured on a fixture pool containing 7900-token chunks, which production
never produces, and it does **not** transfer. Measured on the real distribution
at `-np 1`, client concurrency 2, warm:

| `-ub` | tok/s |
|---|---|
| 8192 (deployed) | 3242 |
| 2048 | 3238 |
| 1024 | 3236 |
| **512** | **2556** |

Everything at or above 1024 is identical, because real chunks are ~900 tokens
and never split. Only 512 hurts, and it hurts by 21%. **`-ub 8192` is correct
and should stay** — it costs nothing and covers the 8192-token policy ceiling
without splitting.

The lesson is about the harness as much as the flag: **a benchmark pool that
does not match production will confidently recommend the wrong thing.**

**D3. The corpus is ~40% non-English.** Document language counts: en 3853,
fr 2559, fi 1462, de 1224, nl 716, it 653, pt 539, es 476, zh 366, el 215.
Chinese tokenizes at roughly one token per character against ~5 characters per
token for English prose, so a fixture pool of English novels does not represent
the work. The pools in `fixtures/` follow this mix.

**D4. `-np 4` is well past saturation, and the cost is large.** This is the
most consequential measurement here. Per-slot context held at 8192, `-ub` at
2048, warm, zero errors anywhere in the grid:

**tokens/sec — client concurrency across, server slots down**

| | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| **`-np 1`** | 3133 | **3241** | **3239** | **3236** | **3234** | **3232** |
| `-np 2` | 3154 | 3238 | 2397 | 2419 | 2421 | 2407 |
| `-np 4` (deployed) | 3155 | 3237 | 2396 | 1953 | 1934 | 1932 |
| `-np 8` | 3152 | 3235 | 2483 | 2224 | 2015 | 2011 |

**worst-case latency, ms**

| | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| **`-np 1`** | 307 | 585 | 1168 | 2327 | 4604 | **9009** |
| `-np 2` | 299 | 586 | 1737 | 3228 | 6299 | 12198 |
| `-np 4` | 299 | 585 | 1782 | **33161** | 23557 | **33517** |
| `-np 8` | 300 | 587 | 1558 | 29126 | 32147 | 32215 |

Read it as three facts:

1. **Peak is ~3240 tok/s at client concurrency 2, for every server config.**
   Two requests in flight is all this card wants for ~900-token prefills.
2. **Running prefills genuinely concurrently costs throughput.** The moment the
   server has enough slots to execute more than two at once, aggregate
   throughput falls — to ~2400 at four, and to ~1930 at eight with `-np 4`.
   That is a 40% loss against the peak, and it is a loss, not a trade: latency
   gets worse at the same time.
3. **`-np 1` cannot be pushed into that regime.** It serialises, so it holds
   ~3232 tok/s from concurrency 2 to 32 and its worst case stays proportional
   to queue depth — 9.0s at 32 deep, against 33.5s for `-np 4` at 8 deep.

**There were zero errors in the entire grid.** The failure mode is silent
degradation, which is why it survived unnoticed: nothing logs, nothing fails,
the throughput is just a third lower than it should be.

The practical consequence is that **the deployed `-np 4 -c 32768` is the worst
of the configurations tested** at any concurrency above two, and `-np 1
-c 8192` is both the fastest and the only one that cannot be degraded by client
behaviour. The caveat is that `-np 1` serialises query embedding behind ingest
work — at concurrency 2 a query waits ~600ms, which is fine, but it is a real
consideration if this endpoint ever serves interactive search under load.

**D5. `-fa off` is catastrophic, not merely slower.** 286.7 tok/s against a
warm ~3240, and **44 of 72 requests failed**. It passed the consistency check first,
which is the only reason we know its vectors are fine — the harness measured it
and refused to record a throughput row. Keep `-fa on`.

---

## E. Upstream issues — status here

| issue | status on this card, b10630 |
|---|---|
| **#26206** garbled output >5000 tokens | **Not reproduced.** cosine 1.000000 at 7901 tokens, `-ub 8192` vs 2048. Re-check every bump. |
| **#25692** GPU hang, FA + quantized KV | **Avoided, not tested.** KV is f16; nothing here quantizes it. 1800s soak clean. |
| **#27038** high CPU, SYCL host-pinned memory | Symptom not observed. `GGML_SYCL_ENABLE_HOST_PINNED_MEM=0` deliberately **not** applied — mitigating an absent symptom only widens the search space when something else breaks. |
| **#24946** cont-batching pins gt-c0 | Power state not measured. But do NOT "stop the container between ingests" — a restart costs ~35% for its first ~20,000 tokens (C7). |
| #25455, #27063, #25203 | Not applicable — MoE-only, Alchemist-only, Q8_0-only respectively. |

`GGML_SYCL_FA_ONEDNN_MAX_KV=8192`, the upstream escape hatch for engine resets
under sustained load, was measured at **2417 ± 29 tok/s** against the deployed
2433 ± 27 — indistinguishable, though both were cold numbers (C7). It is
free to enable if resets ever appear.

**One newer image exists: `server-intel-b10644`.** A GHCR scan of b10631–b10750
found nothing else published. Untested.

See also the README's upstream-reference section on
[jeffgrover/b70-setup](https://github.com/jeffgrover/b70-setup), and C8 —
his source build with `GGML_SYCL_GRAPH` and `GGML_SYCL_DNN` enabled is not
faster than our prebuilt image.

---

## F. Hash-verifiable embeddings

The Louise project treats hash-verifiable embeddings as a requirement, so this
section is about that specifically: not "does the server return the same vector
twice" but "would the same bytes be written down again". Reproduce any of it
with `scripts/hash_stability.sh` and `scripts/storage_reproducibility.py`.

### The storage pipeline does not absorb the drift

The obvious hope is that it does not matter — the collection's space record says
`"normalization": "l2-client", "dtype": "f16"`, and f16 keeps about 11 bits of
mantissa against a drift of ~1e-6 relative. Measured, it does not help. Taking
the ten repeats of each gate chunk, L2-normalising and downcasting exactly as
the pipeline does:

| | chunks reproducible |
|---|---|
| raw f32, as returned | 11/16 |
| **after L2 + f16, as stored** | **11/16 — identical outcome** |

Every chunk that drifted in f32 also hashes to more than one value after
downcast. The rounding does not land on the same side often enough to matter.
So **`GGML_SYCL_ENABLE_DNN=0` is load-bearing for this project, not an
optimisation** — without it, five of sixteen fixture chunks produce a different
stored vector on repeat, and re-embedding cannot be verified by hash at all.

### What byte-identity survives, with `GGML_SYCL_ENABLE_DNN=0`

Same node, same model file, hashes of the stored (L2 + f16) vectors:

| varied | result |
|---|---|
| container restart | **identical, all 16 chunks** |
| `-c 8192` → `-c 32768` | **identical, all 16 chunks** |
| image `b10630` → `b10644` | **identical, all 16 chunks** |
| `-np 1` → `-np 4` | **1 chunk differs** |
| `-ub 8192` → `-ub 2048` | **4 chunks differ** — all of them longer than the ubatch |

Two clean readings:

- **`-c` is free and the image bump was free.** Context size never changes the
  arithmetic, and fourteen build numbers of llama.cpp did not either. That last
  one is one data point, not a guarantee — but it means an upgrade is testable
  rather than a leap, which is the point of having the script.
- **`-np` and `-ub` are part of the identity.** `-ub` changes the bytes exactly
  where it changes the work: a chunk longer than the ubatch gets split, and the
  accumulation order changes with it. `-np` shifts how requests are assembled
  into batches. Neither is currently named anywhere in the space.

### `-np` and `-ub` change the bytes for different reasons, and only one is pinnable

Both appear in the table above as "changes the hash", but they are not the same
kind of problem, and the difference decides whether hash verification is
possible at all.

**`-ub` slices the prefill.** A prompt is processed in slices of at most `-ub`
tokens; each slice is its own set of GEMMs with M = the slice length, and
attention over earlier tokens reads them from the KV cache. For a causal model
with last-token pooling the split is mathematically equivalent — C4 measures
cosine 1.000000 at 7901 tokens between `-ub 8192` and `-ub 2048` — but it is
numerically different, because partitioning the work changes which partial sums
exist and in what order they are added.

The important property is that this is **a pure function of (prompt length,
`-ub`)**. Nothing about it varies at runtime. Two nodes with the same `-ub`
produce the same bytes; a chunk shorter than `-ub` is never sliced at all, and
production chunks (~900 tokens) are never sliced at any `-ub` >= 1024. It is
pinnable, and cheap to pin.

**`-np` mixes requests into shared batches.** With more than one slot and
continuous batching, llama.cpp packs tokens from *different concurrent
requests* into a single ubatch. The M a given request is computed under is then
the sum of whatever else happened to be in flight — which depends on arrival
timing, not on configuration.

Measured with `GGML_SYCL_ENABLE_DNN=0`, gate pool, stored (L2 + f16) hashes:

| server | serial | 8 requests in flight |
|---|---|---|
| `-np 1` | **16/16 stable** | 16/16, 16/16, 14/16 over three runs |
| `-np 4` | **16/16 stable** | **0/16 stable — every chunk** |

At `-np 4` under load, *every single chunk* produced more than one stored
vector. Serially the same server is perfectly reproducible. So this is not the
kernel-selection problem of C3 — oneDNN is already disabled in both columns.
It is batch composition, one level up.

**That asymmetry is the whole point.** `-ub` changes the bytes as a function of
configuration, so it can be recorded and matched. `-np` changes them as a
function of runtime traffic, so it cannot be pinned by any amount of metadata:
the same node, same flags, same chunk, embedded twice, gives different bytes
depending on how busy it was. **`-np 1` is therefore a requirement for
hash-verifiable embeddings, not a throughput preference** — and it happens to
be the fastest setting anyway (D4).

The residual at `-np 1`: two of three concurrent runs were perfectly stable and
one flipped the 33- and 65-token chunks. That is not a property of those
lengths — it is the prompt cache, see below. Under concurrency a short chunk
can be re-requested while its own prompt is still the one held in the slot, and
the cache path returns a different vector than a full prefill.

### The prompt cache is a second, larger source of divergence — and it is the one that will bite a verifier

A request can be served two ways: a full prefill, or a hit against the prompt
still cached in the slot from a previous request. **The two paths return
different vectors**, and each path is perfectly stable in itself:

| | value | responses | timing |
|---|---|---|---|
| full prefill | `e5abb1f148316877` | 1 | 0.193s |
| prompt-cache hit | `8e93ce0ea5f045d5` | 19 | 0.014–0.016s |

Same chunk, same warmed server, `GGML_SYCL_ENABLE_DNN=0`, 20 serial requests.
Every cache hit is byte-identical to every other cache hit; the single full
prefill is byte-identical to full prefills elsewhere. There is no randomness
here at all — there are two answers, and which one you get depends on which
path served the request.

The gap is **cosine 0.999965159, max component difference 1.26e-3** — roughly
forty times larger than the oneDNN GEMM drift of C3, and far above anything f16
rounding could absorb.

**`--no-cache-prompt` does not prevent it.** The flag exists
(`--cache-prompt, --no-cache-prompt`, default enabled) and was tested: cache
hits still came back in 0.014s with the cache-path vector. Either it does not
cover the OpenAI-compatible embeddings handler or it does not do what its name
says. Do not rely on it.

**This is what the "marginal 33- and 65-token chunks" actually were.** They are
not marginal and their length is not the point — earlier notes in this file
described them that way, and that was wrong. Any chunk served from cache
returns the cache-path vector. Short chunks show it more often under
concurrency for a mundane reason: they complete quickly and can be re-requested
while their own prompt is still the one held in the slot, where a long chunk
would have been displaced by other work in the meantime. A first guess that
these two chunks shared a leading token was also tested and was wrong — the
sharing is real but has nothing to do with it.

**What it means for ingest and for verification, which differ:**

- **Ingest is safe.** Every chunk in a corpus is a distinct prompt, so
  consecutive requests never hit the cache, and every stored vector comes from
  a full prefill. This is why the 16-chunk pool captures in this file are
  clean: `capture.py` cycles the whole pool between repeats specifically so
  each repeat is a real recomputation, and that design turns out to model
  production correctly.
- **A naive verification pass is not safe.** Re-embedding a chunk immediately
  after embedding it — the obvious way to write a verifier — takes the cache
  path and returns a vector that will not match the stored one, by a margin
  large enough to fail any sensible hash or tight-cosine check. A verifier must
  ensure the chunk is not the slot's current prompt: interleave other work, or
  embed in batches large enough to displace it.

### How much confidence the measurements actually support

Every reproducibility claim in this file up to 2026-08-27 evening was measured
by repeating a 16- or 24-chunk pool in **fixture order, the same order every
run**. Two things were wrong with that, and both were found by changing it:

- **Ordering is not an implementation detail.** It decides whether a request is
  served from the slot prompt cache, and the cache path returns a different
  vector. A fixed order samples one arbitrary path through that and can report
  a configuration as perfectly reproducible when it is not.
- **Repeating N times bounds the failure rate of THOSE chunks**, not of the
  corpus. Every failure mode found here has been input-dependent, so the sample
  that matters is distinct inputs, not repeats of the same input.

`capture.py` now takes `--pattern {cycle,shuffle,adjacent,reverse}` and
`--seed`, records both, and flushes the slot cache between concurrent waves.
`repro_trial.sh` sweeps patterns, seeds and concurrencies; `scale_trial.py`
does the large-sample version and reports a confidence bound.

**Trial 1 — ordering and concurrency.** 40 captures, 3,600 requests over the
24-chunk bench pool:

| pattern | conc 1 | conc 4 |
|---|---|---|
| cycle | 24/24 | 24/24 |
| reverse | 24/24 | 24/24 |
| shuffle | 24/24 | 22/24 |
| adjacent | 0/24 | 0/24 |

`adjacent` requests each chunk twice in a row and is the cache probe — 0/24 is
the expected result and the clearest demonstration of the two code paths. The
`shuffle` 22/24 was a defect in the harness, not the encoder: in both affected
captures the unstable chunk sat within four positions of a wave boundary, where
completion order diverges from submission order and a chunk can follow itself
into the cache. Three of five seeds were clean. The flush between waves removes
it by construction.

**Trial 2 — content diversity.** 4,976 chunks, each from a different Gutenberg
document, sampled to the corpus language mix (en 3859, fr 263, fi 231, de 151,
und 99, nl 70, it 70, es 56, … zh 28). Two full passes, no chunk ever following
itself, so every response is a full prefill. Token lengths min 425, p50 897,
p95 1435, max 3601. **0 mismatches.**

**Trial 3 — shape coverage.** Every token length from 2 to 1,101 — 1,100
distinct values, one chunk each — six full passes. **0 mismatches over 6,600
embeddings.**

Trial 3 is the strongest of the three and it is worth being explicit about why.
Every mechanism found on this card dispatches on **shape**: the GEMM's M is the
token count, and which kernel and reduction order that selects is a function of
M. A random sample says "we looked at 5,000 inputs and saw nothing". A complete
sweep of the length range says "there is no length at which this happens",
which is a different and much better claim. Production chunks run 425–1000
tokens; the sweep covers 2–1101.

**The pooled bound.** The unit of evidence is a comparison, not a chunk: k
passes over a chunk give k-1 independent sequential comparisons.

| trial | comparisons | 95% bound | projected onto 1,373,515 chunks |
|---|---|---|---|
| content (4,976 x 2) | 4,976 | 6.0e-4 | 827 |
| shape (1,100 x 6) | 5,500 | 5.4e-4 | 748 |
| **pooled** | **10,476** | **2.9e-4** | **393** |

Zero mismatches throughout. The bound is `1 - 0.05**(1/n)`, the exact
one-sided Clopper-Pearson limit.

**393 is the limit of what sampling can prove, not a defect count.** Pushing it
to a level that would make "hash-verifiable" unconditionally safe by random
sampling needs a sample the size of the corpus — at which point it is not a
sample, it is the verification pass. What repetition can do is rule out a
systematic problem, and it did that twice: the oneDNN GEMM issue showed up at
~30% of short chunks, and the cache path appears in every adjacent-pattern run.
Neither hid.

Throughput note, since it came free: 3,145 tok/s held across 9,952 consecutive
real embeddings in trial 2, and again across 6,600 in trial 3. The 3,186 figure
from the short benchmark was not a short-run artefact.

### Does the new configuration invalidate the embeddings computed so far?

Two separate questions, with opposite answers.

**For retrieval: no.** Deployed-config vectors against new-config vectors, over
24 production-length chunks: worst cosine **0.999987**, median 0.9999997, best
1.000000 — every one above 0.9999. Nothing about ranking or scoring changes. A
collection holding both is not degraded, and there is no reason to re-embed for
quality.

**For hash verification: there is nothing to invalidate, because the existing
corpus was never hash-verifiable.** This is the part that matters, and it is
worse than the question assumes.

Re-embedding the same chunk on the **deployed** configuration — `-np 4
-c 32768 -ub 8192`, oneDNN enabled, the configuration that produced the corpus
— does not reproduce the same bytes:

| pool | comparisons | mismatches | rate |
|---|---|---|---|
| bench pool, production lengths, 4 passes | 72 | 5 | **6.9%** |
| length sweep 2–1101, 2 passes | 1,100 | **680** | **62%** |

The instability is banded rather than uniform, which is oneDNN selecting
different GEMM kernels by shape:

| tokens | unstable / tested | | tokens | unstable / tested |
|---|---|---|---|---|
| 0–99 | 39 / 99 | | **400–499** | **0 / 100** |
| 100–399 | 284 / 300 | | 500–899 | 357 / 400 |
| | | | **900–1099** | **0 / 200** |

Two bands are perfectly reproducible — 400–499 and 900–1099 — and most of the
rest is not. The corpus chunk-length distribution is concentrated near the
1000-token cap, so a good deal of it falls in the stable 900–999 band; the
bench pool, built to that distribution, mismatched on 5 of 24, and those five
were exactly the chunks at 751 and 830 tokens. **Roughly a fifth of production
chunks sit below the stable band.** The precise fraction needs per-chunk
lengths, which live in LanceDB rather than the manifest.

So the existing vectors are not wrong, and they are not made wrong by changing
the configuration. They are simply not verifiable and never were: re-embedding
them under the configuration that created them already disagrees about one
chunk in five.

**What follows from that:**

- **Nothing needs re-embedding for correctness or quality.** The corpus is
  semantically sound.
- **If hash-verifiable embeddings are a requirement, the corpus has to be
  re-embedded under the new configuration** — not to repair damage, but because
  the property has never held and cannot be granted retroactively.
- **The change is silent.** `space_id` does not include `-np`, `-ub` or the
  oneDNN setting (see above), so vectors from both configurations will coexist
  in one collection with nothing recording which is which. For cosine retrieval
  that is harmless. For a verifier it means "this chunk failed its hash" is
  indistinguishable from "this chunk predates the change". If a re-embed is not
  done all at once, something has to record the boundary.
- **The sooner the better.** The corpus is 12,784 of 79,282 documents embedded.
  Re-embedding what exists is a small fraction of the work remaining; the same
  decision after the other 66,000 documents is a much larger one.

### What this means for `space_id`

The space's `components` today are `embed_model_hash`, `dtype`, `normalization`,
`pooling`, `chunk_version`, `cleaning_version`, `truncation_policy` and
`query_encoder_hash`. For cosine-with-tolerance verification that is the right
set. For **hash** verification it is not sufficient: two nodes agreeing on every
one of those can still store different bytes by running `-np 4` against `-np 1`,
or `-ub 2048` against `-ub 8192` — and at `-np 4` the same node does not even
agree with itself under load.

Anything that hashes vectors across nodes has to additionally pin, or record:

- `GGML_SYCL_ENABLE_DNN=0` — without it nothing else matters
- `-np 1` — and this one has to be FIXED at 1, not merely recorded: above one
  slot the bytes depend on concurrent traffic rather than on configuration,
  so no metadata can describe them
- `-ub` — recordable, since it is a pure function of configuration
- the image (verified equal across b10630 → b10644, but check each bump)

And separately from anything recorded in metadata: whatever re-embeds a chunk
for verification must not let it be served from the slot prompt cache, which
returns a different vector by 1.26e-3 per component.

### The part that cannot be fixed from one machine

Everything above was measured on one B70 with one driver. **Byte-identity
across different GPUs, drivers or backends was not tested and is very unlikely
to hold** — different hardware runs different kernels with different reduction
orders, which is the same mechanism as C3 one level up. A peer on NVIDIA, on a
CPU backend, or on a different Intel part will almost certainly store different
bytes for the same chunk.

That is not a llama.cpp defect and no flag will fix it. It means hash
verification can only be an intra-cohort property: nodes that have declared an
identical encoder stack can verify each other by hash, and everyone else has to
fall back to cosine with a tolerance. Note that the architecture already leans
this way — invariant 1 deliberately keeps the device out of `space_id` and
relies on "compensating checks at import", and `chunk_verification` carries a
`cosine` column rather than a digest.

Confirming or refuting cross-node byte-identity needs a second machine and is
the single most valuable open test for this requirement.
---

## G. Open

- **b10644 is unevaluated.** Run `scripts/matrix.sh` against it before adopting.
  jeffgrover/b70-setup is the place to look first for whether a build is known
  broken.
- **Cross-node byte-identity is untested and is the most valuable open test.**
  Everything in F was measured on one card. A second machine would settle
  whether hash verification can be a peer-to-peer property at all, or only an
  intra-cohort one.
- **No end-to-end ingest validation.** Everything here is the encoder in
  isolation. Vectors have not been followed through LanceDB into retrieval.
- **The client's concurrency is not established.** `platform.toml` declares
  `concurrency = 2` only under `[embed_defaults.genome]`; what drives the text
  encoder is not recorded there. D4 makes this the single most important number
  to pin down, because the deployed server's throughput varies by 40% across
  the plausible range.
- **Why concurrent prefill costs throughput is not understood** — only that it
  does, reproducibly, by ~40%. Worth an upstream question if it persists on
  b10644.
- **Two things worth reporting upstream.** The oneDNN GEMM non-determinism
  (C3) is a clean single-variable isolation and is undocumented; the
  concurrent-prefill throughput loss is reproducible and unexplained.
- ~~Is the sub-1024-token drift caused by flash attention?~~ **Answered: no** —
  it is oneDNN's GEMM, and `GGML_SYCL_ENABLE_DNN=0` removes it entirely for
  1.7% of throughput (C3).
- ~~Is the prebuilt image leaving throughput on the table?~~ **Answered: no.**
  It matches a tuned source build (C8).

---

