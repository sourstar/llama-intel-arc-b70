# baselines

`history.csv` is one row per deliberate benchmark run. It is only useful as a
comparison against the previous row, so every row must be measured the same
way: the `fixtures/bench.json` pool, warm (`bench.py` warms a full pass by
default), with the image recorded.

`superseded-2026-08-27-old-pool.csv` holds the first six rows ever recorded.
They are kept because they are the evidence for KNOWN-ISSUES D2, and separated
because they are **not comparable to anything else here**: they were measured
against a fixture pool containing 7900-token chunks, which production never
produces, and with a 2-request warmup that leaves the card ~35% cold. Between
them those two faults produced a confident recommendation to lower `-ub` that
the corrected measurements reverse. Do not compare a new row against them.

Other files:

- `grid-<stamp>.csv` — one concurrency grid: server slots x client concurrency.
- `concurrency-<stamp>/` — one concurrency sweep for a single server config,
  with its consistency captures and per-run logs.
- `matrix-<stamp>/` — one configuration sweep, with every capture, comparison
  log and crash log it produced.
- `soak-<stamp>.{log,csv}` — one soak.
- `stack-compare-<stamp>.log` — `llama bench` against a model with a published
  figure elsewhere, to compare our stack rather than our model.
