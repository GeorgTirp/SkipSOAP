# queue_curvature_20261020

Offline study of curvature along the data queue (random reshuffling) on `cifar_ggn_20261012`'s top-free runs
(seeds 1–3, states 50, 100, 250 and 400, replayed bitwise). Read `PROTOCOL.md` first (SHA-256 in
`logs/protocol_sha256.txt`).

- **Part V:** per-example loss and GGN trace against the time since each example's last visit, with a placebo queue.
- **Part S:** curvature sources (current batch, next 8, last 8, the trailing pool at its own weights, random 8, and
  Rademacher probes) scored by subspace capture on the queue's actual curvature and on held-out curvature, and by
  CG steps on the queue's and held-out loss.
- **Part F:** checks the floor-vs-drift ratio from `spectral_rotation_20261020` against Part S.

- [RESULTS.md](RESULTS.md): the visit phase shows in losses but not in curvature estimates; staleness and pooling
  matter; the trailing pool overtakes a fresh batch late in training; probe-row CG doubles the one-step held-out gain.
- [DEVIATIONS.md](DEVIATIONS.md): D1, an out-of-memory crash before any output; fixed and rerun.
- [HANDOFF_PROPOSED_ENTRY.md](HANDOFF_PROPOSED_ENTRY.md): proposed ledger entry.

## Layout

- `scripts/run.py`: replays each unit with queue bookkeeping (positions with the calls of `Replay.next_batch`, last
  visits, a placebo order) and checks gate G1 at steps 50, 100 and 250. Eight steps after each state it runs Parts V
  and S. Outputs go to `$QC_DATA/queue/` (default `~/skipsoap-study-data/queue_curvature_20261020/`); the JSON is
  mirrored to `results/queue/`.
- `scripts/queue_analysis.py`: Parts V and S at one state.
- `scripts/probes.py`: sampled-label and Rademacher probe rows at arbitrary weights for named tensors; per-example
  loss and trace estimates.
- `scripts/summarize.py`: readings into `results/summary.json` and `results/summary.md`.
- `scripts/orchestrate.sh`: waits for `spectral_rotation_20261020`, then manifests and suites before and after, the
  run and the summary.
- `scripts/common.py`, `states.py`, `spectral.py`, `isolation_manifest.sh`: copies (paths changed in `common.py`).
- `tests/`: run by explicit path.

## Implementation choices the protocol leaves open (fixed before any run)

- **Generators:** per state and source (`50000 + 1000·seed + t`, scaled by 100 plus an offset), with the sketch
  generator separate from the probe draws. The visit-phase probes use `990000 + 1000·seed + t`.
- **Pooled rows:** each source's rows are weighted by (number of pooled examples)^−1/2, so ZZᵀ is the average GGN
  estimate.
- **Groups:** tensor groups are sized so that one source's pooled rows (8 × 2,048 columns) fit 2 GiB. Every source
  and target is recomputed per group.
- **Learning rate:** η for the `identity` step's base scale is the run's lr at step t + 1.
- **Pre-run check:** a CPU smoke run at tiny sizes (8-example batches) exercised the whole analysis and the driver's
  windows before any GPU run. It is not part of the results.
