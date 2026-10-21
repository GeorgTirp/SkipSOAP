# Deviations

## D1 — CUDA out-of-memory at the first state; capture normalization fixed; rerun from the start (2026-10-20, before any result)

**What happened.**
- The orchestration started at 19:05:12Z. The before-manifest was taken and the suites passed (16 suites,
  including this study's 7 tests).
- `run.py` replayed seed 1 to step 58 and checked gate G1 at step 50 (passed). It completed Part V for state 50 and
  the six sources of the first tensor group.
- It then failed at 19:08:22Z with `torch.OutOfMemoryError` in `queue_analysis.analyse`, at
  `total = float(z.double().square().sum())`.
- **Cause:** a bug. The target's squared Frobenius norm was computed from a float64 copy of its whole pooled column
  block (16,384 × 16,384 for the largest tensors, 2 GiB). That copy does not fit beside the block itself.
- No output was written. `set -e` stopped the orchestration before the summary and the after-checks.
- **Also seen, not a failure:** a few seconds earlier, one CUDA allocator warning on a 256 MiB request with 204 MiB
  free. The allocator released its cache and retried successfully.

**Fix.**
- The norm is accumulated in float64 over blocks of 1,024 columns (`queue_analysis.squared_norm`). It is the same
  quantity, without the full-size copy.
- A test against the direct computation is added to `tests/test_queue.py`.
- Nothing else changes, so no declared quantity, reading or grouping is affected.
- Because of the allocator warning, the rerun sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in
  `orchestrate.sh`. This changes how CUDA memory is reserved, not any computation.

**Rerun.**
- The whole orchestration is relaunched: a new before-manifest, suites, all three seeds, the summary, the
  after-manifest and the suites.
- This attempt's logs are kept with the suffix `.attempt1`.
