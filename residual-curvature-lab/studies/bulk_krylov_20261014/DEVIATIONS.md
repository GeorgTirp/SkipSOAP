# Deviations

## D1 — crash at the first state; fit helper fixed; rerun from the start (2026-10-14, before any result)

**What happened.**
- The orchestration started at 19:10:29Z. The before-manifest was taken and the suites passed (14 suites, including
  this study's 10 tests).
- `run.py` crashed at about 19:15Z in the first state (seed 1, step 10), at the last tensor of the dense pass:
  `RuntimeError: upper bound and lower bound inconsistent with step sign` in `spectral.loglog_fit`.
- **Cause:** a bug. For `head.bias`, the only tensor with fewer than 17 entries (10), the bulk fit over ranks 17–200
  asked for a range beyond the spectrum. `loglog_fit` built an empty-range `arange` instead of returning "no fit".
- No state output was written. `set -e` stopped the orchestration before the summary and the after-checks.
- **Also seen, not a failure:** one CUDA allocator warning, a failed 1 GiB allocation with 1.0 GB free. The
  allocator freed its cache and retried successfully, and the run went on until the crash.

**Fix.**
- `loglog_fit` returns `(None, None)` when the requested first rank exceeds the number of values, as it already does
  when fewer than 3 positive values remain.
- A test for that case is added to `tests/test_bulk.py`.
- Nothing else changes, so no other tensor's quantities and no declared reading are affected. P1 and P2 use only
  matrix tensors (smallest 640 entries), and `head.bias` has no bulk fit in any case.

**Rerun.**
- The whole orchestration is relaunched: a new before-manifest, suites, all 12 states, the summary, the
  after-manifest and the suites.
- This attempt's logs are kept with the suffix `.attempt1`.

**Cost.** In the rerun a state takes about 1 min (62 s for the first), within the protocol's estimate. An earlier
version of this note said 3.5 min; that timing wrongly included the suites.
