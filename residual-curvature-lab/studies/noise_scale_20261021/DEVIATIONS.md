# Deviations

## D1 — Pre-launch smoke ran twice; Lanczos storage layout changed for speed (2026-10-21, before the run)

**What happened.**
- The declared smoke (PROTOCOL.md, section 5) ran topfree seed 1, state 10, into the scratchpad. G2 passed (worst
  relative error 9.8e-5 against the 1e-2 tolerance; peak memory 4.4 GiB), but the state took 478 s against the
  estimated 150–180 s.
- **Cause:** a scratch profile showed 428 s in the Lanczos recursion. The stored basis had the layout
  (step, dimension, column), so each reorthogonalization made `einsum` copy up to 0.9 GB.
- **Change:** `scripts/lanczos.py` now stores the basis as (column, step, dimension) and reorthogonalizes with
  batched matrix products. The arithmetic is the same (two classical Gram–Schmidt passes per step). The 8 tests
  pass unchanged.
- **Second smoke:** states 10 and 25 of topfree seed 1, again into the scratchpad. 104 s and 98 s per state, G2
  unchanged to the reported digits, peak 4.1 GiB.

**What was seen.** The smoke outputs include the evaluation-batch decreases for the five powers at those two
states of one trajectory. They are excluded from the results. No protocol quantity, reading or threshold was
changed. The Lanczos step count stays at 300.

**Consequence.** The cost estimate falls to about 100 s per state, about 75 GPU-minutes in all.
