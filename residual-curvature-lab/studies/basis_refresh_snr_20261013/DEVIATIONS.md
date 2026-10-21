# Deviations: basis_refresh_snr_20261013

## D1. Part C crashed on loading the first top-free checkpoint (loader bug; rerun of Part C only)

- **What happened:** the queue (`scripts/queue.sh`) completed Parts A (15 states, 08:30–08:41Z) and B (9 windows,
  08:41–09:04Z). Part C then failed at its first state, seed 1 step 50, before writing any output:
  `TypeError: RNG state must be a torch.ByteTensor` in `PARSECL.load_state_dict`.
- **Cause:** `states.to()` moves every tensor in the checkpoint to the GPU, including PARSEC-L's saved CPU
  generator state, which `torch.Generator.set_state` accepts only as a CPU ByteTensor. SOAP states (Parts A and B)
  hold no generator state, so they were not affected.
- **Fix:** `states.topfree` keeps the `parsec` entry of the optimizer state dict on the CPU. The generator only
  draws probe sketches, which Part C does not use. A test now loads a stored top-free checkpoint onto the GPU, when
  one is available.
- **Rerun:** Part C only (`run.py snr`). Parts A and B are unaffected and are not rerun. No protocol change.

## D2. A3's step grid never bracketed; a post-hoc transfer check with an extended grid (2026-10-13)

- **Finding:** in Part A's transfer comparison, every one of the 105 selections (15 states × 7 candidates) chose the
  smallest multiplier of the pre-declared grid, 2⁻³ of α₀. 99 of the 105 evaluation-batch "decreases" are negative:
  even the exact GGN step at 1/8 of its own ray-optimal scale raises the held-out loss.
  - α₀ is the ray-optimal global scalar on the *summed per-tensor* quadratic, which ignores the cross-tensor
    curvature. Along SOAP steps that curvature was 15–21× the per-tensor terms (`cifar_ggn`, `cifar_sn_soap`).
  - The pre-declared grid therefore did not reach the useful step sizes. **A3 stays as pre-declared: "not shown",
    0 of 9 for both comparisons.** It is reported as not measuring transfer at useful step sizes.
- **Post-hoc check, defined here before it runs:** `run.py ladder --posthoc` recomputes each Part A state.
  - The candidates and the quadratic results are recomputed identically and must reproduce the first run bitwise
    (an added check).
  - The transfer grid is extended to α₀·2^k for k ∈ {−14, …, 0}. k is chosen on the same selection batch and
    evaluated on the same evaluation batch.
  - Output goes to `results/ladder_posthoc/`. It is labeled post hoc and has no verdict.
  - Reported: for each candidate, the chosen k (and whether it is interior to the grid) and the evaluation-batch
    decrease. At steps 10, 25 and 50, the counts of `probe_exact` > `soap_exact` and `probe_probe` > `soap_map`.
