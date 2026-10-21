# Deviations: cifar_ggn_20261012

## D1. Action-audit gate failure in `ggn` seed 1, step 100 (2026-10-12)

- **What happened:** the evaluation run `20261012T110529-d747f9fc` stopped at 12:43Z in unit `ggn__gamma0.01__s1`
  at step 100. This is the second of the four audit steps (2, 100, 250, 500). The block-against-action audit for
  `head.weight` measured 3.5989e-4, against the gate of 1e-4 (PROTOCOL.md, section 6). `ggn.tensor_ggn_step` raised,
  as designed, and the unit wrote `failure.json`. `run_queue.sh` and `orchestrate.sh` stopped (`set -e`). No
  finalize and no summary ran.
- **State at the stop:**
  - Complete: tuning (8 units), extension (1 unit, `topfree` lr 9e-2), and evaluation units `soap` and `topfree`
    for seeds 1–3.
  - Failed: `ggn__gamma0.01__s1` at step 100. Steps 1–99 ran, but records are written only at validation steps, and
    the failure handler writes only `failure.json`. So what is kept is: training records for steps 1–75, validation
    at steps 0–75, measurement rows at steps 10–70, and the step-50 checkpoint.
  - Not started: `ggn` seeds 2 and 3, and `ggn_sn` seeds 1–3.
- **Not changed:** the gate, its tolerance and the code. Nothing has been rerun.
- **Also seen, not a failure:** at the unit's first GPU allocation, the caching allocator logged one failed 1 GiB
  request (`CUDACachingAllocator.cpp:3934`, free 0.94 GB of 8.2 GB). The allocator's retry succeeded, and the unit ran
  normally to step 100.
- **Owner decision:** see *D1 resolution* below.

### D1 diagnosis (2026-10-12, offline; no training rerun)

- **Method:** `scripts/diagnose_audit.py`, from the unit's step-50 checkpoint and the batch of step 51 (same data
  order and augmentation draws). It records the audit for every tensor without raising, then recomputes both sides
  in float64 with the same u and the same F0 (upcast).
  - Output: `results/diagnosis_D1_step51.json` and `logs/diagnosis_D1_step51.log`.
- **The audit already fails at step 51.** `head.weight` measures 2.18e-4, and it is the only tensor over 1e-4. The
  next worst is 2.4e-6 (`blocks.0.w2.weight`). At step 2 the maximum was 2.35e-6. So the failure grew during
  training. It is not specific to step 100.
- **The float32 reference fails, not the solved block.** For `head.weight` (errors relative to the damped action,
  as in the gate):

  | Quantity | Value |
  | --- | ---: |
  | audit as written (float32 block vs float32 reference, this u) | 2.73e-4 |
  | float32 block vs float64 block | 7.4e-7 |
  | float32 reference vs float64 reference | 2.73e-4 |
  | float64 block vs float64 reference | 4.1e-13 |

  The other six tensors checked are at most 2.6e-6 on every line, and their float64 agreement is 1e-14 or better.
- **Cause:** at step 51, `head.weight`'s SOAP denominators span 6.1 decades. For the other tensors checked the span
  is 0.2–4.8 decades. The reference passes through original coordinates: F0 u, then C(F0 u) in float32, then F0ᵀ.
  Along small-denominator directions it amplifies float32 rounding by the spread of d^-1/2 on both sides. The
  block is accumulated from rows already whitened, so it never takes that path.
- **Consequence:** the GGN step solves the whitened block, which is accurate to 7e-7. The run's steps are therefore
  not in doubt. But the audit's float32 reference cannot certify `head.weight` at 1e-4 once its denominators spread
  this far. As written, every later audit step would likely fail.
- **Cost of a float64 reference** (`scripts/time_float64_action.py`, random inputs, batch 2048;
  `logs/diagnosis_D1_float64_timing.log`): one float64 GGN action takes 4.0 s for `ggn` and 12–21 s for `ggn_sn`
  (microbatch 256 or 128), with a peak of 1.1–2.1 GiB.

### D1 options (for the owner)

1. **Float64 recheck of failing tensors (recommended).**
   - Audit in float32 as now. For any tensor over 1e-4, recompute the reference in float64 (same u and F0, float64
     copy of the model, microbatch 128) and gate on that value at the same 1e-4.
   - Both values are logged.
   - Extra cost is seconds per audit step.
2. **Float64 reference for every tensor at the audit steps.**
   - The cleanest version of the check.
   - About 7 min per audit step for `ggn` and 21–36 min for `ggn_sn`: roughly 6–9 GPU-h more in total.
3. **Keep the gate as written.** The `ggn` arms cannot pass audit step 100.

Under options 1 and 2, `ggn__gamma0.01__s1` restarts from step 0 (the failed attempt is preserved as
`*.interrupted-*`). The audit does not touch the training state, so the restart's steps 1–99 must reproduce the failed
attempt's records bitwise. That is an added check. The tolerance stays 1e-4 throughout.

### D1 resolution: owner decision (2026-10-12)

- **Decision:** the owner chose option 3. The gate stays as written, and the GGN arms are stopped. No rerun, no
  change to the audit, and no further GPU training in this study.
- **What the study reports:**
  - Readings that need the GGN arms (R1, R3, R4, R5) are **not available**. R6 is reported only for the units that
    exist.
  - R2 (`topfree` vs `soap`) is complete.
  - `ggn__gamma0.01__s1` steps 1–75 (the records kept) are reported descriptively, as a failed unit. Its steps
    passed every gate that ran: the whitened residual at every step, and the audit at step 2. Per the diagnosis
    above, the unit stopped because the float32 audit reference failed, not the solved block. That is not a
    protocol reading.
- **Closing the evaluation run:** `train.py finalize` refuses runs with a failed unit, so it is not called. The run
  directory is closed with `closed.json` (reason D1) and mirrored to `runs/` by `common.mirror`. The partial summary
  comes from `summarize.py --partial`.

## D2. Bytecode written by Python calls without PYTHONDONTWRITEBYTECODE (2026-10-12, hygiene)

- **What happened:** during the D1 diagnosis, at 12:46Z, a one-off `python -c` call ran without
  `PYTHONDONTWRITEBYTECODE=1`. It listed tensor names and checkpoint keys. It wrote `__pycache__` folders into this
  study's `scripts/`, `src/residual_curvature/` (top level, `optim`, `data`, `models`) and `third_party/soap/`.
- **Found and removed:** found at 16:55Z, while preparing `cifar_sn_soap_20261012`. Every `__pycache__` folder
  outside `.venv` was created today by this session's calls, and all were removed.
- **Effect:** none on results. Bytecode caches are derived from the sources. No source file changed, and the
  isolation manifests exclude `__pycache__`.
