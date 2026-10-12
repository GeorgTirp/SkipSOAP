# Deviations and incidents

1. **Harness logging bug; first launch aborted and relaunched (10 Oct, 17:16–17:20Z).** At step 25 the tuning
   unit `early_newton__gradient_handover1__s0` failed while writing its per-step record. The diagnostic r̂ is
   infinite when the cross-fitted signal is not positive, and the JSON writer rejects non-finite values. This is
   a logging bug, not a numerical gate: no arm, setting or reading changed.
   - **Response:** the queue was stopped. The harness now records non-finite values as null (`json_safe`, with a
     test), and the seed-0 selection treats a missing step-100 loss as infinitely bad. Fresh runs were created
     because changed sources fail the archive check of the old runs.
   - **Aborted runs, kept and not used:** `20261010T171632-ddf1d240` (tune) and `20261010T171634-24575cea`
     (eval_fixed). They ran for about four minutes; no unit completed.

2. **SOAP reproduction gate failed for lr 1e-2, seed 1 (step 300, difference 2.2e-6 > 1e-6).** The unit
   `soap__lr0.01__s1` in run `20261010T171814-46f283ad` stopped as pre-declared. It matched the gap audit to
   1e-9 at step 25 (study test). The difference comes from rounding: the lab chain applies θ + (θ_new − θ) and
   reloads SOAP's state every step, while `PARSECL` updates in place, and the difference grows over 300 steps.
   - **Not done:** the gate was not loosened and the unit was not rerun.
   - **Consequence:** this study's "SOAP lr 1e-2" arm is reported as failed for seed 1. Readings R1 and R2 use
     the gap audit's native SOAP at lr 1e-2 on the same seeds and data order, which is the gate's own reference.
     This study's seeds 2 and 3 are reported against it.
   - **Remaining effect on comparisons:** the PARSEC-L arms run in this study's harness, which rounds like
     in-place SOAP. Differences of order 1e-6 in validation loss are therefore present in all comparisons with
     gap-audit runs.
