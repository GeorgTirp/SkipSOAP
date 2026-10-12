# Geometry follow-up: tuned PARSEC, tuned OT training, full (non-block) GGN and OT

Study ID `geometry_followup_20261008`. Selected by the project owner on 8 October 2026
("Yes do these 3 please") after the gap audit (`studies/gap_audit_20261007/`). This file is the
prospective protocol, written at 2026-10-08T09:15Z before any code or run of this study. It is
not edited after the first run; departures go to `DEVIATIONS.md` with time and reason.

## Questions

1. **PARSEC (`pns_eigenadam`), tuned.** The 6 October pilot ran it once (seed 0, 50 steps,
   lr .003, default branch only). How do both branches compare with the tuned gap-audit
   baselines when tuned the same way?
2. **OT training, tuned.** The tensor graph-Wasserstein arm ran once (seed 0, 50 steps, gamma
   .01). How does it compare with tuned exact tensor GGN on fresh seeds over 500 steps?
3. **Cross-tensor coupling in training.** Do full (non-block) GGN and full OT steps train
   differently from their tensor-block versions at matched damping, metric and search?

## Isolation (as in the gap audit)

Write only to this folder and `$GEO_FOLLOWUP_DATA` (default
`~/skipsoap-study-data/geometry_followup_20261008/`). Read and import only from the main lab;
never modify `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `references/`, `third_party/`,
`runs/`, `datasets/`, `.venv/`, `HANDOFF.md`, `STATUS.md`, `README.md`, `AGENTS.md`,
`pyproject.toml`, `tested-requirements.txt` or `studies/gap_audit_20261007/`. The gap-audit runs
in `~/skipsoap-study-data/gap_audit_20261007/` are read-only inputs. `PYTHONDONTWRITEBYTECODE=1`
for every Python call; pytest with `-p no:cacheprovider`; no git writes; no new packages or
downloads; the official MNIST test split is never accessed. A bug in shared code is reported,
not fixed. `isolation_manifest_before.txt` (written first, 1,157 files, covering the main lab and
the gap-audit folder) must equal `isolation_manifest_after.txt` at the end. `HANDOFF.md` is not
edited; a `HANDOFF_PROPOSED_ENTRY.md` is written here.

## Workload, seeds and conventions (identical to gap-audit check 1)

`configs/mini_vit_allocation.yaml` workload: ViT width 16, expansion 32, 2 blocks, 2 heads,
28x28 MNIST, batch 512, CPU float64, one thread, deterministic algorithms. Model seed s, data
order `DataOrder(training, 102 + 1000 s)`, split seed 101. Validation on the reserved 5,000
examples at steps 0, 10, 25, 50, ..., 500. Checkpoints at 50/200/500 in evaluation. Store the
per-step train-batch loss, step seconds (excluding validation, I/O and numerical audits) and process
CPU seconds, and all search trials. Up to four concurrent single-threaded processes. Numerical
tolerance 1e-8; never loosen a tolerance or gate to make a run pass. Study-internal checks use only
residual, action or energy criteria at 1e-8, never tight Euclidean comparisons between separately
computed solves (gap-audit DEVIATIONS #11). Code from `studies/gap_audit_20261007/scripts/` may be
copied (with recorded source SHA-256), not imported, so both studies stay independent.

## Arms

| Arm | Definition |
| --- | --- |
| `parsec_default` | Repository port `optim.parsec.propose` (default `split_spaces=False`: global GGN Ritz vectors V, g~ = (I−VᵀV)g + Vᵀ diag((λ+δ)^-1/2) V g, then AdamW). Native steps (no search), as in the source. |
| `parsec_split` | Study port of the source's `split_spaces=True` branch: Newton step −lr_top Vᵀ (Vg / (max(λ,0)+δ+1e-12)) in the top space plus Adam with lr_perp on (I−VᵀV)g. The source's bias-correction index t = step+1 on the already incremented step is preserved. Native steps. |
| `ot_tensor` | Exact tensor blocks of the dynamic graph-Wasserstein pullback (`curvature.wasserstein`), as the pilot's `wasserstein_tensor` but with gamma as a parameter (study copy, bitwise identical at gamma .01), shadow-SOAP metric, exact-trace damping, current gradient, global search. |
| `ggn_full` | As the gap audit's `ggn_exact_trace` (gamma .001, its tuned value), but with the full D x D GGN including all cross-tensor blocks, assembled from the same streamed Jacobian factors and solved by dense Cholesky with the same block-diagonal metric mu Omega. |
| `ot_full` | As `ot_tensor`, with the full D x D graph-Wasserstein metric, at the gamma selected for `ot_tensor`. |

All PARSEC arms use rank r Ritz pairs from `optim.parsec.lanczos_ritz` (float64, full
reorthogonalization, 2r Krylov iterations, Lanczos seed 0 + step), refresh at steps divisible by
the refresh interval, damping δ = 1e-4, betas (.9, .999), eps 1e-8, weight decay 0. Curvature arms
take their first consumed step as SOAP initialization, as in the gap audit.

`ggn_full`/`ot_full` versus their block versions is a matched interaction ablation: identical
gradient, damping rule mu = max(gamma trace/D, 1e-8) (the full trace equals the sum of block
traces), metric, solver type, RHS and search; only the off-diagonal tensor blocks differ.

## Step 1: reproduction gates (≤ 1e-9 absolute; > 1e-6 stops the study)

- `parsec_default` lr .003, refresh 10, rank 16 (32 iterations), seed 0, 50 steps, through
  `run_transport_parsec_noise.propose('parsec_native')`: the pilot `parsec_native` validation
  losses at steps 0/10/25/50 (`runs/20261006T100557-f1efb932/validation.json`).
- `ot_tensor` gamma .01, seed 0, 50 steps, through the repository branch: the pilot
  `wasserstein_tensor` losses.
- This study's copied harness: `ggn_exact_trace` gamma .001, seed 1, 50 steps, must match the
  gap-audit evaluation unit `ggn_exact_trace__gamma0.001__s1` at steps 0/10/25/50.

## Step 2: tuning (seed 0 only; validation loss at the end of the horizon)

- `parsec_default`, stage A: lr in {3e-4, 1e-3, 3e-3, 1e-2, 3e-2} x refresh in {10, 100}, rank 16,
  500 steps. Stage B, at the stage-A optimum: rank in {8, 32}.
- `parsec_split`, stage A: lr_perp = lr_top = lr in the same five-point grid, refresh 10, rank 16,
  500 steps. Stage B: lr_top in {1e-2, 1e-1, 1} at the stage-A lr_perp. Stage C: refresh 100 at the
  stage-B optimum.
- `ot_tensor`: gamma in {.001, .01, .1}, 200 steps (as GGN in the gap audit).
- `ggn_full`, `ot_full`: no tuning (matched ablations).
- Edge rule: if a selected lr/lr_perp/gamma is a grid edge, add one log-spaced point beyond it
  (lr 1e-4 or 1e-1; gamma 1e-4 or 1), once per arm; categorical axes are not extended. A
  divergent run (nonfinite parameters, update or loss) is a failed configuration.

## Step 3: evaluation (fresh seeds 1, 2, 3; 500 consumed steps)

All five arms with the selected settings. Baselines are the gap-audit evaluation units (same
seeds, data order, harness and validation schedule): `ggn_exact_trace` (gamma .001),
`searched_soap_tuned`, `native_soap` (lr 1e-2), `adam_native` (lr 1e-2). Fallback if the projected
cost exceeds the budget: 300 steps for `ggn_full` and `ot_full`.

## Pre-declared readings (descriptive, not significance tests)

For a comparison X versus Y at a step: X is *better* if its validation loss is lower in all three
seeds, *worse* if higher in all three, otherwise *mixed*. Report at steps 50, 200 and 500:

1. `parsec_default` and `parsec_split` each versus tuned `adam_native`, and versus tuned
   `searched_soap_tuned` and `ggn_exact_trace` (descriptive).
2. `ot_tensor` versus `ggn_exact_trace`.
3. `ggn_full` versus `ggn_exact_trace`, and `ot_full` versus `ot_tensor`.

Also report mean ± sd tables, steps- and seconds-to-target for the gap audit's T50 = 0.355037 and
T100 = 0.265401, wall-clock (measured, CPU and sequential-estimate seconds), PARSEC Ritz residuals
and refresh diagnostics, the split branch's top/complement step norms and top-space orthogonality,
and full-versus-block off-diagonal curvature energy fractions at audit steps.

## Checks

Study tests (run by explicit path): the study OT copy equals the repository branch bitwise at
gamma .01; the split-branch port matches the source equations on analytic cases (zero curvature
reduces to Adam with the source's bias correction; with V spanning the space, the top step is
−lr_top (C+δI)^-1 g); full GGN/OT matrices act like `GGN.action` / `wasserstein.action` to 1e-8,
and their diagonal blocks equal the block constructions to 1e-8 (relative Frobenius). In runs:
dense-solve residuals ≤ 1e-8 every step; full-matrix action checks ≤ 1e-8 at steps 2/50/200/500;
the gradient/metric/search code paths are the repository's.

## Budget, cost dry run and stop conditions

Dry run first: three steps of every arm, then projected CPU-hours. Budget for the whole study:
18 CPU-h; if the projection exceeds it, use the 300-step fallback for the full arms; if it still
exceeds it, stop. Stop also on: a reproduction gate failing by more than 1e-6, a numerical check
failing at 1e-8, the baseline suite failing, or a required change outside this folder or its data
directory. On stopping, record the state in `RESULTS.md`, mark the item stopped and end.

## Outputs

`README.md`, `RESULTS.md` (summary ≤ 12 lines, settings, tables, figures, verdicts, limitations,
deviations, isolation audit, final suite, reproduction commands), `DEVIATIONS.md`,
`HANDOFF_PROPOSED_ENTRY.md`, light mirrors of each run (json/md/png/yaml, ≤ 30 MB) in `runs/`.
