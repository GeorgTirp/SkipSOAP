# Proposed HANDOFF entry (SN curvature study, 8 October 2026)

Draft for a human to merge into `HANDOFF.md`; nothing was written to `HANDOFF.md` or `STATUS.md`.
Study `studies/sn_curvature_20261008/`; external runs `~/skipsoap-study-data/sn_curvature_20261008/runs/`.

## Ledger row

| Question / experiment | Finding | Artifacts |
| --- | --- | --- |
| SNGP spectral normalization (exact σ, residual-branch weights, c ∈ {0.95, 2, 6}) versus no SN: distance preservation, GGN structure and estimation quality, ViT and residual MLP, seeds 0/1, states at 10/50/200 searched-SOAP steps (parts A–C) | Mostly *mixed* by the six-pair rule. Bounds 2/6 are inactive in the ViT for most of 200 steps (σ(W) < c). At c = 0.95: ViT diagonal and empirical-Fisher damped-inverse errors consistently lower (EF .36–.84 vs .42–.92); MLP GGN effective rank consistently lower. In raw parameters every SN-active tensor has an exact GGN null direction (≤ 4.5e-16). SN lowers representation distortion only at step 10. No Laplace or training comparison run. | `20261008T130821-44610eee`; `studies/sn_curvature_20261008/RESULTS.md` |

## Candidate insights to preserve

1. **SN acts only where σ(W) exceeds its bound.** In short ViT runs the common bounds (2, 6) never
   activate, so A–C diagnostics at those bounds are null results by construction. Longer training is
   needed before SNGP-style bounds can matter.
2. **In raw parameters, SN adds exact curvature null directions** (scale invariance along each
   normalized W). Damping, priors and Laplace precision act differently on these directions than on
   effective-weight curvature.
3. **Tight SN improves how well cheap estimators approximate the damped GGN inverse in the ViT**
   (diagonal and empirical Fisher, all six matched states at c = 0.95). That is a candidate mechanism
   for a Laplace benefit, not a demonstrated one.
