# Proposed HANDOFF entry (gap audit study, 7 October 2026)

Draft for a human to merge into `HANDOFF.md`. Nothing here has been written to `HANDOFF.md` or
`STATUS.md`. Study: `studies/gap_audit_20261007/`; external runs under
`~/skipsoap-study-data/gap_audit_20261007/runs/`; report `studies/gap_audit_20261007/RESULTS.md`.

## Ledger rows

| Question / experiment | Finding | Artifacts |
| --- | --- | --- |
| Gap audit check 1 (+2a/2b): exact GGN versus tuned baselines, ViT batch 512, fresh seeds 1–3, 500 steps | Verdict C (mixed). Tuned-searched-SOAP/GGN validation-loss ratio 1.79–2.05 at step 50 but 1.09–1.20 at step 500; GGN .182 ± .002 at 500, tuned searched SOAP .210 ± .007, tuned native SOAP (lr 1e-2) .161 ± .010 (lower than GGN in all seeds). GGN needs fewest steps to T50/T100 but never fewest seconds (native SOAP 15 s versus GGN 350 s to T50). Dense empirical-Fisher training does not close the gap (2a: no); SOAP map without momentum trains far worse (2b). Seed-0 tuning only; reproduction gate bitwise. | `20261007T121557-dfef403c` (gate), `20261007T122018-998da6a3` (tuning), `20261007T143302-baba2664` (evaluation); `studies/gap_audit_20261007/RESULTS.md` |
| Gap audit check 2c: oracle structured families, six saved ViT states | Best cheap family, the exact diagonal in the nearest-Kronecker eigenbasis, recovers 46.0% of aggregate current-gradient exact-GGN headroom (insufficient, < 60%); Kronecker sums rank 1/2/3 recover 15.5/15.9/18.5%, the empirical-Fisher rank-2 fit 25.6%. Full-GGN PCG: 5.4% (2 matvecs), 67.1% (5), 132.7% (10, beats the block-diagonal teacher). Sanity rows reproduce the capacity screen to 5.2e-12. | `20261007T123608-7f39721a`; `studies/gap_audit_20261007/RESULTS.md` |
| Gap audit check 3: matched OT/GGN capacity protocol (first execution) | 420 searches, 204 operator states, all audits at 1e-8 passed. Pre-declared reading met: OT own-teacher recovery exceeds GGN's by 10.4 pp (state_4: 47.7 vs 37.3%) and 10.5 pp (combined_state_3: 57.0 vs 46.4%) with ~.001 larger absolute decreases. Thin margins; OT exact headroom is smaller (.0235 vs .0266) and OT inverse ray errors are slightly worse. No family reaches 80%. | `20261007T124647-1d8487a2`; [protocol](docs/ot_ggn_capacity_protocol_20261007.md) |
| Gap audit check 4a/4b: nine-ViT family, 22 trajectories, 66 saved states | Exact tensor-GGN current-gradient headroom is positive in every architecture (0.035–0.081). Oracle Kronecker-basis diagonal recovers 36–75% (main ViT 35.6%, train split 48.7%); 5-step full-GGN PCG 68–114%; SOAP-basis diagonal 9–13%. 4b stopped at 21/22 units on a study-internal solver-agreement assertion (1.2e-11 vs 1e-12; residuals 1e-15). 4c (predictor) skipped: gate failed. | `20261007T125950-63edd820`, `20261007T143410-b579ef62` (partial) |

## Candidate insights to preserve

1. **The 50-step exact-GGN gap on the small ViT is mainly early.** On fresh seeds with tuned
   baselines, GGN's validation loss is about half of tuned searched SOAP's at step 50, but within
   10–20% by step 500. Tuned native SOAP (lr 1e-2) is then lower than GGN, and GGN is never faster in
   wall-clock. Compare tuned baselines at matched horizons before attributing a training gap to
   curvature.
2. **Local oracle gains need not carry over to trajectories.** Dense per-example empirical Fisher
   recovered most of the early local GGN gain at saved SOAP states, yet with GGN's damping, solver and
   search it trains worse than tuned searched SOAP. The momentum-free SOAP map is competitive in
   current-gradient fixed-state scores but trains far worse than SOAP with momentum.
3. **Kronecker structure fitted to exact GGN blocks carries little of the headroom.** With oracle
   parameters, the diagonal in the nearest-Kronecker eigenbasis recovers 46% on the six main states
   and 36–75% across nine ViTs. Kronecker sums of rank 1–3 recover 15–19%, and their symmetrized
   rank-2/3 sums are usually indefinite. Matrix-free full-GGN PCG with about 5 matvecs (67%; 68–114%
   across the family) is the only local candidate near the bar; it is untested in training.
4. **OT versus GGN compressibility is a thin result.** The pre-declared reading favours OT
   (+10.4/+10.5 pp recovery for state_4 and combined_state_3), but partly because the OT teacher has
   less headroom. OT's own-teacher inverse errors are not smaller, and no family reaches 80% under
   either teacher.
5. **Euclidean agreement between separately computed solves must scale with conditioning.** On
   this CPU backend, `cholesky_solve` gives buffer-dependent last bits. Two valid solves of a block
   with condition number 6.8e7 differed by 1.2e-11 relative while both had residuals of 1e-15. Use
   residual, energy or backward-error checks at 1e-8 rather than tight forward comparisons.
