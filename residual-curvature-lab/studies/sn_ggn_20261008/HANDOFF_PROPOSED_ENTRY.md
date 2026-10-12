# Proposed HANDOFF entry (SN x GGN training, 8 October 2026)

Draft for a human to merge into `HANDOFF.md`; nothing was written to `HANDOFF.md` or `STATUS.md`.
Study `studies/sn_ggn_20261008/`; external run `~/skipsoap-study-data/sn_ggn_20261008/runs/20261008T182801-03c6c5c2`.

## Ledger row

| Question / experiment | Finding | Artifacts |
| --- | --- | --- |
| Does SNGP spectral normalization (c = 0.95, exact σ, 12 residual-branch weights) change training with exact per-tensor GGN (gamma .001, exact trace, shadow-SOAP metric, same-batch ray search) or native SOAP (lr 1e-2)? Main ViT, batch 512, seeds 1–3, 250 steps, settings tuned without SN | All nine readings *mixed* (SN vs no SN at steps 50/100/250 for each optimizer; GGN/SOAP ratio). Step-250 means: GGN .212 vs .213, SOAP .213 vs .221. GGN reaches T50 in 58 steps in both conditions. SN raises GGN's steps-1–50 next-batch transfer from 19% to 32% in every seed by shrinking the own-batch decrease (.21 → .14); next-batch progress unchanged (.040 vs .044 ± .005). After step 50 GGN's next-batch progress is 0 ± .001 per update in both. No-SN units reproduce the gap audit exactly (0.0). | `20261008T182801-03c6c5c2`; `studies/sn_ggn_20261008/RESULTS.md` |

## Candidate insights to preserve

1. **At matched settings, SN does not change optimizer quality here.** Neither per-tensor GGN nor
   native SOAP trains consistently better or worse with SN at c = 0.95 over 250 steps, and GGN's
   step-count advantage over SOAP is unchanged. Retuned or longer comparisons remain open.
2. **SN reduces same-batch overfitting of searched GGN steps early, without adding real progress.**
   The effect is on the own-batch decrease, not the next-batch decrease.
3. **In raw parameters, GGN under SN takes twice as long steps on half the curvature trace** and
   doubles the raw spectral norms. Diagnostics and damping that use raw-parameter scales differ by
   these factors under SN.
