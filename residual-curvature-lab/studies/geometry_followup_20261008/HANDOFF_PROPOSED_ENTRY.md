# Proposed HANDOFF entry (geometry follow-up, 8 October 2026)

Draft for a human to merge into `HANDOFF.md`; nothing was written to `HANDOFF.md` or `STATUS.md`.
Study `studies/geometry_followup_20261008/`; external runs `~/skipsoap-study-data/geometry_followup_20261008/runs/`.

## Ledger rows

| Question / experiment | Finding | Artifacts |
| --- | --- | --- |
| Tuned PARSEC (`pns_eigenadam`), both branches, ViT batch 512, seeds 1–3, 500 steps | Default branch (lr 3e-3, refresh 10, rank 32) .186 ± .010 and split branch (lr 1e-2, refresh 100) .192 ± .013 at step 500: better than tuned Adam (.202) and tuned searched SOAP (.210) in all seeds, mixed versus exact tensor GGN (.182), behind tuned native SOAP (.161); slowest early. Split-branch Newton steps with lr_top ≥ 0.1 diverge; the tuned split step is ~99% complement Adam. | `20261008T092438-2c2b66dd` (tuning), `20261008T103332-f9ee97b7` (evaluation); `studies/geometry_followup_20261008/RESULTS.md` |
| Tuned tensor graph-Wasserstein (OT) training, same harness | OT (gamma .1) worse than tensor GGN (gamma .001) at step 50, better at 200 and 500 in all seeds (.165 versus .182). Different tuned damping; at matched gamma .1 (seed-0 tuning) the two were close. | `20261008T103332-f9ee97b7` |
| Full (non-block) GGN and OT versus tensor blocks, matched damping/metric/search | Full GGN worse than block GGN at 50/200/500 in all seeds (.330 versus .182 at 500). Full OT better than tensor OT at step 50, worse at 200/500. Cross-tensor blocks carry 54–92% of the full matrices' energy. | `20261008T092440-b7620df1`, `20261008T103332-f9ee97b7` |

## Candidate insights to preserve

1. **Cross-tensor curvature hurts training here at matched damping.** Exact full GGN, with 54–91% of its
   energy in cross-tensor blocks, trains worse than tensor-block GGN throughout 500 steps. Per-tensor
   curvature is the useful scope, consistent with the gap audit's finding that the gain lies within
   tensors.
2. **PARSEC's gains over Adam need tuning and time.** Tuned, both branches beat tuned Adam and searched
   SOAP at step 500 but trail early. The split branch's Newton top-space step is unstable at useful sizes
   and negligible at stable ones.
3. **OT ends better than GGN only with heavier damping.** Tensor OT at gamma .1 ends below tensor GGN at
   gamma .001; at matched gamma the two were close in tuning, so damping, not geometry, may explain it.
