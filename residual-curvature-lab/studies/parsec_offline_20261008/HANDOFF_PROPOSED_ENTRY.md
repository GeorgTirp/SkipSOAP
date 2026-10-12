# Proposed HANDOFF entry (PARSEC-L offline tests, 8 October 2026)

Draft for a human to merge into `HANDOFF.md`; nothing was written to `HANDOFF.md` or `STATUS.md`.
Study `studies/parsec_offline_20261008/`; external runs `~/skipsoap-study-data/parsec_offline_20261008/runs/`.

## Ledger row

| Question / experiment | Finding | Artifacts |
| --- | --- | --- |
| PARSEC-L offline (idea bank stages B, C): probe-estimated whitened top-r corrections (I2), one-step tracking (I3), SN stream-writer Kronecker structure (SN-1 partial); 72 + 31 + 48 saved states, no training | I2 passes: s = 4 sampled labels keep 97.8% of exact PARSEC-L's local gain (r = 16, momentum; 93–104% across settings). PARSEC-L (exact or s = 4 probes) recovers ~63% of exact tensor-GGN headroom on 69 shared states, on gradient and momentum (exploratory). I3 killed: one-step tracking is worse than fresh extraction for r = 4/8/16. S2 absent by the declared rule (stream-writer Kronecker error not consistently lower under SN; 5/6 ViT pairs lower). Unplanned: the residual MLP's branches all die by step 200 under searched SOAP without SN or at c = 6; c ≤ 2 keeps them alive. | `20261008T162657-67bfad3a`, `20261008T162659-e0b80fae`, `20261008T162701-af2b0339`; `studies/parsec_offline_20261008/RESULTS.md` |

## Candidate insights to preserve

1. **Sampled-label probes suffice for whitened top-r curvature corrections.** Four labels per example
   recover the exact PARSEC-L operator's local gain; per-block extraction from shared probes is accurate,
   and the probe rows are ordinary backprop gradients with sampled labels.
2. **PARSEC-L is the first deployable-looking family above 60% of exact tensor-GGN local headroom**
   (~63%, gradient and momentum), versus 46% for the best Kronecker family. It is still local evidence.
3. **The whitened top subspaces rotate too fast for one-step tracking** (~0.05 sin²Θ per step);
   re-extract at refresh instead.
4. **Searched SOAP can kill the residual branches of a small MLP** (dead GELUs by step 200); tight SN
   prevents it. Check branch activity before interpreting late-training curvature diagnostics.
