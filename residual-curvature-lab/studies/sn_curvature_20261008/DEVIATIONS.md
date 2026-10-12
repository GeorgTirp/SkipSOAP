# Deviations from PROTOCOL.md

| # | Time (UTC) | Section | Departure | Reason |
| --- | --- | --- | --- | --- |
| 1 | 2026-10-08T13:18Z | Budget (scheduling) | After the first four ViT units, the scheduler was changed from four to two concurrent workers for the remaining twelve units (the four running units were left to finish). | Diagnostic states took ~215 s under four-way load versus 74 s alone (2.9x, dense eigendecompositions and full-matrix products contending for the shared matrix hardware), so the four-worker re-projection (~4.3 CPU-h) exceeded the 4 CPU-h budget. Two workers keep both seeds within budget at similar wall time; no unit, setting or check changed. |
| 2 | 2026-10-08T13:04Z | Isolation | The agent edited `studies/geometry_followup_20261008/RESULTS.md` (two timestamp corrections in that finished study's report) at 13:04:04Z, eight seconds after this study's before-manifest (13:03:56Z). The after-manifest therefore differs in that file. | Finishing the sibling report; no code of this study wrote outside its folder or data directory. The other manifest difference, `ideas/PARSEC_ideas.md` (created 13:27:28Z), was not written by this session. |
