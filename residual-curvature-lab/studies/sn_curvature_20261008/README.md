Isolated side study. Nothing here changes the main lab, its scripts or its runs, or the other studies. Large artifacts live outside the synced folder at `~/skipsoap-study-data/sn_curvature_20261008/` (`$SN_CURVATURE_DATA`). Start with RESULTS.md.

# Spectral normalization and curvature (`sn_curvature_20261008`)

How SNGP's spectral normalization (distance preservation) interacts with curvature and curvature
estimation in the lab's ViT and residual MLP: parts A (distance preservation), B (curvature
structure) and C (estimation quality), as selected by the project owner on 8 October 2026. No Laplace
predictive (D) and no training comparisons (E). Protocol: [PROTOCOL.md](PROTOCOL.md). Departures:
[DEVIATIONS.md](DEVIATIONS.md). Results: [RESULTS.md](RESULTS.md).

## Status

- [x] Isolation manifest (1,563 files), baseline suite (106 passed), study tests (5 passed)
- [x] Cost dry run `20261008T130452-a6beee47`: projected 2.7 CPU-h (budget 4)
- [x] Main run `20261008T130821-44610eee`: 16 units (2 models x 4 conditions x 2 seeds), 48 states; gates passed
- [x] Summary, final suite (106 passed), study tests (5 passed), isolation audit (two explained differences), reports
