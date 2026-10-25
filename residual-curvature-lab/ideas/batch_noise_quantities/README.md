# batch_noise_quantities — three quantities for the batch size of curvature steps

A short derivation note (`three_quantities.tex` / `three_quantities.pdf`) for `../batch_noise.md`:
1. the noise-to-signal ratio r: batch overfitting of same-batch-fitted steps and the share of the quadratic model's
   gain that survives;
2. the compute-optimal batch B* = sqrt(K · B_ov) once a step has a fixed price;
3. noise in the curvature estimate itself (inverse-Wishart moments), with its own threshold near the block dimension.

`make_figure.py` (seeded, synthetic, reads no lab data) writes `figures/survival_cost.pdf` and `generated/numbers.tex`,
the macros for every synthetic number and check quoted in the note.

Rebuild (from `residual-curvature-lab/`):

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ideas/batch_noise_quantities/make_figure.py
    cd ideas/batch_noise_quantities && ~/.local/bin/tectonic three_quantities.tex
