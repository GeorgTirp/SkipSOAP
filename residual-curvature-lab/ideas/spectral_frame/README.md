# spectral_frame — working notes on spectrum and frame

A LaTeX note built in parts, jointly with the owner: a probabilistic model of the curvature matrices that
preconditioned optimizers (SOAP) estimate, split into spectrum (a distribution) and frame (eigenvectors).

- `workbook.tex` / `workbook.pdf`: Part I as a workbook to work through: background reading, warm-ups, exercises,
  guided step-by-step proofs and small numpy tasks, with hints in Appendix A. Start here.
- `workbook_solutions.tex` / `workbook_solutions.pdf`: the same workbook with full solutions (Appendix B). The
  wrapper defines `\withsolutions` and inputs `workbook.tex`, which then inputs `solutions.tex`.
- `hints.tex`, `solutions.tex`: Appendices A and B of the workbook.
- `spectral_frame.tex` / `spectral_frame.pdf`: Part I as a compact note (14 pages), for reference.
- `make_figures.py`: seeded synthetic figures and numerical checks. It writes `figures/*.pdf` and
  `generated/numbers.tex` (LaTeX macros for every synthetic number the note and the workbook quote). It reads no
  lab data.
- Lab numbers are quoted from the studies' `RESULTS.md` files, with the study named.

Rebuild (from `residual-curvature-lab/`):

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ideas/spectral_frame/make_figures.py
    cd ideas/spectral_frame
    tectonic workbook.tex && tectonic workbook_solutions.tex && tectonic spectral_frame.tex

Tectonic reruns TeX until the table of contents and cross-references settle. With TeX Live instead, run
`pdflatex` twice on each file.
