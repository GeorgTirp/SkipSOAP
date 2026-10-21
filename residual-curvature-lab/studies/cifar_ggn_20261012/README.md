# cifar_ggn_20261012

CIFAR-10 on a depth-6 ViT (width 64, 0.31M parameters), batch 2048, GPU float32. The study compares exact per-tensor
GGN with and without SN (c = 1), tuned SOAP and top-free PARSEC-L. Along the steps taken it measures distance
preservation through depth and quadratic-model fidelity (full GGN, per-tensor GGN, Hessian).

- [PROTOCOL.md](PROTOCOL.md): questions, arms, tuning, diagnostics, gates, readings R1–R6.
- [RESULTS.md](RESULTS.md): closed early. R2 is complete; the GGN arms stopped at the step-100 action audit.
- [DEVIATIONS.md](DEVIATIONS.md): D1, the audit failure, its float64 diagnosis and the owner's decision.
- `scripts/`:
  - `train.py`: arms, tuning and evaluation runs, dry run;
  - `ggn.py`: the exact per-tensor GGN step, whitened, in memory groups;
  - `micro.py`: microbatched gradient, GGN action, Hessian quadratic and lean probes;
  - `measure.py`: SN, quadratic checks, distance preservation, block Jacobians;
  - `cifar.py`: split, augmentation and models;
  - `summarize.py`: tuning table and readings (`--partial` for the closed evaluation run);
  - `diagnose_audit.py`, `time_float64_action.py`: the D1 diagnosis and the float64 action timing;
  - `run_queue.sh`: runs the queue; `orchestrate.sh` chains tuning, extension, evaluation and summary;
  - `common.py`: paths, archives and CUDA determinism.
- `tests/`: the GGN step against the lab reference (float64), whitened blocks, lean probes, microbatched quantities,
  quadratic-check terms against dense references, SN, augmentation and the unit plan.
- External data: `~/skipsoap-study-data/cifar_ggn_20261012/` (`$CIFAR_GGN_DATA`). Light mirrors go to `runs/`.

Commands, from `residual-curvature-lab/`:

```sh
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -m pytest -p no:cacheprovider -q studies/cifar_ggn_20261012/tests
.venv/bin/python studies/cifar_ggn_20261012/scripts/train.py create --phase tune           # prints the run directory
sh studies/cifar_ggn_20261012/scripts/run_queue.sh <tuning run>
.venv/bin/python studies/cifar_ggn_20261012/scripts/train.py create --phase eval --tuning <tuning run>
sh studies/cifar_ggn_20261012/scripts/run_queue.sh <evaluation run>
.venv/bin/python studies/cifar_ggn_20261012/scripts/summarize.py --tuning <tuning run> --eval <evaluation run>
```
