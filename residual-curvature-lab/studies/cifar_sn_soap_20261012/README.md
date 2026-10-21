# cifar_sn_soap_20261012

SOAP with and without SN (c = 1) on CIFAR-10, using the depth-6 ViT of `cifar_ggn_20261012` (0.31M parameters,
batch 2048, GPU float32). It asks whether SN changes training, distance preservation through depth, the accuracy of
the quadratic model along SOAP's step, and the cross-tensor coupling along that step.

- [PROTOCOL.md](PROTOCOL.md): questions, arms, tuning, diagnostics, gates G0/G1, readings S1–S12.
- [RESULTS.md](RESULTS.md): SN makes SOAP worse here. It slows and smooths representation change, but does not
  improve the quadratic model, and it raises cross-tensor coupling.
- [DEVIATIONS.md](DEVIATIONS.md): D1 (bytecode from a pre-run test call), D2 (manifest difference).
- `scripts/`:
  - `train.py`: the arms `soap` and `soap_sn`, and the tuning and evaluation runs. Derived from `cifar_ggn`'s
    runner; its `soap` path is unchanged.
  - `nonlinear.py`: measures A–D, which split the error into network and loss-Taylor parts, measure the network
    nonlinearity at matched function-space step, the curvature persistence, and the representations' linearization
    and drift by depth.
  - `g1.py`: gate G1, bitwise reproduction of `cifar_ggn`'s `soap` units.
  - `summarize.py`: tuning table and readings.
  - `common.py`: paths, archives and CUDA determinism. It appends `studies/cifar_ggn_20261012/scripts` (`cifar`,
    `measure`, `micro`, `ggn`, imported unchanged).
  - `run_queue.sh` and `orchestrate.sh`: the queue and the phase chain.
- `tests/`: unit plan, arm models, `soap` equal to `cifar_ggn`'s bitwise, SN with an inactive bound equal to plain
  SOAP, the helpers, and measures A–D against float64 references.
- External data: `~/skipsoap-study-data/cifar_sn_soap_20261012/` (`$CIFAR_SN_DATA`). Light mirrors go to `runs/`.

Commands, from `residual-curvature-lab/`:

```sh
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -m pytest -p no:cacheprovider -q studies/cifar_sn_soap_20261012/tests
sh studies/cifar_sn_soap_20261012/scripts/orchestrate.sh > studies/cifar_sn_soap_20261012/logs/orchestrate.log 2>&1
```
