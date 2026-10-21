# gpu_port_20261012

The first study on the GPU machine. It checks that PARSEC-L gives the same results on the GPU as on the CPU, and
what float32 changes, on the established MNIST workload. It also measures the cost of the candidate CIFAR-10 ViTs
for the next study.

- [PROTOCOL.md](PROTOCOL.md): questions, gates G0–G3, readings R0–R3 and C1, decision rule.
- `gpu_parsec/`: device-ready copy of `studies/parsec_l_20261009/parsec_l/`. Random numbers come from CPU
  generators, so CPU and GPU runs draw the same numbers. On the CPU it is the original, bitwise.
- `scripts/`:
  - `train.py`: MNIST units, full-state windows and the one-step replay;
  - `run_all.sh`: runs all units, then the replays;
  - `cost.py` and `run_cost.sh`: C1;
  - `cifar.py`: CIFAR-10 split and ViT;
  - `summarize.py`: gates and readings;
  - `common.py`: paths, archives and CUDA determinism.
- `tests/`: the 22 PARSEC-L tests against the copy, plus port tests (unit plan, copy = original, bitwise CPU
  replay, one GPU step against the CPU, CUDA repeatability, CIFAR split and models).
- External data: `~/skipsoap-study-data/gpu_port_20261012/` (`$GPU_PORT_DATA`). Light mirrors go to `runs/`.

Commands, from `residual-curvature-lab/`:

```sh
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -m pytest -p no:cacheprovider -q studies/gpu_port_20261012/tests
.venv/bin/python studies/gpu_port_20261012/scripts/train.py create          # prints the run directory
sh studies/gpu_port_20261012/scripts/run_all.sh <run directory>
.venv/bin/python studies/gpu_port_20261012/scripts/cost.py create
sh studies/gpu_port_20261012/scripts/run_cost.sh <cost run directory>
.venv/bin/python studies/gpu_port_20261012/scripts/summarize.py --run <run directory> --cost <cost run directory>
```
