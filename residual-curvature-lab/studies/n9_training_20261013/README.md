# n9_training_20261013

Training test of N9 on CIFAR-10 (depth-6 ViT, batch 2048, 500 steps): SOAP and top-free PARSEC-L with SOAP's
eigenbases rebuilt from the probe-GGN factors, against the baselines and a schedule control (`soap_eigh`).

- [PROTOCOL.md](PROTOCOL.md): arms, tuning, gates G0–G2, readings N1–N4.
- [RESULTS.md](RESULTS.md):
  - N9 does not help SOAP;
  - under top-free PARSEC-L it helps modestly (raw better at steps 25 and 200 in every seed, EMA 500 in 2 of 3).
- `scripts/`:
  - `n9.py`: rebuilds the bases from probe or own factors, carries momentum and second moments, checks G2;
  - `train.py`: arms and runs; the `soap` and `topfree` paths are `cifar_ggn`'s, unchanged;
  - `g1.py`: bitwise reproduction of `cifar_ggn`'s baseline units;
  - `summarize.py`: readings;
  - `run_queue.sh`, `orchestrate.sh`: the queue and the phase chain;
  - `common.py`: paths and archives.
- `tests/`: the rebuild against dense references, the baselines equal to `cifar_ggn`'s bitwise, and the rebuild
  schedule and arms.
- External data: `~/skipsoap-study-data/n9_training_20261013/` (`$N9_DATA`).

```sh
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -m pytest -p no:cacheprovider -q studies/n9_training_20261013/tests
sh studies/n9_training_20261013/scripts/orchestrate.sh > studies/n9_training_20261013/logs/orchestrate.log 2>&1
```
