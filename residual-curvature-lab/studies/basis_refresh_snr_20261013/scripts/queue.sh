#!/bin/sh
# Runs Parts A, B and C one GPU process at a time (from residual-curvature-lab/); stops at the first failure.
set -e
PY=.venv/bin/python
R=studies/basis_refresh_snr_20261013/scripts/run.py
L=studies/basis_refresh_snr_20261013/logs
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "$(date -u +%FT%TZ) start" >> $L/queue.log
$PY $R ladder --seeds 1 --steps 25 50 100 250 >> $L/ladder.log 2>&1
$PY $R ladder --seeds 2 3 >> $L/ladder.log 2>&1
echo "$(date -u +%FT%TZ) ladder done" >> $L/queue.log
$PY $R refresh >> $L/refresh.log 2>&1
echo "$(date -u +%FT%TZ) refresh done" >> $L/queue.log
$PY $R snr >> $L/snr.log 2>&1
echo "$(date -u +%FT%TZ) snr done" >> $L/queue.log
