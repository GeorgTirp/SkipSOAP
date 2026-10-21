#!/bin/sh
# Runs every pending unit of a long_run run one GPU process at a time, then finalizes (from residual-curvature-lab/).
# Stops at the first failing unit (gates stop units; see PROTOCOL.md section 5).
set -e
RUN=$1
PY=.venv/bin/python
T=studies/long_run_20261013/scripts/train.py
LOGS=studies/long_run_20261013/logs/$(basename "$RUN")
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$LOGS"
for u in $($PY $T pending --run "$RUN"); do
    echo "$(date -u +%FT%TZ) start $u" >> "$LOGS/queue.log"
    $PY $T work --run "$RUN" --unit "$u" > "$LOGS/$u.log" 2>&1
    echo "$(date -u +%FT%TZ) done $u" >> "$LOGS/queue.log"
done
$PY $T finalize --run "$RUN" > "$LOGS/finalize.log" 2>&1
