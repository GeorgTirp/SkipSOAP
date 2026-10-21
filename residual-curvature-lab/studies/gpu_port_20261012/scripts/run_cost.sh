#!/bin/sh
# Runs every cell of a gpu_port cost run (C1) one process at a time, from residual-curvature-lab/.
set -e
RUN=$1
PY=.venv/bin/python
C=studies/gpu_port_20261012/scripts/cost.py
LOGS=studies/gpu_port_20261012/logs/$(basename "$RUN")
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$LOGS"
for c in $($PY -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['cells']))" "$RUN/protocol.json"); do
    $PY $C cell --run "$RUN" --cell "$c" > "$LOGS/$c.log" 2>&1
done
$PY $C finalize --run "$RUN" > "$LOGS/finalize.log" 2>&1
