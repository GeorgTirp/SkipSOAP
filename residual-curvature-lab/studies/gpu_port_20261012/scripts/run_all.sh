#!/bin/sh
# Runs every unit of a gpu_port run (PROTOCOL.md, section 7), from residual-curvature-lab/:
# CPU units on up to four workers with CUDA hidden, GPU units one at a time alongside them,
# then the one-step replays (G3 on the CPU, R1 on the GPU) and the finalization.
set -e
RUN=$1
PY=.venv/bin/python
T=studies/gpu_port_20261012/scripts/train.py
LOGS=studies/gpu_port_20261012/logs/$(basename "$RUN")
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$LOGS"
(CUDA_VISIBLE_DEVICES= $PY $T pending --run "$RUN" --device cpu | CUDA_VISIBLE_DEVICES= xargs -P 4 -I{} \
    sh -c "$PY $T work --run '$RUN' --unit {} > '$LOGS/{}.log' 2>&1") &
CPU=$!
for u in $($PY $T pending --run "$RUN" --device cuda); do
    $PY $T work --run "$RUN" --unit "$u" > "$LOGS/$u.log" 2>&1
done
wait $CPU
for u in soap__cpu64__s0 topfree__cpu64__s0; do
    CUDA_VISIBLE_DEVICES= $PY $T replay --run "$RUN" --unit $u --device cpu >> "$LOGS/replay.log" 2>&1
    $PY $T replay --run "$RUN" --unit $u --device cuda >> "$LOGS/replay.log" 2>&1
done
$PY $T finalize --run "$RUN" > "$LOGS/finalize.log" 2>&1
