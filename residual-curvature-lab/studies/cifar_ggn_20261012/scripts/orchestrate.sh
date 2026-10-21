#!/bin/sh
# Chains the phases of PROTOCOL.md section 4 (from residual-curvature-lab/): seed-0 tuning, an extension run if the
# selected lr lies at a grid edge, then the evaluation on seeds 1-3, then the summary. Stops at the first failure.
set -e
PY=.venv/bin/python
T=studies/cifar_ggn_20261012/scripts/train.py
S=studies/cifar_ggn_20261012
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
TUNE=$($PY $T create --phase tune | tail -1)
echo "$TUNE" > $S/logs/run_tune.txt
sh $S/scripts/run_queue.sh "$TUNE"
EXT=$($PY $T create --phase extend --tuning "$TUNE" | tail -1)
TUNINGS="$TUNE"
if [ "$EXT" != "none" ]; then
    echo "$EXT" > $S/logs/run_extend.txt
    sh $S/scripts/run_queue.sh "$EXT"
    TUNINGS="$TUNE $EXT"
fi
EVAL=$($PY $T create --phase eval --tuning $TUNINGS | tail -1)
echo "$EVAL" > $S/logs/run_eval.txt
sh $S/scripts/run_queue.sh "$EVAL"
$PY $S/scripts/summarize.py --tuning $TUNINGS --eval "$EVAL" > $S/logs/summary_stdout.txt 2>&1
echo "orchestration complete $(date -u +%FT%TZ)"
