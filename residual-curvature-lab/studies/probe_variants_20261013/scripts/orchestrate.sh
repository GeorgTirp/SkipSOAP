#!/bin/sh
# Chains the phases of PROTOCOL.md section 4 (from residual-curvature-lab/): seed-0 tuning of topfree_ot (c), an
# extension run if the selected c lies at a grid edge, the evaluation on seeds 1-3, gate G1, then the summary.
set -e
PY=.venv/bin/python
T=studies/probe_variants_20261013/scripts/train.py
S=studies/probe_variants_20261013
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
# A G1 failure is reported (PROTOCOL.md, section 5), not a stop: the summary still runs.
$PY $S/scripts/g1.py --run "$EVAL" > $S/logs/g1_stdout.txt 2>&1 || echo "G1 FAILED $(date -u +%FT%TZ)" > $S/logs/g1_failed.txt
$PY $S/scripts/summarize.py --tuning $TUNINGS --eval "$EVAL" > $S/logs/summary_stdout.txt 2>&1
echo "orchestration complete $(date -u +%FT%TZ)"
