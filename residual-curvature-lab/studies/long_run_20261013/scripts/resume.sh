#!/bin/sh
# Resumes the evaluation run after DEVIATIONS.md D1 (from residual-curvature-lab/): the pending units only (completed
# units are verified by checksum and skipped; the interrupted one is kept as *.interrupted-* and rerun from step 0),
# then the summary over both tuning runs.
set -e
PY=.venv/bin/python
S=studies/long_run_20261013
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
EVAL=$(cat $S/logs/run_eval.txt)
TUNINGS="$(cat $S/logs/run_tune.txt) $(cat $S/logs/run_extend.txt)"
echo "$(date -u +%FT%TZ) resume" >> $S/logs/orchestrate.log
sh $S/scripts/run_queue.sh "$EVAL"
$PY $S/scripts/summarize.py --tuning $TUNINGS --eval "$EVAL" > $S/logs/summary_stdout.txt 2>&1
echo "orchestration complete $(date -u +%FT%TZ)" >> $S/logs/orchestrate.log
