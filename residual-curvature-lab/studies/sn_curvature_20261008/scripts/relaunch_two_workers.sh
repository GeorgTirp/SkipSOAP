#!/bin/sh
# DEVIATIONS #1: wait for the four running units to exit, then run the remaining units with two workers.
S=studies/sn_curvature_20261008; R="$1"; shift
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 10; done; done
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python $S/scripts/run.py pending --run "$R" | sed "s|^|$S/scripts/run.py $R |" > $S/logs/queue2.txt
echo "$(date -u +%FT%TZ) relaunch with $(wc -l < $S/logs/queue2.txt) units, 2 workers"
sh $S/scripts/queue_runner.sh $S/logs/queue2.txt 2 &
sleep 3
caffeinate -is -w $! &
wait
