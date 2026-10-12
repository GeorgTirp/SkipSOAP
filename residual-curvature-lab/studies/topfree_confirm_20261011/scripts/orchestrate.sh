#!/bin/sh
# Waits for the tuning run, finalizes it (seed-0 selection), creates eval_tuned, queues its units and STOP.
# Only sequences the decisions train.py makes under PROTOCOL.md. Usage: sh orchestrate.sh TUNE_RUN QUEUE_FILE
T="$1"; Q="$2"; S=studies/topfree_confirm_20261011
export PYTHONDONTWRITEBYTECODE=1
until [ "$(ls "$T"/*/complete.json 2>/dev/null | wc -l | tr -d ' ')" -ge 5 ] || [ -f "$T/failure.json" ]; do sleep 20; done
if [ -f "$T/failure.json" ]; then echo "$(date -u +%FT%TZ) tuning failed" >> "$S/logs/orchestrate.log"; echo STOP >> "$Q"; exit 1; fi
.venv/bin/python $S/scripts/train.py finalize --run "$T" >> "$S/logs/orchestrate.log" 2>&1
E=$(.venv/bin/python $S/scripts/train.py create --phase eval_tuned --tuning "$T" 2>/dev/null | tail -1)
echo "$(date -u +%FT%TZ) eval_tuned run $E" >> "$S/logs/orchestrate.log"
for u in $(.venv/bin/python $S/scripts/train.py pending --run "$E" 2>/dev/null); do echo "$S/scripts/train.py $E $u" >> "$Q"; done
echo STOP >> "$Q"
