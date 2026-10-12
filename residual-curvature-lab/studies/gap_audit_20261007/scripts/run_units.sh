#!/bin/sh
# Run every pending unit of a check-1 run with up to four single-threaded workers.
# Usage (from residual-curvature-lab/): sh studies/gap_audit_20261007/scripts/run_units.sh RUN_DIR [WORKERS]
set -e
export RUN="$1" S=studies/gap_audit_20261007 PYTHONDONTWRITEBYTECODE=1
WORKERS="${2:-4}"
export LOGS="$S/logs/$(basename "$RUN")"
mkdir -p "$LOGS"
.venv/bin/python $S/scripts/check1_train.py pending --run "$RUN" > "$LOGS/pending.txt"
echo "$(date -u +%FT%TZ) start $(wc -l < "$LOGS/pending.txt") units" >> "$LOGS/driver.log"
xargs -P "$WORKERS" -n 1 sh -c '.venv/bin/python "$S/scripts/check1_train.py" work --run "$RUN" --unit "$1" > "$LOGS/$1.log" 2>&1 || echo "FAILED $1" >> "$LOGS/driver.log"' _ < "$LOGS/pending.txt"
echo "$(date -u +%FT%TZ) done" >> "$LOGS/driver.log"
