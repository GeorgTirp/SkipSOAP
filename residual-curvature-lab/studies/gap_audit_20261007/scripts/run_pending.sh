#!/bin/sh
# Run every pending unit of a run with up to four single-threaded workers.
# Usage (from residual-curvature-lab/): sh studies/gap_audit_20261007/scripts/run_pending.sh SCRIPT RUN_DIR [WORKERS]
set -e
export SCRIPT="$1" RUN="$2" S=studies/gap_audit_20261007 PYTHONDONTWRITEBYTECODE=1
WORKERS="${3:-4}"
export LOGS="$S/logs/$(basename "$RUN")"
mkdir -p "$LOGS"
.venv/bin/python "$SCRIPT" pending --run "$RUN" > "$LOGS/pending.txt"
echo "$(date -u +%FT%TZ) start $(wc -l < "$LOGS/pending.txt") units" >> "$LOGS/driver.log"
xargs -P "$WORKERS" -n 1 sh -c '.venv/bin/python "$SCRIPT" work --run "$RUN" --unit "$1" > "$LOGS/$1.log" 2>&1 || echo "FAILED $1" >> "$LOGS/driver.log"' _ < "$LOGS/pending.txt"
echo "$(date -u +%FT%TZ) done" >> "$LOGS/driver.log"
