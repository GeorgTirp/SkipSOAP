#!/bin/sh
# Global unit queue across runs. Each appended line is "SCRIPT RUN_DIR UNIT"; at most
# WORKERS units run at once (section 3: <= 4 single-threaded processes in total,
# counting any process started outside the queue). Append "STOP" to drain and exit.
# Usage (from residual-curvature-lab/): sh studies/damping_sweep_20261011/scripts/queue_runner.sh QUEUE_FILE WORKERS
QUEUE="$1"; WORKERS="${2:-4}"
export S=studies/damping_sweep_20261011 PYTHONDONTWRITEBYTECODE=1
touch "$QUEUE"
tail -n +1 -f "$QUEUE" | while read -r line; do [ "$line" = "STOP" ] && break; echo "$line"; done | \
xargs -P "$WORKERS" -L 1 sh -c 'L="$S/logs/$(basename "$1")"; mkdir -p "$L"; echo "$(date -u +%FT%TZ) start $2" >> "$L/driver.log"; if .venv/bin/python "$0" work --run "$1" --unit "$2" > "$L/$2.log" 2>&1; then echo "$(date -u +%FT%TZ) done $2" >> "$L/driver.log"; else echo "$(date -u +%FT%TZ) FAILED $2" >> "$L/driver.log"; fi'
echo "$(date -u +%FT%TZ) queue drained" >> "$QUEUE.log"
