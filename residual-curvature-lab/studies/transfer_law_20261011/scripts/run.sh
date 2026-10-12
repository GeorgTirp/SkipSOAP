#!/bin/sh
# Runs Part A (six states) and Part B with at most four processes (from residual-curvature-lab/).
S=studies/transfer_law_20261011
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "corpus.py 0" "corpus.py 1" "corpus.py 2" "corpus.py 3" "steps.py" "corpus.py 4" "corpus.py 5" | \
xargs -P 4 -L 1 sh -c 'n=$(echo "$*" | tr " ." "__"); echo "$(date -u +%FT%TZ) start $*" >> "$0/logs/driver.log"; if .venv/bin/python "$0/scripts/$1" ${2:-} > "$0/logs/$n.log" 2>&1; then echo "$(date -u +%FT%TZ) done $*" >> "$0/logs/driver.log"; else echo "$(date -u +%FT%TZ) FAILED $*" >> "$0/logs/driver.log"; fi' "$S"
echo "$(date -u +%FT%TZ) all done" >> "$S/logs/driver.log"
