#!/bin/sh
# Applies the staged tuning protocol (PROTOCOL.md step 2) without manual polling: advance each arm's
# next stage when the previous one completes, then finalize the tuning run, create the remaining
# evaluation run and queue it. All decisions are made by train.py; this script only sequences them.
# Usage (from residual-curvature-lab/): sh studies/geometry_followup_20261008/scripts/orchestrate.sh TUNING_RUN
N=studies/geometry_followup_20261008; T="$1"; Q=$N/logs/queue.txt
export PYTHONDONTWRITEBYTECODE=1
while true; do
  if new=$(.venv/bin/python $N/scripts/train.py advance --run "$T" 2>>$N/logs/orchestrate.err); then
    if [ -n "$new" ] && [ "$new" != "nothing to add" ]; then
      echo "$new" | sed "s|^|$N/scripts/train.py $T |" >> $Q
      echo "$(date -u +%FT%TZ) queued: $(echo $new)"
    fi
  fi
  if .venv/bin/python -c "import sys; sys.path.insert(0, '$N/scripts'); import train; from pathlib import Path; sys.exit(0 if train.tuning_complete(Path('$T')) else 1)" 2>>$N/logs/orchestrate.err; then
    break
  fi
  sleep 60
done
.venv/bin/python $N/scripts/train.py finalize --run "$T"
E=$(.venv/bin/python $N/scripts/train.py create --phase eval --arms parsec_default,parsec_split,ot_tensor,ot_full --tuning "$T")
.venv/bin/python $N/scripts/train.py pending --run "$E" | sed "s|^|$N/scripts/train.py $E |" >> $Q
echo "$(date -u +%FT%TZ) evaluation run $E queued"
