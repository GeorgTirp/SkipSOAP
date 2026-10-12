#!/bin/sh
# Rule 10 isolation manifest. Run from residual-curvature-lab/; writes to stdout.
# Identical command to the one that produced isolation_manifest_before.txt.
find . \( -path ./runs -o -path ./datasets -o -path ./.venv -o -path ./build -o -name '*.egg-info' -o -name .pytest_cache -o -name __pycache__ -o -path ./studies/gap_audit_20261007 \) -prune -o \( -type f -o -type l \) -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256
