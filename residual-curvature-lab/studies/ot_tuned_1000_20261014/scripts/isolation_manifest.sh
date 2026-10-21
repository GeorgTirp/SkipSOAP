#!/bin/sh
# Isolation manifest for studies/ot_tuned_1000_20261014 (run from residual-curvature-lab/).
# Same exclusions as the earlier studies, with this study folder excluded instead.
find . \( -path ./runs -o -path ./datasets -o -path ./.venv -o -path ./build -o -name '*.egg-info' -o -name .pytest_cache -o -name __pycache__ -o -path ./studies/ot_tuned_1000_20261014 \) -prune -o \( -type f -o -type l \) -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256
