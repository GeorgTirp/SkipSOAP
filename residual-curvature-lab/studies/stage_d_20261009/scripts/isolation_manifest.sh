#!/bin/sh
# Isolation manifest for studies/stage_d_20261009 (run from residual-curvature-lab/).
# Same exclusions as the gap audit's rule 10, with this study folder excluded instead.
find . \( -path ./runs -o -path ./datasets -o -path ./.venv -o -path ./build -o -name '*.egg-info' -o -name .pytest_cache -o -name __pycache__ -o -path ./studies/stage_d_20261009 \) -prune -o \( -type f -o -type l \) -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256
