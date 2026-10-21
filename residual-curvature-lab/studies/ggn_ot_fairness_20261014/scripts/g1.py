"""Gate G1 (PROTOCOL.md, section 4): the rerun topfree_ot seed-1 unit reproduces ot_tuned_1000_20261014's unit bitwise in
validation.json and training.json (timings excluded).

Usage (from residual-curvature-lab/): g1.py --run EVAL_DIR. Writes results/g1.json; exits 1 if any field differs.
"""
import argparse
import json
import sys
from pathlib import Path

import common

import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

TIMING = {'training_seconds', 'step_seconds', 'seconds'}
REFERENCE = Path.home()/'skipsoap-study-data'/'ot_tuned_1000_20261014'/'runs'/'20261014T162934-085e376e'


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    name = train.name_of(train.unit(train.OT_REFERENCE['arm'], 1, train.OT_REFERENCE['lr'], train.OT_REFERENCE['strength']))
    checks = {f'{name}/{r}': strip(json.loads((args.run/name/r).read_text())) == strip(json.loads((REFERENCE/name/r).read_text()))
              for r in ('validation.json', 'training.json')}
    out = {'run': str(args.run), 'reference': str(REFERENCE), 'checks': checks, 'passed': all(checks.values())}
    write_json(common.STUDY/'results'/'g1.json', out)
    print(json.dumps(out, indent=1))
    return 0 if out['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
