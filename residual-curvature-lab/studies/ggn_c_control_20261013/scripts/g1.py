"""Gate G1 (PROTOCOL.md, section 3):
- the topfree_ot evaluation units reproduce probe_variants_20261013's bitwise in validation.json and training.json
  (timings excluded);
- the topfree_c c = 64 seed-0 tuning unit reproduces cifar_ggn_20261012's tuning unit topfree__lr0.03__s0 bitwise in
  validation.json (timings excluded).

Usage (from residual-curvature-lab/): g1.py --run EVAL_DIR --tuning TUNE_DIR
Writes results/g1.json; exits 1 if any field differs.
"""
import argparse
import json
import sys
from pathlib import Path

import common

import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

TIMING = {'training_seconds', 'step_seconds', 'seconds'}
DATA = Path.home()/'skipsoap-study-data'
PROBE_VARIANTS = DATA/'probe_variants_20261013'/'runs'/'20261013T111924-eae7ad48'
GGN_TUNING = DATA/'cifar_ggn_20261012'/'runs'/'20261012T103803-7d7d6274'


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def same(a, b):
    return strip(json.loads(Path(a).read_text())) == strip(json.loads(Path(b).read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tuning', type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    for seed in train.EVAL_SEEDS:
        name = train.name_of(train.spec_of('topfree_ot', seed, train.OT_C))
        for record in ('validation.json', 'training.json'):
            checks[f'{name}/{record}'] = same(args.run/name/record, PROBE_VARIANTS/name/record)
    name = train.name_of(train.spec_of('topfree_c', 0, 64.))
    checks[f'{name}/validation.json'] = same(args.tuning/name/'validation.json', GGN_TUNING/'topfree__lr0.03__s0'/'validation.json')
    out = {'run': str(args.run), 'tuning': str(args.tuning), 'checks': checks, 'passed': all(checks.values())}
    write_json(common.STUDY/'results'/'g1.json', out)
    print(json.dumps(out, indent=1))
    return 0 if out['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
