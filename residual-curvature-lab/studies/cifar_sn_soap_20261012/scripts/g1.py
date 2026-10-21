"""Gate G1 (PROTOCOL.md, section 6): this study's `soap` units reproduce the `soap` units of cifar_ggn_20261012's
evaluation run bitwise, in every field that study recorded: wall-clock timings and this study's added measures
('nonlinear') are excluded.

Usage (from residual-curvature-lab/): g1.py --run DIR --reference DIR
Writes results/g1.json; exits 1 if any field differs.
"""
import argparse
import json
import sys
from pathlib import Path

import common
from common import read

import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

TIMING = {'training_seconds', 'step_seconds', 'seconds'}
ADDED = {'nonlinear'}


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIMING | ADDED}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def first_difference(a, b, path=''):
    if type(a) is not type(b):
        return path or '/'
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f'{path}/keys'
        return next((d for k in a if (d := first_difference(a[k], b[k], f'{path}/{k}'))), None)
    if isinstance(a, list):
        if len(a) != len(b):
            return f'{path}/length'
        return next((d for i, (x, y) in enumerate(zip(a, b)) if (d := first_difference(x, y, f'{path}[{i}]'))), None)
    return None if a == b else path or '/'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    out, ok = {'run': str(args.run), 'reference': str(args.reference), 'units': {}}, True
    for seed in train.EVAL_SEEDS:
        name = f'soap__lr{train.SOAP_LR:g}__s{seed}'
        row = {}
        for record in ('validation.json', 'training.json', 'measurements.json'):
            a = strip(json.loads((args.run/name/record).read_text()))
            b = strip(json.loads((args.reference/name/record).read_text()))
            difference = first_difference(a, b)
            row[record] = 'bitwise equal' if difference is None else f'differs at {difference}'
            ok &= difference is None
        out['units'][name] = row
    out['passed'] = ok
    write_json(common.STUDY/'results'/'g1.json', out)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
