"""Readings F1-F3 (PROTOCOL.md, section 5): tuned GGN top-free (this study) vs ot_tuned_1000_20261014's tuned OT top-free
and tuned SOAP (references, licensed by G1), seeds 1-3; and the seed-0 tuning table.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): summarize.py --tuning DIR [DIR] --eval DIR
Writes studies/ggn_ot_fairness_20261014/results/summary.json and prints the readings.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import common
from common import read

import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

REFERENCE = Path.home()/'skipsoap-study-data'/'ot_tuned_1000_20261014'/'runs'/'20261014T162934-085e376e'
REFERENCE_UNITS = {'topfree_ot': ('topfree_ot__lr0.09__c4096__s{}', .98), 'soap': ('soap__lr0.03__s{}', .995)}


def compare(a, b):
    if all(x < y for x, y in zip(a, b)):
        return 'better'
    if all(x > y for x, y in zip(a, b)):
        return 'worse'
    return 'mixed'


def record(directory, name, step):
    return {r['step']: r for r in read(directory/name/'validation.json')}[step]


def tuning_table(directory):
    rows = []
    for name, spec in sorted(read(directory/'protocol.json')['units'].items()):
        final = {r['step']: r for r in read(directory/name/'validation.json')}.get(train.STEPS)
        rows.append({'unit': name, 'lr': spec['lr'], 'c': spec.get('strength'), 'raw': final and final['validation_loss'],
                     **{f'ema{d:g}': final and final[f'ema{d:g}']['validation_loss'] for d in train.EMA_DECAYS}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tuning', type=Path, nargs='+', required=True)
    parser.add_argument('--eval', type=Path, required=True)
    args = parser.parse_args()
    chosen = train.selection(args.tuning)['topfree']
    seeds = train.EVAL_SEEDS
    ggn = train.name_of(train.unit('topfree', 0, chosen['lr'], chosen['strength'])).replace('__s0', '__s{}')
    arms = {'topfree': (args.eval, ggn, chosen['ema_decay']),
            **{arm: (REFERENCE, pattern, d) for arm, (pattern, d) in REFERENCE_UNITS.items()}}
    values = {}
    for arm, (directory, pattern, d) in arms.items():
        rows = {s: record(directory, pattern.format(s), train.STEPS) for s in seeds}
        values[arm] = {'ema_1000': [rows[s][f'ema{d:g}']['validation_loss'] for s in seeds],
                       'raw_1000': [rows[s]['validation_loss'] for s in seeds],
                       'accuracy_1000': [rows[s]['validation_accuracy'] for s in seeds],
                       **{f'raw_{t}': [record(directory, pattern.format(s), t)['validation_loss'] for s in seeds] for t in (200, 500)}}
    readings = {
        'F1_ot_vs_tuned_ggn': {'reading': compare(values['topfree_ot']['ema_1000'], values['topfree']['ema_1000']),
                               'difference': [a-b for a, b in zip(values['topfree_ot']['ema_1000'], values['topfree']['ema_1000'])]},
        'F2_tuned_ggn_vs_soap': {'reading': compare(values['topfree']['ema_1000'], values['soap']['ema_1000']),
                                 'difference': [a-b for a, b in zip(values['topfree']['ema_1000'], values['soap']['ema_1000'])]}}
    readings['F1_ot_vs_tuned_ggn']['verdict'] = ('the OT metric matters' if readings['F1_ot_vs_tuned_ggn']['reading'] == 'better'
                                                else 'advantage attributed to tuning the brake and lr')
    out = {'tuning': [r for run in args.tuning for r in tuning_table(run)], 'selection': chosen, 'readings': readings,
           'values': values, 'means': {arm: {k: statistics.mean(v) for k, v in vals.items()} for arm, vals in values.items()}}
    results = common.STUDY/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'summary.json', out)
    for row in out['tuning']:
        print(row)
    print(json.dumps({k: out[k] for k in ('selection', 'readings', 'means')}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
