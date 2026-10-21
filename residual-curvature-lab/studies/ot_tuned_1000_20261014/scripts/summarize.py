"""Readings T1-T3 (PROTOCOL.md, section 5) of an ot_tuned_1000 evaluation run, and the seed-0 tuning table.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): summarize.py --tuning DIR [DIR] [--eval DIR]
Writes studies/ot_tuned_1000_20261014/results/summary.json and prints the readings.
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


def compare(a, b):
    pairs = [(x if x is not None else float('inf'), y if y is not None else float('inf')) for x, y in zip(a, b)]
    if all(x < y for x, y in pairs):
        return 'better'
    if all(x > y for x, y in pairs):
        return 'worse'
    return 'mixed'


def loss_at(record, decay=None):
    if record is None:
        return None
    value = record['validation_loss'] if decay is None else record[f'ema{decay:g}']['validation_loss']
    return value if value == value else None


def tuning_table(directory):
    rows = []
    for name, spec in sorted(read(directory/'protocol.json')['units'].items()):
        final = {r['step']: r for r in read(directory/name/'validation.json')}.get(train.STEPS)
        rows.append({'unit': name, 'arm': spec['arm'], 'lr': spec['lr'], 'c': spec.get('strength'), 'raw': loss_at(final),
                     **{f'ema{d:g}': loss_at(final, d) for d in train.EMA_DECAYS}})
    return rows


def summarize(directory):
    protocol = read(directory/'protocol.json')
    chosen, seeds = protocol['chosen'], train.EVAL_SEEDS
    units = {(s['arm'], s['seed']): n for n, s in protocol['units'].items()}
    v = {k: {r['step']: r for r in read(directory/n/'validation.json')} for k, n in units.items()}
    h = {k: json.loads((directory/n/'training.json').read_text()) for k, n in units.items()}

    def at(arm, step, ema):
        return [loss_at(v[(arm, s)].get(step), chosen[arm]['ema_decay'] if ema else None) for s in seeds]

    out = {'run': str(directory), 'chosen': chosen, 'readings': {}}
    for code, ema in (('T1_ema_1000', True), ('T2_raw_1000', False)):
        a, b = at('topfree_ot', train.STEPS, ema), at('soap', train.STEPS, ema)
        diff = [x-y for x, y in zip(a, b)]
        out['readings'][code] = {'reading': compare(a, b), 'topfree_ot': a, 'soap': b, 'difference': diff,
                                 'mean_difference': statistics.mean(diff)}
    out['readings']['T1_ema_1000']['verdict'] = ('OT top-free beats SOAP' if out['readings']['T1_ema_1000']['reading'] == 'better'
                                                else 'not shown')
    out['curves'] = {arm: {step: [loss_at(v[(arm, s)].get(step)) for s in seeds] for step in (100, 200, 500, 800, 1000)}
                     for arm in train.ARMS}
    out['accuracy'] = {arm: [v[(arm, s)][train.STEPS]['validation_accuracy'] for s in seeds] for arm in train.ARMS}
    out['seconds_per_step'] = {arm: statistics.median(r['step_seconds'] for s in seeds for r in h[(arm, s)][1:])
                               for arm in train.ARMS}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tuning', type=Path, nargs='+', required=True)
    parser.add_argument('--eval', type=Path)
    args = parser.parse_args()
    out = {'tuning': [row for run in args.tuning for row in tuning_table(run)], 'selection': train.selection(args.tuning)}
    for row in out['tuning']:
        print(row)
    print('selection', out['selection'])
    if args.eval:
        out['evaluation'] = summarize(args.eval)
        print(json.dumps(out['evaluation']['readings'], indent=1))
        print(json.dumps(out['evaluation']['curves'], indent=1))
    results = common.STUDY/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'summary.json', out)


if __name__ == '__main__':
    sys.exit(main())
