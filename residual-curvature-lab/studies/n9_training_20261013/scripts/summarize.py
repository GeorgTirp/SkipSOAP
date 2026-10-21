"""Readings N1-N4 (PROTOCOL.md, section 6) of an n9_training evaluation run, and the seed-0 tuning table.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): summarize.py --tuning DIR [DIR] [--eval DIR]
Writes studies/n9_training_20261013/results/summary.json and prints the readings.
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

POINTS = ['raw_25', 'raw_50', 'raw_200', 'ema_500']
COMPARISONS = {'N1': ('soap_n9', 'soap'), 'N2': ('soap_n9', 'soap_eigh'), 'N3': ('topfree_n9', 'topfree'),
               'N4': ('soap_eigh', 'soap')}


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
        rows.append({'unit': name, 'arm': spec['arm'], 'lr': spec['lr'], 'raw_500': loss_at(final),
                     **{f'ema{d:g}_500': loss_at(final, d) for d in train.EMA_DECAYS},
                     'accuracy_500': final and final['validation_accuracy']})
    return rows


def summarize(directory):
    protocol = read(directory/'protocol.json')
    chosen, seeds = protocol['chosen'], train.EVAL_SEEDS
    units = {(s['arm'], s['seed']): n for n, s in protocol['units'].items()}
    v = {k: {r['step']: r for r in read(directory/n/'validation.json')} for k, n in units.items()}
    h = {k: json.loads((directory/n/'training.json').read_text()) for k, n in units.items()}

    def point(arm, name):
        kind, step = name.split('_')
        step = int(step)
        decay = chosen[arm]['ema_decay'] if kind == 'ema' else None
        return [loss_at(v[(arm, s)].get(step), decay) for s in seeds]

    values = {arm: {p: point(arm, p) for p in POINTS} for arm in train.ARMS}
    readings = {}
    for code, (a, b) in COMPARISONS.items():
        readings[code] = {p: compare(values[a][p], values[b][p]) for p in POINTS}
    readings['N1']['verdict'] = ('N9 speeds up SOAP early' if readings['N1']['raw_25'] == readings['N1']['raw_50'] == 'better'
                                 else 'not shown')
    readings['N2']['verdict'] = ('the probe source matters' if readings['N2']['raw_25'] == readings['N2']['raw_50'] == 'better'
                                 else 'not shown')
    curves = {arm: {step: statistics.mean(loss_at(v[(arm, s)].get(step)) for s in seeds) for step in train.VALIDATION}
              for arm in train.ARMS}
    targets = {step: curves['soap'][step] for step in (50, 200)}
    reach = {arm: {f'T{step}': [next((t for t in train.VALIDATION if t > 0 and loss_at(v[(arm, s)].get(t)) is not None
                                      and loss_at(v[(arm, s)][t]) <= target), None) for s in seeds]
                   for step, target in targets.items()} for arm in train.ARMS}
    rebuilds = {}
    for arm in train.ARMS:
        rows = [r['rebuild'] for s in seeds for r in h[(arm, s)] if 'rebuild' in r]
        if rows:
            rebuilds[arm] = {'count': len(rows), 'max_momentum': max(r['momentum'] for r in rows),
                             'max_second_moment_sum': max(r['second_moment_sum'] for r in rows),
                             'min_second_moment': min(r['min_second_moment'] for r in rows),
                             'median_seconds': statistics.median(r['seconds'] for r in rows)}
    seconds = {arm: statistics.median(r['step_seconds'] for s in seeds for r in h[(arm, s)][1:]) for arm in train.ARMS}
    accuracy = {arm: [v[(arm, s)][train.STEPS]['validation_accuracy'] for s in seeds] for arm in train.ARMS}
    return {'run': str(directory), 'chosen': chosen, 'readings': readings, 'values': values, 'curves': curves,
            'targets': targets, 'reach': reach, 'rebuilds': rebuilds, 'seconds_per_step': seconds, 'accuracy_500': accuracy}


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
        print(json.dumps(out['evaluation']['values'], indent=1))
    results = common.STUDY/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'summary.json', out)


if __name__ == '__main__':
    sys.exit(main())
