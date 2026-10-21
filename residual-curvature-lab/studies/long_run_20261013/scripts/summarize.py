"""Readings R1-R5 (PROTOCOL.md, section 5) of a long_run evaluation run, and the seed-0 tuning table.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): summarize.py --tuning DIR [DIR] [--eval DIR]
Writes studies/long_run_20261013/results/summary.json and prints the readings.
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

ALONG = [500, 1000, 1500, 2000]


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
        rows.append({'unit': name, 'arm': spec['arm'], 'lr': spec['lr'], 'raw': loss_at(final),
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

    out = {'run': str(directory), 'chosen': chosen, 'readings': {}, 'along': {}}
    def reading(first, second, ema):
        a, b = at(first, train.STEPS, ema), at(second, train.STEPS, ema)
        diff = [x-y for x, y in zip(a, b)]
        return {'reading': compare(a, b), first: a, second: b, 'difference': diff,
                'seeds_lower': sum(d < 0 for d in diff), 'mean_difference': statistics.mean(diff)}

    out['readings']['R1_ema_2000'] = reading('topfree', 'soap', True)
    out['readings']['R2_raw_2000'] = reading('topfree', 'soap', False)
    out['readings']['R5_n9'] = {'ema_vs_topfree': reading('topfree_n9', 'topfree', True),
                                'raw_vs_topfree': reading('topfree_n9', 'topfree', False),
                                'ema_vs_soap': reading('topfree_n9', 'soap', True),
                                'raw_vs_soap': reading('topfree_n9', 'soap', False)}
    for step in ALONG:
        out['along'][step] = {f'{a}_minus_{b}_{kind}': [x-y for x, y in zip(at(a, step, kind == 'ema'), at(b, step, kind == 'ema'))]
                              for a, b in (('topfree', 'soap'), ('topfree_n9', 'topfree'), ('topfree_n9', 'soap'))
                              for kind in ('ema', 'raw')}
    out['curves'] = {arm: {step: statistics.mean(loss_at(v[(arm, s)].get(step)) for s in seeds) for step in train.VALIDATION}
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
        print(json.dumps(out['evaluation']['along'], indent=1))
    results = common.STUDY/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'summary.json', out)


if __name__ == '__main__':
    sys.exit(main())
