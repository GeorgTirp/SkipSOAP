"""Readings C1-C3 (PROTOCOL.md, section 4) of a ggn_c_control evaluation run, and the seed-0 tuning table over c.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): summarize.py --tuning DIR [DIR] [--eval DIR]
Writes studies/ggn_c_control_20261013/results/summary.json and prints the readings.
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

POINTS = ['raw_50', 'raw_200', 'ema_500']
REFERENCE = common.LAB/'studies'/'probe_variants_20261013'/'results'  # topfree (c = 64), seeds 1-3


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
        rows.append({'unit': name, 'c': spec.get('strength'), 'raw_500': loss_at(final),
                     **{f'ema{d:g}_500': loss_at(final, d) for d in train.EMA_DECAYS}})
    return rows


def summarize(directory):
    protocol = read(directory/'protocol.json')
    chosen, seeds = protocol['chosen'], train.EVAL_SEEDS
    units = {(s['arm'], s['seed']): n for n, s in protocol['units'].items()}
    v = {k: {r['step']: r for r in read(directory/n/'validation.json')} for k, n in units.items()}
    h = {k: json.loads((directory/n/'training.json').read_text()) for k, n in units.items()}

    def point(arm, name):
        kind, step = name.split('_')
        decay = chosen[arm]['ema_decay'] if kind == 'ema' else None
        return [loss_at(v[(arm, s)].get(int(step)), decay) for s in seeds]

    values = {arm: {p: point(arm, p) for p in POINTS} for arm in train.ARMS}
    reference = json.loads((REFERENCE/'summary.json').read_text())['evaluation']['values']['topfree']
    values['topfree_c64_reference'] = reference
    readings = {'C1_ot_vs_ggn_matched': {p: compare(values['topfree_ot'][p], values['topfree_c'][p]) for p in POINTS},
                'C2_tuned_c_vs_c64': {p: compare(values['topfree_c'][p], reference[p]) for p in POINTS}}
    readings['C1_ot_vs_ggn_matched']['verdict'] = ('OT is the better curvature source' if readings['C1_ot_vs_ggn_matched']['ema_500'] == 'better'
                                                  else 'P1 attributed to tuning c; OT line closed for PARSEC-L')
    curves = {arm: {step: statistics.mean(loss_at(v[(arm, s)].get(step)) for s in seeds) for step in train.VALIDATION}
              for arm in train.ARMS}
    probes = {arm: {'correction_energy_median': statistics.median(r['correction_energy'] for s in seeds for r in h[(arm, s)]
                                                                   if r['probe'] and r['correction_energy'] is not None),
                    'eta_lambda_max_median': (statistics.median(r['eta_lambda_max'] for s in seeds for r in h[(arm, s)]
                                                                if r.get('eta_lambda_max') is not None)
                                              if any(r.get('eta_lambda_max') is not None for s in seeds for r in h[(arm, s)]) else None)}
              for arm in train.ARMS}
    seconds = {arm: statistics.median(r['step_seconds'] for s in seeds for r in h[(arm, s)][1:]) for arm in train.ARMS}
    accuracy = {arm: [v[(arm, s)][train.STEPS]['validation_accuracy'] for s in seeds] for arm in train.ARMS}
    return {'run': str(directory), 'chosen': chosen, 'readings': readings, 'values': values, 'curves': curves,
            'probes': probes, 'seconds_per_step': seconds, 'accuracy_500': accuracy}


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
