"""Readings R1-R6 (PROTOCOL.md, section 7) of a cifar_ggn evaluation run, and the seed-0 tuning table.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  summarize.py --tuning DIR [DIR] [--eval DIR [--partial]]
--partial summarizes the evaluation run closed after the D1 gate failure (DEVIATIONS.md).
Writes studies/cifar_ggn_20261012/results/summary.json and prints the tables.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from scipy.stats import spearmanr

import common
from common import read

import train  # noqa: E402
from residual_curvature.provenance import write_json  # noqa: E402

WINDOWS = {'10-100': (10, 100), '110-500': (110, 500)}


def validation(directory, name):
    return {r['step']: r for r in read(directory/name/'validation.json')}


def measurements(directory, name):
    return json.loads((directory/name/'measurements.json').read_text())


def units_by(directory):
    """{(arm, seed): unit name}."""
    return {(s['arm'], s['seed']): n for n, s in read(directory/'protocol.json')['units'].items()}


def compare(a, b):
    """'better' if every a < b, 'worse' if every a > b, else 'mixed' (a, b: per-seed lists; None counts as worse)."""
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
        v = validation(directory, name)
        final = v.get(train.STEPS)
        rows.append({'unit': name, 'arm': spec['arm'], 'setting': spec.get('lr', spec.get('gamma')),
                     'raw_500': loss_at(final), **{f'ema{d:g}_500': loss_at(final, d) for d in train.EMA_DECAYS},
                     'accuracy_500': final and final['validation_accuracy'],
                     'outcome': read(directory/name/'outcome.json')['status'] if (directory/name/'outcome.json').exists() else None})
    return rows


def summarize(directory):
    protocol = read(directory/'protocol.json')
    chosen, units = protocol['chosen'], units_by(directory)
    seeds = train.EVAL_SEEDS
    decay = {'soap': chosen['soap']['ema_decay'], 'topfree': chosen['topfree']['ema_decay'],
             'ggn': chosen['ggn']['ema_decay'], 'ggn_sn': chosen['ggn']['ema_decay']}
    v = {key: validation(directory, name) for key, name in units.items()}
    m = {key: measurements(directory, name) for key, name in units.items()}

    def raw(arm, step):
        return [loss_at(v[(arm, s)].get(step)) for s in seeds]

    def ema(arm, step=train.STEPS):
        return [loss_at(v[(arm, s)].get(step), decay[arm]) for s in seeds]

    out = {'run': str(directory), 'chosen': chosen, 'decay': decay, 'readings': {}, 'values': {}}
    r = out['readings']
    r['R1_ggn_vs_soap'] = {'raw_50': compare(raw('ggn', 50), raw('soap', 50)), 'raw_200': compare(raw('ggn', 200), raw('soap', 200)),
                           'ema_500': compare(ema('ggn'), ema('soap'))}
    r['R2_topfree_vs_soap'] = {'ema_500': compare(ema('topfree'), ema('soap'))}
    r['R3_sn_vs_ggn'] = {'raw_50': compare(raw('ggn_sn', 50), raw('ggn', 50)), 'raw_200': compare(raw('ggn_sn', 200), raw('ggn', 200)),
                         'ema_500': compare(ema('ggn_sn'), ema('ggn'))}
    for arm in ('soap', 'topfree', 'ggn', 'ggn_sn'):
        out['values'][arm] = {'raw_50': raw(arm, 50), 'raw_200': raw(arm, 200), 'raw_500': raw(arm, train.STEPS),
                              'ema_500': ema(arm), 'accuracy_500': [v[(arm, s)][train.STEPS]['validation_accuracy']
                                                                     if train.STEPS in v[(arm, s)] else None for s in seeds]}

    def at(key, step):
        return next((row for row in m[key] if row['step'] == step), None)

    def distortion(key, step, representation):
        row = at(key, step)
        return row and row['distance']['local'][representation]['distortion_q95_q5']

    r4 = {}
    for step in (100, 500):
        for representation in ('head', 'depth6_stream'):
            sn = [distortion(('ggn_sn', s), step, representation) for s in seeds]
            plain = [distortion(('ggn', s), step, representation) for s in seeds]
            r4[f'{representation}_{step}'] = {'reading': compare(sn, plain), 'ggn_sn': sn, 'ggn': plain}
    r['R4_sn_preservation'] = {'parts': r4, 'verdict': 'SN raises preservation' if all(x['reading'] == 'better' for x in r4.values())
                               else 'not shown'}

    def deviation(key, batch, model, window):
        low, high = WINDOWS[window]
        values = [abs(1-row[batch][f'ratio_{model}']) for row in m[key]
                  if low <= row['step'] <= high and row[batch][f'ratio_{model}'] is not None]
        return (statistics.mean(values), statistics.median(values)) if values else (None, None)

    r5, fidelity = {}, {}
    for window in WINDOWS:
        sn = [deviation(('ggn_sn', s), 'next', 'full', window)[0] for s in seeds]
        plain = [deviation(('ggn', s), 'next', 'full', window)[0] for s in seeds]
        r5[window] = {'reading': compare(sn, plain), 'ggn_sn': sn, 'ggn': plain}
    r['R5_sn_quadratic_model'] = {'parts': r5, 'verdict': 'SN improves the quadratic model'
                                  if all(x['reading'] == 'better' for x in r5.values()) else 'not shown'}
    for key in units:
        fidelity[f'{key[0]}_s{key[1]}'] = {
            f'{batch}_{model}_{window}': deviation(key, batch, model, window)
            for batch in ('own', 'next') for model in ('full', 'block', 'hessian') for window in WINDOWS}
        fidelity[f'{key[0]}_s{key[1]}']['coupling_median'] = statistics.median(
            row['next']['coupling'] for row in m[key] if row['next']['coupling'] is not None) if m[key] else None
        fidelity[f'{key[0]}_s{key[1]}']['gauss_newton_gap_median'] = statistics.median(
            row['next']['gauss_newton_gap'] for row in m[key] if row['next']['gauss_newton_gap'] is not None) if m[key] else None
    out['fidelity'] = fidelity

    def pairs(key):
        rows = [(row['distance']['local']['head']['distortion_q95_q5'], abs(1-row['next']['ratio_full']))
                for row in m[key] if row['next']['ratio_full'] is not None and row['distance']['local']['head']['distortion_q95_q5'] is not None]
        return rows

    r6 = {}
    for key in units:
        rows = pairs(key)
        r6[f'{key[0]}_s{key[1]}'] = float(spearmanr(*zip(*rows)).statistic) if len(rows) > 2 else None
    for label, arms in (('pooled_ggn', ('ggn', 'ggn_sn')), ('pooled_all', ('soap', 'topfree', 'ggn', 'ggn_sn'))):
        rows = [p for key in units if key[0] in arms for p in pairs(key)]
        r6[label] = float(spearmanr(*zip(*rows)).statistic) if len(rows) > 2 else None
    r['R6_interaction_spearman'] = r6

    targets = {step: statistics.mean(raw('soap', step)) for step in (50, 200)}
    reach = {}
    for key in units:
        history = {row['step']: row['training_seconds'] for row in json.loads((directory/units[key]/'training.json').read_text())}
        reach[f'{key[0]}_s{key[1]}'] = {}
        for step, target in targets.items():
            hit = next((s for s in sorted(v[key]) if s > 0 and loss_at(v[key][s]) is not None and loss_at(v[key][s]) <= target), None)
            reach[f'{key[0]}_s{key[1]}'][f'T{step}'] = {'step': hit, 'seconds': history.get(hit) if hit else None}
    out['targets'], out['reach'] = targets, reach
    out['seconds_per_step'] = {
        arm: statistics.median(row['step_seconds'] for s in seeds
                               for row in json.loads((directory/units[(arm, s)]/'training.json').read_text())[1:])
        for arm in ('soap', 'topfree', 'ggn', 'ggn_sn')}
    return out


def summarize_partial(directory):
    """The closed evaluation run of DEVIATIONS.md D1: R2 in full, R6 for the units that exist, and the failed
    `ggn` seed-1 unit descriptively (its records end at step 75). R1, R3, R4 and R5 need the GGN arms: not available."""
    protocol = read(directory/'protocol.json')
    chosen, units = protocol['chosen'], units_by(directory)
    seeds = train.EVAL_SEEDS
    decay = {'soap': chosen['soap']['ema_decay'], 'topfree': chosen['topfree']['ema_decay'], 'ggn': chosen['ggn']['ema_decay']}
    present = {key: name for key, name in units.items() if (directory/name/'validation.json').exists()}
    status = {f'{k[0]}_s{k[1]}': (read(directory/n/'outcome.json')['status'] if (directory/n/'outcome.json').exists()
                                  else 'failed' if (directory/n/'failure.json').exists() else 'missing')
              for k, n in units.items()}
    v = {key: validation(directory, name) for key, name in present.items()}
    m = {key: measurements(directory, name) for key, name in present.items()}
    history = {key: json.loads((directory/name/'training.json').read_text()) for key, name in present.items()}

    def raw(arm, step):
        return [loss_at(v[(arm, s)].get(step)) for s in seeds]

    def ema(arm, step=train.STEPS):
        return [loss_at(v[(arm, s)].get(step), decay[arm]) for s in seeds]

    out = {'run': str(directory), 'closed': 'DEVIATIONS.md D1', 'chosen': chosen, 'decay': decay, 'status': status,
           'readings': {'R1_ggn_vs_soap': 'not available (D1)', 'R2_topfree_vs_soap': {'ema_500': compare(ema('topfree'), ema('soap'))},
                        'R3_sn_vs_ggn': 'not available (D1)', 'R4_sn_preservation': 'not available (D1)',
                        'R5_sn_quadratic_model': 'not available (D1)'}, 'values': {}}
    for arm in ('soap', 'topfree'):
        out['values'][arm] = {'raw_50': raw(arm, 50), 'raw_200': raw(arm, 200), 'raw_500': raw(arm, train.STEPS),
                              'ema_500': ema(arm), 'accuracy_500': [v[(arm, s)][train.STEPS]['validation_accuracy'] for s in seeds],
                              'ema_500_minus_soap': None}
    out['values']['topfree']['ema_500_minus_soap'] = [a-b for a, b in zip(ema('topfree'), ema('soap'))]

    # The failed ggn seed-1 unit, beside soap and topfree on the same seed and data order, at its validation steps.
    g = ('ggn', 1)
    steps = sorted(v[g])
    out['ggn_s1'] = {
        'steps': steps,
        **{arm: {'raw': [loss_at(v[(arm, 1)].get(s)) for s in steps], 'ema': [loss_at(v[(arm, 1)].get(s), decay[arm]) if s > 0 else None for s in steps],
                 'accuracy': [v[(arm, 1)][s]['validation_accuracy'] for s in steps]} for arm in ('ggn', 'soap', 'topfree')},
        'trace': [{k: row.get(k) for k in ('step', 'loss', 'mu', 'alpha', 'solve_residual', 'solve_residual_original', 'kappa_max',
                                          'max_action_check', 'search_bracketed', 'step_seconds', 'norm_without_key_bias')}
                  for row in history[g]],
        'max_whitened_residual': max(row['solve_residual'] for row in history[g] if row['solve_residual'] is not None),
        'all_bracketed': all(row['search_bracketed'] for row in history[g] if row['search_bracketed'] is not None)}
    targets = {step: statistics.mean(raw('soap', step)) for step in (50, 200)}
    out['targets'] = targets
    out['ggn_s1']['reach_T50'] = next((s for s in steps if s > 0 and loss_at(v[g][s]) <= targets[50]), None)

    def window(key, batch, model, low, high, absolute=True):
        values = [row[batch][f'ratio_{model}'] for row in m[key] if low <= row['step'] <= high and row[batch][f'ratio_{model}'] is not None]
        values = [abs(1-x) for x in values] if absolute else values
        return {'mean': statistics.mean(values), 'median': statistics.median(values), 'n': len(values)} if values else None

    fidelity = {}
    for key in present:
        label, rows = f'{key[0]}_s{key[1]}', m[key]
        fidelity[label] = {'steps': [rows[0]['step'], rows[-1]['step']] if rows else None}
        for low, high in ((10, 70), (10, 100), (110, 500)):
            for batch in ('own', 'next'):
                for model in ('full', 'block', 'hessian'):
                    fidelity[label][f'{batch}_{model}_{low}-{high}_abs1m'] = window(key, batch, model, low, high)
                    fidelity[label][f'{batch}_{model}_{low}-{high}_ratio'] = window(key, batch, model, low, high, absolute=False)
            for quantity in ('coupling', 'gauss_newton_gap'):
                values = [row['next'][quantity] for row in rows if low <= row['step'] <= high and row['next'][quantity] is not None]
                fidelity[label][f'next_{quantity}_{low}-{high}_median'] = statistics.median(values) if values else None
    out['fidelity'] = fidelity

    distance = {}
    for key in present:
        label = f'{key[0]}_s{key[1]}'
        distance[label] = {row['step']: {r: row['distance']['local'][r]['distortion_q95_q5']
                                         for r in ('depth1_stream', 'depth3_stream', 'depth6_stream', 'head')}
                           for row in m[key] if row['step'] in (10, 50, 70, 100, 250, 500)}
        distance[label]['block_jacobian'] = {row['step']: [b['rms'] for b in row['block_jacobian']]
                                             for row in m[key] if 'block_jacobian' in row}
    out['distance'] = distance

    def pairs(key):
        return [(row['distance']['local']['head']['distortion_q95_q5'], abs(1-row['next']['ratio_full']))
                for row in m[key] if row['next']['ratio_full'] is not None and row['distance']['local']['head']['distortion_q95_q5'] is not None]

    r6 = {}
    for key in present:
        rows = pairs(key)
        r6[f'{key[0]}_s{key[1]}'] = {'spearman': float(spearmanr(*zip(*rows)).statistic) if len(rows) > 2 else None, 'n': len(rows)}
    rows = [p for key in present for p in pairs(key)]
    r6['pooled_present'] = {'spearman': float(spearmanr(*zip(*rows)).statistic), 'n': len(rows), 'units': len(present)}
    out['readings']['R6_interaction_spearman'] = r6
    out['seconds_per_step'] = {f'{k[0]}_s{k[1]}': statistics.median(row['step_seconds'] for row in history[k][1:]) for k in present}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tuning', type=Path, nargs='+', required=True)
    parser.add_argument('--eval', type=Path)
    parser.add_argument('--partial', action='store_true', help='closed evaluation run (DEVIATIONS.md D1)')
    args = parser.parse_args()
    out = {'tuning': [row for run in args.tuning for row in tuning_table(run)], 'selection': train.selection(args.tuning)}
    for row in out['tuning']:
        print(row)
    print('selection', out['selection'])
    if args.eval and args.partial:
        out['evaluation'] = summarize_partial(args.eval)
        print(json.dumps({k: out['evaluation'][k] for k in ('status', 'readings', 'values', 'targets', 'seconds_per_step')}, indent=1))
    elif args.eval:
        out['evaluation'] = summarize(args.eval)
        print(json.dumps(out['evaluation']['readings'], indent=1))
        print(json.dumps(out['evaluation']['values'], indent=1))
    results = common.STUDY/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'summary.json', out)


if __name__ == '__main__':
    sys.exit(main())
