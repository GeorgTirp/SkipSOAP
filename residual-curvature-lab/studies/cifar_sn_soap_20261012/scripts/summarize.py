"""Readings S1-S12 (PROTOCOL.md, section 7) of a cifar_sn_soap evaluation run, and the seed-0 tuning table.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  summarize.py --tuning DIR [DIR] [--eval DIR]
Writes studies/cifar_sn_soap_20261012/results/summary.json and prints the tables.
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
MATCHED = [f'{scale:g}' for scale in train.MATCHED_RMS]
REPRESENTATIONS = ('head', 'depth6_stream')


def compare(a, b):
    """'better' if every a < b, 'worse' if every a > b, else 'mixed' (per-seed lists; None counts as worse)."""
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
                     'accuracy_500': final and final['validation_accuracy'],
                     'outcome': read(directory/name/'outcome.json')['status'] if (directory/name/'outcome.json').exists() else None})
    return rows


def summarize(directory):
    protocol = read(directory/'protocol.json')
    chosen, seeds = protocol['chosen'], train.EVAL_SEEDS
    units = {(s['arm'], s['lr'], s['seed']): n for n, s in protocol['units'].items()}
    v = {k: {r['step']: r for r in read(directory/n/'validation.json')} for k, n in units.items()}
    m = {k: json.loads((directory/n/'measurements.json').read_text()) for k, n in units.items()}
    h = {k: json.loads((directory/n/'training.json').read_text()) for k, n in units.items()}
    soap = ('soap', train.SOAP_LR)
    tuned = ('soap_sn', chosen['soap_sn']['lr'])
    matched = ('soap_sn', train.SOAP_LR)
    arms = {'soap': soap, 'soap_sn_tuned': tuned, 'soap_sn_matched': matched}
    if matched == tuned:
        del arms['soap_sn_matched']
    decay = {soap: chosen['soap']['ema_decay'], tuned: chosen['soap_sn']['ema_decay'], matched: chosen['soap_sn']['ema_decay']}

    def raw(arm, step):
        return [loss_at(v[(*arm, s)].get(step)) for s in seeds]

    def ema(arm, step=train.STEPS):
        return [loss_at(v[(*arm, s)].get(step), decay[arm]) for s in seeds]

    def row_at(arm, seed, step):
        return next((r for r in m[(*arm, seed)] if r['step'] == step), None)

    def distortion(arm, step, representation, kind='local'):
        return [(r := row_at(arm, s, step)) and r['distance'][kind][representation]['distortion_q95_q5'] for s in seeds]

    def median_over(arm, seed, window, value):
        low, high = WINDOWS[window]
        values = [x for r in m[(*arm, seed)] if low <= r['step'] <= high and (x := value(r)) is not None]
        return statistics.median(values) if values else None

    def model_error(model, batch='next'):
        return lambda r: abs(1-r[batch][f'ratio_{model}']) if r[batch][f'ratio_{model}'] is not None else None

    out = {'run': str(directory), 'chosen': chosen, 'readings': {}, 'values': {}}
    r = out['readings']
    comparisons = {'S1': tuned, 'S2': matched}
    for code, arm in comparisons.items():
        if code == 'S2' and matched == tuned:
            r['S2_matched_lr'] = 'same units as S1 (tuned lr equals SOAP lr)'
            continue
        r[f'{code}_{"tuned" if code == "S1" else "matched"}_lr'] = {
            'raw_50': compare(raw(arm, 50), raw(soap, 50)), 'raw_200': compare(raw(arm, 200), raw(soap, 200)),
            'ema_500': compare(ema(arm), ema(soap))}

    present = {label: arm for label, arm in arms.items() if all((*arm, s) in units for s in seeds)}
    for label, arm in present.items():
        out['values'][label] = {'lr': arm[1], 'raw_50': raw(arm, 50), 'raw_200': raw(arm, 200), 'raw_500': raw(arm, train.STEPS),
                                'ema_500': ema(arm), 'accuracy_500': [v[(*arm, s)][train.STEPS]['validation_accuracy'] for s in seeds]}

    def sn_readings(arm):
        s3 = {f'{rep}_{step}': {'reading': compare(distortion(arm, step, rep), distortion(soap, step, rep)),
                                'soap_sn': distortion(arm, step, rep), 'soap': distortion(soap, step, rep)}
              for step in (100, 500) for rep in REPRESENTATIONS}
        s4 = {w: {'reading': compare(a := [median_over(arm, s, w, model_error('full')) for s in seeds],
                                     b := [median_over(soap, s, w, model_error('full')) for s in seeds]), 'soap_sn': a, 'soap': b}
              for w in WINDOWS}
        coupling = lambda r: r['next']['coupling']
        s5 = {w: {'reading': compare(a := [median_over(arm, s, w, coupling) for s in seeds],
                                     b := [median_over(soap, s, w, coupling) for s in seeds]), 'soap_sn': a, 'soap': b}
              for w in WINDOWS}
        return {
            'S3_distance_preservation': {'parts': s3, 'verdict': 'SN raises preservation'
                                         if all(x['reading'] == 'better' for x in s3.values()) else 'not shown'},
            'S4_full_ggn_model': {'parts': s4, 'verdict': 'SN improves the full-GGN model'
                                  if all(x['reading'] == 'better' for x in s4.values()) else 'not shown'},
            'S5_coupling': {'parts': s5, 'verdict': 'SN lowers cross-tensor coupling along the step'
                            if all(x['reading'] == 'better' for x in s5.values()) else 'not shown'}}

    def nl(row, *path):
        value = row['nonlinear']
        for key in path:
            value = value[key] if value is not None else None
        return value

    def added_readings(arm):
        s8 = {f'{scale}_{w}': {'reading': compare(
                  a := [median_over(arm, s, w, lambda row, c=scale: nl(row, 'next', 'matched', c, 'linearization_error')) for s in seeds],
                  b := [median_over(soap, s, w, lambda row, c=scale: nl(row, 'next', 'matched', c, 'linearization_error')) for s in seeds]),
                  'soap_sn': a, 'soap': b}
              for scale in MATCHED for w in WINDOWS}
        stale = lambda row: abs(1-x) if (x := nl(row, 'persistence', 'ten_step_ratio')) is not None else None
        s10 = {w: {'reading': compare(a := [median_over(arm, s, w, stale) for s in seeds], b := [median_over(soap, s, w, stale) for s in seeds]),
                   'soap_sn': a, 'soap': b} for w in WINDOWS}
        return {'S8_network_nonlinearity_matched': {'parts': s8, 'verdict': 'SN lowers network nonlinearity at matched step'
                                                    if all(x['reading'] == 'better' for x in s8.values()) else 'not shown'},
                'S10_curvature_persistence': {'parts': s10, 'verdict': 'SN keeps the directional curvature closer over ten steps'
                                              if all(x['reading'] == 'better' for x in s10.values()) else 'not shown'}}

    r['tuned'] = {**sn_readings(tuned), **added_readings(tuned)}
    if matched != tuned:
        r['matched'] = {**sn_readings(matched), **added_readings(matched)}

    # S11 (prediction 3): which tracks the next-batch quadratic-model error along the step, the logit linearization
    # error or the head input's local distortion?
    s11 = {}
    for k in units:
        rows = [(abs(1-row['next']['ratio_full']), nl(row, 'next', 'step', 'linearization_error'),
                 row['distance']['local']['head']['distortion_q95_q5']) for row in m[k]
                if row['next']['ratio_full'] is not None and nl(row, 'next', 'step', 'linearization_error') is not None]
        error, linearization, distortion = zip(*rows)
        s11[f'{k[0]}_lr{k[1]:g}_s{k[2]}'] = {'linearization': float(spearmanr(error, linearization).statistic),
                                             'distortion': float(spearmanr(error, distortion).statistic), 'rows': len(rows)}
    r['S11_prediction3'] = {'units': s11, 'verdict': 'linearization error tracks quadratic-model error better than distortion'
                            if all(x['linearization'] > 0 and x['linearization'] > abs(x['distortion']) for x in s11.values())
                            else 'not shown'}

    # Descriptive: every model, both batches, coupling, GN gap, step norm, SN activity, depth profile, block Jacobians.
    fidelity, profile, activity = {}, {}, {}
    for key in units:
        label = f'{key[0]}_lr{key[1]:g}_s{key[2]}'
        fidelity[label] = {f'{batch}_{model}_{w}_median_abs1m': median_over(key[:2], key[2], w, model_error(model, batch))
                           for batch in ('own', 'next') for model in ('full', 'block', 'hessian') for w in WINDOWS}
        for w in WINDOWS:
            for quantity in ('coupling', 'gauss_newton_gap', 'step_norm'):
                fidelity[label][f'next_{quantity}_{w}_median'] = median_over(key[:2], key[2], w, lambda r, q=quantity: r['next'][q])
            fidelity[label][f'next_full_{w}_median_ratio'] = median_over(key[:2], key[2], w, lambda r: r['next']['ratio_full'])
        profile[label] = {row['step']: {rep: x['distortion_q95_q5'] for rep, x in row['distance']['local'].items()}
                          for row in m[key] if row['step'] in (10, 50, 100, 250, 500)}
        profile[label]['global'] = {row['step']: {rep: row['distance']['global'][rep]['distortion_q95_q5'] for rep in REPRESENTATIONS}
                                    for row in m[key] if row['step'] in (100, 500)}
        profile[label]['block_jacobian'] = {row['step']: [b['rms'] for b in row['block_jacobian']] for row in m[key] if 'block_jacobian' in row}
        activity[label] = {row['step']: {'active': sum(x['active'] for x in row['spectral'].values()),
                                         'sigma_raw_median': statistics.median(x['sigma_raw'] for x in row['spectral'].values()),
                                         'sigma_raw_max': max(x['sigma_raw'] for x in row['spectral'].values())}
                           for row in m[key] if row['step'] in (10, 50, 100, 250, 500)}
    out['fidelity'], out['distance_profile'], out['sn_activity'] = fidelity, profile, activity

    # S9, S12 and the transfer decomposition (descriptive).
    added = {}
    for key in units:
        label, a, seed = f'{key[0]}_lr{key[1]:g}_s{key[2]}', key[:2], key[2]
        row = {}
        for w in WINDOWS:
            for batch in ('next', 'own'):
                row[f'{batch}_network_share_{w}'] = median_over(a, seed, w, lambda r, b=batch: nl(r, b, 'step', 'network_share'))
                row[f'{batch}_linearization_error_{w}'] = median_over(a, seed, w, lambda r, b=batch: nl(r, b, 'step', 'linearization_error'))
                row[f'{batch}_jd_rms_{w}'] = median_over(a, seed, w, lambda r, b=batch: nl(r, b, 'jd_rms'))
            for scale in MATCHED:
                row[f'matched{scale}_network_share_{w}'] = median_over(a, seed, w, lambda r, c=scale: nl(r, 'next', 'matched', c, 'network_share'))
                row[f'matched{scale}_ratio_full_{w}'] = median_over(a, seed, w, lambda r, c=scale: nl(r, 'next', 'matched', c, 'ratio_full'))
            row[f'one_step_ratio_{w}'] = median_over(a, seed, w, lambda r: nl(r, 'persistence', 'one_step_ratio'))
            row[f'ten_step_ratio_{w}'] = median_over(a, seed, w, lambda r: nl(r, 'persistence', 'ten_step_ratio'))
            row[f'transfer_gamma_{w}'] = median_over(a, seed, w, lambda r: r['next']['gd']/r['own']['gd'] if r['own']['gd'] else None)
            row[f'transfer_rho_full_{w}'] = median_over(a, seed, w, lambda r: r['next']['full']/r['own']['full'] if r['own']['full'] > 0 else None)
            for name in ('depth1_stream', 'depth3_stream', 'depth6_stream', 'head'):
                for q in ('linearization_error', 'drift', 'distance_change', 'cka'):
                    row[f'{name}_{q}_{w}'] = median_over(a, seed, w, lambda r, n=name, qq=q: nl(r, 'representation', n).get(qq))
        added[label] = row
    out['added_measures'] = added

    def pairs(key):
        return [(row['distance']['local']['head']['distortion_q95_q5'], abs(1-row['next']['ratio_full']))
                for row in m[key] if row['next']['ratio_full'] is not None and row['distance']['local']['head']['distortion_q95_q5'] is not None]

    s7 = {f'{k[0]}_lr{k[1]:g}_s{k[2]}': float(spearmanr(*zip(*pairs(k))).statistic) for k in units}
    for label, arm in present.items():
        rows = [p for s in seeds for p in pairs((*arm, s))]
        s7[f'pooled_{label}'] = float(spearmanr(*zip(*rows)).statistic)
    rows = [p for k in units for p in pairs(k)]
    s7['pooled_all'] = float(spearmanr(*zip(*rows)).statistic)
    r['S7_interaction_spearman'] = s7
    out['seconds_per_step'] = {label: statistics.median(row['step_seconds'] for s in seeds for row in h[(*arm, s)][1:])
                               for label, arm in present.items()}
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
        print(json.dumps(out['evaluation']['values'], indent=1))
    results = common.STUDY/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'summary.json', out)


if __name__ == '__main__':
    sys.exit(main())
