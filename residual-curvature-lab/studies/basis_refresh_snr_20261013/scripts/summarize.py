"""Readings A1-A4, B1-B3 and C1-C3 (PROTOCOL.md, sections 4-6) from results/{ladder,refresh,snr}/*.json.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): summarize.py
Writes studies/basis_refresh_snr_20261013/results/summary.json and prints the readings.
"""
import json
import math
import statistics
import sys
from pathlib import Path

from scipy.stats import spearmanr

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY/'results'
CANDIDATES = ['soap_map', 'soap_exact', 'soapeigh_exact', 'probe_exact', 'probe_probe', 'kron_exact', 'ggn']
EARLY = [10, 25, 50]
SEEDS = [1, 2, 3]
SIGNALS = ['age', 'factor_rotation', 'probe_rho_rise', 'spread', 'probe_rho']
BANDS = {'1-5': (1, 5), '6-10': (6, 10), '11-20': (11, 20), '21-30': (21, 30)}


def load(part):
    return {p.stem: json.loads(p.read_text()) for p in sorted((RESULTS/part).glob('*.json'))}


def rho(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    value = spearmanr(*zip(*pairs)).statistic
    return None if value != value else float(value)


def kind(name):
    if name.startswith('blocks.'):
        for key in ('attention.query.weight', 'attention.key.weight', 'attention.value.weight', 'attention.output.weight',
                    'w1.weight', 'w2.weight'):
            if name.endswith(key):
                return key.replace('.weight', '')
        return '1-D'
    return {'input.position_embedding': 'embedding', 'input.projection.weight': 'embedding', 'input.cls_token': 'embedding',
            'head.weight': 'head'}.get(name, '1-D')


def pooled(tensors, names=None):
    names = names or list(tensors)
    total = {c: sum(tensors[n]['candidates'][c]['ray'] for n in names) for c in CANDIDATES}
    optimum = sum(tensors[n]['optimum'] for n in names)
    base = total['soap_map']
    return {c: {'headroom': (total[c]-base)/(optimum-base) if optimum > base else None,
                'absolute': total[c]/optimum if optimum > 0 else None} for c in CANDIDATES}


def part_a():
    states = load('ladder')
    if not states:
        return None
    per_state, by_kind = {}, {}
    for key, s in states.items():
        p = pooled(s['tensors'])
        ggn = s['transfer']['ggn']['evaluation_decrease']
        per_state[key] = {'seed': s['seed'], 'step': s['step'], 'pooled': p,
                          'transfer': {c: s['transfer'][c]['evaluation_decrease'] for c in CANDIDATES},
                          'transfer_relative_to_ggn': {c: s['transfer'][c]['evaluation_decrease']/ggn if ggn else None for c in CANDIDATES},
                          'multipliers': {c: s['transfer'][c]['multiplier'] for c in CANDIDATES}}
        kinds = {}
        for name in s['tensors']:
            kinds.setdefault(kind(name), []).append(name)
        by_kind[key] = {k: pooled(s['tensors'], names) for k, names in kinds.items()}
    early = [v for v in per_state.values() if v['step'] in EARLY]
    a1 = sum(v['pooled']['probe_exact']['headroom'] > v['pooled']['soap_exact']['headroom'] for v in early)
    a2 = sum(v['pooled']['probe_probe']['headroom'] > 0 for v in early)
    a3_exact = sum(v['transfer']['probe_exact'] > v['transfer']['soap_exact'] for v in early)
    a3_deploy = sum(v['transfer']['probe_probe'] > v['transfer']['soap_map'] for v in early)
    complete = len(early) == 9
    readings = {
        'A1_probe_basis_vs_soap_basis': {'better': a1, 'of': len(early),
                                          'verdict': 'probe basis better' if complete and a1 == 9 else 'not shown'},
        'A2_deployable_vs_soap_map': {'positive': a2, 'of': len(early),
                                      'verdict': 'deployable N9 beats SOAP map' if complete and a2 == 9 else 'not shown'},
        'A3_transfer': {'probe_exact_vs_soap_exact': a3_exact, 'probe_probe_vs_soap_map': a3_deploy, 'of': len(early),
                        'verdict_exact': 'better' if complete and a3_exact == 9 else 'not shown',
                        'verdict_deployable': 'better' if complete and a3_deploy == 9 else 'not shown'}}
    table = {}
    for step in sorted({v['step'] for v in per_state.values()}):
        rows = [v for v in per_state.values() if v['step'] == step]
        table[step] = {c: {'headroom': [r['pooled'][c]['headroom'] for r in sorted(rows, key=lambda r: r['seed'])],
                           'absolute': [r['pooled'][c]['absolute'] for r in sorted(rows, key=lambda r: r['seed'])],
                           'transfer_rel_ggn': [r['transfer_relative_to_ggn'][c] for r in sorted(rows, key=lambda r: r['seed'])]}
                       for c in CANDIDATES}
    return {'readings': readings, 'table': table, 'per_state': per_state, 'by_kind': by_kind}


def part_b():
    windows = load('refresh')
    if not windows:
        return None
    out, b1, b2 = {}, 0, {s: {'above': 0, 'beats_age': 0} for s in SIGNALS if s != 'age'}
    band_values = {b: [] for b in BANDS}
    global_by_age = {}
    for key, w in windows.items():
        ages, means = [], []
        pairs = {s: [] for s in SIGNALS}
        benefits, cross = [], {s: [] for s in SIGNALS}
        for row in w['rows']:
            values = [t['benefit'] for t in row['tensors'].values() if t['benefit'] is not None]
            ages.append(row['age'])
            means.append(statistics.mean(values) if values else None)
            for band, (lo, hi) in BANDS.items():
                if lo <= row['age'] <= hi:
                    band_values[band] += values
            step_signal = {s: [] for s in SIGNALS}
            step_benefit = []
            for t in row['tensors'].values():
                benefits.append(t['benefit'])
                step_benefit.append(t['benefit'])
                for s in SIGNALS:
                    value = row['age'] if s == 'age' else t[s]
                    pairs[s].append(value)
                    step_signal[s].append(value)
            for s in SIGNALS:
                if s != 'age':
                    cross[s].append(rho(step_signal[s], step_benefit))
            nl = row['next_batch_decrease']
            global_by_age.setdefault(row['age'], []).append(nl['refreshed']-nl['held'])
        trend = rho(ages, means)
        b1 += trend is not None and trend > 0
        correlations = {s: rho(pairs[s], benefits) for s in SIGNALS}
        for s in b2:
            c, a = correlations[s], correlations['age']
            b2[s]['above'] += c is not None and c >= .3
            b2[s]['beats_age'] += c is not None and (a is None or c > a)
        out[key] = {'age_trend': trend, 'mean_benefit_by_age': dict(zip(ages, means)), 'pooled_spearman': correlations,
                    'cross_sectional_median': {s: statistics.median([v for v in cross[s] if v is not None])
                                               if any(v is not None for v in cross[s]) else None for s in cross if s != 'age'}}
    n = len(windows)
    readings = {'B1_staleness_grows': {'positive_windows': b1, 'of': n,
                                       'verdict': 'refresh benefit grows with age' if n == 9 and b1 == 9 else 'not shown'},
                'B2_signals': {s: {**v, 'of': n, 'verdict': 'fires when stale' if n == 9 and v['above'] >= 7 and v['beats_age'] >= 7
                                   else 'not shown'} for s, v in b2.items()}}
    return {'readings': readings, 'windows': out,
            'median_benefit_by_band': {b: statistics.median(v) if v else None for b, v in band_values.items()},
            'global_next_batch_benefit_by_age': {a: statistics.median(v) for a, v in sorted(global_by_age.items())}}


def part_c():
    states = load('snr')
    if not states:
        return None
    out, c1, c2 = {}, 0, 0
    for key, s in states.items():
        below, total, kappas, snrs = 0, 0, [], []
        top_snr, control_snr, alphas, eff_tf, eff_soap, signal_share, noise_share, dim_share = [], [], [], [], [], [], [], []
        for name, t in s['tensors'].items():
            su = [v for v in t['snr'] if v is not None]
            sc = [v for v in t['snr_control'] if v is not None]
            if su and sc:
                total += 1
                below += statistics.median(su) < statistics.median(sc)
            kappas += t['kappa']
            snrs += t['snr']
            top_snr += su
            control_snr += sc
            alphas += [a for a in t['alpha_star'] if a is not None]
            eff_tf += [v for v in t['efficiency_topfree'] if v is not None and v > 0]
            eff_soap += [v for v in t['efficiency_soap'] if v is not None and v > 0]
            if t['signal_share_top'] is not None:
                signal_share.append(t['signal_share_top'])
            if t['noise_share_top'] is not None:
                noise_share.append(t['noise_share_top'])
            dim_share.append(t['dimension_share_top'])
        fraction = below/total if total else None
        correlation = rho(kappas, snrs)
        c1 += fraction is not None and fraction >= .75
        c2 += correlation is not None and correlation <= -.3
        quart = lambda v: [statistics.quantiles(v, n=4)[i] for i in range(3)] if len(v) >= 4 else None
        out[key] = {'seed': s['seed'], 'step': s['step'], 'fraction_top_noisier': fraction, 'tensors': total,
                    'spearman_kappa_snr': correlation,
                    'snr_top_quartiles': quart(top_snr), 'snr_control_quartiles': quart(control_snr),
                    'alpha_star_quartiles': quart(alphas), 'alpha_star_zero_fraction': sum(a == 0 for a in alphas)/len(alphas),
                    'median_abs_log10_efficiency_topfree': statistics.median(abs(math.log10(v)) for v in eff_tf) if eff_tf else None,
                    'median_abs_log10_efficiency_soap': statistics.median(abs(math.log10(v)) for v in eff_soap) if eff_soap else None,
                    'median_log10_efficiency_topfree': statistics.median(math.log10(v) for v in eff_tf) if eff_tf else None,
                    'median_log10_efficiency_soap': statistics.median(math.log10(v) for v in eff_soap) if eff_soap else None,
                    'median_signal_share_top': statistics.median(signal_share) if signal_share else None,
                    'median_noise_share_top': statistics.median(noise_share) if noise_share else None,
                    'median_dimension_share_top': statistics.median(dim_share)}
    n = len(states)
    readings = {'C1_braked_directions_noisier': {'states': c1, 'of': n,
                                                 'verdict': 'braked directions are the noisy ones' if n == 9 and c1 == 9 else 'not shown'},
                'C2_brake_follows_noise': {'states': c2, 'of': n,
                                           'verdict': 'stronger brakes sit on noisier directions' if n == 9 and c2 == 9 else 'not shown'}}
    return {'readings': readings, 'states': out}


def main():
    out = {'A': part_a(), 'B': part_b(), 'C': part_c()}
    (RESULTS/'summary.json').write_text(json.dumps(out, indent=1, allow_nan=False)+'\n')
    for part, value in out.items():
        if value is not None:
            print(part, json.dumps(value['readings'], indent=1))


if __name__ == '__main__':
    sys.exit(main())
