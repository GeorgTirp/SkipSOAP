"""Readings L1-L5 and N1-N3 (PROTOCOL.md, sections 3-4) from results/{ladder,control}/*.json.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): summarize.py
Writes studies/ladder_mechanism_20261013/results/summary.json and prints the readings.
"""
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]/'results'
EARLY = [10, 25, 50]
RUNGS = ['topfree', 'held', 'fresh', 'exact_top', 'ggn']
SHARES = {'coefficients': ('held', 'topfree'), 'staleness': ('fresh', 'held'), 'estimation': ('exact_top', 'fresh'),
          'missing_directions': ('ggn', 'exact_top')}


def load(part):
    return {p.stem: json.loads(p.read_text()) for p in sorted((RESULTS/part).glob('*.json'))}


def part_l():
    states = load('ladder')
    per_state = {}
    for key, s in states.items():
        total = {r: sum(t['reductions'][r] for t in s['tensors'].values()) for r in RUNGS+['held_only', 'soap']}
        gap = total['ggn']-total['topfree']
        per_state[key] = {'seed': s['seed'], 'step': s['step'], 'totals': total, 'gap': gap,
                          'shares': {k: (total[a]-total[b])/gap if gap > 0 else None for k, (a, b) in SHARES.items()},
                          'absolute': {r: total[r]/total['ggn'] for r in total},
                          'transfer': {r: s['transfer'][r]['evaluation_decrease'] for r in s['transfer']},
                          'interior': {r: s['transfer'][r]['interior'] for r in s['transfer']}}
    early = [v for v in per_state.values() if v['step'] in EARLY]
    complete = len(early) == 9

    def count(condition):
        return sum(bool(condition(v)) for v in early)

    held_closes = count(lambda v: v['shares']['coefficients'] is not None and v['shares']['coefficients'] >= .5)
    stale = count(lambda v: v['shares']['staleness'] is not None and v['shares']['staleness'] >= .1)
    estimate = count(lambda v: v['shares']['estimation'] is not None and v['shares']['estimation'] >= .1)
    missing = count(lambda v: v['shares']['missing_directions'] is not None and v['shares']['missing_directions'] >= .5)
    readings = {
        'L1_information_present': {'states': held_closes, 'of': len(early), 'verdict': 'held subspace closes >= 50% of the gap'
                                   if complete and held_closes == 9 else 'not shown'},
        'L2_staleness_matters': {'states': stale, 'of': len(early), 'verdict': 'staleness matters' if complete and stale == 9 else 'not shown'},
        'L3_estimation_matters': {'states': estimate, 'of': len(early), 'verdict': 'estimation matters' if complete and estimate == 9 else 'not shown'},
        'L4_directions_missing': {'states': missing, 'of': len(early), 'verdict': 'directions are missing' if complete and missing == 9 else 'not shown'}}
    readings['L4_directions_missing']['consequence'] = (
        "'close the GGN gap with PARSEC-L' formally dropped" if readings['L4_directions_missing']['verdict'] == 'directions are missing'
        and readings['L1_information_present']['verdict'] == 'not shown' else 'no formal consequence')
    interior = [ok for v in per_state.values() for ok in v['interior'].values()]
    readings['L5_transfer'] = {'interior_fraction': sum(interior)/len(interior) if interior else None}
    return {'readings': readings, 'per_state': per_state}


def part_n():
    points = load('control')
    out, n1, n2 = {}, 0, 0
    for key, p in points.items():
        rho = {k: v['ratio'] for k, v in p['directions'].items()}
        randoms = [v for k, v in rho.items() if k.startswith('random_') and v is not None]
        median_random = statistics.median(randoms)
        deviation_random = statistics.median(abs(1-v) for v in randoms)
        deviation_step = abs(1-rho['step']) if rho['step'] is not None else None
        selection = deviation_step is not None and deviation_random < .5*deviation_step
        n1 += selection
        n2 += median_random < .7
        out[key] = {'arm': p['arm'], 'seed': p['seed'], 't': p['t'], 'rho_step': rho['step'], 'rho_gradient': rho['gradient'],
                    'rho_random_quartiles': statistics.quantiles(randoms, n=4), 'rho_random_median': median_random,
                    'selection_criterion': selection}
    n = len(points)
    readings = {'N1_selection': {'points': n1, 'of': n, 'verdict': 'staleness is mostly a property of the step direction'
                                 if n == 18 and n1 == 18 else 'not shown'},
                'N2_global_change': {'points': n2, 'of': n, 'verdict': 'curvature grows along random directions too'
                                     if n == 18 and n2 == 18 else 'not shown'}}
    return {'readings': readings, 'points': out}


def main():
    out = {'L': part_l() if (RESULTS/'ladder').exists() else None, 'N': part_n() if (RESULTS/'control').exists() else None}
    (RESULTS/'summary.json').write_text(json.dumps(out, indent=1, allow_nan=False)+'\n')
    for part, value in out.items():
        if value is not None:
            print(part, json.dumps(value['readings'], indent=1))


if __name__ == '__main__':
    sys.exit(main())
