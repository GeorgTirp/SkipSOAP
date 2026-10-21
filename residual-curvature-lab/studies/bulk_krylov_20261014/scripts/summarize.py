"""Readings P1-P3 and K1-K5 (PROTOCOL.md, sections 3-4) from results/bulk/*.json.

Usage (from residual-curvature-lab/): summarize.py
Writes studies/bulk_krylov_20261014/results/summary.json and summary.md.

Definitions fixed here, before any result: a *matrix tensor* has a two-dimensional shape with both sides above 1
(cls_token (1, 64) is a vector). The early states are steps 10, 25 and 50 of seeds 1-3; step 100 is descriptive.
Shares are pooled over all tensors of a state: (sum_b R_X - sum_b R_topfree) / (sum_b R* - sum_b R_topfree).
"""
import json
import statistics
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY/'results'
LADDER = Path.home()/'skipsoap-study-data'/'ladder_mechanism_20261013'/'ladder'
SEEDS, EARLY, LATE = [1, 2, 3], [10, 25, 50], [100]
KRYLOV, CHEBYSHEV = [2, 4, 8], [4, 8]
R2_MIN, ALPHA_TOLERANCE, SHARE_P = .95, .15, .8
K1_MAX, HALF = .2, .5


def load(seed, step):
    return json.loads((RESULTS/'bulk'/f'topfree_s{seed}_t{step:04d}.json').read_text())


def is_matrix(row):
    return len(row['shape']) == 2 and min(row['shape']) > 1


def kind(name):
    for key in ['position_embedding', 'projection', 'query', 'key', 'value', 'output', 'w1', 'w2', 'head']:
        if key in name:
            return key
    return 'other'


def depth(name):
    parts = name.split('.')
    return int(parts[1]) if parts[0] == 'blocks' else (-1 if parts[0] == 'input' else 99)


def shares(out):
    rows = out['tensors'].values()
    rungs = list(next(iter(rows))['reductions'])
    total = {r: sum(row['reductions'][r] for row in rows) for r in rungs}
    gap = total['ggn']-total['topfree']
    return gap, {r: (total[r]-total['topfree'])/gap if gap > 0 else None for r in rungs}, total


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def power_law(out):
    matrices = {n: r for n, r in out['tensors'].items() if is_matrix(r)}
    r2 = [r['spectrum']['r2_all'] for r in matrices.values()]
    close = [r['probe']['alpha_trace'] is not None and r['spectrum']['alpha_bulk'] is not None
             and abs(r['probe']['alpha_trace']-r['spectrum']['alpha_bulk']) <= ALPHA_TOLERANCE for r in matrices.values()]
    return {'matrix_tensors': len(matrices),
            'fraction_r2_all': sum(v is not None and v >= R2_MIN for v in r2)/len(matrices),
            'fraction_alpha_close': sum(close)/len(matrices),
            'median_r2_all': median(r2), 'median_alpha_top': median(r['spectrum']['alpha_top'] for r in matrices.values()),
            'median_alpha_bulk': median(r['spectrum']['alpha_bulk'] for r in matrices.values()),
            'median_alpha_trace': median(r['probe']['alpha_trace'] for r in matrices.values()),
            'median_alpha_slope': median(r['probe']['alpha_slope'] for r in matrices.values()),
            'median_abs_alpha_trace_minus_bulk': median(
                abs(r['probe']['alpha_trace']-r['spectrum']['alpha_bulk']) for r in matrices.values()
                if r['probe']['alpha_trace'] is not None and r['spectrum']['alpha_bulk'] is not None),
            'max_ritz_residual_last': max((r['spectrum']['ritz_residual_last'] or 0.) for r in matrices.values())}


def by_kind(outs):
    """P3: median alpha_top and alpha_bulk by tensor kind and by depth, per step (over seeds)."""
    table = {}
    for (seed, step), out in outs.items():
        for name, row in out['tensors'].items():
            if not is_matrix(row):
                continue
            for key in [f'kind:{kind(name)}', f'depth:{depth(name)}']:
                entry = table.setdefault(key, {}).setdefault(str(step), {'alpha_top': [], 'alpha_bulk': [], 'alpha_trace': []})
                entry['alpha_top'].append(row['spectrum']['alpha_top'])
                entry['alpha_bulk'].append(row['spectrum']['alpha_bulk'])
                entry['alpha_trace'].append(row['probe']['alpha_trace'])
    return {key: {step: {q: median(v) for q, v in entry.items()} for step, entry in steps.items()} for key, steps in table.items()}


def ladder_consistency(seed, step, out):
    """R* and top-free's reductions against ladder_mechanism's at the same state (same blocks by construction)."""
    path = LADDER/f'topfree_s{seed}_t{step:04d}.json'
    if not path.exists():
        return None
    ladder = json.loads(path.read_text())
    worst = {}
    for rung in ['ggn', 'topfree']:
        worst[rung] = max(abs(out['tensors'][n]['reductions'][rung]-r['reductions'][rung])/max(abs(r['reductions'][rung]), 1e-300)
                          for n, r in ladder['tensors'].items())
    return {'mu_equal': out['mu'] == ladder['mu'], 'trace_equal': out['trace'] == out['dense_trace'],
            'max_relative_difference': worst,
            'bitwise': all(out['tensors'][n]['reductions'][rung] == r['reductions'][rung]
                           for n, r in ladder['tensors'].items() for rung in ['ggn', 'topfree'])}


def main():
    outs = {(seed, step): load(seed, step) for seed in SEEDS for step in EARLY+LATE
            if (RESULTS/'bulk'/f'topfree_s{seed}_t{step:04d}.json').exists()}
    missing = [(s, t) for s in SEEDS for t in EARLY+LATE if (s, t) not in outs]
    states = {}
    for (seed, step), out in sorted(outs.items()):
        gap, share, total = shares(out)
        states[f's{seed}_t{step}'] = {
            'seed': seed, 'step': step, 'gap': gap, 'shares': share, 'totals': total, 'power_law': power_law(out),
            'evaluation_decrease': {r: v['evaluation_decrease'] for r, v in out['transfer'].items()},
            'interior': {r: v['interior'] for r, v in out['transfer'].items()},
            'timing': out['timing'], 'seconds': out['seconds'], 'peak_cuda_bytes': out.get('peak_cuda_bytes'),
            'ladder': ladder_consistency(seed, step, out)}
    early = [v for v in states.values() if v['step'] in EARLY]
    complete = len(early) == len(SEEDS)*len(EARLY)
    readings = {'complete': complete, 'missing': missing}
    if complete:
        readings['P1'] = all(v['power_law']['fraction_r2_all'] >= SHARE_P for v in early)
        readings['P2'] = all(v['power_law']['fraction_alpha_close'] >= SHARE_P for v in early)
        readings['K1'] = all(v['shares']['diag_powerlaw'] is not None and v['shares']['diag_powerlaw'] < K1_MAX for v in early)
        readings['K2'] = next((m for m in KRYLOV if all((v['shares'][f'deflated_krylov_{m}'] or 0.) >= HALF for v in early)), None)
        readings['K3'] = all((v['shares']['deflated_chebyshev_8'] or 0.) >= HALF for v in early)
        readings['K4'] = all(v['evaluation_decrease']['full_cg_sub'] > v['evaluation_decrease']['topfree'] for v in early)
    rungs = list(next(iter(states.values()))['shares']) if states else []
    readings['K5'] = {r: {'median_share_early': median(v['shares'][r] for v in early),
                          'min_share_early': min((v['shares'][r] for v in early if v['shares'][r] is not None), default=None),
                          'median_evaluation_decrease_early': median(v['evaluation_decrease'][r] for v in early),
                          'wins_vs_topfree_early': sum(v['evaluation_decrease'][r] > v['evaluation_decrease']['topfree'] for v in early)}
                      for r in rungs}
    readings['P3'] = by_kind(outs)
    readings['timing'] = {key: median(v['timing'][key] for v in states.values())
                          for key in ['full_cg_sub_seconds_per_product', 'full_cg_full_seconds_per_product']}
    summary = {'readings': readings, 'states': states}
    (RESULTS/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    lines = ['# bulk_krylov_20261014 summary (generated by scripts/summarize.py)', '',
             f"Complete: {complete}; missing: {missing}", '']
    for code in ['P1', 'P2', 'K1', 'K2', 'K3', 'K4']:
        if code in readings:
            lines.append(f'- {code}: {readings[code]}')
    lines += ['', '| state | G | ' + ' | '.join(rungs) + ' |', '| --- | --- | ' + ' | '.join('---' for _ in rungs) + ' |']
    for key, v in states.items():
        lines.append(f"| {key} | {v['gap']:.3g} | " + ' | '.join('' if v['shares'][r] is None else f"{v['shares'][r]:.3f}" for r in rungs) + ' |')
    lines += ['', '| state | ' + ' | '.join(rungs) + ' |', '| --- | ' + ' | '.join('---' for _ in rungs) + ' |']
    for key, v in states.items():
        lines.append(f'| {key} | ' + ' | '.join(f"{v['evaluation_decrease'][r]:.4f}" for r in rungs) + ' |')
    lines += ['', '| state | P1 fraction | P2 fraction | median R² | α_top | α_bulk | α_trace | α_slope | max Ritz residual | ladder bitwise |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, v in states.items():
        p = v['power_law']
        cells = [p['fraction_r2_all'], p['fraction_alpha_close'], p['median_r2_all'], p['median_alpha_top'], p['median_alpha_bulk'],
                 p['median_alpha_trace'], p['median_alpha_slope'], p['max_ritz_residual_last']]
        lines.append(f'| {key} | ' + ' | '.join('' if c is None else f'{c:.3g}' for c in cells)
                     + f" | {None if v['ladder'] is None else v['ladder']['bitwise']} |")
    (RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    sys.exit(main())
