"""Readings A1-A3 and B1-B2 (PROTOCOL.md section 5) and tables. Usage: summarize.py (after corpus.py x6 and steps.py)"""
import json
import math
import statistics
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY/'results'
STATES = [f'{t}-{s}' for t in ['searched_soap', 'exact_tensor_ggn'] for s in [10, 25, 50]]
CORPUS_OPS = ['sgd', 'soap_map', 'exact_ggn_N']
STEP_OPS = ['sgd', 'soap_map', 'exact_ggn', 'parsec_l']


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.]*len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    return cov/(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**.5


def calibration(pairs):
    pairs = [(p, m) for p, m in pairs if m is not None]
    rho, mae = spearman([p for p, _ in pairs], [m for _, m in pairs]), statistics.mean(abs(p-m) for p, m in pairs)
    verdict = 'supported' if rho >= .8 and mae <= .15 else 'not supported' if rho < .5 else 'partial'
    return {'spearman': rho, 'mae': mae, 'points': len(pairs), 'verdict': verdict}


def slope(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((a-mx)*(b-my) for a, b in zip(xs, ys))/sum((a-mx)**2 for a in xs)


def part_a():
    data = {s: json.loads((RESULTS/f'corpus_{s}.json').read_text()) for s in STATES}
    points = [(s, op, int(size), e) for s, d in data.items() for op in CORPUS_OPS for size, e in d['summary'][op].items()]
    a1 = calibration([(e['predicted']['corpus'], e['share_corpus']) for *_, e in points])
    a3 = calibration([(e['predicted']['validation_conditional'], e['share_validation']) for *_, e in points])
    a3_population = calibration([(e['predicted']['validation_population'], e['share_validation']) for *_, e in points])
    slopes, ratios, used = {}, [], 0
    inside = lambda v: v is not None and -.9 < v < .9
    for s in STATES:
        for op in CORPUS_OPS:
            valid = [(e['predicted']['r_train'], (1-e['share_corpus'])/(1+e['share_corpus'])) for t, o, _, e in points
                     if t == s and o == op and inside(e['share_corpus']) and inside(e['predicted']['corpus'])]
            used += len(valid)
            ratios += [m/p for p, m in valid]
            if len(valid) >= 3:
                slopes[f'{s}|{op}'] = slope([math.log(p) for p, _ in valid], [math.log(m) for _, m in valid])
    median_slope = statistics.median(slopes.values()) if slopes else None
    median_ratio = statistics.median(ratios) if ratios else None
    ok_slope = median_slope is not None and .8 <= median_slope <= 1.25
    ok_ratio = median_ratio is not None and .5 <= median_ratio <= 2
    a2 = {'pairs_with_3_points': len(slopes), 'points': used, 'median_slope': median_slope, 'median_ratio': median_ratio,
          'slopes': slopes, 'verdict': 'holds' if ok_slope and ok_ratio else 'partial' if ok_slope or ok_ratio else 'fails'}
    mechanism = calibration([(2*e['gamma_realized']-1, e['share_corpus']) for *_, e in points])
    theory = {s: {op: d['theory'][op] for op in CORPUS_OPS} for s, d in data.items()}
    table = {s: {op: {int(size): {'pred': e['predicted']['corpus'], 'corpus': e['share_corpus'], 'val_cond': e['predicted']['validation_conditional'],
                                  'val_pop': e['predicted']['validation_population'], 'validation': e['share_validation'],
                                  'gamma_share': 2*e['gamma_realized']-1, 'boundary': e['boundary_excluded'], 'rejected': e['rejected']}
                      for size, e in d['summary'][op].items()} for op in CORPUS_OPS} for s, d in data.items()}
    checks = {s: d['checks'] for s, d in data.items()}
    return {'A1_corpus_calibration': a1, 'A2_one_over_B': a2, 'A3_new_data_calibration': a3,
            'reported': {'validation_population_form': a3_population, 'realized_mechanism_corpus': mechanism},
            'theory': theory, 'table': table, 'checks': checks}


def part_b():
    data = json.loads((RESULTS/'steps.json').read_text())['states']
    held = {(r['state'], r['operator'], r['policy']): r for s in data for r in s['rows']}
    mean_over = lambda policy, ops: statistics.mean(held[(s, op, policy)]['held_234'] for s in STATES for op in ops)
    second24, second_ggn = mean_over('second', STEP_OPS), mean_over('second', ['exact_ggn'])
    b1 = {'gamma_x': mean_over('gamma_x', STEP_OPS), 'second': second24}
    b1['ratio'] = b1['gamma_x']/second24
    b1['verdict'] = 'passes' if b1['ratio'] >= .9 else 'fails'
    b2 = {'wiener': mean_over('wiener', ['exact_ggn']), 'second': second_ggn}
    b2['ratio'] = b2['wiener']/second_ggn
    b2['identity_differences'] = {s['state']: s['checks']['wiener_identity_vs_exact_ggn'] for s in data}
    b2['residuals'] = {s['state']: {k: s['checks'][k] for k in ['residual_newton_eigenbasis', 'residual_exact_ggn_reference']} for s in data}
    # DEVIATIONS #1: a failed Wiener gate voids B2; the ratio is kept only as a labelled diagnostic.
    if any(v > 1e-8 for v in b2['identity_differences'].values()):
        b2['verdict'] = 'not evaluated (gate failed)'
    else:
        b2['verdict'] = 'passes' if b2['ratio'] >= .9 else 'fails'
    policies = ['same', 'second', 'oracle', 'gamma_x', 'gamma_b', 'slope_x']
    per_operator = {op: {p: mean_over(p, [op]) for p in policies} for op in STEP_OPS}
    per_operator['exact_ggn'].update({p: mean_over(p, ['exact_ggn']) for p in ['newton_full', 'wiener', 'wiener_x', 'wiener_search']})
    wins = {p: sum(held[(s, op, p)]['held_234'] > held[(s, op, 'second')]['held_234'] for s in STATES for op in STEP_OPS)
            for p in ['gamma_x', 'gamma_b', 'slope_x']}
    wins.update({p: sum(held[(s, 'exact_ggn', p)]['held_234'] > held[(s, 'exact_ggn', 'second')]['held_234'] for s in STATES)
                 for p in ['newton_full', 'wiener', 'wiener_x', 'wiener_search']})
    curvature = {s: {'batch_curvature_share': held[(s, 'exact_ggn', 'same')]['share_234'], 'panel1_curvature_share': held[(s, 'exact_ggn_xc', 'same')]['share_234'],
                     'batch_curvature_held': held[(s, 'exact_ggn', 'same')]['held_234'], 'panel1_curvature_held': held[(s, 'exact_ggn_xc', 'same')]['held_234']}
                 for s in STATES}
    alphas = {s: {op: {p: held[(s, op, p)]['alpha_raw'] for p in ['same', 'second', 'oracle', 'gamma_x', 'gamma_b', 'slope_x']} for op in STEP_OPS} for s in STATES}
    stats = {s: {op: {k: held[(s, op, 'gamma_x')][k] for k in ['q', 'S_x', 'O_hat', 'kappa_1', 'kappa_b', 'gamma_hat']} for op in STEP_OPS} for s in STATES}
    shrink = {s: {k: held[(s, 'exact_ggn', 'wiener')][k] for k in ['phi_energy_weighted', 'phi_x_energy_weighted']} for s in STATES}
    per_state = {s: {op: {p: held[(s, op, p)]['held_234'] for p in policies if (s, op, p) in held} for op in STEP_OPS} for s in STATES}
    for s in STATES:
        per_state[s]['exact_ggn'].update({p: held[(s, 'exact_ggn', p)]['held_234'] for p in ['newton_full', 'wiener', 'wiener_x', 'wiener_search']})
    checks = {s['state']: s['checks'] for s in data}
    return {'B1_gamma_x': b1, 'B2_wiener': b2, 'per_operator_mean_held_234': per_operator, 'wins_vs_second': wins,
            'curvature_optimism': curvature, 'alpha_raw': alphas, 'statistics': stats, 'wiener_shrink': shrink,
            'per_state_held_234': per_state, 'checks': checks}


def main():
    out = {'part_a': part_a(), 'part_b': part_b()}
    (RESULTS/'summary.json').write_text(json.dumps(out, indent=1, default=str)+'\n')
    a, b = out['part_a'], out['part_b']
    print(json.dumps({k: a[k] for k in ['A1_corpus_calibration', 'A3_new_data_calibration']}, indent=1))
    print(json.dumps({k: v for k, v in a['A2_one_over_B'].items() if k != 'slopes'}, indent=1), json.dumps(a['reported'], indent=1))
    print(json.dumps({k: b[k] for k in ['B1_gamma_x', 'B2_wiener', 'wins_vs_second']}, indent=1))
    print('per operator mean held-out (panels 2-4):')
    for op, v in b['per_operator_mean_held_234'].items():
        print(f'  {op:10s}', ' '.join(f'{p} {x:+.4f}' for p, x in v.items()))
    print('noise scale B_noise / f* per state (sgd, soap_map, exact_ggn_N):')
    for s, t in a['theory'].items():
        print(f'  {s:20s}', ' | '.join(f"{t[op]['noise_scale']:7.0f} / {t[op]['f_star']:.3f}" for op in CORPUS_OPS))
    print('corpus share measured (predicted) by size:')
    for s, ops in a['table'].items():
        for op, sizes in ops.items():
            print(f'  {s:20s} {op:11s}', ' '.join(f"{v['corpus']:+.2f}({v['pred']:+.2f})" for v in sizes.values()))
    print('validation share measured (conditional / population) by size:')
    for s, ops in a['table'].items():
        for op, sizes in ops.items():
            print(f'  {s:20s} {op:11s}', ' '.join(f"{v['validation']:+.2f}({v['val_cond']:+.2f}/{v['val_pop']:+.2f})" for v in sizes.values()))
    print('curvature optimism (exact GGN, share on panels 2-4: batch curvature vs panel-1 curvature):')
    for s, v in b['curvature_optimism'].items():
        print(f"  {s:20s} {v['batch_curvature_share']:+.2f} vs {v['panel1_curvature_share']:+.2f} | held {v['batch_curvature_held']:+.4f} vs {v['panel1_curvature_held']:+.4f}")


if __name__ == '__main__':
    main()
