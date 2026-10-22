"""Readings N1-N3, F1, F2, P1 and descriptive tables (PROTOCOL.md, section 4) from results/noise/*.json.

Usage (from residual-curvature-lab/): summarize.py
Writes studies/noise_scale_20261021/results/summary.json and summary.md. All forms are summed over the matrix tensors
(block-diagonal model). Noise forms from partition batches of b = 1,024 are scaled to batch B by
(1/B − 1/N)/(1/b − 1/N).
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY/'results'
ARMS, SEEDS, STATES = ['topfree', 'soap'], [1, 2, 3], [10, 25, 50, 100, 200, 400]
EARLY, LATE = [10, 25, 50], [100, 200, 400]
N, b, B = 45000, 1024, 2048
POWERS = [round(-1+0.05*i, 2) for i in range(41)]
GRID = [0., .25, .5, .75, 1.]
FINE = [round(0.05*i, 2) for i in range(21)]
NEED = 24


def index(q):
    return int(round((q+1)/0.05))


def pooled(state):
    tensors = state['tensors'].values()
    corpus = [sum(t['corpus'][i] for t in tensors) for i in range(len(POWERS))]
    noise = [sum(t['noise_mean'][i] for t in tensors) for i in range(len(POWERS))]
    return corpus, noise


def theory(state, batch=B):
    """Φ_ν, K_ν (examples), N_ν(batch), Ψ_ν(batch) for ν on FINE (keys are ν)."""
    corpus, noise = pooled(state)
    factor = (1/batch-1/N)/(1/b-1/N)
    out = {}
    for nu in FINE:
        z_neg, z_metric = corpus[index(-nu)], corpus[index(1-2*nu)]
        noise_metric = noise[index(1-2*nu)]
        phi = z_neg**2/(2*z_metric)
        ratio = factor*noise_metric/z_metric
        out[nu] = {'phi': phi, 'K': noise_metric/(z_metric*(1/b-1/N)), 'N': ratio, 'psi': phi/(1+ratio)}
    return out


def crossings(th):
    pairs = {}
    for lo, hi in zip(GRID, GRID[1:]):
        g = th[hi]['phi']/th[lo]['phi']
        denominator = th[hi]['K']-g*th[lo]['K']
        crit = 1/(1/N+(g-1)/denominator) if denominator > 0 else 0.
        pairs[f'{lo}->{hi}'] = {'G': g, 'B_crit': crit}
    return pairs, max(p['B_crit'] for p in pairs.values())


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.]*len(v)
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]:
                j += 1
            for k in range(i, j+1):
                r[order[k]] = (i+j)/2
            i = j+1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    cov = sum((a-mx)*(c-my) for a, c in zip(rx, ry))
    return cov/((sum((a-mx)**2 for a in rx)*sum((c-my)**2 for c in ry))**.5)


def main():
    outs = {(a, s): json.loads((RESULTS/'noise'/f'{a}_s{s}.json').read_text()) for a in ARMS for s in SEEDS
            if (RESULTS/'noise'/f'{a}_s{s}.json').exists()}
    complete = len(outs) == len(ARMS)*len(SEEDS) and all(len(o['states']) == len(STATES) for o in outs.values())
    cases = {}
    for (a, s), out in outs.items():
        for t in STATES:
            state = out['states'].get(str(t))
            if state is None:
                continue
            th = theory(state)
            pairs, required = crossings(th)
            evaluation = {nu: state['score']['steps'][str(nu)]['evaluation'] for nu in GRID}
            best = max(evaluation, key=evaluation.get)
            cases[f'{a}_s{s}_t{t}'] = {
                'arm': a, 'seed': s, 'step': t, 'theory': {str(k): v for k, v in th.items()}, 'pairs': pairs,
                'B_required': required, 'theory_best_grid': max(GRID, key=lambda nu: th[nu]['psi']),
                'theory_best_fine': max(FINE, key=lambda nu: th[nu]['psi']), 'evaluation': evaluation, 'best': best,
                'validation': {nu: state['score']['steps'][str(nu)]['validation'] for nu in GRID},
                'interior': {nu: state['score']['steps'][str(nu)]['interior'] for nu in GRID},
                'multiplier': {nu: state['score']['steps'][str(nu)]['multiplier'] for nu in GRID},
                'g2': state['g2'], 'delta': state['delta'],
                'min_ritz_over_delta': min(x['min_ritz_over_delta'] for x in state['tensors'].values())}
    readings = {'complete': complete}
    if complete:
        c = cases.values()
        readings['N1'] = all(x['theory']['1.0']['K'] > x['theory']['0.5']['K'] > x['theory']['0.0']['K'] for x in c)
        rho = {f'{a}_s{s}': spearman(STATES, [cases[f'{a}_s{s}_t{t}']['theory']['1.0']['K'] for t in STATES])
               for a in ARMS for s in SEEDS}
        readings['N2_spearman'] = rho
        readings['N2'] = all(v >= .8 for v in rho.values())
        breaks = {}
        n3 = True
        for a in ARMS:
            for s in SEEDS:
                req = [cases[f'{a}_s{s}_t{t}']['B_required'] for t in STATES]
                n3 &= req[0] < B <= req[-1]
                breaks[f'{a}_s{s}'] = next((t for t, r in zip(STATES, req) if r >= B), None)
        readings['N3'] = n3
        readings['t_break'] = breaks
        interior = sum(x['best'] in (.25, .5, .75) for x in c)
        readings['F1_count'] = interior
        readings['F1'] = interior >= NEED
        f2 = {}
        for a in ARMS:
            early = [cases[f'{a}_s{s}_t{t}']['best'] for s in SEEDS for t in EARLY]
            late = [cases[f'{a}_s{s}_t{t}']['best'] for s in SEEDS for t in LATE]
            f2[a] = {'early_mean': sum(early)/len(early), 'late_mean': sum(late)/len(late)}
        readings['F2_means'] = f2
        readings['F2'] = all(v['early_mean'] > v['late_mean'] for v in f2.values())
        close = sum(abs(GRID.index(x['theory_best_grid'])-GRID.index(x['best'])) <= 1 for x in c)
        readings['P1_count'] = close
        readings['P1'] = close >= NEED
    summary = {'readings': readings, 'cases': {k: {**v, 'evaluation': {str(n): e for n, e in v['evaluation'].items()},
                                                  'validation': {str(n): e for n, e in v['validation'].items()},
                                                  'interior': {str(n): e for n, e in v['interior'].items()},
                                                  'multiplier': {str(n): e for n, e in v['multiplier'].items()}}
                                              for k, v in cases.items()}}
    (RESULTS/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    f = lambda v, d=3: '' if v is None else (f'{v:.{d}g}' if isinstance(v, float) else str(v))
    lines = ['# noise_scale_20261021 summary (generated by scripts/summarize.py)', '', f'Complete: {complete}', '']
    for k, v in readings.items():
        lines.append(f'- {k}: {v}')
    lines += ['', '| case | K_0 | K_½ | K_1 | N_1(2048) | B_req | theory ν* (grid / fine) | best ν (eval) | eval decrease ν=0 / ½ / 1 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for key, x in cases.items():
        th = x['theory']
        lines.append(f"| {key} | {f(th['0.0']['K'])} | {f(th['0.5']['K'])} | {f(th['1.0']['K'])} | {f(th['1.0']['N'])} | "
                     f"{f(x['B_required'])} | {x['theory_best_grid']} / {x['theory_best_fine']} | {x['best']} | "
                     f"{f(x['evaluation'][0.0])} / {f(x['evaluation'][0.5])} / {f(x['evaluation'][1.0])} |")
    (RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    sys.exit(main())
