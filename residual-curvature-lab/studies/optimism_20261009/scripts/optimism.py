"""Signal/optimism test of the transfer share on the six banked states (PROTOCOL.md). No training.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): optimism.py
Writes results/optimism.json and results/optimism_tables.md in this study folder.
"""
import copy
import json
import statistics
import sys
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
LAB = STUDY.parents[1]
sys.path.insert(0, str(LAB/'scripts'))

from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.curvature.layerwise import solve_blocks  # noqa: E402
from residual_curvature.curvature.operators import GGN, loss  # noqa: E402
from residual_curvature.curvature.solves import relative_error  # noqa: E402
from residual_curvature.data import dataset  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry, make_model  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402
from residual_curvature.optim.soap import SOAPMetric, frozen_apply  # noqa: E402
from run_large_batch_oracle import measurement, state_equal  # noqa: E402
from run_mini_vit_allocation import search_candidate  # noqa: E402

PARENT = Path('runs/20261005T143029-9b582ed5')
CHECK2C = Path.home()/'skipsoap-study-data'/'gap_audit_20261007'/'runs'/'20261007T123608-7f39721a'
STATES = [(t, s) for t in ['searched_soap', 'exact_tensor_ggn'] for s in [10, 25, 50]]
OPERATORS = ['sgd', 'soap_map', 'exact_ggn', 'ef_newton']
CHECK2C_NAMES = {'soap_map': 'identity', 'exact_ggn': 'exact'}


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


def analyse_state(trajectory, step, c, x, y):
    source = PARENT/f'{trajectory}-{step:04d}'
    arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
    info = json.loads((source/'diagnostics.json').read_text())
    flat, g, momentum, contexts = [arrays[k] for k in ['parameters', 'gradient', 'momentum_rhs', 'contexts']]
    ggn, ef, native = arrays['ggn_blocks'], arrays['empirical_fisher_blocks'], arrays['raw_soap']
    model = make_model(c)
    registry = Registry(model)
    commit(model, registry, flat)
    model_before, contexts_before, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(contexts), torch.get_rng_state().clone()
    construction, panel_ids = info['construction_ids'], info['panel_ids']
    batch = len(construction)
    fn = functional_logits(model, registry, x[construction])
    objective = lambda w: loss(fn(w), y[construction])
    op = GGN(fn, flat, y[construction])
    tolerance = c['solver']['tolerance64']
    checks = {'gradient_error': relative_error(op.gradient, g)}
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in ggn)/registry.size, c['damping']['floor'])
    mu_ef = max(.01*sum(float(b.trace()) for b in ef)/registry.size, c['damping']['floor'])
    omega = [metric.scale*frozen_apply(torch.eye(s.count).reshape(s.shape+(s.count,)), contexts[s.name], inverse=True).reshape(s.count, s.count)
             for s in registry.specs]
    operators = {'sgd': [torch.eye(s.count) for s in registry.specs], 'soap_map': omega,
                 'exact_ggn': [b+mu*o for b, o in zip(ggn, omega)], 'ef_newton': [b+mu_ef*o for b, o in zip(ef, omega)]}
    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    panel_gradients = [GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient for ids in panel_ids]
    mean_panel = torch.stack(panel_gradients).mean(0)
    sigma = [batch/(batch-1)*(f-g[s.slice, None]@g[None, s.slice]) for s, f in zip(registry.specs, ef)]
    checks['sigma_min_eigenvalue_relative'] = min(float(torch.linalg.eigvalsh((m+m.T)/2).min())/max(float(torch.linalg.eigvalsh((m+m.T)/2).max()), 1e-300)
                                                  for m in sigma)

    def inverse(blocks, v):
        out = torch.empty_like(v)
        for s, a in zip(registry.specs, blocks):
            out[s.slice] = torch.linalg.solve(a, v[s.slice])
        return out

    theory = {}
    for name, blocks in operators.items():
        optimism = sum(float(torch.linalg.solve(a, m).diagonal().sum()) for a, m in zip(blocks, sigma))/batch
        cross = [float(panel_gradients[p]@inverse(blocks, panel_gradients[q])) for p in range(4) for q in range(p+1, 4)]
        signal = statistics.mean(cross)
        signal_batch = float(g@inverse(blocks, g))-optimism
        r = optimism/signal if signal > 0 else float('inf')
        theory[name] = {'optimism': optimism, 'signal': signal, 'signal_from_batch': signal_batch, 'r': r,
                        'noise_scale_B_r': batch*r, 'predicted_search_share': (1-r)/(1+r) if signal > 0 else -1.,
                        'predicted_fixed_share': 1/(1+r) if signal > 0 else 0.}
    directions = {}
    for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
        directions[('sgd', rhs_name)] = -rhs
        for name, blocks, damping in [('soap_map', [torch.zeros_like(b) for b in ggn], mu), ('exact_ggn', ggn, mu), ('ef_newton', ef, mu_ef)]:
            values, solve = solve_blocks(blocks, registry, rhs, metric, [damping], tolerance)
            directions[(name, rhs_name)] = values[damping]
    stored = {(r['name'], r['rhs']): r['selected'] for r in json.loads((CHECK2C/f'{trajectory}-{step:04d}'/'candidates.json').read_text())}
    rows = []
    for (name, rhs_name), delta in directions.items():
        selected, search = search_candidate(flat, delta, native, objective)
        values = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
        own, held = values['actual_decrease'], values['independent_mean_decrease']
        realized = float(mean_panel@selected)/float(g@selected) if float(g@selected) != 0 else float('nan')
        row = {'state': f'{trajectory}-{step}', 'operator': name, 'rhs': rhs_name, 'own_decrease': own, 'held_out_decrease': held,
               'measured_share': held/own if own != 0 else float('nan'), 'realized_rho': realized,
               'realized_search_share': 2*realized-1, 'selected_alpha': search.get('selected_alpha'),
               'bracketed': search.get('bracketed'), 'rejected': search.get('rejected')}
        if name in CHECK2C_NAMES:
            reference = stored[(CHECK2C_NAMES[name], rhs_name)]
            row['gate_difference'] = max(abs(own-reference['actual_decrease']), abs(held-reference['independent_mean_decrease']))
            checks[f'gate_{name}_{rhs_name}'] = row['gate_difference']
        rows.append(row)
    if not (state_equal(model_before, model.state_dict()) and state_equal(contexts_before, contexts) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('State mutated')
    if checks['gradient_error'] > 1e-12 or max(v for k, v in checks.items() if k.startswith('gate_')) > 1e-12:
        raise RuntimeError(f'Gate failed: {checks}')
    return {'state': f'{trajectory}-{step}', 'mu': mu, 'mu_ef': mu_ef, 'theory': theory, 'rows': rows, 'checks': checks}


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, _ = dataset(c)
    states = []
    for trajectory, step in STATES:
        states.append(analyse_state(trajectory, step, c, x, y))
        print(f"{trajectory}-{step}: " + ', '.join(f"{n} r={states[-1]['theory'][n]['r']:.2f}" for n in OPERATORS), flush=True)
    points = [(s['theory'][r['operator']], r) for s in states for r in s['rows'] if r['rhs'] == 'gradient']
    predicted, measured = [t['predicted_search_share'] for t, _ in points], [r['measured_share'] for _, r in points]
    order_r = sum(all(s['theory'][a]['r'] > s['theory'][b]['r'] for a, b in zip(OPERATORS[::-1], OPERATORS[::-1][1:])) for s in states)
    def share(s, name):
        return next(r['measured_share'] for r in s['rows'] if r['operator'] == name and r['rhs'] == 'gradient')
    order_share = sum(all(share(s, a) < share(s, b) for a, b in zip(OPERATORS[::-1], OPERATORS[::-1][1:])) for s in states)
    above = [r for t, r in points if t['r'] > 1]
    momentum = [r for s in states for r in s['rows'] if r['rhs'] == 'momentum']
    readings = {
        'calibration_spearman': spearman(predicted, measured),
        'calibration_mae': statistics.mean(abs(a-b) for a, b in zip(predicted, measured)),
        'ordering_r_states': order_r, 'ordering_share_states': order_share,
        'sign_points_r_above_1': len(above), 'sign_negative_held_out': sum(r['held_out_decrease'] < 0 for r in above),
        'mechanism_spearman_gradient': spearman([r['realized_search_share'] for _, r in points], measured),
        'mechanism_mae_gradient': statistics.mean(abs(r['realized_search_share']-r['measured_share']) for _, r in points),
        'mechanism_spearman_momentum': spearman([r['realized_search_share'] for r in momentum], [r['measured_share'] for r in momentum]),
    }
    s, m = readings['calibration_spearman'], readings['calibration_mae']
    readings['calibration_verdict'] = 'supported' if s >= .6 and m <= .2 else 'not supported' if s < .3 else 'partial'
    out = STUDY/'results'
    out.mkdir(exist_ok=True)
    (out/'optimism.json').write_text(json.dumps({'readings': readings, 'states': states}, indent=1, default=str)+'\n')
    lines = ['| State | Operator | r | B·r | Predicted share | Realized 2ρ−1 | Measured share | Own | Held-out |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for st in states:
        for r in st['rows']:
            if r['rhs'] != 'gradient':
                continue
            t = st['theory'][r['operator']]
            lines.append(f"| {st['state']} | {r['operator']} | {t['r']:.2f} | {t['noise_scale_B_r']:.0f} | {t['predicted_search_share']:+.2f} | "
                         f"{r['realized_search_share']:+.2f} | {r['measured_share']:+.2f} | {r['own_decrease']:.4f} | {r['held_out_decrease']:+.4f} |")
    lines += ['', '| State | Operator (momentum RHS) | Realized 2ρ−1 | Measured share | Own | Held-out |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for st in states:
        for r in st['rows']:
            if r['rhs'] == 'momentum':
                lines.append(f"| {st['state']} | {r['operator']} | {r['realized_search_share']:+.2f} | {r['measured_share']:+.2f} | {r['own_decrease']:.4f} | {r['held_out_decrease']:+.4f} |")
    (out/'optimism_tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(readings, indent=1))
    print('\n'.join(lines))
    print('checks:', {st['state']: {k: (f'{v:.1e}' if isinstance(v, float) else v) for k, v in st['checks'].items()} for st in states})


if __name__ == '__main__':
    main()
