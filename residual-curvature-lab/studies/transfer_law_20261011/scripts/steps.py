"""Part B of PROTOCOL.md: step sizes computed from statistics only, against second-batch selection. No training.

Directions and reference policies are built exactly as in studies/optimism_20261009/scripts/followup.py and must
reproduce its results/followup.json. Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): steps.py
Writes results/steps.json.
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
from residual_curvature.curvature.layerwise import solve_blocks, streamed_blocks  # noqa: E402
from residual_curvature.curvature.operators import GGN, loss  # noqa: E402
from residual_curvature.curvature.solves import relative_error  # noqa: E402
from residual_curvature.data import dataset  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry, make_model  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402
from residual_curvature.optim import capacity  # noqa: E402
from residual_curvature.optim.soap import SOAPMetric, frozen_apply  # noqa: E402
from run_large_batch_oracle import measurement, state_equal  # noqa: E402
from run_mini_vit_allocation import search_candidate  # noqa: E402

PARENT = Path('runs/20261005T143029-9b582ed5')
FOLLOWUP = Path('studies/optimism_20261009/results/followup.json')
STATES = [(t, s) for t in ['searched_soap', 'exact_tensor_ggn'] for s in [10, 25, 50]]
OPERATORS = ['sgd', 'soap_map', 'exact_ggn', 'parsec_l']
RANK = 16


def bounded_search(flat, delta, native, objective):
    """search_candidate for reported-only policies: a grid-boundary selection gives no step (DEVIATIONS #2)."""
    try:
        return search_candidate(flat, delta, native, objective)
    except RuntimeError as error:
        if 'boundary' not in str(error):
            raise
        return torch.zeros_like(delta), {'selected_alpha': None, 'boundary': True}


def triangular(l, v, upper=False):
    return torch.linalg.solve_triangular(l, v, upper=upper)


def state_steps(trajectory, step, c, x, y, reference):
    name = f'{trajectory}-{step}'
    source = PARENT/f'{trajectory}-{step:04d}'
    arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
    info = json.loads((source/'diagnostics.json').read_text())
    flat, g, contexts = [arrays[k] for k in ['parameters', 'gradient', 'contexts']]
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
    checks = {'gradient_error': relative_error(op.gradient, g)}
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in ggn)/registry.size, c['damping']['floor'])
    beta = 1/(mu*metric.scale)
    eye = lambda s: torch.eye(s.count).reshape(s.shape+(s.count,))
    omega = [metric.scale*frozen_apply(eye(s), contexts[s.name], inverse=True).reshape(s.count, s.count) for s in registry.specs]
    factors = [capacity.factor(eye(s), contexts[s.name]).reshape(s.count, s.count) for s in registry.specs]
    # As followup.py: explicit A^-1 per tensor for the statistics; Cholesky solves (check 2c) for SOAP-map and GGN directions.
    inverses = {'sgd': [torch.eye(s.count) for s in registry.specs],
                'soap_map': [torch.linalg.inv(o) for o in omega],
                'exact_ggn': [torch.linalg.inv(b+mu*o) for b, o in zip(ggn, omega)]}
    parsec = []
    for f, block in zip(factors, ggn):
        eig, q = torch.linalg.eigh(beta*f.T@block@f)
        r = min(RANK, len(eig))
        w, kappa = q[:, -r:], eig[-r:].clamp_min(0)
        parsec.append(f@(torch.eye(len(eig))-w@torch.diag(kappa/(1+kappa))@w.T)@f.T)
    inverses['parsec_l'] = parsec

    def apply(op_name, v):
        out = torch.empty_like(v)
        for s, a in zip(registry.specs, inverses[op_name]):
            out[s.slice] = a@v[s.slice]
        return out

    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    g1 = GGN(functional_logits(model, registry, x[panel_ids[0]]), flat, y[panel_ids[0]]).gradient
    op1 = GGN(functional_logits(model, registry, x[panel_ids[0]]), flat, y[panel_ids[0]])
    sigma = [batch/(batch-1)*(f-g[s.slice, None]@g[None, s.slice]) for s, f in zip(registry.specs, ef)]
    panel1, panels234 = panels[0], (lambda w: statistics.mean(float(f(w)) for f in panels[1:]))
    rows = []

    def measure(op_name, policy, selected, raw_direction, extra=None):
        values = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
        own, held = values['actual_decrease'], values['independent_decreases']
        row = {'state': name, 'operator': op_name, 'policy': policy, 'own': own, 'held_all': statistics.mean(held),
               'held_234': statistics.mean(held[1:]), 'norm': float(selected.norm()),
               'alpha_raw': float(selected@raw_direction)/float(raw_direction@raw_direction)}
        row['share_234'] = row['held_234']/own if own > 0 else None
        row.update(extra or {})
        rows.append(row)
        return row

    solved = {'soap_map': ([torch.zeros_like(b) for b in ggn], mu), 'exact_ggn': (ggn, mu)}
    directions = {}
    for op_name in OPERATORS:
        if op_name in solved:
            blocks, damping = solved[op_name]
            delta = solve_blocks(blocks, registry, g, metric, [damping], c['solver']['tolerance64'])[0][damping]
        else:
            delta = -apply(op_name, g)
        directions[op_name] = delta
        for policy, target in [('same', objective), ('second', lambda w: panel1(w)), ('oracle', panels234)]:
            selected, search = search_candidate(flat, delta, native, target)
            row = measure(op_name, policy, selected, delta, {'alpha_search': search.get('selected_alpha')})
            ref = reference[(name, op_name, policy)]
            checks[f'gate_{op_name}_{policy}'] = max(abs(row['own']-ref['own']), abs(row['held_234']-ref['held_234']))
        # Statistics-only step sizes (no loss evaluations).
        q = -float(g@delta)
        s_x = -float(g1@delta)
        o_hat = sum(float((a@m).diagonal().sum()) for a, m in zip(inverses[op_name], sigma))/batch
        scale = q/float(g@apply(op_name, g))  # DEVIATIONS #3: the direction's A (1/mu for soap_map, 1 otherwise)
        o_hat *= scale
        kappa_1, kappa_b = float(delta@op1.action(delta)), float(delta@op.action(delta))
        gamma = s_x/(s_x+o_hat) if s_x > 0 else 0.
        stats = {'q': q, 'S_x': s_x, 'O_hat': o_hat, 'kappa_1': kappa_1, 'kappa_b': kappa_b, 'gamma_hat': gamma, 'operator_scale': scale}
        for policy, alpha in [('gamma_x', gamma*q/kappa_1 if kappa_1 > 0 else 0.),
                              ('gamma_b', max(q-o_hat, 0.)/kappa_b if kappa_b > 0 else 0.),
                              ('slope_x', max(s_x, 0.)/kappa_1 if kappa_1 > 0 else 0.)]:
            measure(op_name, policy, alpha*delta, delta, dict(stats, alpha_rule=alpha))

    # Wiener step for exact GGN: eigenbasis of the metric-whitened batch GGN, Omega_b = L L^T, K = L^-1 C L^-T.
    wiener, wiener_x, newton = (torch.empty_like(g) for _ in range(3))
    shrink_mean, shrink_x_mean, weights = [], [], []
    for s, block, o, sig in zip(registry.specs, ggn, omega, sigma):
        l = torch.linalg.cholesky((o+o.T)/2)
        k = triangular(l, triangular(l, block).T)
        lam, w = torch.linalg.eigh((k+k.T)/2)
        h = w.T@triangular(l, g[s.slice, None])[:, 0]
        h1 = w.T@triangular(l, g1[s.slice, None])[:, 0]
        m = triangular(l, triangular(l, sig).T)
        noise = (w.T@((m+m.T)/2)@w).diagonal()/batch
        signal = (h.square()-noise).clamp_min(0)
        signal_x = (h*h1).clamp_min(0)
        phi = torch.where(signal+noise > 0, signal/(signal+noise), torch.zeros_like(h))
        phi_x = torch.where(signal_x+noise > 0, signal_x/(signal_x+noise), torch.zeros_like(h))
        coefficient = h/(lam+mu)
        back = lambda z: triangular(l.T, (w@z)[:, None], upper=True)[:, 0]
        wiener[s.slice], wiener_x[s.slice], newton[s.slice] = -back(phi*coefficient), -back(phi_x*coefficient), -back(coefficient)
        shrink_mean.append(float((phi*coefficient.square()).sum()))
        shrink_x_mean.append(float((phi_x*coefficient.square()).sum()))
        weights.append(float(coefficient.square().sum()))
    checks['wiener_identity_vs_exact_ggn'] = relative_error(newton, directions['exact_ggn'])
    # DEVIATIONS #1: this gate voids reading B2 only; it is recorded here, with both directions' residuals.
    system = lambda v: torch.cat([((b+mu*o+(b+mu*o).T)/2)@v[s.slice] for s, b, o in zip(registry.specs, ggn, omega)])
    checks['residual_newton_eigenbasis'] = relative_error(system(newton), -g)
    checks['residual_exact_ggn_reference'] = relative_error(system(directions['exact_ggn']), -g)
    wiener_gate_failed = checks['wiener_identity_vs_exact_ggn'] > 1e-8
    energy = {'phi_energy_weighted': sum(shrink_mean)/sum(weights), 'phi_x_energy_weighted': sum(shrink_x_mean)/sum(weights),
              'wiener_gate_failed': wiener_gate_failed}
    measure('exact_ggn', 'newton_full', newton, directions['exact_ggn'], {'wiener_gate_failed': wiener_gate_failed})
    measure('exact_ggn', 'wiener', wiener, directions['exact_ggn'], energy)
    measure('exact_ggn', 'wiener_x', wiener_x, directions['exact_ggn'], energy)
    selected, search = bounded_search(flat, wiener, native, objective)
    measure('exact_ggn', 'wiener_search', selected, directions['exact_ggn'], {'alpha_search': search.get('selected_alpha'),
                                                                            'boundary': search.get('boundary', False),
                                                                            'wiener_gate_failed': wiener_gate_failed})

    # Curvature from panel 1 instead of the construction batch (same damping), same-batch search.
    blocks1, _ = streamed_blocks(model, registry, flat, x[panel_ids[0]], 8, c['resources']['budget_mib'])
    delta_xc = torch.empty_like(g)
    for s, block, o in zip(registry.specs, blocks1, omega):
        a = block+mu*o
        delta_xc[s.slice] = -torch.cholesky_solve(g[s.slice, None], torch.linalg.cholesky((a+a.T)/2))[:, 0]
    selected, search = bounded_search(flat, delta_xc, native, objective)
    measure('exact_ggn_xc', 'same', selected, delta_xc, {'alpha_search': search.get('selected_alpha'), 'boundary': search.get('boundary', False)})

    gates = [v for k, v in checks.items() if k.startswith('gate_')]
    if checks['gradient_error'] > 1e-12 or max(gates) > 1e-12:
        raise RuntimeError(f'Gate failed: {checks}')
    if not (state_equal(model_before, model.state_dict()) and state_equal(contexts_before, contexts) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('State mutated')
    return {'state': name, 'mu': mu, 'rows': rows, 'checks': checks}


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, _ = dataset(c)
    followup = json.loads(FOLLOWUP.read_text())
    reference = {(r['state'], r['operator'], r['policy']): r for s in followup['states'] for r in s['rows'] if r['rhs'] == 'gradient'}
    states = []
    for trajectory, step in STATES:
        states.append(state_steps(trajectory, step, c, x, y, reference))
        s = states[-1]
        held = {(r['operator'], r['policy']): r['held_234'] for r in s['rows']}
        print(f"{s['state']}: gates max {max(v for k, v in s['checks'].items() if k.startswith('gate_')):.1e}, "
              f"wiener identity {s['checks']['wiener_identity_vs_exact_ggn']:.1e} | exact_ggn second {held[('exact_ggn', 'second')]:+.4f} "
              f"gamma_x {held[('exact_ggn', 'gamma_x')]:+.4f} wiener {held[('exact_ggn', 'wiener')]:+.4f}", flush=True)
    (STUDY/'results').mkdir(exist_ok=True)
    (STUDY/'results'/'steps.json').write_text(json.dumps({'states': states}, indent=1, default=str)+'\n')


if __name__ == '__main__':
    main()
