"""Follow-up (PROTOCOL_FOLLOWUP.md): refined theory, step-size remedies, momentum drift and PARSEC-L. No training.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): followup.py
Writes results/followup.json and results/followup_tables.md in this study folder.
"""
import copy
import json
import statistics
import sys
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY/'scripts'))
from optimism import CHECK2C, PARENT, STATES, spearman  # noqa: E402
from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.curvature.layerwise import solve_blocks  # noqa: E402
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

OPERATORS = ['sgd', 'soap_map', 'exact_ggn', 'ef_newton', 'parsec_l']
LABEL_FREE = ['sgd', 'soap_map', 'exact_ggn', 'parsec_l']
RANK = 16


def nanmean(values):
    values = [v for v in values if v == v]
    return statistics.mean(values) if values else float('nan')


def state_followup(trajectory, step, c, x, y):
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
    checks = {'gradient_error': relative_error(op.gradient, g)}
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in ggn)/registry.size, c['damping']['floor'])
    mu_ef = max(.01*sum(float(b.trace()) for b in ef)/registry.size, c['damping']['floor'])
    beta = 1/(mu*metric.scale)
    eye = lambda s: torch.eye(s.count).reshape(s.shape+(s.count,))
    omega = [metric.scale*frozen_apply(eye(s), contexts[s.name], inverse=True).reshape(s.count, s.count) for s in registry.specs]
    factors = [capacity.factor(eye(s), contexts[s.name]).reshape(s.count, s.count) for s in registry.specs]
    # Explicit A^-1 per tensor (scale is irrelevant for r, shares and searched directions).
    inverses = {'sgd': [torch.eye(s.count) for s in registry.specs],
                'soap_map': [torch.linalg.inv(o) for o in omega],
                'exact_ggn': [torch.linalg.inv(b+mu*o) for b, o in zip(ggn, omega)],
                'ef_newton': [torch.linalg.inv(b+mu_ef*o) for b, o in zip(ef, omega)]}
    parsec, kappas = [], []
    for f, block in zip(factors, ggn):
        eig, q = torch.linalg.eigh(beta*f.T@block@f)
        r = min(RANK, len(eig))
        w, kappa = q[:, -r:], eig[-r:].clamp_min(0)
        parsec.append(f@(torch.eye(len(eig))-w@torch.diag(kappa/(1+kappa))@w.T)@f.T)
        kappas.append(float(kappa.max()))
    inverses['parsec_l'] = parsec

    def apply(name, v):
        out = torch.empty_like(v)
        for s, a in zip(registry.specs, inverses[name]):
            out[s.slice] = a@v[s.slice]
        return out

    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    pg = [GGN(functional_logits(model, registry, x[ids]), flat, y[ids]).gradient for ids in panel_ids]
    mean_all, mean_12 = torch.stack(pg).mean(0), torch.stack(pg[:2]).mean(0)
    sigma = [batch/(batch-1)*(f-g[s.slice, None]@g[None, s.slice]) for s, f in zip(registry.specs, ef)]
    # 1. EF ceiling, per tensor.
    ceiling = [float(g[s.slice]@torch.linalg.solve(f+mu_ef*o, g[s.slice])) for s, f, o in zip(registry.specs, ef, omega)]
    theory = {}
    for name in OPERATORS:
        optimism = sum(float((a@m).diagonal().sum()) for a, m in zip(inverses[name], sigma))/batch
        signal = statistics.mean(float(pg[p]@apply(name, pg[q])) for p in range(4) for q in range(p+1, 4))
        r = optimism/signal if signal > 0 else float('inf')
        am = apply(name, momentum)
        theory[name] = {'optimism': optimism, 'signal': signal, 'r': r,
                        'predicted_search_share': (1-r)/(1+r) if signal > 0 else -1.,
                        'predicted_fixed_share': 1/(1+r) if signal > 0 else 0.,
                        'momentum_cosine': float(mean_all@am)/(max(signal, 1e-300)*float(momentum@am))**.5}
    rows = []

    def measure(name, rhs_name, policy, delta, selected=None, search=None):
        if selected is None:
            selected = delta
        values = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
        own, held = values['actual_decrease'], values['independent_decreases']
        row = {'state': f'{trajectory}-{step}', 'operator': name, 'rhs': rhs_name, 'policy': policy, 'own': own,
               'held_all': statistics.mean(held), 'held_234': statistics.mean(held[1:]), 'held_34': statistics.mean(held[2:]),
               'norm': float(selected.norm()), 'selected_alpha': None if search is None else search.get('selected_alpha')}
        row['share_all'] = row['held_all']/own if own > 0 else float('nan')
        row['share_34'] = row['held_34']/own if own > 0 else float('nan')
        denominator = float(g@selected)
        row['realized_12'] = 2*float(mean_12@selected)/denominator-1 if denominator != 0 else float('nan')
        rows.append(row)
        return row

    stored = {(r['name'], r['rhs']): r['selected'] for r in json.loads((CHECK2C/f'{trajectory}-{step:04d}'/'candidates.json').read_text())}
    panel1, panels234 = panels[0], (lambda w: statistics.mean(float(f(w)) for f in panels[1:]))
    solved = {'soap_map': ([torch.zeros_like(b) for b in ggn], mu), 'exact_ggn': (ggn, mu), 'ef_newton': (ef, mu_ef)}
    for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
        for name in OPERATORS:
            if name in solved:
                # Check 2c's Cholesky solves, so the stored candidates are reproduced bit for bit.
                blocks, damping = solved[name]
                delta = solve_blocks(blocks, registry, rhs, metric, [damping], c['solver']['tolerance64'])[0][damping]
            else:
                delta = -apply(name, rhs)
            selected, search = search_candidate(flat, delta, native, objective)
            row = measure(name, rhs_name, 'same', delta, selected, search)
            if name in ['soap_map', 'exact_ggn']:
                ref = stored[({'soap_map': 'identity', 'exact_ggn': 'exact'}[name], rhs_name)]
                checks[f'gate_{name}_{rhs_name}'] = max(abs(row['own']-ref['actual_decrease']), abs(row['held_all']-ref['independent_mean_decrease']))
            sel, srch = search_candidate(flat, delta, native, lambda w: panel1(w))
            measure(name, rhs_name, 'second', delta, sel, srch)
            sel, srch = search_candidate(flat, delta, native, panels234)
            measure(name, rhs_name, 'oracle', delta, sel, srch)
            if rhs_name == 'gradient' and name in LABEL_FREE:
                o, dd = theory[name]['optimism'], float(delta@delta)
                corrected = lambda w, d=delta, o=o, dd=dd: objective(w)+o*float((w-flat)@d)/dd
                sel, srch = search_candidate(flat, delta, native, corrected)
                measure(name, rhs_name, 'corrected', delta, sel, srch)
    # Fixed steps: native SOAP's step size s.
    pu = torch.empty_like(momentum)
    for spec in registry.specs:
        ctx = contexts[spec.name]
        pu[spec.slice] = capacity.factor(capacity.factor(momentum[spec.slice].reshape(spec.shape), ctx, transpose=True), ctx).reshape(-1)
    checks['native_cosine_deficit'] = 1-float(native@(-pu))/float(native.norm()*pu.norm())
    size = float(native.norm()/pu.norm())
    for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
        soap_step = -size*torch.cat([(f@f.T@rhs[s.slice]) for s, f in zip(registry.specs, factors)])
        parsec_step = -size*apply('parsec_l', rhs)
        measure('soap_map', rhs_name, 'fixed', soap_step)
        measure('parsec_l', rhs_name, 'fixed', parsec_step)
        measure('parsec_l_blend1', rhs_name, 'fixed', parsec_step*float(soap_step.norm()/parsec_step.norm()))
    checks['fixed_soap_momentum_vs_native'] = relative_error(-size*torch.cat([(f@f.T@momentum[s.slice]) for s, f in zip(registry.specs, factors)]), native)
    gates = [v for k, v in checks.items() if k.startswith('gate_')]
    if checks['gradient_error'] > 1e-12 or max(gates) > 1e-12 or checks['native_cosine_deficit'] > 1e-10:
        raise RuntimeError(f'Gate failed: {checks}')
    if not (state_equal(model_before, model.state_dict()) and state_equal(contexts_before, contexts) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('State mutated')
    return {'state': f'{trajectory}-{step}', 'theory': theory, 'rows': rows, 'checks': checks, 'ceiling': ceiling,
            'parsec_kappa_max': max(kappas), 'step_size': size}


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, _ = dataset(c)
    states = []
    for trajectory, step in STATES:
        states.append(state_followup(trajectory, step, c, x, y))
        print(f"{trajectory}-{step}: " + ', '.join(f"{n} r={states[-1]['theory'][n]['r']:.2f}" for n in OPERATORS)
              + f" | checks {{{', '.join(f'{k}: {v:.1e}' for k, v in states[-1]['checks'].items())}}}", flush=True)
    row = lambda s, op, rhs, pol: next(r for r in s['rows'] if r['operator'] == op and r['rhs'] == rhs and r['policy'] == pol)
    readings = {}
    readings['1_ef_ceiling_max'] = max(max(s['ceiling']) for s in states)
    same24 = [row(s, op, 'gradient', 'same') for s in states for op in ['sgd', 'soap_map', 'exact_ggn', 'ef_newton']]
    readings['2_independent_mechanism_spearman'] = spearman([r['realized_12'] for r in same24], [r['share_34'] for r in same24])
    readings['3_second_beats_same_exact_ggn_gradient_states'] = sum(row(s, 'exact_ggn', 'gradient', 'second')['held_234'] > row(s, 'exact_ggn', 'gradient', 'same')['held_234'] for s in states)
    pairs = [(row(s, op, rhs, 'second'), row(s, op, rhs, 'same')) for s in states for op in OPERATORS for rhs in ['gradient', 'momentum']]
    readings['3_second_beats_same_all'] = f"{sum(a['held_234'] > b['held_234'] for a, b in pairs)}/{len(pairs)}"
    readings['3_mean_held_234'] = {pol: nanmean([row(s, op, rhs, pol)['held_234'] for s in states for op in OPERATORS for rhs in ['gradient', 'momentum']])
                                   for pol in ['same', 'second', 'oracle']}
    corrected18 = [(row(s, op, 'gradient', 'corrected'), row(s, op, 'gradient', 'same')) for s in states for op in ['sgd', 'soap_map', 'exact_ggn']]
    readings['4_corrected_beats_same_18'] = sum(a['held_234'] > b['held_234'] for a, b in corrected18)
    readings['4_corrected_beats_same_parsec_l'] = sum(row(s, 'parsec_l', 'gradient', 'corrected')['held_234'] > row(s, 'parsec_l', 'gradient', 'same')['held_234'] for s in states)
    by = {s['state']: s for s in states}
    drift = {op: all(by[f'exact_tensor_ggn-{k}']['theory'][op]['momentum_cosine'] < by[f'searched_soap-{k}']['theory'][op]['momentum_cosine'] for k in [25, 50])
             for op in ['sgd', 'soap_map', 'exact_ggn']}
    readings['5_drift_holds'] = drift
    readings['6a_parsec_between_states'] = sum(s['theory']['soap_map']['r'] < s['theory']['parsec_l']['r'] < s['theory']['exact_ggn']['r'] for s in states)
    for variant in ['parsec_l', 'parsec_l_blend1']:
        readings[f'6_{variant}_keeps_protection_states'] = sum(row(s, variant, 'momentum', 'fixed')['share_all'] >= row(s, 'soap_map', 'momentum', 'fixed')['share_all']-.1 for s in states)
        readings[f'6_{variant}_more_heldout_than_soap_states'] = sum(row(s, variant, 'momentum', 'fixed')['held_all'] > row(s, 'soap_map', 'momentum', 'fixed')['held_all'] for s in states)
    labelfree = [(s['theory'][r['operator']]['predicted_search_share'], r['share_all']) for s in states for r in s['rows']
                 if r['rhs'] == 'gradient' and r['policy'] == 'same' and r['operator'] in LABEL_FREE]
    readings['exploratory_labelfree_calibration'] = {'spearman': spearman([a for a, _ in labelfree], [b for _, b in labelfree]),
                                                     'mae': statistics.mean(abs(a-b) for a, b in labelfree), 'points': len(labelfree)}
    out = STUDY/'results'
    (out/'followup.json').write_text(json.dumps({'readings': readings, 'states': states}, indent=1, default=str)+'\n')
    lines = ['| State | r: sgd / soap / ggn / ef / parsec-L | cos_A(m, ḡ) sgd / soap / ggn | EF ceiling max |', '| --- | --- | --- | ---: |']
    for s in states:
        t = s['theory']
        lines.append(f"| {s['state']} | {' / '.join(f'{t[o]['r']:.2f}' for o in OPERATORS)} | "
                     f"{' / '.join(f'{t[o]['momentum_cosine']:+.2f}' for o in ['sgd', 'soap_map', 'exact_ggn'])} | {max(s['ceiling']):.3f} |")
    lines += ['', '| State | Operator | RHS | Held-out 2–4: same / second / corrected / oracle |', '| --- | --- | --- | --- |']
    for s in states:
        for op in OPERATORS:
            for rhs in ['gradient', 'momentum']:
                cells = []
                for pol in ['same', 'second', 'corrected', 'oracle']:
                    match = [r for r in s['rows'] if r['operator'] == op and r['rhs'] == rhs and r['policy'] == pol]
                    cells.append(f"{match[0]['held_234']:+.4f}" if match else '—')
                lines.append(f"| {s['state']} | {op} | {rhs} | {' / '.join(cells)} |")
    lines += ['', '| State | RHS | Fixed step (own, held-out all, share): SOAP | PARSEC-L | PARSEC-L blend 1 |', '| --- | --- | --- | --- | --- |']
    for s in states:
        for rhs in ['gradient', 'momentum']:
            cells = [row(s, op, rhs, 'fixed') for op in ['soap_map', 'parsec_l', 'parsec_l_blend1']]
            lines.append(f"| {s['state']} | {rhs} | " + ' | '.join(f"{r['own']:.4f}, {r['held_all']:+.4f}, {r['share_all']:+.2f}" for r in cells) + ' |')
    (out/'followup_tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(readings, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
