"""Training arms for checks 1, 2a and 2b.

Existing proposal code is imported unchanged. Study-local code:
- `study_ggn_propose(gamma)`: copy of the `ggn_exact_trace` branch of
  `run_transport_parsec_noise.propose`, which hard-codes gamma .01 (pre-approved in
  section 5; logged in DEVIATIONS.md). With gamma .01 it is bitwise identical
  (study test).
- `dense_ef`: the same branch with dense per-example empirical-Fisher tensor blocks.
- `soap_gradient_map`: the current gradient through SOAP's frozen preconditioner.
"""
import copy
import time

import torch
from torch.func import functional_call, grad, jvp, vmap

import common  # noqa: F401  (puts the repository scripts/ on sys.path)
from residual_curvature.curvature.layerwise import solve_blocks, streamed_blocks
from residual_curvature.curvature.operators import GGN, loss, residual
from residual_curvature.curvature.solves import relative_error
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from run_mini_vit_allocation import propose as mini_propose, search_candidate
from run_transport_parsec_noise import optimizer_settings, propose as transport_propose

ARMS = ['native_soap', 'adam_native', 'searched_soap', 'searched_soap_tuned',
        'ggn_exact_trace', 'dense_ef', 'soap_gradient_map']


def arm_config(c, arm, settings):
    """Per-arm config: only SOAP lr/betas/precondition_frequency change here."""
    c = copy.deepcopy(c)
    if arm == 'native_soap' and 'lr' in settings:
        c['soap']['lr'] = settings['lr']
    if arm == 'searched_soap_tuned':
        c['soap']['betas'] = list(settings['betas'])
        c['soap']['precondition_frequency'] = settings['precondition_frequency']
    policy = 'native' if arm in ['native_soap', 'adam_native'] else 'broad_direction_search'
    c['training'].update(method=arm, policy=policy)
    return c


def adam_settings(settings):
    value = optimizer_settings('adam_native')
    value['lr'] = settings.get('lr', value['lr'])
    return value


def per_example_gradients(model, registry, flat, x, y):
    """Rows g_i = grad of example i's cross-entropy, so mean_i g_i = g."""
    def single(w, xi, yi):
        logits = functional_call(model, registry.unflatten(w), (xi[None],))
        return torch.nn.functional.cross_entropy(logits, yi[None])
    return vmap(grad(single), in_dims=(None, 0, 0))(flat, x, y).detach()


def empirical_fisher_blocks(model, registry, flat, x, y):
    """F_bb = (1/B) sum_i g_ib g_ib^T for every registered tensor."""
    started = time.perf_counter()
    rows = per_example_gradients(model, registry, flat, x, y)
    blocks = [rows[:, s.slice].T @ rows[:, s.slice] / len(x) for s in registry.specs]
    return blocks, rows.mean(0), {'construction_seconds': time.perf_counter()-started,
                                  'trace': sum(float(b.diag().sum()) for b in blocks),
                                  'batch_size': len(x), 'route': 'torch.func.vmap(grad) per-example gradients'}


def empirical_fisher_action(op, v):
    """Independent JVP/VJP route: F v = J^T diag_i(e_i e_i^T) J v / B, e_i = p_i - onehot(y_i)."""
    tangent = jvp(op.fn, (op.params,), (v,))[1]
    e = residual(op.logits, op.targets)*len(op.targets)
    return op.pullback(e*(e*tangent).sum(-1, keepdim=True)/len(op.targets))[0].detach()


def _curvature_step(kind, gamma, model, registry, flat, state, x, y, c, step, audit):
    """Shared body of `ggn_exact_trace` (study copy) and `dense_ef`.

    For kind == 'ggn' this follows run_transport_parsec_noise.propose line by line,
    with .01 replaced by gamma.
    """
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    op = GGN(fn, flat, y)
    contexts, algorithm, audit_seconds, search = None, {}, 0., None
    proposal, next_state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
    native = registry.flatten(proposal)
    delta = native
    if all(value is not None for value in contexts.values()):
        metric = SOAPMetric(registry, contexts)
        if kind == 'ggn':
            blocks, construction = streamed_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
        else:
            blocks, mean_gradient, construction = empirical_fisher_blocks(model, registry, flat, x, y)
            started = time.perf_counter()
            construction['mean_gradient_error'] = relative_error(mean_gradient, op.gradient)
            if construction['mean_gradient_error'] > c['solver']['tolerance64']:
                raise RuntimeError(f"Per-example gradient mean check failed: {construction['mean_gradient_error']}")
            audit_seconds += time.perf_counter()-started
        mu = max(gamma*construction['trace']/registry.size, c['damping']['floor'])
        values, solve = solve_blocks(blocks, registry, op.gradient, metric, [mu], c['solver']['tolerance64'])
        delta = values[mu]
        algorithm = {'curvature': construction, 'mu': mu, 'gamma': gamma, 'solve': solve[mu]}
        if audit:
            started = time.perf_counter()
            gen = torch.Generator().manual_seed(c['analysis_seed']+step)
            checks = {}
            for spec, block in zip(registry.specs, blocks):
                v = torch.zeros_like(flat)
                v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
                product = op.action(v) if kind == 'ggn' else empirical_fisher_action(op, v)
                checks[spec.name] = relative_error(block@v[spec.slice], product[spec.slice])
            if max(checks.values()) > c['solver']['tolerance64']:
                raise RuntimeError('Curvature action audit failed')
            algorithm['action_checks'] = checks
            audit_seconds += time.perf_counter()-started
    raw_norm = float(delta.norm())
    if float(native.norm()) > 0:
        delta, search = search_candidate(flat, delta, native, objective)
    row = {'method': 'ggn_exact_trace' if kind == 'ggn' else 'dense_ef', 'algorithm': algorithm, 'search': search,
           'audit_seconds': audit_seconds, 'native_radius': float(native.norm()), 'raw_norm': raw_norm,
           'norm': float(delta.norm()), 'gradient_norm': float(op.gradient.norm()), 'slope': float(op.gradient@delta),
           'loss_before': float(objective(flat)), 'loss_after': float(objective(flat+delta)),
           'operator_counts': op.counts}
    if not torch.isfinite(delta).all():
        raise FloatingPointError('Nonfinite update')
    return delta, next_state, contexts, row


def study_ggn_propose(gamma, model, registry, flat, state, x, y, c, step, audit=False):
    return _curvature_step('ggn', gamma, model, registry, flat, state, x, y, c, step, audit)


def dense_ef_propose(gamma, model, registry, flat, state, x, y, c, step, audit=False):
    return _curvature_step('ef', gamma, model, registry, flat, state, x, y, c, step, audit)


def soap_gradient_map_propose(model, registry, flat, state, x, y, c, step):
    """SOAP statistics advance once; candidate -P g without momentum; native-radius search."""
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    op = GGN(fn, flat, y)
    proposal, next_state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
    native = registry.flatten(proposal)
    delta, search = native, None
    if all(value is not None for value in contexts.values()):
        delta = -SOAPMetric(registry, contexts).map(op.gradient)
    raw_norm = float(delta.norm())
    if float(native.norm()) > 0:
        delta, search = search_candidate(flat, delta, native, objective)
    row = {'method': 'soap_gradient_map', 'algorithm': {}, 'search': search, 'audit_seconds': 0.,
           'native_radius': float(native.norm()), 'raw_norm': raw_norm, 'norm': float(delta.norm()),
           'gradient_norm': float(op.gradient.norm()), 'slope': float(op.gradient@delta),
           'loss_before': float(objective(flat)), 'loss_after': float(objective(flat+delta)),
           'operator_counts': op.counts}
    return delta, next_state, contexts, row


def propose_step(arm, settings, implementation, model, registry, flat, state, x, y, c, step, audit=False):
    """One consumed step. `implementation='original'` routes GGN through the repository runner."""
    if arm in ['native_soap', 'searched_soap', 'searched_soap_tuned']:
        method = 'native_soap' if arm == 'native_soap' else 'searched_soap'
        delta, state, contexts, row = mini_propose(method, model, registry, flat, state, x, y, c, step)
        row['audit_seconds'] = 0.
    elif arm == 'adam_native':
        delta, state, contexts, row = transport_propose('adam_native', model, registry, flat, state, x, y, c, step,
                                                        adam_settings(settings), None)
    elif arm == 'ggn_exact_trace' and implementation == 'original':
        if settings['gamma'] != .01:
            raise ValueError('The repository branch hard-codes gamma .01')
        delta, state, contexts, row = transport_propose('ggn_exact_trace', model, registry, flat, state, x, y, c, step,
                                                        optimizer_settings('ggn_exact_trace'), None, audit=audit)
    elif arm == 'ggn_exact_trace':
        delta, state, contexts, row = study_ggn_propose(settings['gamma'], model, registry, flat, state, x, y, c, step, audit)
    elif arm == 'dense_ef':
        delta, state, contexts, row = dense_ef_propose(settings['gamma'], model, registry, flat, state, x, y, c, step, audit)
    elif arm == 'soap_gradient_map':
        delta, state, contexts, row = soap_gradient_map_propose(model, registry, flat, state, x, y, c, step)
    else:
        raise ValueError(arm)
    return delta, state, contexts, row


def compact_row(row):
    """Per-step summary for training.json; full rows (all search trials) go to history_full.json.gz."""
    search = row.get('search') or {}
    algorithm = row.get('algorithm') or {}
    solve = algorithm.get('solve') or row.get('solve') or {}
    curvature = algorithm.get('curvature') or {}
    return {'loss_before': row['loss_before'], 'loss_after': row['loss_after'], 'norm': row['norm'],
            'raw_norm': row['raw_norm'], 'slope': row['slope'],
            'native_radius': row.get('native_radius', row.get('native_norm')),
            'audit_seconds': row.get('audit_seconds', 0.),
            'mu': algorithm.get('mu'), 'trace': curvature.get('trace'),
            'curvature_seconds': curvature.get('construction_seconds'),
            'solve_residual': solve.get('residual'),
            'mean_gradient_error': curvature.get('mean_gradient_error'),
            'max_action_check': max(algorithm['action_checks'].values()) if algorithm.get('action_checks') else None,
            'search': {k: search.get(k) for k in ['selected_alpha', 'evaluations', 'bracketed', 'boundary', 'rejected',
                                                  'grid_best_alpha']} if search else None}
