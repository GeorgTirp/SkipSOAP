"""Training arms for the geometry follow-up.

Repository code is imported unchanged (PARSEC default branch, Lanczos/Ritz, GGN and
graph-Wasserstein operators, damped block solves, search). Study-local code:
- `_curvature_step`: copied from studies/gap_audit_20261007/scripts/arms.py (SHA-256
  e2807259...), which follows `run_transport_parsec_noise.propose` with gamma as a parameter;
  the 'ot' kind is the repository `wasserstein_tensor` branch (bitwise at gamma .01, study test).
- `full_ggn`, `full_ot`, `_full_step`: full D x D matrices from the same streamed Jacobian
  factors as `layerwise.streamed_blocks` / `wasserstein.streamed_wasserstein_blocks`, solved
  densely with the same block-diagonal metric mu Omega as `layerwise.solve_blocks`.
- `parsec_split_propose`: port of the source's `split_spaces=True` branch
  (references/user_optimizers/parsec_pasted.py, `_apply_eigenadam_split_spaces`).
"""
import copy
import time

import torch
from torch.func import jacrev

import common  # noqa: F401  (puts the repository scripts/ on sys.path)
from residual_curvature.curvature import wasserstein
from residual_curvature.curvature.layerwise import solve_blocks, streamed_blocks, workspace_estimate
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.parsec import lanczos_ritz
from residual_curvature.optim.soap import SOAPMetric, frozen_apply, soap_propose
from run_mini_vit_allocation import search_candidate
from run_transport_parsec_noise import optimizer_settings, propose as transport_propose

ARMS = ['parsec_default', 'parsec_split', 'ot_tensor', 'ggn_full', 'ot_full']


def arm_config(c, arm, settings):
    c = copy.deepcopy(c)
    policy = 'native' if arm.startswith('parsec') else 'broad_direction_search'
    c['training'].update(method=arm, policy=policy)
    return c


def parsec_settings(settings):
    """Repository PARSEC settings with the tuned lr, refresh, rank (2 x rank Krylov iterations)."""
    value = optimizer_settings('parsec_native')
    for key in ['lr', 'refresh_every', 'rank', 'lr_top', 'lr_perp']:
        if key in settings:
            value[key] = settings[key]
    value['lanczos_iters'] = 2*value['rank']
    return value


def parsec_split_propose(gradient, params, state, settings, operator):
    """Source split_spaces=True: Newton step in the top Ritz space, Adam on the complement.

    The source passes the already incremented step and adds one again for bias
    correction (t = step + 1); this port preserves that. No search, no weight decay.
    """
    if state is None:
        state = {'step': 0, 'm_perp': torch.zeros_like(gradient), 'v_perp': torch.zeros_like(gradient),
                 'eigenvalues': gradient.new_empty(0), 'vectors': gradient.new_empty(0, len(gradient)),
                 'last_refresh': None}
    state = copy.deepcopy(state)
    step = state['step']+1
    refresh = None
    if settings['refresh_every'] > 0 and step % settings['refresh_every'] == 0:
        eigen, vectors, refresh = lanczos_ritz(operator.action, gradient, settings['rank'],
                                               settings['lanczos_iters'], settings['seed']+step)
        state.update(eigenvalues=eigen, vectors=vectors, last_refresh=step)
    v, lam = state['vectors'], state['eigenvalues']
    proj = v@gradient
    g_perp = gradient-v.T@proj
    b1, b2 = settings['betas']
    t = step+1
    state['m_perp'] = b1*state['m_perp']+(1-b1)*g_perp
    state['v_perp'] = b2*state['v_perp']+(1-b2)*g_perp.square()
    step_perp = -settings['lr_perp']*(state['m_perp']/(1-b1**t))/((state['v_perp']/(1-b2**t)).sqrt()+settings['eps'])
    lam_eff = lam.clamp_min(0)+settings['damping']
    step_top = -settings['lr_top']*(v.T@(proj/(lam_eff+1e-12)))
    state['step'] = step
    delta = step_top+step_perp
    if not torch.isfinite(delta).all():
        raise FloatingPointError('Nonfinite PARSEC split proposal')
    perp_norm = float(step_perp.norm())
    return delta, state, {'refresh': refresh, 'active_rank': len(lam), 'last_refresh': state['last_refresh'],
                          'gradient_norm': float(gradient.norm()), 'top_step_norm': float(step_top.norm()),
                          'perp_step_norm': perp_norm,
                          'perp_step_top_space_fraction': float((v@step_perp).norm())/perp_norm if perp_norm and len(lam) else 0.,
                          'bias_correction_index': t}


def _streamed_factor(kind, model, registry, flat, x, microbatch, budget_mib):
    """Rows R with R^T R = full GGN (or full graph-Wasserstein metric); same factors as the block routines."""
    estimate = workspace_estimate(registry, model.head.out_features, microbatch, flat.element_size())
    if estimate['estimated_workspace_bytes'] > budget_mib*2**20:
        raise MemoryError('Full curvature workspace exceeds budget')
    rows = []
    for start in range(0, len(x), microbatch):
        xb = x[start:start+microbatch]
        fn = functional_logits(model, registry, xb)
        p = fn(flat).detach().softmax(-1)
        j = jacrev(fn, chunk_size=16)(flat).detach()
        if kind == 'ggn':
            centered = j-(p[..., None]*j).sum(1, keepdim=True)
            rows.append((centered*(p/len(x)).sqrt()[..., None]).reshape(-1, registry.size))
        else:
            probability_jac = p[..., None]*(j-(p[..., None]*j).sum(1, keepdim=True))
            lap = wasserstein.graph_laplacian(p)
            chol = torch.linalg.cholesky(lap+torch.ones_like(lap)/p.shape[-1])
            rows.append((torch.linalg.solve_triangular(chol, probability_jac, upper=False)/len(x)**.5).reshape(-1, registry.size))
        del j
    return torch.cat(rows)


def full_matrix(kind, model, registry, flat, x, microbatch=8, budget_mib=768):
    started = time.perf_counter()
    factor = _streamed_factor(kind, model, registry, flat, x, microbatch, budget_mib)
    full = factor.T@factor
    return full, {'construction_seconds': time.perf_counter()-started, 'trace': float(full.diag().sum()),
                  'jacobian_rows': len(x)*model.head.out_features, 'microbatch': microbatch, 'batch_size': len(x),
                  'scope': 'full parameter space, all cross-tensor blocks'}


def full_metric(registry, metric, dtype, tolerance):
    """Block-diagonal mu-free damping metric Omega, identical blocks to layerwise.solve_blocks."""
    omega = torch.zeros(registry.size, registry.size, dtype=dtype)
    for spec in registry.specs:
        eye = torch.eye(spec.count, dtype=dtype)
        block = metric.scale*frozen_apply(eye.reshape(spec.shape+(spec.count,)), metric.contexts[spec.name],
                                          inverse=True).reshape(spec.count, spec.count)
        if relative_error(block, block.T) > tolerance:
            raise RuntimeError(f'Asymmetric damping metric for {spec.name}')
        omega[spec.slice, spec.slice] = block
    return omega


def off_diagonal_energy(full, registry):
    total = float(full.square().sum())
    diagonal = sum(float(full[s.slice, s.slice].square().sum()) for s in registry.specs)
    return (total-diagonal)/total if total else 0.


def _curvature_step(kind, gamma, model, registry, flat, state, x, y, c, step, audit):
    """Block (tensor) curvature step; copied from the gap audit, with an 'ot' kind."""
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
            blocks, construction = wasserstein.streamed_wasserstein_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
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
                product = op.action(v) if kind == 'ggn' else wasserstein.action(model, registry, flat, x, v)
                checks[spec.name] = relative_error(block@v[spec.slice], product[spec.slice])
            if max(checks.values()) > c['solver']['tolerance64']:
                raise RuntimeError('Curvature action audit failed')
            algorithm['action_checks'] = checks
            audit_seconds += time.perf_counter()-started
    raw_norm = float(delta.norm())
    if float(native.norm()) > 0:
        delta, search = search_candidate(flat, delta, native, objective)
    row = {'method': kind+'_tensor', 'algorithm': algorithm, 'search': search, 'audit_seconds': audit_seconds,
           'native_radius': float(native.norm()), 'raw_norm': raw_norm, 'norm': float(delta.norm()),
           'gradient_norm': float(op.gradient.norm()), 'slope': float(op.gradient@delta),
           'loss_before': float(objective(flat)), 'loss_after': float(objective(flat+delta)), 'operator_counts': op.counts}
    if not torch.isfinite(delta).all():
        raise FloatingPointError('Nonfinite update')
    return delta, next_state, contexts, row


def _full_step(kind, gamma, model, registry, flat, state, x, y, c, step, audit):
    """Full-matrix curvature step at matched damping, metric, RHS and search."""
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    op = GGN(fn, flat, y)
    tolerance = c['solver']['tolerance64']
    algorithm, audit_seconds, search = {}, 0., None
    proposal, next_state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
    native = registry.flatten(proposal)
    delta = native
    if all(value is not None for value in contexts.values()):
        metric = SOAPMetric(registry, contexts)
        full, construction = full_matrix(kind, model, registry, flat, x, 8, c['resources']['budget_mib'])
        mu = max(gamma*construction['trace']/registry.size, c['damping']['floor'])
        started = time.perf_counter()
        system = full+mu*full_metric(registry, metric, flat.dtype, tolerance)
        chol = torch.linalg.cholesky((system+system.T)/2)
        delta = torch.cholesky_solve(-op.gradient[:, None], chol)[:, 0]
        residual = relative_error(system@delta, -op.gradient)
        if residual > tolerance:
            raise RuntimeError(f'Full dense solve failed: {residual}')
        algorithm = {'curvature': construction, 'mu': mu, 'gamma': gamma,
                     'solve': {'residual': residual, 'seconds': time.perf_counter()-started}}
        if audit:
            started = time.perf_counter()
            gen = torch.Generator().manual_seed(c['analysis_seed']+step)
            v = torch.randn(registry.size, generator=gen, dtype=flat.dtype)
            product = op.action(v) if kind == 'ggn' else wasserstein.action(model, registry, flat, x, v)
            if kind == 'ggn':
                blocks, _ = streamed_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
            else:
                blocks, _ = wasserstein.streamed_wasserstein_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
            diag = (sum(float((full[s.slice, s.slice]-b).square().sum()) for s, b in zip(registry.specs, blocks))
                    / sum(float(b.square().sum()) for b in blocks))**.5
            checks = {'full_action': relative_error(full@v, product), 'diagonal_blocks_frobenius': diag}
            if max(checks.values()) > tolerance:
                raise RuntimeError(f'Full curvature audit failed: {checks}')
            algorithm.update(audit_checks=checks, off_diagonal_energy_fraction=off_diagonal_energy(full, registry))
            audit_seconds += time.perf_counter()-started
        del full, system, chol
    raw_norm = float(delta.norm())
    if float(native.norm()) > 0:
        delta, search = search_candidate(flat, delta, native, objective)
    row = {'method': kind+'_full', 'algorithm': algorithm, 'search': search, 'audit_seconds': audit_seconds,
           'native_radius': float(native.norm()), 'raw_norm': raw_norm, 'norm': float(delta.norm()),
           'gradient_norm': float(op.gradient.norm()), 'slope': float(op.gradient@delta),
           'loss_before': float(objective(flat)), 'loss_after': float(objective(flat+delta)), 'operator_counts': op.counts}
    if not torch.isfinite(delta).all():
        raise FloatingPointError('Nonfinite update')
    return delta, next_state, contexts, row


def _parsec_step(arm, settings, model, registry, flat, state, x, y, c, step):
    ps = parsec_settings(settings)
    if arm == 'parsec_default':
        return transport_propose('parsec_native', model, registry, flat, state, x, y, c, step, ps, None)
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    op = GGN(fn, flat, y)
    delta, next_state, algorithm = parsec_split_propose(op.gradient, flat, state, ps, op)
    row = {'method': 'parsec_split', 'algorithm': algorithm, 'search': None, 'audit_seconds': 0.,
           'native_radius': float(delta.norm()), 'raw_norm': float(delta.norm()), 'norm': float(delta.norm()),
           'gradient_norm': float(op.gradient.norm()), 'slope': float(op.gradient@delta),
           'loss_before': float(objective(flat)), 'loss_after': float(objective(flat+delta)), 'operator_counts': op.counts}
    return delta, next_state, None, row


def propose_step(arm, settings, implementation, model, registry, flat, state, x, y, c, step, audit=False):
    if arm.startswith('parsec'):
        return _parsec_step(arm, settings, model, registry, flat, state, x, y, c, step)
    if arm == 'ot_tensor' and implementation == 'original':
        if settings['gamma'] != .01:
            raise ValueError('The repository branch hard-codes gamma .01')
        return transport_propose('wasserstein_tensor', model, registry, flat, state, x, y, c, step,
                                 optimizer_settings('wasserstein_tensor'), None, audit=audit)
    if arm in ['ggn_exact_trace', 'ot_tensor']:
        return _curvature_step('ggn' if arm == 'ggn_exact_trace' else 'ot', settings['gamma'], model, registry, flat,
                               state, x, y, c, step, audit)
    if arm in ['ggn_full', 'ot_full']:
        return _full_step(arm.split('_')[0], settings['gamma'], model, registry, flat, state, x, y, c, step, audit)
    raise ValueError(arm)


def compact_row(row):
    """Per-step summary for training.json; full rows go to history_full.json.gz."""
    search = row.get('search') or {}
    algorithm = row.get('algorithm') or {}
    solve = algorithm.get('solve') or {}
    curvature = algorithm.get('curvature') or {}
    refresh = algorithm.get('refresh') or {}
    checks = algorithm.get('action_checks') or algorithm.get('audit_checks')
    return {'loss_before': row['loss_before'], 'loss_after': row['loss_after'], 'norm': row['norm'],
            'raw_norm': row['raw_norm'], 'slope': row['slope'], 'native_radius': row.get('native_radius'),
            'audit_seconds': row.get('audit_seconds', 0.), 'mu': algorithm.get('mu'), 'trace': curvature.get('trace'),
            'curvature_seconds': curvature.get('construction_seconds'), 'solve_residual': solve.get('residual'),
            'max_audit_check': max(checks.values()) if checks else None,
            'off_diagonal_energy_fraction': algorithm.get('off_diagonal_energy_fraction'),
            'max_ritz_residual': max(refresh['relative_ritz_residuals']) if refresh.get('relative_ritz_residuals') else None,
            'ritz_eigenvalues': refresh.get('eigenvalues'), 'active_rank': algorithm.get('active_rank'),
            'top_step_norm': algorithm.get('top_step_norm'), 'perp_step_norm': algorithm.get('perp_step_norm'),
            'perp_step_top_space_fraction': algorithm.get('perp_step_top_space_fraction'),
            'search': {k: search.get(k) for k in ['selected_alpha', 'evaluations', 'bracketed', 'boundary', 'rejected',
                                                  'grid_best_alpha']} if search else None}
