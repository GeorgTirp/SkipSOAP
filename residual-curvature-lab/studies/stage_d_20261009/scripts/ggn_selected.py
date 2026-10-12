"""Arm D: the gap audit's exact tensor GGN step (`arms._curvature_step('ggn')`) with the ray search run on a
selection batch instead of the training batch, plus the optimism diagnostic r for GGN and SOAP's metric."""
import time

import torch

import common  # noqa: F401  (puts the repository scripts/ on sys.path)
import arms
from residual_curvature.curvature.layerwise import solve_blocks, streamed_blocks
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, frozen_apply, soap_propose
from run_mini_vit_allocation import search_candidate


def optimism(model, registry, flat, x, y, blocks, metric, mu):
    """r = tr(A^-1 Sigma)/B / (g1^T A^-1 g2) from true-label per-example gradients (halves cross-fitted)."""
    rows = arms.per_example_gradients(model, registry, flat, x, y)
    batch, half = len(rows), len(rows)//2
    centered = rows-rows.mean(0)
    a, b = rows[:half].mean(0), rows[half:2*half].mean(0)
    out = {'optimism_soap': float((centered.T*metric.map(centered.T)).sum())/(batch-1)/batch,
           'signal_soap': float(a@metric.map(b)), 'optimism_ggn': 0., 'signal_ggn': 0.}
    for spec, block in zip(registry.specs, blocks):
        omega = metric.scale*frozen_apply(torch.eye(spec.count, dtype=flat.dtype).reshape(spec.shape+(spec.count,)),
                                          metric.contexts[spec.name], inverse=True).reshape(spec.count, spec.count)
        system = block+mu*omega
        chol = torch.linalg.cholesky((system+system.T)/2)
        part = centered[:, spec.slice]
        out['optimism_ggn'] += float((part.T*torch.cholesky_solve(part.T, chol)).sum())/(batch-1)/batch
        out['signal_ggn'] += float(a[spec.slice]@torch.cholesky_solve(b[spec.slice, None], chol)[:, 0])
    for name in ['soap', 'ggn']:
        out[f'r_{name}'] = out[f'optimism_{name}']/out[f'signal_{name}'] if out[f'signal_{name}'] > 0 else float('inf')
    return out


def propose(gamma, model, registry, flat, state, x, y, xs, ys, c, step, audit=False, diagnose=False):
    """`arms._curvature_step('ggn')` line by line, except that `search_candidate` uses the selection batch (xs, ys)."""
    fn = functional_logits(model, registry, x)
    objective = lambda v: loss(fn(v), y)
    selection_fn = functional_logits(model, registry, xs)
    selection = lambda v: loss(selection_fn(v), ys)
    op = GGN(fn, flat, y)
    contexts, algorithm, audit_seconds, diagnostic_seconds, search, diagnostic = None, {}, 0., 0., None, None
    proposal, next_state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(op.gradient), state, c['soap'])
    native = registry.flatten(proposal)
    delta = native
    if all(value is not None for value in contexts.values()):
        metric = SOAPMetric(registry, contexts)
        blocks, construction = streamed_blocks(model, registry, flat, x, 8, c['resources']['budget_mib'])
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
                checks[spec.name] = relative_error(block@v[spec.slice], op.action(v)[spec.slice])
            if max(checks.values()) > c['solver']['tolerance64']:
                raise RuntimeError('Curvature action audit failed')
            algorithm['action_checks'] = checks
            audit_seconds += time.perf_counter()-started
        if diagnose:
            started = time.perf_counter()
            diagnostic = optimism(model, registry, flat, x, y, blocks, metric, mu)
            diagnostic_seconds = time.perf_counter()-started
    raw_norm = float(delta.norm())
    if float(native.norm()) > 0:
        delta, search = search_candidate(flat, delta, native, selection)
    row = {'method': 'ggn_selected', 'algorithm': algorithm, 'search': search, 'audit_seconds': audit_seconds,
           'diagnostic_seconds': diagnostic_seconds, 'native_radius': float(native.norm()), 'raw_norm': raw_norm,
           'norm': float(delta.norm()), 'gradient_norm': float(op.gradient.norm()), 'slope': float(op.gradient@delta),
           'loss_before': float(objective(flat)), 'loss_after': float(objective(flat+delta)),
           'selection_before': float(selection(flat)), 'selection_after': float(selection(flat+delta)),
           'operator_counts': op.counts}
    if not torch.isfinite(delta).all():
        raise FloatingPointError('Nonfinite update')
    return delta, next_state, contexts, row, diagnostic
