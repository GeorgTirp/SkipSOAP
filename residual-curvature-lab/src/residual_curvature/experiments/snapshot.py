import resource
import sys
import time

import numpy as np
import torch

from ..curvature.operators import GGN, guard, loss
from ..curvature.sketch import categorical_probes, factors_from_probes, microbatch_factors
from ..curvature.solves import IdentityMetric, cg, diagonal_solve, factor_solve, relative_error, sketch_sweeps
from ..models.mlp import functional_logits
from ..optim.allocation import allocation_candidates, ray_value
from ..optim.soap import SOAPMetric, soap_propose
from ..provenance import checksum, write_json
from .search import local_direction_grid


def candidate_metrics(delta, g, u, mu, metric, registry):
    norm2 = float(delta @ delta)
    curvature = float((u.T @ delta).square().sum())
    slope = float(-g @ delta)
    return {'norm': norm2 ** .5, 'slope': slope,
            'undamped_decrease': slope - .5 * curvature,
            'damped_decrease': slope - .5 * curvature - .5 * mu * float(delta @ metric.apply(delta)),
            'exposure': curvature / norm2 if norm2 else None,
            'shares': {s.name: float(delta[s.slice].square().sum()) / norm2 if norm2 else 0.0 for s in registry.specs},
            'optimized_ray': ray_value(g, delta, lambda v: u @ (u.T @ v) + mu * metric.apply(v))}


def snapshot(model, registry, optimizer_state, x, y, construction_ids, panel_ids, c, step, directory):
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    flat = registry.flatten(dict(model.named_parameters())).detach()
    xb, yb = x[construction_ids].to(c['device']), y[construction_ids].to(c['device'])
    fn = functional_logits(model, registry, xb)
    tolerance = c['solver']['tolerance64' if flat.dtype == torch.float64 else 'tolerance32']
    allocations = guard(registry.size, yb.numel() * c['model']['classes'], max(c['sketch']['ranks']), c['resources'], flat.dtype)
    operator = GGN(fn, flat, yb)
    j, u, eigen_tolerance = operator.factors()
    g = operator.gradient
    soap_start = time.perf_counter()
    native_dict, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), optimizer_state, c['soap'])
    soap_seconds = time.perf_counter() - soap_start
    native = registry.flatten(native_dict)
    soap_metric = SOAPMetric(registry, contexts)
    metric = soap_metric if c['damping']['metric'] == 'soap' else IdentityMetric()
    groups = registry.groups(c['partition'])
    trace = float(u.square().sum())
    base = float(loss(fn(flat), yb))
    panels = [(functional_logits(model, registry, x[ids].to(c['device'])), y[ids].to(c['device'])) for ids in panel_ids]
    panel_base = [float(loss(f(flat), target)) for f, target in panels]
    # Each repeat's rank grid is nested; repeats have independent RNG streams.
    sketches = []
    arrays = {'jacobian': j.cpu().numpy(), 'output_factor': u.cpu().numpy(), 'gradient': g.cpu().numpy(),
              'parameters': flat.cpu().numpy()}
    diagnostics = {'output_eigen_tolerance': eigen_tolerance, 'exact_trace': trace, 'checks': {},
                   'soap_proposal_seconds': soap_seconds, 'probe_refresh_seconds': [],
                   'action_error_convention': 'relative L2; absolute L2 when reference norm < 1e-12'}
    validation_gen = torch.Generator().manual_seed(c['analysis_seed'] + step)
    validation = torch.randn(registry.size, generator=validation_gen, dtype=flat.dtype).to(flat.device)
    diagnostics['checks']['ggn_action'] = relative_error(operator.action(validation), u @ (u.T @ validation))
    for repeat in range(c['sketch']['repeats']):
        probe_seed = c['probe_seed'] + 100000 * step + repeat
        maxrank = max(c['sketch']['ranks'])
        xi = categorical_probes(operator.logits, maxrank, probe_seed)
        probe_start = time.perf_counter()
        yy = factors_from_probes(fn, flat, xi)
        diagnostics['probe_refresh_seconds'].append(time.perf_counter() - probe_start)
        micro = microbatch_factors(model, registry, flat, xb, xi, c['sketch']['microbatch'])
        diagnostics['checks'][f'microbatch_{repeat}'] = relative_error(micro, yy)
        arrays[f'probes_{repeat}'] = xi.cpu().numpy()
        for rank in c['sketch']['ranks']:
            factor = yy[:, :rank] * (maxrank / rank) ** .5
            arrays[f'sketch_{repeat}_{rank}'] = factor.cpu().numpy()
            # Frobenius error computed through small Gram matrices, without D^2.
            denom = float((u.T @ u).square().sum())
            error2 = float((factor.T @ factor).square().sum()) + denom - 2 * float((u.T @ factor).square().sum())
            sketches.append((rank, repeat, probe_seed, factor, (max(0, error2) / max(denom, 1e-30)) ** .5))
    records = []
    for gamma in c['damping']['gammas']:
        mu = max(gamma * trace / registry.size, c['damping']['floor'])
        action = lambda v: operator.action(v) + mu * metric.apply(v)
        solve_start = time.perf_counter()
        full = factor_solve(g, u, mu, metric.inverse)
        # Woodbury subtracts potentially large terms. Refine the solution using
        # the independent matrix-free residual, retaining the declared tolerance.
        refinements = 0
        for _ in range(4):
            error = action(full) + g
            if error.norm() / g.norm().clamp_min(1e-14) <= tolerance:
                break
            full += factor_solve(error, u, mu, metric.inverse)
            refinements += 1
        solve_seconds = time.perf_counter() - solve_start
        full_residual = float((action(full) + g).norm() / g.norm().clamp_min(1e-14))
        if full_residual > tolerance:
            raise RuntimeError(f'Output-dual reference failed declared residual tolerance: {full_residual}')
        iterative = cg(action, -g, soap_metric.map, tolerance, c['solver']['max_iterations'])
        allocation, qp, e, k = allocation_candidates(g, u, metric, mu, native, registry)
        arrays[f'K_{gamma}'] = k.cpu().numpy()
        arrays[f'E_{gamma}'] = e.cpu().numpy()
        dual = mu * torch.eye(u.shape[1], dtype=flat.dtype, device=flat.device) + u.T @ metric.inverse(u)
        eigen = torch.linalg.eigvalsh(dual)
        common = {'gamma': gamma, 'mu': mu, 'trace_source': 'exact output factor', 'metric': c['damping']['metric'],
                  'metric_scale': metric.scale, 'full_solve_residual': full_residual,
                  'full_solve_seconds': solve_seconds, 'dual_condition': float(eigen[-1] / eigen[0]),
                  'dual_refinements': refinements,
                  'cg_iterations': iterative.iterations, 'cg_residual': iterative.residual,
                  'cg_converged': iterative.converged, 'allocation': qp}
        candidates = [('native_soap', native, {}), ('frozen_minus_Pg', -soap_metric.map(g), {}),
                      ('native_ray', native * ray_value(g, native, action)['alpha'], {}),
                      ('exact_diagonal', diagonal_solve(g, u, mu, metric.inverse, groups), {}),
                      ('exact_full', full, {}),
                      ('cg_full' if iterative.converged else 'cg_approximate', iterative.delta, {})]
        candidates += [(name, value, {}) for name, value in allocation.items()]
        for rank, repeat, probe_seed, factor, error in sketches:
            extra = {'rank': rank, 'repeat': repeat, 'probe_seed': probe_seed, 'probe_type': 'shared categorical',
                     'sketch_frobenius_error': error,
                     'validation_action_error': relative_error(factor @ (factor.T @ validation), operator.action(validation))}
            arrays[f'ETY_{gamma}_{repeat}_{rank}'] = (e.T @ factor).cpu().numpy()
            candidates.extend([
                ('sketch_coupled', factor_solve(g, factor, mu, metric.inverse), extra),
                ('sketch_diagonal', diagonal_solve(g, factor, mu, metric.inverse, groups), extra),
                ('sketch_one_sweep', sketch_sweeps(g, factor, mu, metric, groups), extra)])
        for index, (name, delta, extra) in enumerate(candidates):
            selected, search = local_direction_grid(flat, delta, float(native.norm()), lambda v: loss(fn(v), yb), c['search']['grid_exponents'])
            raw = candidate_metrics(delta, g, u, mu, metric, registry)
            selected_metrics = candidate_metrics(selected, g, u, mu, metric, registry)
            raw['actual_decrease'] = base - float(loss(fn(flat + delta), yb))
            selected_metrics['actual_decrease'] = base - float(loss(fn(flat + selected), yb))
            selected_metrics['independent_decreases'] = [initial - float(loss(f(flat + selected), target)) for initial, (f, target) in zip(panel_base, panels)]
            raw['independent_decreases'] = [initial - float(loss(f(flat + delta), target)) for initial, (f, target) in zip(panel_base, panels)]
            if extra:
                # Candidate-relevant product is evaluated against the exact operator.
                factor = next(f for k_, r_, _, f, _ in sketches if k_ == extra['rank'] and r_ == extra['repeat'])
                extra = dict(extra, candidate_action_error=relative_error(factor @ (factor.T @ delta), operator.action(delta)))
            records.append(dict(common, **extra, name=name, step=step, scope='all parameters', partition=c['partition'],
                                allocation_partition='tensor' if name.startswith('allocation_') else None,
                                gradient_norm=float(g.norm()), raw=raw, selected=selected_metrics, search=search))
            arrays[f'delta_{gamma}_{index}'] = delta.cpu().numpy()
            arrays[f'selected_{gamma}_{index}'] = selected.cpu().numpy()
        # Native amplitudes reconstruct exposure through unordered pairs with factor two.
        tau = e.T @ native
        pair = 2 * torch.triu(k * tau[:, None] * tau[None, :], diagonal=1)
        diagonal_value = float((k.diag() * tau.square()).sum())
        coupling = {'gamma': gamma, 'diagonal': diagonal_value, 'positive_pairs': float(pair.clamp_min(0).sum()),
                    'negative_pairs': float(pair.clamp_max(0).sum()), 'pair_convention': 'unordered, multiplied by two',
                    'quadratic': float((u.T @ native).square().sum())}
        coupling['reconstruction_error'] = abs(coupling['diagonal'] + coupling['positive_pairs'] + coupling['negative_pairs'] - coupling['quadratic'])
        diagnostics.setdefault('coupling', []).append(coupling)
        arrays[f'pair_contributions_{gamma}'] = pair.cpu().numpy()
    # Tiny independent dense reference only after its separate allocation guard.
    if registry.size <= c['resources']['dense_cap']:
        guard(registry.size, u.shape[1], max(c['sketch']['ranks']), c['resources'], flat.dtype, dense=True)
        dense = u @ u.T
        eig = torch.linalg.eigvalsh(dense)
        psd_tol = 100 * torch.finfo(flat.dtype).eps * max(1.0, float(eig.abs().max()))
        diagnostics['dense'] = {'min_eigenvalue': float(eig[0]), 'psd_tolerance': psd_tol,
                                'symmetry_error': relative_error(dense, dense.T)}
        if eig[0] < -psd_tol:
            raise RuntimeError('Dense GGN failed PSD check')
        system = dense + mu * metric.apply(torch.eye(registry.size, dtype=flat.dtype, device=flat.device))
        system_eig = torch.linalg.eigvalsh(system)
        condition = float(system_eig[-1] / system_eig[0])
        direct = torch.linalg.solve(system, -g)
        diagnostics['dense'].update(damped_condition=condition,
                                    dense_solve_residual=float((action(direct) + g).norm() / g.norm().clamp_min(1e-14)),
                                    dual_forward_error_bound=max(1e-9, 100 * torch.finfo(flat.dtype).eps * condition))
        diagnostics['checks']['dense_dual'] = relative_error(full, direct)
        arrays['dense_ggn'] = dense.cpu().numpy()
    else:
        diagnostics['dense'] = {'status': 'not allocated; parameter dimension cap', 'reference': 'exact output-dual'}
    diagnostics.update({'construction_ids': construction_ids, 'panel_ids': panel_ids, 'step': step,
                        'partition': c['partition'], 'scope': 'all parameters', 'counts': operator.counts,
                        'extra_probe_vjps': sum(max(c['sketch']['ranks']) * (1 + (len(xb) + c['sketch']['microbatch'] - 1) // c['sketch']['microbatch']) for _ in range(c['sketch']['repeats'])),
                        'loss_forward_evaluations': sum(r['search']['evaluations'] + 2 + 2 * len(panels) for r in records) + 1 + len(panels),
                        'wall_seconds': time.perf_counter() - start, 'allocation_estimate': allocations,
                        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                        'peak_memory_scope': 'process lifetime RSS; includes interpreter, libraries and previous snapshots'})
    np.savez_compressed(directory / 'arrays.npz', **arrays)
    torch.save({'pre_proposal_optimizer': optimizer_state, 'used_context': contexts}, directory / 'frozen_state.pt')
    write_json(directory / 'artifacts.json', {name: checksum(directory / name) for name in ['arrays.npz', 'frozen_state.pt']})
    write_json(directory / 'registry.json', registry.json())
    write_json(directory / 'diagnostics.json', diagnostics)
    write_json(directory / 'candidates.json', records)
    return diagnostics
