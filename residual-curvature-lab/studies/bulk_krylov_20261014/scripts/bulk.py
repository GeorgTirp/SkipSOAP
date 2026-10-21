"""Parts P and K (PROTOCOL.md, sections 3-4) at one top-free state.

Per tensor, in the state's SOAP-whitened coordinates (as ladder_mechanism_20261013's ladder.py): the exact whitened GGN
block K, A = K + mu scale I, the whitened gradient z_g of the next batch, the exact step y* and top-free's own step.
Part P fits power laws to the exact top-200 spectrum and to the probe sketch. Part K scores the rungs (see
RUNGS) on the damped per-tensor quadratic, and every rung's global step on held-out data.
"""
import copy
import time

import torch

import algebra
import common
import ggn
import micro
import spectral
import states
from gpu_parsec.optimizer import randomized_top, rotate
from residual_curvature.experiments.training import commit
from residual_curvature.optim.soap import SOAPMetric

GAMMA, FLOOR, RANK, SPECTRUM, CHUNK, GROUP_CHUNK, BUDGET = 1e-2, 1e-8, 16, 200, 64, 128, int(2.5*2**30)
CONSISTENCY = 1e-3  # gate L0
KRYLOV = [2, 4, 8]
CHEBYSHEV = [4, 8]
CG_STEPS, SUB_BATCH = 5, 256
RUNGS = (['topfree', 'ggn', 'diag_exact', 'diag_powerlaw'] + [f'krylov_{m}' for m in KRYLOV]
         + [f'deflated_krylov_{m}' for m in KRYLOV] + [f'deflated_chebyshev_{m}' for m in CHEBYSHEV]
         + ['full_cg_sub', 'full_cg_full'])
MULTIPLIERS = [2.**k for k in range(-14, 3)]


def proposal(replay, x, y):
    """Top-free's next step with weight decay 0 from a copy of the state (as ladder_mechanism's ladder.py)."""
    optimizer, flat = replay.arm.optimizer, replay.flat().clone()
    saved, generator = copy.deepcopy(optimizer.state_dict()), replay.arm.probe_generator.get_state()
    decays = [g['weight_decay'] for g in optimizer.param_groups]
    try:
        for g in optimizer.param_groups:
            g['weight_decay'] = 0.
        replay.arm.step(replay.t+1, x, y, None)
        return replay.flat()-flat
    finally:
        for g, decay in zip(optimizer.param_groups, decays):
            g['weight_decay'] = decay
        commit(replay.arm.model, replay.arm.registry, flat)
        optimizer.load_state_dict(states.to(saved, flat.device))
        replay.arm.probe_generator.set_state(generator)


def whitened(vector, context):
    return ggn.whiten(vector[None], context)[0].reshape(-1)


def coordinates(step, context):
    return (rotate(step[None], context['basis'], transpose=True)[0]*context['denominator'].sqrt()).reshape(-1)


def powerlaw_step(zg, top, alpha, probe_diagonal, damping):
    """-z_g / (lambda_rank(i) + damping), lambda_k = top k^-alpha assigned by the rank of the probe diagonal (rank 1 the
    largest)."""
    assumed = spectral.power_law_values(top, alpha, len(zg), zg.dtype, zg.device)
    ranks = torch.argsort(torch.argsort(probe_diagonal, descending=True, stable=True), stable=True)
    return -zg/(assumed[ranks]+damping)


def whitened_operator(model, registry, flat, x, contexts, damping):
    """v -> (F0^T C F0 + damping I) v over all tensors at once, with C the full GGN of batch x (matrix-free)."""
    action = micro.ggn_action(model, registry, flat, x, states.MICRO)

    def apply(v):
        u = torch.zeros_like(flat)
        for spec in registry.specs:
            u[spec.slice] = ggn.unwhiten(v[spec.slice].reshape(spec.shape), contexts[spec.name]).reshape(-1)
        cu = action(u)
        out = torch.zeros_like(flat)
        for spec in registry.specs:
            out[spec.slice] = whitened(cu[spec.slice].reshape(spec.shape), contexts[spec.name])
        return out+damping*v
    return apply


def whitened_full_cg(model, registry, flat, x, contexts, zg_full, damping, steps):
    """`steps` CG steps on (F0^T C F0 + damping I) y = -z_g from y = 0."""
    return spectral.conjugate_gradient(whitened_operator(model, registry, flat, x, contexts, damping), -zg_full, steps)


def run_state(replay, probe_seed):
    started = time.perf_counter()
    model, registry, optimizer = replay.arm.model, replay.arm.registry, replay.arm.optimizer
    x, y = replay.next_batch()
    xs, ys = replay.following(0)
    xe, ye = replay.following(1)
    flat = replay.flat()
    contexts = states.contexts(optimizer, registry)
    metric = SOAPMetric(registry, contexts)
    _, g = micro.gradient(model, registry, flat, x, y, states.MICRO)
    step_tf = proposal(replay, x, y)
    params = {spec.name: p for p, spec in zip(model.parameters(), registry.specs)}

    # Probes: top-16 directions and eigenvalues, trace and diagonal per tensor (whitened); then free the rows.
    probes = micro.lean_probes(model, x, samples=1, generator=torch.Generator().manual_seed(probe_seed))
    sketch = torch.Generator().manual_seed(probe_seed+1)
    probe = {}
    for spec in registry.specs:
        rows = probes[params[spec.name]]
        z = ggn.whiten(rows, contexts[spec.name]).reshape(len(rows), -1).T
        u, kappa = randomized_top(z, RANK, sketch, 8, 2)
        probe[spec.name] = {'u': u, 'kappa': kappa.double().cpu(), 'trace': float(z.square().sum()), 'diag': z.square().sum(1)}
    del probes

    # mu needs tr C, which the ladder takes from the first group of the dense pass. The full-GGN CG rungs need mu before
    # that pass, so tr C is taken here by the same call without blocks. The dense pass below repeats the ladder's calls
    # exactly (its blocks, R* and top-free reductions are compared with the ladder's in summarize.py) and its trace is
    # recorded against this one.
    _, trace = ggn.accumulate(model, registry, flat, x, [], CHUNK, all_parameters=True, contexts=contexts)
    mu = max(GAMMA*trace/registry.size, FLOOR)
    damping = mu*metric.scale
    zg_full = torch.zeros_like(flat)
    for spec in registry.specs:
        zg_full[spec.slice] = whitened(g[spec.slice].reshape(spec.shape), contexts[spec.name])
    timing = {}
    full = {}
    for label, batch in (('full_cg_sub', x[:SUB_BATCH]), ('full_cg_full', x)):
        common.synchronize(flat.device)
        began = time.perf_counter()
        full[label] = whitened_full_cg(model, registry, flat, batch, contexts, zg_full, damping, CG_STEPS)
        common.synchronize(flat.device)
        timing[f'{label}_seconds_per_product'] = (time.perf_counter()-began)/CG_STEPS

    top_generator = torch.Generator().manual_seed(probe_seed+2)
    vectors = {name: torch.zeros_like(flat) for name in RUNGS}
    rows, failures = {}, []
    dense_trace = None
    for index, specs in enumerate(ggn.groups_for(registry, BUDGET, flat.element_size())):
        blocks, group_trace = ggn.accumulate(model, registry, flat, x, specs, CHUNK if index == 0 else GROUP_CHUNK,
                                             all_parameters=index == 0, contexts=contexts)
        if index == 0:
            dense_trace = group_trace
        for spec in specs:
            context, pr = contexts[spec.name], probe[spec.name]
            system = ggn.symmetrize_(blocks.pop(spec.name))
            # Part P: exact spectrum and fits; probe-based prediction.
            values, eigenvectors = spectral.top_eigenpairs(system, SPECTRUM, top_generator, oversample=SPECTRUM)
            last = eigenvectors[:, -1]
            ritz_residual = float((system@last-values[-1]*last).norm()/values[-1]) if float(values[-1]) > 0 else None
            del eigenvectors
            values = values.double().cpu()
            trace_k = float(system.diagonal().double().sum())
            diag_k = system.diagonal().clone()
            alpha_all, r2_all = spectral.loglog_fit(values, 1, SPECTRUM)
            alpha_top, r2_top = spectral.loglog_fit(values, 1, RANK)
            alpha_bulk, r2_bulk = spectral.loglog_fit(values, RANK+1, SPECTRUM)
            kappa = pr['kappa']
            alpha_slope, _ = spectral.loglog_fit(kappa, 1, RANK)
            alpha_trace = spectral.alpha_from_trace(float(kappa[0]), pr['trace'], spec.count) if len(kappa) else None
            lambda17 = float(kappa[0])*(RANK+1)**-alpha_trace if alpha_trace is not None else None
            # Part K.
            system.diagonal().add_(damping)
            zg = zg_full[spec.slice]
            chol = torch.linalg.cholesky(system)
            ystar = torch.cholesky_solve(-zg[:, None], chol)[:, 0]
            del chol
            optimum = -.5*float((zg.double()*ystar.double()).sum())
            apply = lambda v: system@v
            u = pr['u']
            projector = lambda v: v-u@(u.T@v)
            steps = {'topfree': coordinates(step_tf[spec.slice].reshape(spec.shape), context), 'ggn': ystar,
                     'diag_exact': -zg/(diag_k+damping), 'full_cg_sub': full['full_cg_sub'][spec.slice],
                     'full_cg_full': full['full_cg_full'][spec.slice]}
            steps['diag_powerlaw'] = (powerlaw_step(zg, float(kappa[0]), alpha_trace, pr['diag'], damping)
                                      if alpha_trace is not None else torch.zeros_like(zg))
            reductions = {'ggn': optimum}
            for m in KRYLOV:
                steps[f'krylov_{m}'], reductions[f'krylov_{m}'] = algebra.galerkin(system, zg, spectral.krylov_basis(apply, zg, m))
                bulk = spectral.krylov_basis(lambda v: projector(apply(projector(v))), projector(zg), m)
                basis = algebra.orth(torch.cat([u, bulk], 1))
                steps[f'deflated_krylov_{m}'], reductions[f'deflated_krylov_{m}'] = algebra.galerkin(system, zg, basis)
            y_top, _ = algebra.galerkin(system, zg, u)
            for m in CHEBYSHEV:
                if lambda17 is not None and lambda17 > 0:
                    xb = spectral.chebyshev_solve(lambda v: projector(apply(projector(v))), -projector(zg), damping, lambda17+damping, m)
                    steps[f'deflated_chebyshev_{m}'] = y_top+projector(xb)
                else:
                    steps[f'deflated_chebyshev_{m}'] = y_top
            for label in RUNGS:
                if label not in reductions:
                    reductions[label] = algebra.ray(system, zg, steps[label])
                if label != 'ggn' and reductions[label] > optimum*(1+CONSISTENCY)+1e-12:
                    failures.append({'tensor': spec.name, 'rung': label, 'reduction': reductions[label], 'optimum': optimum})
            linear = {label: float((zg.double()*steps[label].double()).sum()) for label in RUNGS}
            curvature = {label: float((steps[label].double()*(system@steps[label]).double()).sum()) for label in RUNGS}
            for label in RUNGS:
                vectors[label][spec.slice] = ggn.unwhiten(steps[label].reshape(spec.shape), context).reshape(-1)
            rows[spec.name] = {
                'count': spec.count, 'shape': list(spec.shape), 'reductions': reductions, 'linear': linear, 'curvature': curvature,
                'spectrum': {'top': values[:SPECTRUM].tolist(), 'trace': trace_k, 'alpha_all': alpha_all, 'r2_all': r2_all,
                             'alpha_top': alpha_top, 'r2_top': r2_top, 'alpha_bulk': alpha_bulk, 'r2_bulk': r2_bulk,
                             'ritz_residual_last': ritz_residual},
                'probe': {'kappa': kappa.tolist(), 'trace': pr['trace'], 'alpha_slope': alpha_slope, 'alpha_trace': alpha_trace,
                          'lambda17': lambda17}}
            del system
        del blocks
    if failures:
        raise RuntimeError(f'Gate L0 failed: {failures[:5]}')
    base_selection = micro.batch_loss(model, registry, flat, xs, ys, states.MICRO)
    base_evaluation = micro.batch_loss(model, registry, flat, xe, ye, states.MICRO)
    transfer = {}
    for label in RUNGS:
        lin = sum(r['linear'][label] for r in rows.values())
        cur = sum(r['curvature'][label] for r in rows.values())
        alpha0 = -lin/cur if lin < 0 and cur > 0 else 0.
        selection = {m: base_selection-micro.batch_loss(model, registry, flat+alpha0*m*vectors[label], xs, ys, states.MICRO)
                     for m in MULTIPLIERS}
        chosen = max(selection, key=selection.get)
        transfer[label] = {'alpha0': alpha0, 'multiplier': chosen, 'interior': MULTIPLIERS[0] < chosen < MULTIPLIERS[-1],
                           'evaluation_decrease': base_evaluation-micro.batch_loss(model, registry, flat+alpha0*chosen*vectors[label],
                                                                                   xe, ye, states.MICRO)}
    return {'arm': replay.arm_name, 'seed': replay.seed, 'step': replay.t, 'mu': mu, 'scale': metric.scale, 'damping': damping,
            'trace': trace, 'dense_trace': dense_trace,
            'tensors': rows, 'transfer': transfer, 'timing': timing, 'seconds': time.perf_counter()-started}
