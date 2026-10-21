"""Part L (PROTOCOL.md, section 3): the reduced-subspace ladder at one top-free PARSEC-L state.

Per tensor, in the state's SOAP-whitened coordinates (F0 = Q diag(den^-1/2), before the step): the exact whitened GGN
block K (ggn.accumulate with whitening), A = K + mu scale I with mu = gamma tr C / D, the whitened gradient z_g of the
next batch, the whitened momentum after this step's update z_m = F0^T (beta1 m + (1 - beta1) g). Rungs:
  topfree    top-free's actual next step (optimizer copy, weight decay 0), whitened by F0^-1 (ray reduction);
  held       Galerkin in orth[U_held, P z_m, P z_g] (P projects out U_held);
  fresh      the same with the top-16 probe subspace extracted now (PARSEC-L's randomized SVD);
  exact_top  the same with the exact top-16 eigenvectors of K;
  ggn        y* = -A^-1 z_g;
and, descriptively, held_only (Galerkin in U_held) and soap (direction -z_m).
"""
import copy
import time

import torch

import algebra
import ggn
import micro
import states
from gpu_parsec.optimizer import randomized_top, rotate
from residual_curvature.experiments.training import commit
from residual_curvature.optim.soap import SOAPMetric

GAMMA, FLOOR, RANK, CHUNK, GROUP_CHUNK, BUDGET = 1e-2, 1e-8, 16, 64, 128, int(2.5*2**30)
CONSISTENCY = 1e-3  # gate L0
RUNGS = ['topfree', 'held', 'fresh', 'exact_top', 'ggn']
DESCRIPTIVE = ['held_only', 'soap']
MULTIPLIERS = [2.**k for k in range(-14, 3)]


def proposal(replay, x, y):
    """Top-free's next step with weight decay 0 from a copy of the state; parameters, optimizer and probe generator
    restored."""
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
    """F0^T v for a gradient-like (covector) v."""
    return ggn.whiten(vector[None], context)[0].reshape(-1)


def coordinates(step, context):
    """F0^-1 d for a step d: sqrt(den) Q^T d."""
    return (rotate(step[None], context['basis'], transpose=True)[0]*context['denominator'].sqrt()).reshape(-1)


def complement_basis(u, zm, zg):
    pm = zm-u@(u.T@zm)
    pg = zg-u@(u.T@zg)
    return algebra.orth(torch.cat([u, pm[:, None], pg[:, None]], 1))


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
    lr_next = states.LRS['topfree']*states.GGN_TRAIN.lr_factor(replay.t+1)
    beta1 = optimizer.param_groups[0]['betas'][0]
    held, momenta = {}, {}
    for spec in registry.specs:
        p, state = params[spec.name], optimizer.state[params[spec.name]]
        brake = algebra.brake_directions(optimizer, p, lr_next, state['step']+1)
        held[spec.name] = brake[0] if brake is not None else torch.zeros(spec.count, 0, device=flat.device)
        momenta[spec.name] = beta1*optimizer.project_back(state['exp_avg'], state)+(1-beta1)*g[spec.slice].reshape(spec.shape)
    probes = micro.lean_probes(model, x, samples=1, generator=torch.Generator().manual_seed(probe_seed))
    sketch = torch.Generator().manual_seed(probe_seed+1)
    fresh = {}
    for spec in registry.specs:
        rows = probes[params[spec.name]]
        z = ggn.whiten(rows, contexts[spec.name]).reshape(len(rows), -1).T
        fresh[spec.name], _ = randomized_top(z, RANK, sketch, 8, 2)
    del probes
    top_generator = torch.Generator().manual_seed(probe_seed+2)
    vectors = {name: torch.zeros_like(flat) for name in RUNGS+DESCRIPTIVE}
    rows, failures, mu = {}, [], None
    for index, specs in enumerate(ggn.groups_for(registry, BUDGET, flat.element_size())):
        blocks, trace = ggn.accumulate(model, registry, flat, x, specs, CHUNK if index == 0 else GROUP_CHUNK,
                                       all_parameters=index == 0, contexts=contexts)
        if index == 0:
            mu = max(GAMMA*trace/registry.size, FLOOR)
        damping = mu*metric.scale
        for spec in specs:
            context = contexts[spec.name]
            system = ggn.symmetrize_(blocks.pop(spec.name))
            exact = algebra.top_eigenvectors(system, RANK, top_generator)
            system.diagonal().add_(damping)
            zg = whitened(g[spec.slice].reshape(spec.shape), context)
            zm = whitened(momenta[spec.name], context)
            chol = torch.linalg.cholesky(system)
            ystar = torch.cholesky_solve(-zg[:, None], chol)[:, 0]
            del chol
            optimum = -.5*float((zg.double()*ystar.double()).sum())
            steps = {'topfree': coordinates(step_tf[spec.slice].reshape(spec.shape), context), 'soap': -zm, 'ggn': ystar}
            reductions = {'topfree': algebra.ray(system, zg, steps['topfree']), 'soap': algebra.ray(system, zg, -zm),
                          'ggn': optimum}
            for label, u in (('held', held[spec.name]), ('fresh', fresh[spec.name]), ('exact_top', exact)):
                steps[label], reductions[label] = algebra.galerkin(system, zg, complement_basis(u, zm, zg))
            steps['held_only'], reductions['held_only'] = algebra.galerkin(system, zg, held[spec.name])
            for label, value in reductions.items():
                if label != 'ggn' and value > optimum*(1+CONSISTENCY)+1e-12:
                    failures.append({'tensor': spec.name, 'rung': label, 'reduction': value, 'optimum': optimum})
            linear = {label: float((zg.double()*s.double()).sum()) for label, s in steps.items()}
            curvature = {label: float((s.double()*(system@s).double()).sum()) for label, s in steps.items()}
            for label, s in steps.items():
                vectors[label][spec.slice] = ggn.unwhiten(s.reshape(spec.shape), context).reshape(-1)
            rows[spec.name] = {'count': spec.count, 'reductions': reductions, 'linear': linear, 'curvature': curvature,
                               'held_rank': held[spec.name].shape[1]}
            del system
        del blocks
    if failures:
        raise RuntimeError(f'Gate L0 failed: {failures[:5]}')
    base_selection = micro.batch_loss(model, registry, flat, xs, ys, states.MICRO)
    base_evaluation = micro.batch_loss(model, registry, flat, xe, ye, states.MICRO)
    transfer = {}
    for label in RUNGS+DESCRIPTIVE:
        lin = sum(r['linear'][label] for r in rows.values())
        cur = sum(r['curvature'][label] for r in rows.values())
        alpha0 = -lin/cur if lin < 0 and cur > 0 else 0.
        selection = {m: base_selection-micro.batch_loss(model, registry, flat+alpha0*m*vectors[label], xs, ys, states.MICRO)
                     for m in MULTIPLIERS}
        chosen = max(selection, key=selection.get)
        transfer[label] = {'alpha0': alpha0, 'multiplier': chosen,
                           'interior': MULTIPLIERS[0] < chosen < MULTIPLIERS[-1],
                           'evaluation_decrease': base_evaluation-micro.batch_loss(model, registry, flat+alpha0*chosen*vectors[label],
                                                                                   xe, ye, states.MICRO)}
    return {'arm': replay.arm_name, 'seed': replay.seed, 'step': replay.t, 'mu': mu, 'scale': metric.scale,
            'tensors': rows, 'transfer': transfer, 'seconds': time.perf_counter()-started}
