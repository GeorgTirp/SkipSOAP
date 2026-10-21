"""Part B (PROTOCOL.md, section 5): along a SOAP window with refresh switched off, the benefit of refreshing now and
three cheap signals, per matrix tensor.

At each window state (after step t; basis age a = t + 1 - s for the step t + 1):
- candidates: SOAP's step t + 1 with the held basis, and after applying SOAP's refresh now (states.refresh_now), both
  with weight decay 0, from copies of the optimizer state;
- ground truth: per-tensor ray reductions of both on the damped quadratic (one per-example-Jacobian pass), and the
  next-batch loss decrease of both at their real size;
- signals: I15a factor rotation of SOAP's own factors, N8 probe off-diagonal ratio (and its rise), I10 spread of the
  probe factors' top-8 eigenvectors over the held basis.
The trajectory then continues with the held step and the real weight decay.
"""
import copy
import time

import torch

import micro
import quad
import states
from residual_curvature.optim.soap import SOAPMetric

GAMMA, FLOOR, WINDOW, TOP, PROBE_SEED = 1e-2, 1e-8, 30, 8, 9200


def offdiagonal(matrix):
    """(off-diagonal Frobenius^2, total Frobenius^2)."""
    total = float(matrix.square().sum())
    return total-float(matrix.diagonal().square().sum()), total


def rotation(factors, bases):
    """sqrt(sum off^2 / sum total^2) of Q^T F Q over the rotated modes."""
    off = total = 0.
    for f, q in zip(factors, bases):
        if len(f) and len(q) and len(q) > 1:
            o, t = offdiagonal(q.T@f@q)
            off, total = off+o, total+t
    return (off/total)**.5 if total > 0 else None


def spread(factors, bases, top=TOP):
    """Mean over the top eigenvectors v of each probe factor of 1 - max_i (q_i'v)^2, pooled over the rotated modes."""
    values = []
    for f, q in zip(factors, bases):
        if len(f) and len(q) and len(q) > 1:
            v = quad.eigenbasis(f)[:, :min(top, len(q))]
            values += (1-(q.T@v).square().max(0).values).tolist()
    return sum(values)/len(values) if values else None


def proposal(replay, x, y, refresh):
    """SOAP's next step (weight decay 0) from a copy of the state, with or without refreshing now; state restored."""
    optimizer, flat = replay.arm.optimizer, replay.flat().clone()
    saved, t = copy.deepcopy(optimizer.state_dict()), replay.t
    decays = [g['weight_decay'] for g in optimizer.param_groups]
    try:
        for g in optimizer.param_groups:
            g['weight_decay'] = 0.
        if refresh:
            states.refresh_now(optimizer)
        replay.arm.step(t+1, x, y)
        return replay.flat()-flat
    finally:
        for g, decay in zip(optimizer.param_groups, decays):
            g['weight_decay'] = decay
        from residual_curvature.experiments.training import commit
        commit(replay.arm.model, replay.arm.registry, flat)
        optimizer.load_state_dict(copy.deepcopy(saved))


def window(replay, start_probe_ratio=None):
    """One window from the replay's current state (right after a refresh); returns the per-step rows."""
    replay.freeze_bases()
    model, registry, optimizer = replay.arm.model, replay.arm.registry, replay.arm.optimizer
    params = {spec.name: p for p, spec in zip(model.parameters(), registry.specs)}
    matrices = [spec for spec in registry.specs if len(spec.shape) == 2]
    start, rows, base = replay.t, [], {}
    for _ in range(WINDOW):
        began = time.perf_counter()
        x, y = replay.next_batch()
        xn, yn = replay.following(0)
        flat = replay.flat()
        contexts = states.soap_contexts(optimizer, registry)
        held = proposal(replay, x, y, refresh=False)
        refreshed = proposal(replay, x, y, refresh=True)
        _, g = micro.gradient(model, registry, flat, x, y, states.MICRO)
        probes = micro.lean_probes(model, x, samples=1, generator=torch.Generator().manual_seed(PROBE_SEED+replay.t))
        signals = {}
        for spec in matrices:
            q = contexts[spec.name]['basis']
            modes = {k for k, m in enumerate(q) if len(m)}
            pf = quad.probe_factors(probes[params[spec.name]], modes)
            rho = rotation(pf, q)
            if replay.t == start:
                base[spec.name] = rho
            signals[spec.name] = {'factor_rotation': rotation(optimizer.state[params[spec.name]]['GG'], q),
                                  'probe_rho': rho, 'probe_rho_rise': rho-base[spec.name] if rho is not None else None,
                                  'spread': spread(pf, q)}
        del probes
        quadratics, trace = quad.tensor_quadratics(model, registry, flat, x, [held, refreshed])
        mu = max(GAMMA*trace/registry.size, FLOOR)
        metric = SOAPMetric(registry, contexts)
        tensors = {}
        for spec in matrices:
            gb = g[spec.slice].reshape(spec.shape)
            out = {}
            for k, (label, d) in enumerate((('held', held), ('refreshed', refreshed))):
                db = d[spec.slice].reshape(spec.shape)
                damping = quad.damping_quadratic(db, contexts[spec.name], mu, metric.scale)
                out[label] = quad.ray(float((gb*db).sum()), quadratics[spec.name][k]+damping)
            top = max(out['held'], out['refreshed'])
            out['benefit'] = (out['refreshed']-out['held'])/top if top > 0 else None
            tensors[spec.name] = {**out, **signals[spec.name]}
        base_loss = micro.batch_loss(model, registry, flat, xn, yn, states.MICRO)
        nonlinear = {label: base_loss-micro.batch_loss(model, registry, flat+d, xn, yn, states.MICRO)
                     for label, d in (('held', held), ('refreshed', refreshed))}
        rows.append({'t': replay.t, 'age': replay.t+1-start, 'mu': mu, 'tensors': tensors, 'next_batch_decrease': nonlinear,
                     'seconds': time.perf_counter()-began})
        replay.step(x, y)  # the trajectory continues with the held basis and the real weight decay
    return rows
