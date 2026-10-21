"""Part N (PROTOCOL.md, section 4): the N21 control at one point (arm, seed, t).

d = theta_t - theta_{t-1} (the step just taken); C_t at theta_{t-1}, C_old at theta_{t-11}, both on the next batch
(batch t + 1, unaugmented). Directions: d, the gradient g of batch t at theta_{t-1}, and 8 random directions
r = F0 xi (xi ~ N(0, I) in the SOAP-whitened coordinates at theta_{t-1}), scaled to |d|.
Ratio rho(u) = u'C_old u / u'C_t u.
"""
import time

import torch

import algebra
import ggn
import micro
import states

RANDOM = 8


def run_point(replay, snapshot_t, snapshot_old, seed_offset):
    """`replay` restored to theta_{t-1}; snapshots after step t and after step t - 11."""
    started = time.perf_counter()
    model, registry, optimizer = replay.arm.model, replay.arm.registry, replay.arm.optimizer
    t = replay.t+1
    if snapshot_t['t'] != t or snapshot_old['t'] != t-11:
        raise ValueError('snapshot steps do not match')
    x, y = replay.next_batch()
    xn, _ = replay.following(0)
    previous = replay.flat()
    _, g = micro.gradient(model, registry, previous, x, y, states.MICRO)
    contexts = states.contexts(optimizer, registry)
    d = snapshot_t['parameters'].to(previous.device)-previous
    old = snapshot_old['parameters'].to(previous.device)
    generator = torch.Generator().manual_seed(seed_offset)
    directions = {'step': d, 'gradient': g}
    for k in range(RANDOM):
        xi = torch.randn(registry.size, generator=generator, dtype=torch.float64).to(device=previous.device, dtype=previous.dtype)
        r = torch.zeros_like(previous)
        for spec in registry.specs:
            r[spec.slice] = ggn.unwhiten(xi[spec.slice].reshape(spec.shape), contexts[spec.name]).reshape(-1)
        directions[f'random_{k}'] = r*(d.norm()/r.norm())
    rows = {}
    for name, u in directions.items():
        now = algebra.directional_ggn(model, registry, previous, u, xn, states.MICRO)
        then = algebra.directional_ggn(model, registry, old, u, xn, states.MICRO)
        rows[name] = {'now': now, 'ten_steps_earlier': then, 'ratio': then/now if now > 0 else None}
    return {'arm': replay.arm_name, 'seed': replay.seed, 't': t, 'directions': rows, 'step_norm': float(d.norm()),
            'seconds': time.perf_counter()-started}
