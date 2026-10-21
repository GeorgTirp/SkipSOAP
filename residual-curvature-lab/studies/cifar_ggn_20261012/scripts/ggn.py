"""Exact per-tensor GGN step on the GPU within a memory budget (PROTOCOL.md, section 3).

The step is the one of residual_curvature.curvature.layerwise (streamed_blocks, solve_blocks), as used by the gap
audit's and stage D's exact tensor GGN:

    C_bb = (1/N) sum_i J_ib^T (diag p_i - p_i p_i^T) J_ib          (mean loss over the complete batch)
    delta_b = -(C_bb + mu Omega_b)^-1 g_b,   mu = max(gamma tr C / D, floor),
    Omega = scale P^-1, the normalized inverse SOAP metric (SOAPMetric).

It is solved in SOAP-whitened coordinates. With SOAP's own factor F0 = Q diag(d^-1/2), P = F0 F0^T (no orthogonality
of Q is assumed), the identity

    C + mu scale P^-1 = F0^-T (F0^T C F0 + mu' I) F0^-1,   mu' = mu scale,

gives delta_b = -F0 (K_b + mu' I)^-1 F0^T g_b with K_b = F0^T C_bb F0, the whitened GGN block that PARSEC-L
approximates. K_b is accumulated from whitened Jacobian rows. In float32 the original-coordinate systems are not
solvable early in training: at step 2 of the CIFAR dry run their condition numbers were 1.7e7 to 1.8e8, because
SOAP's denominators span seven decades. The whitened system's condition number is about 1 + kappa_max.

Implementation differences from the lab code, for memory, speed and float32 only:
- Per-example Jacobians come from vmap over examples of jacrev of one example, which costs about K backward passes
  of the batch. jacrev over a microbatch costs the microbatch size times more.
- Tensors are processed in groups whose blocks fit `budget_bytes`. The first pass differentiates all parameters
  (for tr C, hence mu, from unwhitened rows); later passes differentiate only their group's. Each block is solved
  and freed as soon as its group is complete.
- The whitened residual |(K_b + mu' I) z + F0^T g_b| / |F0^T g_b| is checked against `tolerance` (lab config:
  1e-8 in float64, 1e-4 in float32). The original-coordinate residual |F0^-T r_w| / |g| is reported.
"""
import time

import torch
from torch.func import functional_call, jacrev, vmap

from gpu_parsec.optimizer import rotate
from residual_curvature.optim.soap import mode


def groups_for(registry, budget_bytes, element_bytes):
    """Consecutive tensors whose dense blocks fit the budget together (a larger single block is its own group)."""
    groups, current, used = [], [], 0
    for spec in registry.specs:
        size = spec.count**2*element_bytes
        if current and used+size > budget_bytes:
            groups.append(current)
            current, used = [], 0
        current.append(spec)
        used += size
    if current:
        groups.append(current)
    return groups


def whiten(rows, context):
    """F0^T v for each row v (rows of shape (n, *shape)): Q^T along every preconditioned mode, then / sqrt(d)."""
    return rotate(rows, context['basis'], transpose=True)/context['denominator'].sqrt()


def unwhiten(z, context):
    """F0 z = Q (z / sqrt(d)) for one tensor-shaped z."""
    return rotate((z/context['denominator'].sqrt())[None], context['basis'], transpose=False)[0]


def inverse_transpose(r, context):
    """F0^-T r = Q^-T (sqrt(d) r), by solves (the lab's frozen_apply convention for approximately orthogonal Q)."""
    v = r*context['denominator'].sqrt()
    for axis, basis in enumerate(context['basis']):
        v = mode(v, basis, axis, 'solve_transpose')
    return v


def example_jacobians(model, registry, flat, x, names, chunk):
    """Per chunk of examples: (logits (n, K), {name: Jacobian of each example's logits, (n, K, *shape)}) with
    respect to the named parameters, all others fixed at `flat`."""
    params = {k: v.detach() for k, v in registry.unflatten(flat).items()}
    fixed = {k: v for k, v in params.items() if k not in names}
    sub = {k: params[k] for k in names}

    def logits(part, xi):
        return functional_call(model, {**fixed, **part}, (xi[None],))[0]

    jacobian = vmap(jacrev(logits), in_dims=(None, 0))
    for start in range(0, len(x), chunk):
        xb = x[start:start+chunk]
        with torch.no_grad():
            z = functional_call(model, params, (xb,))
        yield z, jacobian(sub, xb)


def accumulate(model, registry, flat, x, specs, chunk, all_parameters, contexts=None):
    """Dense blocks for `specs`: C_bb, or the whitened K_b = F0^T C_bb F0 if SOAP `contexts` are given; with
    all_parameters also tr C (unwhitened) over every tensor."""
    names = [s.name for s in registry.specs] if all_parameters else [s.name for s in specs]
    wanted = {s.name: s for s in specs}
    blocks = {s.name: flat.new_zeros(s.count, s.count) for s in specs}
    trace, total = flat.new_zeros(()), len(x)
    for z, jacobians in example_jacobians(model, registry, flat, x, names, chunk):
        p = z.softmax(-1)
        n, classes = p.shape
        weight = (p/total).sqrt()[..., None]
        for name in names:
            j = jacobians[name].reshape(n, classes, -1)
            response = ((j-(p[..., None]*j).sum(1, keepdim=True))*weight).reshape(n*classes, -1)
            if all_parameters:
                trace += response.square().sum()
            if name in wanted:
                spec = wanted[name]
                if contexts is not None:
                    response = whiten(response.reshape((n*classes,)+spec.shape), contexts[name]).reshape(n*classes, -1)
                blocks[name].addmm_(response.T, response)
        del jacobians
    return blocks, float(trace) if all_parameters else None


def symmetrize_(s, width=2048):
    """s <- (s + s^T)/2 in place, a block pair at a time (memory width^2 instead of a full transposed copy)."""
    n = len(s)
    for i in range(0, n, width):
        for j in range(i, n, width):
            upper, lower = s[i:i+width, j:j+width], s[j:j+width, i:i+width]
            mean = (upper+lower.T)/2
            upper.copy_(mean)
            lower.copy_(mean.T)
    return s


def top_eigenvalue(s, generator, iterations=20):
    """Largest eigenvalue of a PSD matrix by power iteration (a lower bound; logged at audit steps only)."""
    v = torch.randn(len(s), generator=generator, dtype=s.dtype).to(s.device)
    for _ in range(iterations):
        v = s@v
        v = v/v.norm()
    return float(v@(s@v))


def tensor_ggn_step(model, registry, flat, x, gradient, metric, gamma, floor, tolerance, budget_bytes, chunk=64,
                    action=None, audit_generator=None, group_chunk=128):
    """delta = -(C_bb + mu Omega_b)^-1 g_b for every tensor b, solved in whitened coordinates; returns (delta, info).

    With `action` (the full GGN action v -> C v) and `audit_generator`, every whitened block is checked against
    F0^T C F0 u from the action, for a random u supported on its tensor, before damping is added; the error is
    relative to the damped system's action |F0^T C F0 u + mu' u|."""
    started = time.perf_counter()
    groups = groups_for(registry, budget_bytes, flat.element_size())
    delta = torch.zeros_like(gradient)
    info = {'groups': len(groups), 'whitened_residuals': {}, 'residuals': {}, 'action_checks': {}, 'kappa_max': {},
            'mu': None, 'trace': None, 'scale': metric.scale}
    squared = squared_rhs = squared_original = 0.
    for index, specs in enumerate(groups):
        # The first pass differentiates every tensor (for tr C) and uses the smaller chunk to fit memory.
        blocks, trace = accumulate(model, registry, flat, x, specs, chunk if index == 0 else group_chunk,
                                   all_parameters=index == 0, contexts=metric.contexts)
        if index == 0:
            info['trace'], info['mu'] = trace, max(gamma*trace/registry.size, floor)
        damping = info['mu']*metric.scale
        for spec in specs:
            block, context = blocks.pop(spec.name), metric.contexts[spec.name]
            if action is not None:
                u = torch.randn(spec.count, generator=audit_generator, dtype=flat.dtype).to(flat.device)
                v = torch.zeros_like(flat)
                v[spec.slice] = unwhiten(u.reshape(spec.shape), context).reshape(-1)
                expected = whiten(action(v)[spec.slice].reshape((1,)+spec.shape), context).reshape(-1)
                # Relative to the damped system's action (K_b + mu' I) u: an error below that cannot move the step by
                # more than the tolerance. For live tensors it is the plain relative error (curvature dominates mu');
                # the attention key biases have zero gradient and pure-rounding blocks.
                info['action_checks'][spec.name] = float((block@u-expected).norm())/float((expected+damping*u).norm())
                if info['action_checks'][spec.name] > tolerance:
                    raise RuntimeError(f'Curvature action audit failed for {spec.name}: {info["action_checks"][spec.name]}')
            system = symmetrize_(block)
            if action is not None:
                info['kappa_max'][spec.name] = top_eigenvalue(system, audit_generator)/damping
            system.diagonal().add_(damping)
            rhs = whiten(gradient[spec.slice].reshape((1,)+spec.shape), context).reshape(-1)
            chol = torch.linalg.cholesky(system)
            z = torch.cholesky_solve(-rhs[:, None], chol).squeeze(1)
            del chol
            whitened = system@z+rhs
            del system, block
            rhs_norm = float(rhs.norm())
            residual = float(whitened.norm())/rhs_norm if rhs_norm > 0 else float(whitened.norm())
            if residual > tolerance:
                raise RuntimeError(f'Exact tensor solve failed for {spec.name}: whitened residual {residual}')
            original = inverse_transpose(whitened.reshape(spec.shape), context).reshape(-1)
            g_norm = float(gradient[spec.slice].norm())
            delta[spec.slice] = unwhiten(z.reshape(spec.shape), context).reshape(-1)
            info['whitened_residuals'][spec.name] = residual
            info['residuals'][spec.name] = float(original.norm())/g_norm if g_norm > 0 else float(original.norm())
            squared += float(whitened.square().sum())
            squared_rhs += rhs_norm**2
            squared_original += float(original.square().sum())
    info['residual'] = (squared/squared_rhs)**.5 if squared_rhs > 0 else 0.
    info['residual_original'] = squared_original**.5/float(gradient.norm()) if float(gradient.norm()) > 0 else 0.
    info['seconds'] = time.perf_counter()-started
    return delta, info
