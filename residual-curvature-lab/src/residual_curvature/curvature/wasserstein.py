"""Dynamic graph-Wasserstein pullback on categorical predictions.

This is a probability-dependent tangent metric, not static discrete OT cost
and not a fixed norm on signed parameter matrices.
"""
import time

import torch
from torch.func import jacrev, jvp, vjp

from .layerwise import workspace_estimate
from ..models.mlp import functional_logits


def graph_laplacian(probabilities, conductance=None):
    classes = probabilities.shape[-1]
    if classes < 2 or (probabilities < 0).any() or not torch.isfinite(probabilities).all():
        raise ValueError('Finite categorical probabilities required')
    if conductance is None:
        conductance = torch.ones(classes, classes, dtype=probabilities.dtype, device=probabilities.device)
        conductance.fill_diagonal_(0)
    if conductance.shape != (classes, classes) or (conductance < 0).any() or not torch.allclose(conductance, conductance.T):
        raise ValueError('Symmetric nonnegative graph conductance required')
    adjacency = conductance*(probabilities[..., :, None]+probabilities[..., None, :])/2
    return torch.diag_embed(adjacency.sum(-1))-adjacency


def probability_metric_solve(p, tangent, conductance=None):
    """L(p)^dagger tangent for zero-sum tangents; fix the constant gauge."""
    lap = graph_laplacian(p, conductance)
    gauge = torch.ones_like(lap)/p.shape[-1]
    return torch.linalg.solve(lap+gauge, tangent)


def action(model, registry, flat, x, v):
    fn = lambda w: functional_logits(model, registry, x)(w).softmax(-1)
    p, pullback = vjp(fn, flat)
    dp = jvp(fn, (flat,), (v,))[1]
    potential = probability_metric_solve(p.detach(), dp[..., None]).squeeze(-1)/len(x)
    return pullback(potential.detach())[0].detach()


def streamed_wasserstein_blocks(model, registry, flat, x, microbatch=8, budget_mib=768):
    estimate = workspace_estimate(registry, model.head.out_features, microbatch, flat.element_size())
    if estimate['estimated_workspace_bytes'] > budget_mib*2**20:
        raise MemoryError('Wasserstein tensor workspace exceeds budget')
    if microbatch < 1 or len(x) == 0:
        raise ValueError('Positive microbatch and construction batch required')
    started = time.perf_counter()
    blocks = [flat.new_zeros(s.count, s.count) for s in registry.specs]
    for start in range(0, len(x), microbatch):
        fn = functional_logits(model, registry, x[start:start+microbatch])
        p = fn(flat).detach().softmax(-1)
        jac = jacrev(fn, chunk_size=16)(flat).detach()
        # J_p = (diag(p)-p p^T) J_z; constants in output logits are annihilated.
        probability_jac = p[..., None]*(jac-(p[..., None]*jac).sum(1, keepdim=True))
        lap = graph_laplacian(p)
        chol = torch.linalg.cholesky(lap+torch.ones_like(lap)/p.shape[-1])
        factors = torch.linalg.solve_triangular(chol, probability_jac, upper=False)/len(x)**.5
        factors = factors.reshape(-1, registry.size)
        for spec, block in zip(registry.specs, blocks):
            local = factors[:, spec.slice]
            block.addmm_(local.T, local)
    return blocks, {'construction_seconds': time.perf_counter()-started,
                    'trace': sum(float(b.trace()) for b in blocks),
                    'jacobian_rows': len(x)*model.head.out_features,
                    'workspace': estimate, 'microbatch': microbatch,
                    'metric': 'dynamic graph Wasserstein, arithmetic edge mobility, complete unit-conductance graph',
                    'normalization': 'mean over complete construction batch'}
