"""Exact tensor GGN blocks, accumulated without a full-batch Jacobian/factor."""
import time

import torch
from torch.func import jacrev

from ..models.mlp import functional_logits
from ..optim.soap import frozen_apply
from .solves import relative_error


def workspace_estimate(registry, classes, microbatch, element_bytes=8):
    sizes = [s.count for s in registry.specs]
    blocks = sum(n * n for n in sizes) * element_bytes
    largest = max(sizes) ** 2 * element_bytes
    jacobian = microbatch * classes * registry.size * element_bytes
    # Stored blocks, metric transforms, system/Cholesky/work arrays and microbatch
    # Jacobian intermediates. This is tensor workspace, not total process RSS.
    return {'block_bytes': blocks, 'largest_block_bytes': largest,
            'microbatch_jacobian_bytes': jacobian,
            'estimated_workspace_bytes': blocks + 7 * largest + 6 * jacobian,
            'scope': 'conservative tensor workspace; excludes dataset/interpreter/library RSS'}


def streamed_blocks(model, registry, flat, x, microbatch, budget_mib):
    if microbatch <= 0 or len(x) == 0:
        raise ValueError('Positive microbatch and nonempty batch required')
    estimate = workspace_estimate(registry, model.head.out_features, microbatch, flat.element_size())
    if estimate['estimated_workspace_bytes'] > budget_mib * 2**20:
        raise MemoryError('Exact tensor GGN exceeds declared workspace budget')
    started = time.perf_counter()
    blocks = [flat.new_zeros(s.count, s.count) for s in registry.specs]
    for start in range(0, len(x), microbatch):
        xb = x[start:start + microbatch]
        fn = functional_logits(model, registry, xb)
        p = fn(flat).detach().softmax(-1)
        j = jacrev(fn, chunk_size=16)(flat).detach()
        # H = diag(p) - pp^T. Weighted centered rows give a PSD square root
        # action without a vocabulary or full-batch output matrix. Every chunk
        # uses the TOTAL batch size, including a shorter final chunk.
        centered = j - (p[..., None] * j).sum(1, keepdim=True)
        response = (centered * (p / len(x)).sqrt()[..., None]).reshape(-1, registry.size)
        for spec, block in zip(registry.specs, blocks):
            local = response[:, spec.slice]
            block.addmm_(local.T, local)
        del j, centered, response
    return blocks, {'construction_seconds': time.perf_counter() - started,
                    'jacobian_rows': len(x) * model.head.out_features,
                    'microbatch': microbatch, 'batch_size': len(x),
                    'trace': sum(float(b.diag().sum()) for b in blocks),
                    'workspace': estimate, 'normalization': 'mean over complete construction batch'}


def block_action(blocks, registry, v):
    return torch.cat([block @ v[s.slice] for s, block in zip(registry.specs, blocks)])


def solve_blocks(blocks, registry, gradient, metric, mus, tolerance):
    """Direct Cholesky solves. Materialize only one tensor's metric at a time."""
    outputs = {mu: torch.zeros_like(gradient) for mu in mus}
    diagnostics = {mu: {'tensor_residuals': {}, 'seconds': 0.0} for mu in mus}
    for spec, block in zip(registry.specs, blocks):
        eye = torch.eye(spec.count, dtype=gradient.dtype, device=gradient.device)
        if hasattr(metric, 'contexts'):
            omega = metric.scale * frozen_apply(eye.reshape(spec.shape + (spec.count,)),
                        metric.contexts[spec.name], inverse=True).reshape(spec.count, spec.count)
        else:
            omega = eye
        symmetry = relative_error(omega, omega.T)
        if symmetry > tolerance:
            raise RuntimeError(f'Asymmetric damping metric for {spec.name}: {symmetry}')
        for mu in mus:
            started = time.perf_counter()
            system = block + mu * omega
            chol = torch.linalg.cholesky((system + system.T) / 2)
            value = torch.cholesky_solve(-gradient[spec.slice, None], chol).squeeze(1)
            residual = relative_error(system @ value, -gradient[spec.slice])
            if residual > tolerance:
                raise RuntimeError(f'Exact tensor solve failed for {spec.name}: {residual}')
            outputs[mu][spec.slice] = value
            diagnostics[mu]['tensor_residuals'][spec.name] = residual
            diagnostics[mu]['seconds'] += time.perf_counter() - started
            del system, chol
        del eye, omega
    for mu, value in outputs.items():
        residual = relative_error(block_action(blocks, registry, value) + mu * metric.apply(value), -gradient)
        diagnostics[mu]['residual'] = residual
        if residual > tolerance:
            raise RuntimeError(f'Exact combined tensor solve failed: {residual}')
    return outputs, diagnostics
