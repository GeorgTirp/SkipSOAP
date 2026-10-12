"""Controlled output-information diagnostics, with the true network unchanged."""
import time

import torch
from torch.func import jacrev

from .layerwise import workspace_estimate
from ..models.mlp import functional_logits
from ..optim.soap import mode


def information_blocks(model, registry, flat, x, y, ranks=(1, 4, 16), repeats=2,
                       seed=303, microbatch=8, budget_mib=768):
    if not ranks or min(ranks) < 1 or repeats < 1 or microbatch < 1 or len(x) == 0:
        raise ValueError('Positive sampling ranks, repeats, microbatch and batch required')
    estimate = workspace_estimate(registry, model.head.out_features, microbatch, flat.element_size())
    collections = 2 + len(ranks)*repeats
    total_workspace = estimate['estimated_workspace_bytes'] + (collections-1)*estimate['block_bytes']
    if total_workspace > budget_mib*2**20:
        raise MemoryError('Output-information diagnostic exceeds tensor-workspace budget')
    started = time.perf_counter()
    names = ['ggn', 'empirical_fisher'] + [f'sampled_{rank}_repeat_{repeat}'
                                         for repeat in range(repeats) for rank in ranks]
    blocks = {name: [flat.new_zeros(s.count, s.count) for s in registry.specs] for name in names}
    gen = torch.Generator(device=flat.device).manual_seed(seed)
    uniforms = torch.rand(repeats, len(x), max(ranks), generator=gen, dtype=flat.dtype, device=flat.device)
    labels = torch.empty_like(uniforms, dtype=torch.long)
    gradient = torch.zeros_like(flat)
    for start in range(0, len(x), microbatch):
        xb, yb = x[start:start+microbatch], y[start:start+microbatch]
        fn = functional_logits(model, registry, xb)
        p = fn(flat).detach().softmax(-1)
        jac = jacrev(fn, chunk_size=16)(flat).detach()
        mean_jac = (p[..., None]*jac).sum(1)
        observed = mean_jac - jac[torch.arange(len(xb), device=flat.device), yb]
        gradient += observed.sum(0)/len(x)
        factors = {
            'ggn': ((jac-mean_jac[:, None])* (p/len(x)).sqrt()[..., None]).reshape(-1, registry.size),
            'empirical_fisher': observed/len(x)**.5,
        }
        for repeat in range(repeats):
            chosen = torch.searchsorted(p.cumsum(-1).contiguous(), uniforms[repeat, start:start+len(xb)].contiguous())
            chosen.clamp_max_(p.shape[-1]-1)
            labels[repeat, start:start+len(xb)] = chosen
            scores = mean_jac[:, None] - jac[torch.arange(len(xb), device=flat.device)[:, None], chosen]
            for rank in ranks:
                factors[f'sampled_{rank}_repeat_{repeat}'] = scores[:, :rank].reshape(-1, registry.size)/(len(x)*rank)**.5
        for name, factor in factors.items():
            for spec, block in zip(registry.specs, blocks[name]):
                local = factor[:, spec.slice]
                block.addmm_(local.T, local)
    return blocks, gradient, labels, {
        'seconds': time.perf_counter()-started, 'seed': seed, 'ranks': list(ranks), 'repeats': repeats,
        'microbatch': microbatch, 'jacobian_rows': len(x)*model.head.out_features,
        'estimated_workspace_bytes': total_workspace,
        'traces': {name: sum(float(b.trace()) for b in bs) for name, bs in blocks.items()},
        'sampled_target': 'E_label~p [(J^T(p-onehot(label)))(J^T(p-onehot(label)))^T] = GGN',
    }


def match_traces(blocks, reference, registry):
    """Oracle trace control; numerically null tensors remain unchanged and labelled."""
    null_threshold = 1e-24*max(1., sum(float(b.trace()) for b in reference))
    result, scales, nulls = [], {}, []
    for spec, block, target in zip(registry.specs, blocks, reference):
        desired, actual = float(target.trace()), float(block.trace())
        if desired <= null_threshold:
            factor = 1.
            nulls.append(spec.name)
        elif actual <= 0:
            raise ValueError(f'Cannot match a zero-curvature tensor to a nonzero trace: {spec.name}')
        else:
            factor = desired/actual
        result.append(block*factor)
        scales[spec.name] = factor
    return result, {'scales': scales, 'numerically_null_tensors': nulls, 'null_threshold': null_threshold}


def coordinate_diagonal_blocks(diagonal, registry, contexts):
    """Materialize Q^-T diag(c) Q^-1 one tensor at a time, using actual SOAP Q."""
    blocks = []
    for spec in registry.specs:
        inverse = torch.eye(spec.count, dtype=diagonal.dtype, device=diagonal.device).reshape(spec.shape+(spec.count,))
        for axis, q in enumerate(contexts[spec.name]['basis']):
            inverse = mode(inverse, q, axis, 'solve')
        inverse = inverse.reshape(spec.count, spec.count)
        blocks.append(inverse.T @ (diagonal[spec.slice, None]*inverse))
    return blocks
