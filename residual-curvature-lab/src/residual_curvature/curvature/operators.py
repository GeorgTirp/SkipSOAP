import torch
from torch.func import jacrev, jvp, vjp


def output_weights(logits, weights=None):
    w = torch.ones(logits.shape[:-1], dtype=logits.dtype, device=logits.device) if weights is None else weights.to(logits)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError('Output weights must be nonnegative with positive sum')
    return w / w.sum()


def loss(logits, targets, weights=None):
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none')
    return (losses * output_weights(logits, weights).reshape(-1)).sum()


def residual(logits, targets, weights=None):
    p = logits.detach().softmax(-1)
    return (p - torch.nn.functional.one_hot(targets, p.shape[-1])) * output_weights(logits, weights)[..., None]


def metric_action(logits, v, weights=None):
    p = logits.detach().softmax(-1)
    return p * (v - (p * v).sum(-1, keepdim=True)) * output_weights(logits, weights)[..., None]


def output_factor(logits, weights=None, eig_rtol=None):
    """Symmetric softmax square root, including its explicitly truncated null mode."""
    p = logits.detach().softmax(-1).reshape(-1, logits.shape[-1])
    w = output_weights(logits, weights).reshape(-1)
    blocks = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    blocks *= w[:, None, None]
    eigen, vectors = torch.linalg.eigh(blocks)
    tol = (100 * torch.finfo(p.dtype).eps if eig_rtol is None else eig_rtol) * eigen.abs().max()
    if eigen.min() < -tol:
        raise ValueError('Output metric is not PSD')
    roots = vectors * torch.where(eigen > tol, eigen.sqrt().nan_to_num(), 0)[:, None, :]
    return torch.block_diag(*roots.unbind()), float(tol)


class GGN:
    def __init__(self, logits_fn, params, targets, weights=None):
        self.fn, self.params, self.targets, self.weights = logits_fn, params.detach(), targets, weights
        self.logits, self.pullback = vjp(self.fn, self.params)
        self.counts = {'forward': 1, 'jvp': 0, 'vjp': 0, 'jacobian_rows': 0}
        self.gradient = self.pullback(residual(self.logits, targets, weights))[0].detach()
        self.counts['vjp'] += 1

    def action(self, v):
        self.counts['jvp'] += 1
        self.counts['vjp'] += 1
        tangent = jvp(self.fn, (self.params,), (v,))[1]
        return self.pullback(metric_action(self.logits, tangent, self.weights))[0].detach()

    def jacobian(self):
        self.counts['jacobian_rows'] += self.logits.numel()
        return jacrev(self.fn)(self.params).reshape(self.logits.numel(), -1).detach()

    def factors(self):
        j = self.jacobian()
        hroot, tolerance = output_factor(self.logits, self.weights)
        return j, j.T @ hroot, tolerance


def allocation_estimate(dimension, outputs, rank, element_bytes=8):
    return {'jacobian_bytes': dimension * outputs * element_bytes,
            'factor_bytes': dimension * outputs * element_bytes,
            'dense_bytes': dimension ** 2 * element_bytes,
            'dual_workspace_bytes': 4 * outputs ** 2 * element_bytes,
            'sketch_bytes': dimension * rank * element_bytes,
            # Includes simultaneously live factors, transforms, output metric and solve work.
            'dual_total_bytes': (6 * dimension * outputs + 8 * outputs ** 2 + 6 * dimension * rank) * element_bytes,
            'dense_total_bytes': (6 * dimension ** 2 + 6 * dimension * outputs + 8 * outputs ** 2) * element_bytes}


def guard(dimension, outputs, rank, resources, dtype, dense=False):
    estimate = allocation_estimate(dimension, outputs, rank, torch.tensor([], dtype=dtype).element_size())
    if outputs > resources['dual_cap'] or (dense and dimension > resources['dense_cap']):
        raise MemoryError('Exact reference dimension exceeds configured cap; use a smaller experiment or matrix-free GGN')
    required = estimate['dense_total_bytes' if dense else 'dual_total_bytes']
    if required > resources['budget_mib'] * 2**20:
        raise MemoryError('Exact reference workspace exceeds allocation budget; no silent approximation is allowed')
    return estimate
