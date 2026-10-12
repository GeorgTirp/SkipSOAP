"""PARSEC-L operator pieces (PROTOCOL.md): sampled-label probes, whitened extraction, cores, angles.

The exact operator is the capacity screen's damping-calibrated teacher_r core
(`optim.capacity.fit_core(K, 'teacher', r)` with K = beta F0^T (C + mu Omega) F0 = I + K_C).
"""
import torch
from torch.func import jacrev

import common  # noqa: F401
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity
from residual_curvature.optim.soap import SOAPMetric, frozen_apply


def setup(registry, contexts, blocks, floor=1e-8):
    """mu, beta, SOAP factors F0_b and whitened K_b = I + K_C,b (as run_learnable_capacity.audit, damping)."""
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in blocks)/registry.size, floor)
    beta = 1/(mu*metric.scale)
    factors, ks = [], []
    for spec, block in zip(registry.specs, blocks):
        eye = torch.eye(spec.count, dtype=block.dtype)
        f = capacity.factor(eye.reshape(spec.shape+(spec.count,)), contexts[spec.name]).reshape(spec.count, spec.count)
        omega = metric.scale*frozen_apply(eye.reshape(spec.shape+(spec.count,)), contexts[spec.name], inverse=True).reshape(spec.count, spec.count)
        factors.append(f)
        raw = f.T@(block+mu*omega)@f
        ks.append(beta*raw)
    return metric, mu, beta, factors, ks


def sampled_factor(model, registry, flat, x, s, seed, microbatch=8):
    """Rows g_ij / sqrt(B s), g_ij = J_i^T (p_i - onehot(xi_ij)), xi_ij ~ categorical(p_i); E[U^T U] = GGN."""
    gen = torch.Generator().manual_seed(seed)
    uniforms = torch.rand(len(x), s, generator=gen, dtype=flat.dtype)
    rows = []
    for start in range(0, len(x), microbatch):
        xb = x[start:start+microbatch]
        fn = functional_logits(model, registry, xb)
        p = fn(flat).detach().softmax(-1)
        jac = jacrev(fn, chunk_size=16)(flat).detach()
        mean_jac = (p[..., None]*jac).sum(1)
        chosen = torch.searchsorted(p.cumsum(-1).contiguous(), uniforms[start:start+len(xb)].contiguous()).clamp_max(p.shape[-1]-1)
        scores = mean_jac[:, None]-jac[torch.arange(len(xb))[:, None], chosen]
        rows.append(scores.reshape(-1, registry.size))
    return torch.cat(rows)/(len(x)*s)**.5


def whitened_probe(u, spec, factor, beta, draws_used):
    """Z_b = sqrt(beta) F0_b^T U_b^T restricted to the first `draws_used` draws per example."""
    total = u.shape[0]
    s_all = total//512 if total % 512 == 0 else None
    block = u[:, spec.slice]
    if s_all and draws_used < s_all:
        block = block.reshape(512, s_all, spec.count)[:, :draws_used].reshape(-1, spec.count)*(s_all/draws_used)**.5
    return beta**.5*factor.T@block.T


def randomized_top(z, rank, seed, oversample=8, power=2):
    """Top-r left singular vectors/values^2 of z (n x m) by a randomized range finder with power steps."""
    n, m = z.shape
    k = min(rank+oversample, n, m)
    gen = torch.Generator().manual_seed(seed)
    y = z@torch.randn(m, k, generator=gen, dtype=z.dtype)
    q, _ = torch.linalg.qr(y)
    for _ in range(power):
        q, _ = torch.linalg.qr(z@(z.T@q))
    u, s, _ = torch.linalg.svd(q.T@z, full_matrices=False)
    r = min(rank, n, m)
    return q@u[:, :r], s[:r].square()


def subspace_core(w, kappa):
    """Core of P_r: I - W diag(kappa/(1+kappa)) W^T (capacity 'subspace' kind)."""
    correction = torch.diag(1/(1+kappa))-torch.eye(len(kappa), dtype=w.dtype)
    return {'kind': 'subspace', 'U': w, 'correction': correction, 'storage': w.numel()+correction.numel(),
            'clipped': 0, 'rank': w.shape[1]}


def exact_top(k, rank):
    """Exact top-r eigenpairs of K_C = K - I."""
    eig, q = torch.linalg.eigh((k+k.T)/2)
    r = min(rank, len(eig))
    return q[:, -r:].flip(1), (eig[-r:].flip(0)-1).clamp_min(0)


def sin2(a, b):
    """||sin Theta||_F^2 / r between orthonormal column spaces a, b (same r)."""
    r = a.shape[1]
    return float(r-(a.T@b).square().sum())/r if r else 0.


def weighted_angle(pairs):
    """Energy-weighted mean of sin^2 over tensors; pairs = [(a, b, weight)]."""
    total = sum(w for _, _, w in pairs)
    return sum(sin2(a, b)*w for a, b, w in pairs)/total if total > 0 else 0.


def transport(w, factor_from, factor_to):
    """Whitened subspace of one step into the next step's whitened coordinates via physical covectors."""
    if w.shape[1] == 0:
        return w
    physical = torch.linalg.solve(factor_from.T, w)
    q, _ = torch.linalg.qr(factor_to.T@physical)
    return q
