"""Per-tensor damped quadratics, Kronecker bases and diagonal operators (PROTOCOL.md, sections 3-4).

Bases are lists with one entry per mode, as SOAP's Q: an orthonormal matrix, or [] for a mode that is not
rotated. For a tensor b with block C (count x count, row-major in the tensor's shape):
    exact diagonal in basis B:  c = diag(B^T C B),  damping diagonal s = mu scale diag(B^T P^-1 B),
    operator:                   d = -B diag(1/(c + s)) B^T g,
    ray reduction of d:         R(d) = (g'd)^2 / (2 d'(C + mu Omega)d) if g'd < 0, else 0.
"""
import torch

import ggn
from gpu_parsec.optimizer import rotate
from residual_curvature.optim.soap import frozen_apply


def eigenbasis(matrix):
    """Orthonormal eigenvectors of a symmetric matrix, by decreasing eigenvalue (float64 eigh, cast back)."""
    _, vectors = torch.linalg.eigh(((matrix+matrix.T)/2).double())
    return vectors.flip(1).to(matrix.dtype)


def factor_bases(factors):
    return [eigenbasis(f) if len(f) else [] for f in factors]


def probe_factors(rows, modes):
    """Per mode: S_k = sum_j unfold_k(G_j) unfold_k(G_j)^T over probe rows (n, *shape), for the modes in `modes`."""
    out = []
    for axis in range(rows.ndim-1):
        if axis not in modes:
            out.append([])
            continue
        u = rows.movedim(axis+1, 1).reshape(rows.shape[0], rows.shape[axis+1], -1)
        out.append(torch.einsum('jak,jbk->ab', u, u))
    return out


def nearest_kronecker(block, shape, iterations=30):
    """Factors [A, B] of the nearest Kronecker product A (x) B to a 2-D tensor's block (Van Loan rearrangement,
    power iteration from B = I)."""
    m, n = shape
    rearranged = block.view(m, n, m, n).permute(0, 2, 1, 3).reshape(m*m, n*n)
    b = torch.eye(n, dtype=block.dtype, device=block.device).reshape(-1)
    b = b/b.norm()
    for _ in range(iterations):
        a = rearranged@b
        a = a/a.norm()
        b = rearranged.T@a
        b = b/b.norm()
    del rearranged
    a, b = a.view(m, m), b.view(n, n)
    return [(a+a.T)/2, (b+b.T)/2]


def exact_diagonal(block, shape, bases, budget=2**26):
    """diag(B^T C B) in the tensor's shape, for tensors of at most two modes (chunked over the first mode's
    columns to keep the intermediate below `budget` elements)."""
    if len(shape) == 1:
        if len(bases[0]):
            b = bases[0]
            return ((block@b)*b).sum(0)
        return block.diagonal().clone()
    m, n = shape
    eye = lambda k: torch.eye(k, dtype=block.dtype, device=block.device)
    left = bases[0] if len(bases[0]) else eye(m)
    right = bases[1] if len(bases[1]) else eye(n)
    x = block.view(m, n*m*n)
    out = torch.empty(m, n, dtype=block.dtype, device=block.device)
    step = max(1, budget//(n*m*n))
    for start in range(0, m, step):
        cols = left[:, start:start+step]
        t1 = (cols.T@x).view(-1, n, m, n)
        t2 = torch.einsum('kajb,jk->kab', t1, cols)
        del t1
        out[start:start+step] = torch.einsum('kab,bl,al->kl', t2, right, right)
    return out


def damping_diagonal(denominator, soap_bases, bases):
    """diag(B^T P^-1 B) for P^-1 = Q^-T diag(den) Q^-1 (SOAPMetric's inverse): per mode M = Q^-1 B, then
    sum_i den_i prod_k M_k[i_k, a_k]^2."""
    out = denominator
    for axis, (q, b) in enumerate(zip(soap_bases, bases)):
        if len(q) == 0 and len(b) == 0:
            continue
        if len(q) == 0:
            m = b
        else:
            m = torch.linalg.solve(q, b if len(b) else torch.eye(q.shape[0], dtype=q.dtype, device=q.device))
        out = torch.tensordot(out, m.square(), dims=([axis], [0])).movedim(-1, axis)
    return out


def apply_diagonal(gradient, bases, diagonal):
    """-B diag(1/diagonal) B^T g for one tensor."""
    z = rotate(gradient[None], bases, transpose=True)[0]/diagonal
    return -rotate(z[None], bases, transpose=False)[0]


def probe_diagonal(rows, bases):
    """sum_j (B^T G_j)^2: the probe estimate of diag(B^T C B)."""
    return rotate(rows, bases, transpose=True).square().sum(0)


def soap_map(gradient, soap_bases, denominator):
    """-Q diag(1/den) Q^T g: SOAP's preconditioner applied to the gradient (direction; SOAP's step size omitted)."""
    return apply_diagonal(gradient, soap_bases, denominator)


def damping_quadratic(d, context, mu, scale):
    """mu d' Omega d with Omega = scale P^-1 (frozen_apply with inverse=True)."""
    return mu*scale*float((d*frozen_apply(d, context, inverse=True)).sum())


def ray(linear, curvature):
    return linear**2/(2*curvature) if linear < 0 and curvature > 0 else 0.


def tensor_quadratics(model, registry, flat, x, directions, chunk=64):
    """({name: [d_b' C_bb d_b for each direction]}, tr C) from per-example Jacobians in one pass over all tensors
    (ggn.example_jacobians; mean loss over the batch)."""
    names = [s.name for s in registry.specs]
    pieces = {s.name: torch.stack([d[s.slice] for d in directions], 1) for s in registry.specs}
    out = {name: torch.zeros(len(directions), dtype=torch.float64) for name in names}
    total, trace = len(x), 0.
    for z, jacobians in ggn.example_jacobians(model, registry, flat, x, names, chunk):
        p = z.softmax(-1)
        n, classes = p.shape
        weight = (p/total).sqrt()[..., None]
        for name in names:
            j = jacobians[name].reshape(n, classes, -1)
            trace += float(((j-(p[..., None]*j).sum(1, keepdim=True))*weight).square().sum())
            jd = j@pieces[name]
            value = (p[..., None]*jd.square()).sum(1)-(p[..., None]*jd).sum(1).square()
            out[name] += value.sum(0).double().cpu()/total
        del jacobians
    return {name: v.tolist() for name, v in out.items()}, trace
