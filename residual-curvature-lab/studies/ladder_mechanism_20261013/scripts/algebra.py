"""Linear algebra of the ladder and the control (PROTOCOL.md, sections 3-4).

Copied with attribution, to avoid importing modules of other studies that share names with this study's:
- `brake_directions` from studies/basis_refresh_snr_20261013/scripts/snr.py (tested there against PARSEC-L's logged
  brakes, and here again);
- `directional_ggn` from studies/cifar_sn_soap_20261012/scripts/nonlinear.py.
"""
import torch
from torch.func import jvp

from gpu_parsec.optimizer import transport
from residual_curvature.curvature.operators import metric_action
from residual_curvature.models.mlp import functional_logits

import micro


def orth(columns, tolerance=1e-6):
    """Orthonormal basis of the span of `columns` (count, k), dropping directions below `tolerance` relative (SVD)."""
    if columns.shape[1] == 0:
        return columns
    u, s, _ = torch.linalg.svd(columns.double(), full_matrices=False)
    keep = s > tolerance*s.max() if float(s.max()) > 0 else s > 0
    return u[:, keep].to(columns.dtype)


def galerkin(system, gradient, basis):
    """(y_B, R_B): the minimizer of g'y + y'Ay/2 over span(B) and its reduction, for orthonormal B. The products with
    the (possibly large) system stay in its dtype; the reduced system is solved in float64."""
    if basis.shape[1] == 0:
        return torch.zeros_like(gradient), 0.
    reduced = (basis.T@(system@basis)).double()
    reduced = (reduced+reduced.T)/2
    rhs = (basis.T@gradient).double()
    coefficients = torch.linalg.solve(reduced, rhs)
    return -(basis@coefficients.to(basis.dtype)), .5*float(rhs@coefficients)


def ray(system, gradient, y):
    """Best reduction along direction y: (g'y)^2 / (2 y'Ay) if g'y < 0, else 0."""
    linear = float((gradient.double()*y.double()).sum())
    curvature = float((y.double()*(system@y).double()).sum())
    return linear**2/(2*curvature) if linear < 0 and curvature > 0 else 0.


def top_eigenvectors(matrix, rank, generator, iterations=12, oversample=8):
    """Top `rank` eigenvectors of a symmetric PSD matrix: exact eigh up to 512 columns, else subspace iteration with
    Rayleigh-Ritz."""
    n = len(matrix)
    rank = min(rank, n)
    if n <= 512:
        _, vectors = torch.linalg.eigh(matrix.double())
        return vectors[:, -rank:].flip(1).to(matrix.dtype)
    v = torch.randn(n, rank+oversample, generator=generator, dtype=torch.float64).to(device=matrix.device, dtype=matrix.dtype)
    for _ in range(iterations):
        v = torch.linalg.qr(matrix@v)[0]
    _, w = torch.linalg.eigh((v.T@matrix@v).double())
    return (v.double()@w[:, -rank:].flip(1)).to(matrix.dtype)


def brake_directions(optimizer, p, lr, beta2_power_step):
    """(U (count, r), kappa (r,), beta) of a parameter's held PARSEC-L brake for a step with learning rate `lr` and
    SOAP step counter `beta2_power_step`; None without a held factor (from basis_refresh_snr's snr.py)."""
    state = optimizer.state[p]
    ps = state.get('parsec', {})
    wt = ps.get('wt')
    if wt is None or len(wt) == 0:
        return None
    group = next(g for g in optimizer.param_groups if any(q is p for q in g['params']))
    denominator = state['exp_avg_sq'].sqrt().add(group['eps'])
    held = transport(wt, ps['bases'], state['Q'])/denominator.sqrt()
    held = held.reshape(len(wt), -1).T
    beta = optimizer.parsec['strength']*lr*(1-group['betas'][1]**beta2_power_step)**.5
    u, s, _ = torch.linalg.svd((beta**.5*held).double(), full_matrices=False)
    return u.to(held.dtype), s.square().to(held.dtype), beta


def directional_ggn(model, registry, flat, d, x, microbatch):
    """d'C d at parameters `flat` for the mean loss of the batch (forward mode; from cifar_sn_soap's nonlinear.py)."""
    total, out = len(x), 0.
    for part in micro.parts(total, microbatch):
        f, jd = jvp(functional_logits(model, registry, x[part]), (flat,), (d,))
        out += (part.stop-part.start)/total*float((jd.detach()*metric_action(f.detach(), jd.detach())).sum())
    return out
