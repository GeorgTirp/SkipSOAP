"""(Copied unchanged from studies/bulk_krylov_20261014/scripts/spectral.py, including its D1 fix.) Spectra, power-law fits, Krylov bases and Chebyshev iteration (PROTOCOL.md, sections 3-4)."""
import math

import torch


def top_eigenpairs(matrix, rank, generator, iterations=16, oversample=16):
    """(values descending, vectors) of the top `rank` eigenpairs of a symmetric PSD matrix: exact eigh up to 512
    columns, else subspace iteration with Rayleigh-Ritz."""
    n = len(matrix)
    rank = min(rank, n)
    if n <= 512:
        values, vectors = torch.linalg.eigh(matrix.double())
        return values.flip(0)[:rank].to(matrix.dtype), vectors.flip(1)[:, :rank].to(matrix.dtype)
    v = torch.randn(n, rank+oversample, generator=generator, dtype=torch.float64).to(device=matrix.device, dtype=matrix.dtype)
    for _ in range(iterations):
        v = torch.linalg.qr(matrix@v)[0]
    values, w = torch.linalg.eigh((v.T@matrix@v).double())
    return values.flip(0)[:rank].to(matrix.dtype), (v.double()@w.flip(1)[:, :rank]).to(matrix.dtype)


def loglog_fit(values, first, last):
    """(alpha, R^2) of log(lambda_k) = c - alpha log k over ranks first..last (1-based, inclusive); None if fewer than
    3 positive values (or first beyond the values; DEVIATIONS.md D1)."""
    if first > len(values):
        return None, None
    k = torch.arange(first, min(last, len(values))+1, dtype=torch.float64)
    lam = values.double().cpu()[first-1:len(k)+first-1]
    keep = lam > 0
    if int(keep.sum()) < 3:
        return None, None
    x, y = torch.log(k[keep]), torch.log(lam[keep])
    xm, ym = x.mean(), y.mean()
    slope = float(((x-xm)*(y-ym)).sum()/((x-xm)**2).sum())
    residual = y-(ym+slope*(x-xm))
    r2 = 1-float((residual**2).sum()/((y-ym)**2).sum())
    return -slope, r2


def alpha_from_trace(top, trace, dimension, low=0., high=12., steps=200):
    """alpha with sum_{k=1}^{D} top k^-alpha = trace (bisection on the decreasing left side); None if not bracketed."""
    k = torch.arange(1, dimension+1, dtype=torch.float64)
    f = lambda a: float(top*(k**-a).sum())-trace
    if not f(low) >= 0 >= f(high):
        return None
    for _ in range(steps):
        mid = (low+high)/2
        if f(mid) > 0:
            low = mid
        else:
            high = mid
    return (low+high)/2


def krylov_basis(apply, b, m):
    """Orthonormal basis of span{b, Ab, ..., A^{m-1} b} (Arnoldi with full reorthogonalization), (count, <= m)."""
    columns = []
    v = b
    for _ in range(m):
        w = v.clone()
        for _ in range(2):
            for q in columns:
                w = w-(q@w)*q
        norm = float(w.norm())
        if norm <= 1e-12*float(b.norm()):
            break
        q = w/norm
        columns.append(q)
        v = apply(q)
    return torch.stack(columns, 1) if columns else b.new_zeros(len(b), 0)


def chebyshev_solve(apply, b, low, high, m):
    """x ~ A^-1 b by m steps of Chebyshev iteration for a symmetric A with spectrum in [low, high] (Saad, Iterative
    Methods, Alg. 12.1); no inner products."""
    theta, delta = (high+low)/2, (high-low)/2
    sigma = theta/delta
    rho = 1/sigma
    x = torch.zeros_like(b)
    r = b.clone()
    d = r/theta
    for _ in range(m):
        x = x+d
        r = r-apply(d)
        rho_next = 1/(2*sigma-rho)
        d = rho_next*rho*d+2*rho_next/delta*r
        rho = rho_next
    return x


def conjugate_gradient(apply, b, m):
    """m steps of CG on A x = b from x = 0 (A symmetric positive definite)."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = float(r@r)
    for _ in range(m):
        ap = apply(p)
        alpha = rr/float(p@ap)
        x = x+alpha*p
        r = r-alpha*ap
        rr_next = float(r@r)
        if rr_next == 0:
            break
        p = r+(rr_next/rr)*p
        rr = rr_next
    return x


def power_law_values(top, alpha, dimension, dtype=torch.float32, device='cpu'):
    """lambda_k = top k^-alpha, k = 1..D."""
    k = torch.arange(1, dimension+1, dtype=torch.float64)
    return (top*k**-alpha).to(dtype=dtype, device=device)


def finite_or_none(value):
    return value if value is not None and math.isfinite(value) else None
