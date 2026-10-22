"""Batched Lanczos quadrature and matrix functions for a symmetric positive-definite operator (PROTOCOL.md, section 3).

For each column v of V, m steps of Lanczos with full reorthogonalization give an orthonormal basis Q_m and a
tridiagonal T_m with A Q_m ≈ Q_m T_m. Then
    vᵀ f(A) v ≈ ||v||² e₁ᵀ f(T_m) e₁            (Gauss quadrature; exact for polynomials of degree ≤ 2m − 1),
    f(A) v ≈ ||v|| Q_m f(T_m) e₁.
The columns run independently in one batch. A column whose Krylov space is exhausted (breakdown) keeps its leading
tridiagonal block; later entries are set to α = 1, β = 0, which decouples them from e₁.
Products with A run in the operator's dtype; the tridiagonal eigendecompositions run in float64.
"""
import torch


class Lanczos:
    def __init__(self, norms, alphas, betas, basis):
        self.norms, self.alphas, self.betas, self.basis = norms, alphas, betas, basis
        tri = torch.diag_embed(alphas.T.double())
        off = betas.T.double()
        tri = tri+torch.diag_embed(off, offset=1)+torch.diag_embed(off, offset=-1)
        self.theta, self.vectors = torch.linalg.eigh(tri)  # (p, m), (p, m, m)
        self.weights = self.vectors[:, 0, :].square()       # e₁ weights of the Ritz pairs

    def min_ritz(self):
        return self.theta.min(dim=1).values

    def forms(self, powers):
        """(len(powers), p) float64: ||v||² e₁ᵀ T^q e₁ for each power q and column."""
        theta = self.theta.clamp_min(torch.finfo(torch.float64).tiny)
        norms2 = self.norms.double().square()
        return torch.stack([norms2*(self.weights*theta**q).sum(1) for q in powers])

    def apply(self, column, power):
        """||v|| Q_m T^q e₁ for one column: the approximation of A^q v (in the basis dtype)."""
        theta = self.theta[column].clamp_min(torch.finfo(torch.float64).tiny)
        coefficients = self.vectors[column]@(theta**power*self.vectors[column, 0, :])
        q = self.basis[column]  # (m, D)
        return float(self.norms[column])*(coefficients.to(q.dtype)@q)


def lanczos(apply, start, steps):
    """Run batched Lanczos for the columns of `start` (D, p) with `apply`: (D, p) -> (D, p)."""
    dim, columns = start.shape
    steps = min(steps, dim)
    eps = 100*torch.finfo(start.dtype).eps
    norms = start.norm(dim=0)
    alive = norms > 0
    q = torch.where(alive, start/norms.clamp_min(torch.finfo(start.dtype).tiny), torch.zeros_like(start))
    basis = start.new_zeros(columns, steps, dim)  # per column, contiguous rows: reorthogonalization by batched products
    alphas = start.new_zeros(steps, columns)
    betas = start.new_zeros(max(steps-1, 0), columns)
    q_prev, beta_prev = torch.zeros_like(q), start.new_zeros(columns)
    for j in range(steps):
        basis[:, j, :] = q.T
        w = apply(q)
        a = (q*w).sum(0)
        w = w-a*q-beta_prev*q_prev
        stored = basis[:, :j+1, :]  # (p, j + 1, D)
        for _ in range(2):  # full reorthogonalization against every stored vector of the column
            c = torch.bmm(stored, w.T.unsqueeze(2))  # (p, j + 1, 1)
            w = w-torch.bmm(stored.transpose(1, 2), c).squeeze(2).T
        b = w.norm(dim=0)
        alphas[j] = torch.where(alive, a, torch.ones_like(a))
        if j == steps-1:
            break
        scale = a.abs()+beta_prev+b
        breakdown = alive & (b <= eps*scale.clamp_min(torch.finfo(start.dtype).tiny))
        keep = alive & ~breakdown
        betas[j] = torch.where(keep, b, torch.zeros_like(b))
        q_prev, q = q, torch.where(keep, w/b.clamp_min(torch.finfo(start.dtype).tiny), torch.zeros_like(w))
        beta_prev = betas[j]
        alive = keep
    return Lanczos(norms, alphas, betas, basis)
