from dataclasses import dataclass

import torch


def relative_error(value, reference):
    """Relative L2 error, switching to absolute L2 for near-zero references."""
    error, norm = torch.linalg.vector_norm(value - reference), torch.linalg.vector_norm(reference)
    return float(error if norm < 1e-12 else error / norm)


class IdentityMetric:
    scale = 1.0

    def apply(self, v):
        return v

    def inverse(self, v):
        return v


def factor_solve(g, u, mu, inverse):
    """Solve -(UU^T + mu Omega)^-1 g without a parameter-space matrix."""
    if mu <= 0:
        raise ValueError('Damping must be strictly positive')
    qg, qu = inverse(g), inverse(u)
    system = mu * torch.eye(u.shape[1], dtype=u.dtype, device=u.device) + u.T @ qu
    chol = torch.linalg.cholesky((system + system.T) / 2)
    return -(qg - qu @ torch.cholesky_solve((u.T @ qg)[:, None], chol).squeeze(1)) / mu


def diagonal_solve(g, u, mu, inverse, groups):
    out = torch.zeros_like(g)
    # Inverse is block diagonal; embedding gives exactly the selected local map.
    for sl in groups:
        def local(v):
            full = v.new_zeros((g.numel(),) + v.shape[1:])
            full[sl] = v
            return inverse(full)[sl]
        out[sl] = factor_solve(g[sl], u[sl], mu, local)
    return out


@dataclass
class CGResult:
    delta: torch.Tensor
    iterations: int
    residual: float
    converged: bool


def cg(action, rhs, precondition=lambda x: x, tolerance=1e-8, max_iterations=1000):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    norm = rhs.norm().clamp_min(1e-14)
    if r.norm() == 0:
        return CGResult(x, 0, 0.0, True)
    z = precondition(r)
    p, rz = z.clone(), r @ z
    for i in range(1, max_iterations + 1):
        ap = action(p)
        if p @ ap <= 0:
            raise ValueError('CG operator/preconditioner is not SPD')
        alpha = rz / (p @ ap)
        x += alpha * p
        r -= alpha * ap
        actual = float((action(x) - rhs).norm() / norm)
        if actual <= tolerance:
            return CGResult(x, i, actual, True)
        z = precondition(r)
        new_rz = r @ z
        p = z + (new_rz / rz) * p
        rz = new_rz
    return CGResult(x, i, actual, False)


def sketch_sweeps(g, y, mu, metric, groups, initial=None, sweeps=1):
    delta = torch.zeros_like(g) if initial is None else initial.clone()
    accumulator = y.T @ delta
    for _ in range(sweeps):
        for sl in reversed(groups):
            without = accumulator - y[sl].T @ delta[sl]
            rhs = g[sl] + y[sl] @ without
            def local(v):
                full = v.new_zeros((g.numel(),) + v.shape[1:])
                full[sl] = v
                return metric.inverse(full)[sl]
            delta[sl] = factor_solve(rhs, y[sl], mu, local)
            accumulator = without + y[sl].T @ delta[sl]
    return delta
