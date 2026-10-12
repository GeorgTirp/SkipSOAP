import numpy as np
import torch
from scipy.optimize import nnls


def direction_matrix(native, registry):
    columns, excluded, names = [], [], []
    for s in registry.specs:
        norm = native[s.slice].norm()
        if norm == 0:
            excluded.append({'name': s.name, 'reason': 'zero native displacement'})
            continue
        col = torch.zeros_like(native)
        col[s.slice] = native[s.slice] / norm
        columns.append(col)
        names.append(s.name)
    e = torch.stack(columns, 1) if columns else native.new_zeros((native.numel(), 0))
    return e, names, excluded


def nonnegative_qp(h, phi, tolerance=1e-8):
    # H=L L^T; ||L^T tau - L^-1 phi||^2/2 differs only by a constant.
    chol = torch.linalg.cholesky(h)
    a = chol.T.detach().cpu().double().numpy()
    b = torch.linalg.solve_triangular(chol, phi[:, None], upper=False).squeeze(1).cpu().double().numpy()
    tau_np, _ = nnls(a, b, maxiter=max(1000, 30 * len(b)))
    tau = torch.as_tensor(tau_np, dtype=h.dtype, device=h.device)
    dual = h @ tau - phi
    scale = max(1.0, float(phi.abs().max()))
    residuals = {
        'primal': float((-tau).clamp_min(0).max()),
        'dual': float((-dual).clamp_min(0).max()) / scale,
        'stationarity': float(dual[tau > 0].abs().max()) / scale if (tau > 0).any() else 0.0,
        'complementarity': float((tau * dual).abs().max()) / max(1.0, float(tau.norm()) * scale),
    }
    if max(residuals.values()) > tolerance:
        raise RuntimeError(f'Allocation KKT check failed: {residuals}')
    return tau, residuals


def allocation_candidates(g, u, metric, mu, native, registry):
    e, names, excluded = direction_matrix(native, registry)
    if e.shape[1] == 0:
        return {}, {'excluded': excluded, 'unavailable': 'zero native SOAP update'}, e, e.T @ e
    phi = -e.T @ g
    eu = e.T @ u
    k = eu @ eu.T
    h = k + mu * e.T @ metric.apply(e)
    tau, residuals = nonnegative_qp(h, phi, 1e-8 if g.dtype == torch.float64 else 1e-4)
    candidates = {'allocation_cone': e @ tau,
                  'allocation_diagonal': e @ (phi / h.diag()).clamp_min(0),
                  'allocation_span': e @ torch.linalg.solve(h, phi)}
    return candidates, {'kkt': residuals, 'names': names, 'excluded': excluded,
                         'uphill_retained': [n for n, p in zip(names, phi) if p < 0]}, e, k


def ray_value(g, delta, action):
    denom = float(delta @ action(delta))
    slope = float(-g @ delta)
    alpha = max(0.0, slope / denom) if denom > 0 else 0.0
    return {'alpha': alpha, 'value': .5 * max(0.0, slope) * alpha}
