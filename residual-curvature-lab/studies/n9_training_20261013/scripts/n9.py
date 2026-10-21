"""Rebuilding SOAP's eigenbases (PROTOCOL.md, section 3): from the probe factors (N9) or from SOAP's own factors
(`soap_eigh`).

For a parameter with bases Q (one per mode; [] = unrotated) and new bases Q', the state is carried as follows:
- momentum (`exp_avg`, kept by upstream SOAP in rotated coordinates): back to original coordinates with Q, into Q';
- second moments (`exp_avg_sq`, rotated coordinates): with Sigma = Q diag(v) Q^T, v' = diag(Q'^T Sigma Q'),
  per mode v' = (M o M) v along that mode with M = Q'^T Q. This preserves sum(v) for orthonormal bases.
Eigendecompositions run in float64 on the CPU (deterministic) and are cast to the bases' dtype and device.
Gate G2 (checked here): momentum preserved to 1e-4 relative, sum(v') = sum(v) to 1e-4 relative, v' >= 0.
"""
import torch

TOLERANCE = 1e-4


def eigenbasis(matrix, like):
    """Orthonormal eigenvectors of a symmetric matrix by decreasing eigenvalue (float64 CPU eigh), cast like `like`."""
    m = matrix.detach().double().cpu()
    _, vectors = torch.linalg.eigh((m+m.T)/2)
    return vectors.flip(1).to(dtype=like.dtype, device=like.device)


def probe_factors(rows, modes):
    """Per mode k in `modes`: S_k = sum_j unfold_k(G_j) unfold_k(G_j)^T over probe rows (n, *shape); [] elsewhere."""
    out = []
    for axis in range(rows.ndim-1):
        if axis not in modes:
            out.append([])
            continue
        u = rows.movedim(axis+1, 1).reshape(rows.shape[0], rows.shape[axis+1], -1)
        out.append(torch.einsum('jak,jbk->ab', u, u))
    return out


def rebase(optimizer, p, factors):
    """Replace the bases of parameter p by the eigenvectors of `factors` (per mode), carrying the state.
    Returns the G2 checks; raises if they fail."""
    state = optimizer.state[p]
    old = state['Q']
    new = [eigenbasis(f, q) if len(q) else [] for f, q in zip(factors, old)]
    momentum = optimizer.project_back(state['exp_avg'], state)
    v = state['exp_avg_sq']
    total = float(v.sum())
    for axis, (a, b) in enumerate(zip(new, old)):
        if len(b):
            mixing = (a.T@b).square()
            v = torch.tensordot(v, mixing.T, dims=([axis], [0])).movedim(-1, axis)
    state['Q'] = new
    state['exp_avg'] = optimizer.project(momentum, state)
    state['exp_avg_sq'] = v
    back = optimizer.project_back(state['exp_avg'], state)
    norm = float(momentum.norm())
    checks = {'momentum': float((back-momentum).norm())/norm if norm > 0 else float((back-momentum).norm()),
              'second_moment_sum': abs(float(v.sum())-total)/total if total > 0 else abs(float(v.sum())),
              'min_second_moment': float(v.min())}
    if checks['momentum'] > TOLERANCE or checks['second_moment_sum'] > TOLERANCE or checks['min_second_moment'] < 0:
        raise RuntimeError(f'Gate G2 failed: {checks}')
    return checks


def rebuild(optimizer, source, probes=None):
    """Rebase every rotated parameter: source 'probe' (factors of the probe rows) or 'own' (SOAP's GG). Returns the
    worst G2 values over parameters."""
    worst = {'momentum': 0., 'second_moment_sum': 0., 'min_second_moment': float('inf'), 'parameters': 0}
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if state.get('Q') is None or not any(len(q) for q in state['Q']):
                continue
            if source == 'probe':
                factors = probe_factors(probes[p], {k for k, q in enumerate(state['Q']) if len(q)})
            else:
                factors = state['GG']
            checks = rebase(optimizer, p, factors)
            worst['momentum'] = max(worst['momentum'], checks['momentum'])
            worst['second_moment_sum'] = max(worst['second_moment_sum'], checks['second_moment_sum'])
            worst['min_second_moment'] = min(worst['min_second_moment'], checks['min_second_moment'])
            worst['parameters'] += 1
    return worst
