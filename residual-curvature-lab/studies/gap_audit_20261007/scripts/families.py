"""Structured per-tensor curvature families with oracle parameters (checks 2c and 4b).

Conventions. The registry flattens an (out, in) tensor row-major, so for A (out x out)
and B (in x in), (A kron B) vec(W) = vec(A W B^T) (study test). Every family is a
per-tensor PSD matrix S_b, damped with the same mu * Omega_b and solved densely here.
1-D tensors keep their exact block in every Kronecker family.
"""
import time

import torch

import common  # noqa: F401
from residual_curvature.curvature.information import coordinate_diagonal_blocks, match_traces
from residual_curvature.curvature.layerwise import block_action
from residual_curvature.curvature.solves import relative_error
from residual_curvature.optim.downstream_soap import exact_coordinate_diagonal
from residual_curvature.optim.soap import frozen_apply

CHEAP = ['kron_basis_diag', 'ks_1', 'ks_2', 'ks_2_ef']
PCG_STEPS = [2, 3, 5, 10]


def is_matrix(spec):
    return len(spec.shape) == 2


def rearrange(c, m, n):
    """Van Loan-Pitsianis: R[(i,k),(j,l)] = C[i*n+j, k*n+l], so R(A kron B) = vec(A) vec(B)^T."""
    return c.reshape(m, n, m, n).permute(0, 2, 1, 3).reshape(m*m, n*n)


def kronecker_terms(c, m, n, rank):
    """Top singular pairs of R(C) as symmetrized factors (A_k, B_k), leading pair with trace(A_1) >= 0."""
    u, s, vh = torch.linalg.svd(rearrange(c, m, n), full_matrices=False)
    terms = []
    for k in range(min(rank, len(s))):
        a = (s[k].sqrt()*u[:, k]).reshape(m, m)
        b = (s[k].sqrt()*vh[k]).reshape(n, n)
        a, b = (a+a.T)/2, (b+b.T)/2
        if k == 0 and float(a.trace()) < 0:
            a, b = -a, -b
        terms.append((a, b))
    return terms, s


def clip_psd(matrix):
    """Clip negative eigenvalues; report the minimum and the clipped share of |eigenvalue| mass."""
    eig, q = torch.linalg.eigh((matrix+matrix.T)/2)
    negative = eig.clamp_max(0).abs().sum()
    info = {'min_eigenvalue': float(eig.min()), 'clipped_trace_fraction': float(negative/eig.abs().sum().clamp_min(1e-300)),
            'clipped': bool(negative > 0)}
    return (q*eig.clamp_min(0))@q.T if negative > 0 else (matrix+matrix.T)/2, info


def kron_sum_block(c, spec, rank):
    m, n = spec.shape
    terms, s = kronecker_terms(c, m, n, rank)
    total = sum(torch.kron(a, b) for a, b in terms)
    block, info = clip_psd(total)
    a1, b1 = terms[0]
    info.update(rank=len(terms), singular_values=s[:max(rank, 3)].tolist(),
                relative_frobenius_error=relative_error(block, c),
                leading_factor_min_eigenvalues=[float(torch.linalg.eigvalsh(a1).min()), float(torch.linalg.eigvalsh(b1).min())],
                storage=len(terms)*(m*m+n*n))
    return block, {'terms': terms}, info


def kron_basis_diag_block(c, spec):
    m, n = spec.shape
    terms, _ = kronecker_terms(c, m, n, 1)
    a1, b1 = terms[0]
    ea, ua = torch.linalg.eigh(a1)
    eb, ub = torch.linalg.eigh(b1)
    u = torch.kron(ua, ub)
    diagonal = ((u.T@c)*u.T).sum(1)
    block = (u*diagonal)@u.T
    info = {'leading_factor_min_eigenvalues': [float(ea.min()), float(eb.min())],
            'min_diagonal': float(diagonal.min()), 'relative_frobenius_error': relative_error(block, c),
            'storage': m*m+n*n+m*n}
    return block, {'A1': a1, 'B1': b1, 'diagonal': diagonal}, info


def family_blocks(name, registry, ggn, ef, contexts):
    """Per-tensor S_b for one family; returns blocks, oracle parameters and per-tensor info."""
    started = time.perf_counter()
    blocks, parameters, info = [], {}, {}
    if name == 'soap_basis_diag':
        diagonal, _ = exact_coordinate_diagonal(ggn, registry, contexts)
        blocks, trace_info = match_traces(coordinate_diagonal_blocks(diagonal, registry, contexts), ggn, registry)
        info['trace_control'] = trace_info
        storage = registry.size
    elif name in ['identity', 'exact', 'exact_1d_only']:
        for spec, c in zip(registry.specs, ggn):
            keep = name == 'exact' or (name == 'exact_1d_only' and not is_matrix(spec))
            blocks.append(c if keep else torch.zeros_like(c))
        storage = sum(s.count**2 for s, b in zip(registry.specs, blocks) if b.abs().sum() > 0)
    else:
        source = ef if name.endswith('_ef') else ggn
        storage = 0
        for spec, c in zip(registry.specs, source):
            if not is_matrix(spec):
                blocks.append(c)
                info[spec.name] = {'exact_1d': True, 'storage': spec.count**2}
                storage += spec.count**2
                continue
            if name == 'kron_basis_diag':
                block, params, local = kron_basis_diag_block(c, spec)
            else:
                rank = int(name.split('_')[1])
                block, params, local = kron_sum_block(c, spec, rank)
            blocks.append(block)
            parameters[spec.name] = params
            info[spec.name] = local
            storage += local['storage']
    return blocks, parameters, {'seconds': time.perf_counter()-started, 'storage_numbers': storage, 'tensors': info}


def soap_factor_storage(registry):
    """SOAP's own factor storage counted as its two eigenbases per 2-D tensor (out^2 + in^2)."""
    return sum(s.shape[0]**2+s.shape[1]**2 for s in registry.specs if is_matrix(s))


class DampedBlocks:
    """Cholesky factors of S_b + mu Omega_b, Omega from the frozen SOAP metric (as solve_blocks)."""

    def __init__(self, blocks, registry, metric, mu, tolerance):
        self.blocks, self.registry, self.metric, self.mu, self.tolerance = blocks, registry, metric, mu, tolerance
        self.systems, self.factors = [], []
        for spec, block in zip(registry.specs, blocks):
            eye = torch.eye(spec.count, dtype=block.dtype)
            omega = metric.scale*frozen_apply(eye.reshape(spec.shape+(spec.count,)), metric.contexts[spec.name],
                                              inverse=True).reshape(spec.count, spec.count)
            if relative_error(omega, omega.T) > tolerance:
                raise RuntimeError(f'Asymmetric damping metric for {spec.name}')
            system = block+mu*omega
            self.systems.append(system)
            self.factors.append(torch.linalg.cholesky((system+system.T)/2))

    def solve(self, rhs):
        """Positive solution S^-1 rhs for a vector or a matrix of RHS columns, residual-checked."""
        matrix = rhs.ndim == 2
        rhs2 = rhs if matrix else rhs[:, None]
        out = torch.empty_like(rhs2)
        for spec, system, chol in zip(self.registry.specs, self.systems, self.factors):
            value = torch.cholesky_solve(rhs2[spec.slice], chol)
            for j in range(rhs2.shape[1]):
                residual = relative_error(system@value[:, j], rhs2[spec.slice, j])
                if residual > self.tolerance:
                    raise RuntimeError(f'Damped block solve failed for {spec.name}: {residual}')
            out[spec.slice] = value
        return out if matrix else out[:, 0]

    def action(self, v):
        return torch.cat([s@v[spec.slice] for spec, s in zip(self.registry.specs, self.systems)])


def pcg(action, precondition, rhs, iterations):
    """Truncated preconditioned CG from x0 = 0; exactly `iterations` operator actions."""
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    z = precondition(r)
    p, rz = z.clone(), float(r@z)
    matvecs = 0
    for i in range(iterations):
        ap = action(p)
        matvecs += 1
        curvature = float(p@ap)
        if curvature <= 0:
            raise RuntimeError('PCG operator is not SPD on its Krylov space')
        alpha = rz/curvature
        x += alpha*p
        r -= alpha*ap
        if i == iterations-1:
            break
        z = precondition(r)
        new = float(r@z)
        p = z+(new/rz)*p
        rz = new
    return x, matvecs


def pcg_operator(kind, op, ggn, registry, metric, mu):
    if kind == 'block':
        return lambda v: block_action(ggn, registry, v)+mu*metric.apply(v)
    return lambda v: op.action(v)+mu*metric.apply(v)
