"""Teacher-fitted inverse capacity diagnostics, with cheap factorized application."""
import time

import torch

from .soap import mode

BOUNDS = (1e-6, 1e6)


def factor(v, context, transpose=False, inverse=False):
    """F0=Q diag(denominator^-1/2); supports vector/multiple RHS and nonorthogonal Q."""
    value = v
    d = context['denominator']
    extra = value.ndim-d.ndim
    root = d.sqrt().reshape(d.shape+(1,)*extra)
    if inverse:
        if transpose:
            value = value*root
            for axis, q in enumerate(context['basis']):
                value = mode(value, q, axis, 'solve_transpose')
        else:
            for axis, q in enumerate(context['basis']):
                value = mode(value, q, axis, 'solve')
            value = value*root
    elif transpose:
        for axis, q in enumerate(context['basis']):
            value = mode(value, q, axis, 'transpose')
        value = value/root
    else:
        value = value/root
        for axis, q in enumerate(context['basis']):
            value = mode(value, q, axis, 'forward')
    return value


def bounded_inverse(matrix):
    eig, q = torch.linalg.eigh((matrix+matrix.T)/2)
    if float(eig.min()) <= 0:
        raise RuntimeError('Nonpositive teacher projection')
    raw = eig.reciprocal()
    clipped = raw.clamp(*BOUNDS)
    return (q*clipped)@q.T, int((raw != clipped).sum())


def core_apply(core, z):
    kind = core['kind']
    if kind == 'diagonal_subspace':
        root = core['root'].reshape((-1,)+(1,)*(z.ndim-1))
        return root*core_apply(core['inner'],root*z)
    if kind == 'identity':
        return z.clone()
    if kind == 'scalar':
        return core['scalar']*z
    if kind == 'coordinate':
        out = z.clone()
        indices = core['indices']
        out[indices] += core['correction']@z[indices]
        return out
    if kind == 'diagonal':
        d = core['diagonal']
        return d.reshape(d.shape+(1,)*(z.ndim-1))*z
    if kind == 'dense':
        return core['matrix']@z
    if kind == 'cholesky':
        value = torch.cholesky_solve(z[:,None] if z.ndim==1 else z,core['cholesky'])
        return value[:,0] if z.ndim==1 else value
    u, correction = core['U'], core['correction']
    return z+u@(correction@(u.T@z))


def dense_core(core, n, template):
    return core_apply(core, torch.eye(n, dtype=template.dtype, device=template.device))


def fit_core(k, family, rank=None, zg=None, zm=None, denominator=None):
    """Minimize the population energy objective in the declared restricted family."""
    n = len(k)
    if family.startswith('combined_'):
        started = time.perf_counter()
        raw = k.diag().reciprocal()
        d = raw.clamp(*BOUNDS)
        root = d.sqrt()
        adjusted = root[:,None]*k*root[None,:]
        inner = fit_core(adjusted,family.removeprefix('combined_'),rank,
                         root*zg,root*zm,denominator/d.reshape(denominator.shape))
        return {'kind':'diagonal_subspace','root':root,'inner':inner,
                'storage':n+inner['storage'],'clipped':int((raw!=d).sum())+inner['clipped'],
                'rank':inner['rank'],'basis_seconds':inner['basis_seconds'],
                'coefficient_fit_seconds':time.perf_counter()-started-inner['basis_seconds']}
    if family == 'identity':
        return {'kind': 'identity', 'storage': 0, 'clipped': 0}
    if family in ['scalar', 'diagonal']:
        raw = n/k.trace() if family == 'scalar' else k.diag().reciprocal()
        return {'kind': family, family: raw.clamp(*BOUNDS),
                'storage': raw.numel(), 'clipped': int(((raw < BOUNDS[0]) | (raw > BOUNDS[1])).sum())}
    if family == 'exact':
        value = torch.linalg.cholesky((k+k.T)/2)
        return {'kind': 'cholesky', 'cholesky': value, 'storage': n*n, 'clipped': 0}
    started = time.perf_counter()
    rank = min(rank, n)
    if family == 'teacher':
        eig, q = torch.linalg.eigh((k+k.T)/2)
        if float(eig.min()) <= 0:
            raise RuntimeError('Nonpositive whitened teacher')
        inv = eig.reciprocal()
        gain = eig*((1-inv).square()-(inv.clamp(*BOUNDS)-inv).square())
        selected = torch.argsort(gain, descending=True, stable=True)[:rank]
        u = q[:, selected]
    elif family == 'coordinate':
        score = zg.square()/zg.square().sum().clamp_min(1e-30)+zm.square()/zm.square().sum().clamp_min(1e-30)
        selected = torch.argsort(score, descending=True, stable=True)[:rank]
        u = torch.eye(n, dtype=k.dtype, device=k.device)[:, selected]
    elif family == 'state':
        t = denominator.reshape(-1).log()
        t = t-t.mean()
        t = t/t.abs().max().clamp_min(1e-30)
        columns = [rhs*t**power for power in range(8) for rhs in [zg, zm]][:rank]
        columns = torch.stack([v/v.norm().clamp_min(1e-30) for v in columns], 1)
        u, singular, _ = torch.linalg.svd(columns, full_matrices=False)
        u = u[:, singular > singular.max()*1e-10]
    else:
        raise ValueError(family)
    basis_seconds = time.perf_counter()-started
    started = time.perf_counter()
    if u.shape[1]:
        t, clipped = bounded_inverse(u.T@k@u)
        correction = t-torch.eye(len(t), dtype=k.dtype, device=k.device)
    else:
        correction, clipped = k.new_empty(0, 0), 0
    if family == 'coordinate':
        return {'kind': 'coordinate', 'indices': selected, 'correction': correction,
                'storage': selected.numel()+correction.numel(), 'clipped': clipped, 'rank': len(selected),
                'basis_seconds': basis_seconds, 'coefficient_fit_seconds':time.perf_counter()-started}
    return {'kind': 'subspace', 'U': u, 'correction': correction,
            'storage': u.numel()+correction.numel(), 'clipped': clipped,
            'rank': u.shape[1], 'basis_seconds':basis_seconds,
            'coefficient_fit_seconds':time.perf_counter()-started}


def apply(cores, registry, contexts, beta, rhs):
    output = torch.empty_like(rhs)
    for spec, core in zip(registry.specs, cores):
        value = rhs[spec.slice].reshape(spec.shape+rhs.shape[1:])
        local = factor(value, contexts[spec.name], transpose=True).reshape((spec.count,)+rhs.shape[1:])
        local = core_apply(core, local).reshape(spec.shape+rhs.shape[1:])
        output[spec.slice] = beta*factor(local, contexts[spec.name]).reshape((spec.count,)+rhs.shape[1:])
    return output


def energy_errors(rhs, proposal, exact, systems, registry):
    """One global nonnegative scalar per complete parameter RHS, never per tensor."""
    if rhs.ndim == 1:
        rhs, proposal, exact = rhs[:, None], proposal[:, None], exact[:, None]
    denominator = (rhs*exact).sum(0)
    linear = (rhs*proposal).sum(0)
    aq = torch.cat([a@proposal[s.slice] for a, s in zip(systems, registry.specs)])
    quadratic = (proposal*aq).sum(0)
    scale = linear.clamp_min(0)/quadratic.clamp_min(1e-300)
    error = proposal-exact
    residual = torch.cat([a@error[s.slice] for a, s in zip(systems, registry.specs)])
    raw_num = (error*residual).sum(0)
    error = proposal*scale-exact
    residual = torch.cat([a@error[s.slice] for a, s in zip(systems, registry.specs)])
    ray_num = (error*residual).sum(0)
    return {'raw': (raw_num/denominator).tolist(), 'ray': (ray_num/denominator).tolist(),
            'global_scales': scale.tolist(), 'denominators': denominator.tolist(),
            'raw_numerator_sum': float(raw_num.sum()), 'ray_numerator_sum': float(ray_num.sum()),
            'denominator_sum': float(denominator.sum())}
