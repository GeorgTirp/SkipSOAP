"""Float64 port of the supplied default PARSEC/PNS precondition-then-Adam rule."""
import copy
import time

import torch


def lanczos_ritz(action, template, rank=16, iterations=32, seed=0):
    if rank < 1 or iterations < rank:
        raise ValueError('Require positive rank <= Krylov iterations')
    start = time.perf_counter()
    gen = torch.Generator(device=template.device).manual_seed(seed)
    q = torch.randn(template.shape, generator=gen, dtype=template.dtype, device=template.device)
    q /= q.norm()
    basis, responses = [], []
    for _ in range(min(iterations, template.numel())):
        basis.append(q)
        z = action(q).detach()
        responses.append(z)
        b = torch.stack(basis, 1)
        residual = z.clone()
        for _ in range(2):
            residual -= b @ (b.T @ residual)
        norm = residual.norm()
        if float(norm) <= 1e-12*max(1., float(z.norm())):
            break
        q = residual/norm
    b, ab = torch.stack(basis, 1), torch.stack(responses, 1)
    projected = b.T@ab
    eigen, rotation = torch.linalg.eigh((projected+projected.T)/2)
    indices = torch.argsort(eigen, descending=True)[:rank]
    eigen, rotation = eigen[indices], rotation[:, indices]
    if float(eigen.min()) < -1e-10*max(1., float(eigen.abs().max())):
        raise RuntimeError('GGN Krylov projection has substantial negative curvature')
    eigen = eigen.clamp_min(0)
    vectors = (b@rotation).T
    error = (ab@rotation-vectors.T*eigen).norm(dim=0)
    relative = error / eigen.abs().clamp_min(max(float(eigen.abs().max())*1e-12, 1e-15))
    orthogonality = float((vectors@vectors.T-torch.eye(len(eigen), dtype=template.dtype, device=template.device)).norm())
    if orthogonality > 1e-8:
        raise RuntimeError('Tracked curvature vectors are not orthonormal')
    return eigen, vectors, {'seconds': time.perf_counter()-start, 'matvecs': len(basis),
                            'eigenvalues': eigen.tolist(), 'relative_ritz_residuals': relative.tolist(),
                            'orthogonality_error': orthogonality,
                            'label': 'approximate global GGN Ritz eigenspace, float64 full reorthogonalization'}


def transform_gradient(gradient, eigenvalues, vectors, damping):
    if damping <= 0:
        raise ValueError('Positive eigenvalue damping required')
    coefficients = vectors@gradient
    return gradient-vectors.T@coefficients + vectors.T@(coefficients/(eigenvalues+damping).sqrt())


def propose(gradient, params, state, settings, operator=None):
    """A zero-rank/base-Adam control uses operator=None and refresh_every=0."""
    if state is None:
        state = {'step': 0, 'm': torch.zeros_like(gradient), 'v': torch.zeros_like(gradient),
                 'eigenvalues': gradient.new_empty(0), 'vectors': gradient.new_empty(0, len(gradient)),
                 'last_refresh': None}
    state = copy.deepcopy(state)
    step = state['step']+1
    refresh = None
    if settings['refresh_every'] > 0 and step % settings['refresh_every'] == 0:
        if operator is None:
            raise ValueError('PARSEC refresh requires a curvature operator')
        eigen, vectors, refresh = lanczos_ritz(operator.action, gradient, settings['rank'],
                                               settings['lanczos_iters'], settings['seed']+step)
        state.update(eigenvalues=eigen, vectors=vectors, last_refresh=step)
    transformed = transform_gradient(gradient, state['eigenvalues'], state['vectors'], settings['damping'])
    b1, b2 = settings['betas']
    state['m'] = b1*state['m']+(1-b1)*transformed
    state['v'] = b2*state['v']+(1-b2)*transformed.square()
    state['step'] = step
    direction = (state['m']/(1-b1**step))/((state['v']/(1-b2**step)).sqrt()+settings['eps'])
    delta = -settings['lr']*(direction+settings.get('weight_decay', 0.)*params)
    if not torch.isfinite(delta).all():
        raise RuntimeError('Nonfinite PARSEC/Adam proposal')
    return delta, state, {'refresh': refresh, 'active_rank': len(state['eigenvalues']),
                          'last_refresh': state['last_refresh'],
                          'gradient_norm': float(gradient.norm()),
                          'transformed_gradient_norm': float(transformed.norm())}
