"""Float64 translation of the user's NI-SOAP, preserving its finite update rules."""
import copy
import math

import torch


def matrix_routes(model):
    names = {f'{name}.weight' for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    return {name: name in names and p.ndim == 2 and min(p.shape) > 1
            for name, p in model.named_parameters()}


def eigensystem(matrix):
    _, q = torch.linalg.eigh((matrix+matrix.T)/2+1e-30*torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device))
    q = q.flip(1)
    return q, (q.T@matrix@q).diag()


def kron_noise(z, ql, qr, el, er):
    rotated = ql.T@z@qr
    return ql@(el.clamp_min(0).sqrt()[:, None]*rotated*er.clamp_min(0).sqrt()[None, :])@qr.T


def refresh(l, r, ql, qr, v):
    left = torch.argsort((ql.T@l@ql).diag(), descending=True)
    right = torch.argsort((qr.T@r@qr).diag(), descending=True)
    v = v[left][:, right]
    ql = torch.linalg.qr(l@ql[:, left], mode='reduced')[0]
    qr = torch.linalg.qr(r@qr[:, right], mode='reduced')[0]
    return ql, qr, (ql.T@l@ql).diag(), (qr.T@r@qr).diag(), v


def decompose(gradient, params, registry, routes, state, settings):
    if state is None:
        state = {'step': 0, 'per_param': {}}
    new_state = copy.deepcopy(state)
    global_step = state['step']+1
    new_state['step'] = global_step
    drift, noise = torch.zeros_like(gradient), torch.zeros_like(gradient)
    b1, b2 = settings['betas']
    shampoo_beta = settings['shampoo_beta']
    coefficient = settings['alpha']*math.sqrt(settings['lr']/settings['batch_size'])
    per_tensor, expected_total, matrix_expected, fallback_expected = {}, 0., 0., 0.
    for i, spec in enumerate(registry.specs):
        g = gradient[spec.slice].reshape(spec.shape)
        p = params[spec.slice].reshape(spec.shape)
        matrix = routes[spec.name]
        if spec.name not in new_state['per_param']:
            s = {'m': torch.zeros_like(g), 'v': torch.zeros_like(g), 'step': -1 if matrix else 0}
            if matrix:
                s.update(L=g.new_zeros(g.shape[0], g.shape[0]), R=g.new_zeros(g.shape[1], g.shape[1]))
            new_state['per_param'][spec.name] = s
        s = new_state['per_param'][spec.name]
        if matrix and s['step'] < 0:
            s['L'] = (1-shampoo_beta)*(g@g.T)
            s['R'] = (1-shampoo_beta)*(g.T@g)
            s['QL'], s['eL'] = eigensystem(s['L'])
            s['QR'], s['eR'] = eigensystem(s['R'])
            s['step'] = 0
            per_tensor[spec.name] = {'route': 'matrix', 'initialization_only': True, 'noise_energy': 0., 'expected_noise_energy': 0.}
            continue
        s['step'] += 1
        t = s['step']
        local = s['QL'].T@g@s['QR'] if matrix else g
        s['m'] = b1*s['m']+(1-b1)*local
        s['v'] = b2*s['v']+(1-b2)*local.square()
        direction = (s['m']/(1-b1**t))/((s['v']/(1-b2**t)).sqrt()+settings['eps'])
        if matrix:
            direction = s['QL']@direction@s['QR'].T
        d = -settings['lr']*(direction+settings.get('weight_decay', 0.)*p)
        # Independent deterministic per-parameter streams; does not consume Torch's global RNG.
        gen = torch.Generator(device=g.device).manual_seed(settings['seed']+1000003*global_step+104729*i)
        z = torch.randn(g.shape, generator=gen, dtype=g.dtype, device=g.device)
        if matrix:
            n = coefficient*kron_noise(z, s['QL'], s['QR'], s['eL'], s['eR'])
            left_trace = float((s['QL'].square().sum(0)*s['eL'].clamp_min(0)).sum())
            right_trace = float((s['QR'].square().sum(0)*s['eR'].clamp_min(0)).sum())
            expected = coefficient**2*left_trace*right_trace
            matrix_expected += expected
        else:
            n = coefficient*z
            expected = coefficient**2*g.numel()
            fallback_expected += expected
        drift[spec.slice], noise[spec.slice] = d.reshape(-1), n.reshape(-1)
        expected_total += expected
        per_tensor[spec.name] = {'route': 'matrix' if matrix else 'fallback', 'initialization_only': False,
                                 'drift_energy': float(d.square().sum()), 'noise_energy': float(n.square().sum()),
                                 'expected_noise_energy': expected}
        if matrix:
            original_m = s['QL']@s['m']@s['QR'].T
            s['L'] = shampoo_beta*s['L']+(1-shampoo_beta)*(g@g.T)
            s['R'] = shampoo_beta*s['R']+(1-shampoo_beta)*(g.T@g)
            if settings['refresh_every'] > 0 and t % settings['refresh_every'] == 0:
                s['QL'], s['QR'], s['eL'], s['eR'], s['v'] = refresh(s['L'], s['R'], s['QL'], s['QR'], s['v'])
            s['m'] = s['QL'].T@original_m@s['QR']
    delta = drift+noise
    if not torch.isfinite(delta).all():
        raise RuntimeError('Nonfinite NI-SOAP proposal')
    drift_energy, noise_energy = float(drift.square().sum()), float(noise.square().sum())
    cross = float(2*drift@noise)
    return drift, noise, new_state, {'drift_energy': drift_energy, 'noise_energy': noise_energy,
                             'drift_noise_cross_term': cross, 'update_energy': float(delta.square().sum()),
                             'expected_noise_energy': expected_total,
                             'expected_matrix_noise_energy': matrix_expected,
                             'expected_fallback_noise_energy': fallback_expected,
                             'per_tensor': per_tensor}


def propose(gradient, params, registry, routes, state, settings):
    drift, noise, state, info = decompose(gradient, params, registry, routes, state, settings)
    return drift+noise, state, info
