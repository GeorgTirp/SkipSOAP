"""Part C (PROTOCOL.md, section 6): signal-to-noise of top-free PARSEC-L's braked directions at a stored state.

Held brake directions: PARSEC-L's held factor `wt` (rotated coordinates of its extraction bases) carried to the current
bases and divided by the current sqrt(den); with beta = c lr sqrt(1 - beta2^t) of the next step, V = sqrt(beta) held^T
as in PARSECL._shrink, whose correction is (I + V^T V)^-1 on the whitened update. The SVD V^T = U diag(sqrt(kappa)) W
gives orthonormal directions u_k with brake factors 1/(1 + kappa_k) and curvatures lambda_k = kappa_k / beta.

Per direction u (held or control), over n per-example true-label gradients g_i whitened as z_i = Q^T g_i / sqrt(den):
mean m_u and variance s_u^2 of u'z_i, signal G^2 = m_u^2 - s_u^2/n (floored at 0), SNR = G^2/(tau s_u^2) with
tau = 1/2048, static optimum alpha* = SNR/(1 + SNR).
"""
import time

import torch

import states
from gpu_parsec.optimizer import rotate, transport
from gpu_parsec.probes import per_example_gradients

SAMPLE, CHUNK, BATCH = 8192, 256, 2048
SAMPLE_SEED, AUGMENT_SEED, CONTROL_SEED = 4242, 4243, 4244


def brake_directions(optimizer, p, lr, beta2_power_step):
    """(U (count, r), kappa (r,), beta) of a parameter's held brake, for a step with learning rate `lr` and SOAP step
    counter `beta2_power_step` (the value state['step'] takes during that step); None without a held factor."""
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


def whiten(rows, state, eps):
    """Per-example rows (n, *shape) to whitened coordinates (n, count): Q^T g / sqrt(den)."""
    denominator = state['exp_avg_sq'].sqrt().add(eps)
    return (rotate(rows, state['Q'], transpose=True)/denominator.sqrt()).reshape(len(rows), -1)


def run_state(seed, step, device, data):
    started = time.perf_counter()
    model, registry, optimizer, _ = states.topfree(seed, step, device)
    lr = 3e-2*states.SNS_TRAIN.lr_factor(step+1)
    params = list(model.parameters())
    names = {id(p): spec.name for p, spec in zip(params, registry.specs)}
    held, accum = {}, {}
    generator = torch.Generator().manual_seed(CONTROL_SEED+seed*1000+step)
    for p in params:
        state = optimizer.state[p]
        brake = brake_directions(optimizer, p, lr, state['step']+1)
        if brake is None:
            continue
        u, kappa, beta = brake
        count, r = u.shape
        control = torch.randn(count, r, generator=generator, dtype=torch.float64).to(device=device, dtype=u.dtype)
        control = control-u@(u.T@control)
        control = torch.linalg.qr(control)[0]
        held[id(p)] = {'u': u, 'kappa': kappa, 'beta': beta, 'control': control}
        accum[id(p)] = {k: torch.zeros(r, dtype=torch.float64, device=device) for k in ('su', 'su2', 'sc', 'sc2')}
        accum[id(p)].update(mean=torch.zeros(count, dtype=torch.float64, device=device), energy=0.)
    order = torch.randperm(len(data.manifest['training']), generator=torch.Generator().manual_seed(SAMPLE_SEED))[:SAMPLE]
    augment = torch.Generator().manual_seed(AUGMENT_SEED)
    for start in range(0, SAMPLE, CHUNK):
        x, y = data.batch(order[start:start+CHUNK].tolist(), augment)
        grads = per_example_gradients(model, x, y, chunk=CHUNK)
        for p in params:
            if id(p) not in held:
                continue
            group = next(g for g in optimizer.param_groups if any(q is p for q in g['params']))
            z = whiten(grads[p], optimizer.state[p], group['eps']).double()
            h, a = held[id(p)], accum[id(p)]
            pu, pc = z@h['u'].double(), z@h['control'].double()
            a['su'] += pu.sum(0)
            a['su2'] += pu.square().sum(0)
            a['sc'] += pc.sum(0)
            a['sc2'] += pc.square().sum(0)
            a['mean'] += z.sum(0)
            a['energy'] += float(z.square().sum())
        del grads
    tau, n = 1/BATCH, SAMPLE
    eta = lambda beta: beta/optimizer.parsec['strength']
    tensors = {}
    for p in params:
        if id(p) not in held:
            continue
        h, a = held[id(p)], accum[id(p)]

        def stats(s, s2):
            mean = s/n
            var = (s2-n*mean.square())/(n-1)
            signal = (mean.square()-var/n).clamp_min(0)
            snr = signal/(tau*var)
            return mean, var, snr

        mu_, var_u, snr_u = stats(a['su'], a['su2'])
        _, var_c, snr_c = stats(a['sc'], a['sc2'])
        kappa = h['kappa'].double()
        lam = kappa/h['beta']
        alpha = snr_u/(1+snr_u)
        population = a['mean']/n
        total_signal = float(population.square().sum())
        total_noise = (a['energy']/n-total_signal)*n/(n-1)
        tensors[names[id(p)]] = {
            'count': h['u'].shape[0], 'rank': h['u'].shape[1], 'beta': h['beta'], 'eta': eta(h['beta']),
            'kappa': kappa.tolist(), 'lambda': lam.tolist(), 'snr': snr_u.tolist(), 'alpha_star': alpha.tolist(),
            'snr_control': snr_c.tolist(),
            'efficiency_topfree': (eta(h['beta'])/(1+kappa)*lam/alpha).tolist(),
            'efficiency_soap': (eta(h['beta'])*lam/alpha).tolist(),
            'signal_share_top': float(mu_.square().sum())/total_signal if total_signal > 0 else None,
            'noise_share_top': float(var_u.sum())/total_noise if total_noise > 0 else None,
            'dimension_share_top': h['u'].shape[1]/h['u'].shape[0]}
    return {'seed': seed, 'step': step, 'lr': lr, 'sample': n, 'tau': tau, 'tensors': tensors,
            'seconds': time.perf_counter()-started}
