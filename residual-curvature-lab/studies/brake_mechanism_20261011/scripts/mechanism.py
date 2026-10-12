"""Checks A-C of PROTOCOL.md: adaptive edge of stability, noise concentration, optimization vs generalization.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): mechanism.py
Writes results/mechanism.json in this study folder. Offline; no training.
"""
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.func import grad, jvp

STUDY = Path(__file__).resolve().parents[1]
LAB = STUDY.parents[1]
sys.path.insert(0, str(LAB/'scripts'))
sys.path.insert(0, str(LAB/'studies'/'parsec_l_20261009'))

from parsec_l import per_example_gradients  # noqa: E402
from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.curvature.layerwise import streamed_blocks  # noqa: E402
from residual_curvature.curvature.operators import GGN, loss  # noqa: E402
from residual_curvature.data import dataset  # noqa: E402
from residual_curvature.models import Registry, make_model  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402
from residual_curvature.optim import capacity  # noqa: E402

GAP = Path.home()/'skipsoap-study-data'/'gap_audit_20261007'/'runs'/'20261007T143302-baba2664'
BRAKE = Path.home()/'skipsoap-study-data'/'stage_d_20261009'/'runs'/'20261010T180135-101694d8'
BETA1, BETA2, RANK, STRENGTH, ITERATIONS = .95, .95, 16, 4., 30
THRESHOLD = 2*(1+BETA1)/(1-BETA1)
UNDERDAMPED = (1-BETA1**.5)**2/(1-BETA1)


def top_eigenvalue(matvec, n, iterations, seed, dtype):
    """Largest Ritz value of a symmetric operator from a fully reorthogonalized Krylov basis."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(n, generator=gen, dtype=dtype)
    basis, images = [], []
    for _ in range(iterations):
        for _ in range(2):
            if basis:
                b = torch.stack(basis, 1)
                q = q-b@(b.T@q)
        q = q/q.norm()
        basis.append(q)
        z = matvec(q)
        images.append(z)
        q = z.clone()
    b, a = torch.stack(basis, 1), torch.stack(images, 1)
    t = b.T@a
    return float(torch.linalg.eigvalsh((t+t.T)/2).max())


def check_state(seed, step, c, x, y, xb, yb):
    state = torch.load(GAP/f'native_soap__lr0.01__s{seed}'/f'checkpoint-{step:04d}.pt', map_location='cpu', weights_only=False)
    model = make_model(c)
    model.load_state_dict(state['model'])
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    contexts = state['used_context']
    context = next(iter(contexts.values()))
    lr, t = context['lr'], context['step']
    scale = lr*(1-BETA2**t)**.5  # lr * lambda(F0^T M F0) * sqrt(bc2) = lr * lambda(P_hat M)
    fn = functional_logits(model, registry, xb)
    objective = lambda w: loss(fn(w), yb)
    op = GGN(fn, flat, yb)
    gradient_of = grad(objective)

    def forward(v):
        out = torch.empty_like(v)
        for s in registry.specs:
            out[s.slice] = capacity.factor(v[s.slice].reshape(s.shape), contexts[s.name]).reshape(-1)
        return out

    def transpose(v):
        out = torch.empty_like(v)
        for s in registry.specs:
            out[s.slice] = capacity.factor(v[s.slice].reshape(s.shape), contexts[s.name], transpose=True).reshape(-1)
        return out

    ggn = lambda v: transpose(op.action(forward(v)))
    hessian = lambda v: transpose(jvp(gradient_of, (flat,), (forward(v),))[1].detach())
    out = {'seed': seed, 'step': step, 'lr': lr,
           'S_hessian': scale*top_eigenvalue(hessian, registry.size, ITERATIONS, 7+seed+step, flat.dtype),
           'S_ggn': scale*top_eigenvalue(ggn, registry.size, ITERATIONS, 7+seed+step, flat.dtype)}
    blocks, _ = streamed_blocks(model, registry, flat, xb, 8, 768)
    params = dict(model.named_parameters())
    gradients = per_example_gradients(model, xb, yb)
    batch = len(xb)
    total_var = top_var = excess_all = excess_top = removed = 0.
    block_max, lambdas, underdamped, top_count = 0., [], 0, 0
    for s, block in zip(registry.specs, blocks):
        f = capacity.factor(torch.eye(s.count).reshape(s.shape+(s.count,)), contexts[s.name]).reshape(s.count, s.count)
        eig, vectors = torch.linalg.eigh(f.T@block@f)
        r = min(RANK, s.count)
        lam, w = scale*eig[-r:].clamp_min(0), vectors[:, -r:]
        block_max = max(block_max, float(lam.max()))
        rows = gradients[params[s.name]].reshape(batch, -1)@f*(1-BETA2**t)**.25  # rows of F_hat^T g_i
        centered = rows-rows.mean(0)
        var_total = float(centered.square().sum())/(batch-1)
        var_k = (centered@w).square().sum(0)/(batch-1)
        total_var += var_total
        top_var += float(var_k.sum())
        sigma2 = var_k/batch
        excess_k = lr*sigma2/4
        excess_all += lr*var_total/batch/4
        excess_top += float(excess_k.sum())
        removed += float((excess_k*(1-1/(1+STRENGTH*lam))).sum())
        braked = lam/(1+STRENGTH*lam)
        underdamped += int((braked > UNDERDAMPED).sum())
        top_count += r
        lambdas += [float(v) for v in lam]
    out.update(S_block_ggn_max=block_max, noise_share_top16=top_var/total_var, dimension_share_top16=top_count/registry.size,
               predicted_excess_soap_all=excess_all, predicted_excess_soap_top16=excess_top, predicted_excess_removed_by_brake=removed,
               underdamped_share_under_brake=underdamped/top_count, top16_lr_lambda_median=statistics.median(lambdas),
               top16_lr_lambda_max=max(lambdas))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out


def check_training_loss():
    rows = {}
    for seed in [1, 2, 3]:
        soap = {r['step']: r['loss_before'] for r in json.loads((GAP/f'native_soap__lr0.01__s{seed}'/'training.json').read_text())}
        brake = {r['step']: r['own_before'] for r in json.loads((BRAKE/f'brake__lr4_lr0.01__s{seed}'/'training.json').read_text())}
        rows[seed] = {f'{lo}-{hi}': statistics.mean(brake[s]-soap[s] for s in range(lo, hi+1)) for lo, hi in [(1, 50), (51, 200), (201, 500)]}
        rows[seed]['soap_201-500_mean'] = statistics.mean(soap[s] for s in range(201, 501))
    return rows


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = load_config('configs/mini_vit_allocation.yaml')
    c['soap']['lr'] = .01
    x, y, manifest = dataset(c)
    ids = manifest['analysis'][:512]
    states = [check_state(seed, step, c, x, y, x[ids], y[ids]) for seed in [1, 2, 3] for step in [50, 200, 500]]
    training = check_training_loss()
    s_h = {(s['seed'], s['step']): s['S_hessian'] for s in states}
    readings = {
        'A_threshold': THRESHOLD,
        'A_at_adaptive_eos': all(s_h[(seed, step)] >= .75*THRESHOLD for seed in [1, 2, 3] for step in [200, 500]),
        'A_far_below': all(v <= 20 for v in s_h.values()),
        'B_noise_concentrated': all(s['noise_share_top16'] >= .3 for s in states),
        'C_optimization_effect': all(training[seed]['201-500'] < 0 for seed in [1, 2, 3]),
    }
    (STUDY/'results').mkdir(exist_ok=True)
    (STUDY/'results'/'mechanism.json').write_text(json.dumps({'readings': readings, 'states': states, 'training_loss': training}, indent=1)+'\n')
    print(json.dumps(readings, indent=1))
    print(json.dumps(training, indent=1))


if __name__ == '__main__':
    main()
