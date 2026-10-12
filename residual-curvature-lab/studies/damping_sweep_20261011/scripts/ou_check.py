"""Part 2 of PROTOCOL.md (with DEVIATIONS.md #1-2): offline Ornstein-Uhlenbeck (noisy-quadratic) check on the
seed-4 checkpoints of topfree_confirm_20261011. No training. Writes results/ou_check.json.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  .venv/bin/python studies/damping_sweep_20261011/scripts/ou_check.py
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401  (import paths)
import train
from parsec_l import per_example_gradients
from parsec_l.optimizer import PARSECL, rotate
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.data import dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model

STUDY = Path(__file__).resolve().parents[1]
CONF = Path.home()/'skipsoap-study-data'/'topfree_confirm_20261011'/'runs'
UNITS = {'soap': CONF/'20261011T135638-221c0b6a'/'soap__constant__lr0.01__s4',
         'brake': CONF/'20261011T135638-221c0b6a'/'brake__constant__lr0.01__s4',
         'topfree': CONF/'20261011T140034-192dd041'/'topfree__constant__lr0.01_c64__s4'}
DYNAMICS = {'soap': ('momentum', 0.), 'brake': ('momentum', 4.), 'topfree': ('gd', 64.)}
BETA1, BETA2, DECAY, RANK, LAG, SEED = .95, .95, .98, 16, 10, 4
STEPS = list(range(20, 481, 20))
BINS = [(0., .01), (.01, .1), (.1, 1.), (1., math.inf)]
TERMS, WINDOW = 4000, 2000  # rho length; EMA-residual filter length (d^2000 ~ 3e-18)
SOAP_PROJECT_BACK = PARSECL.__mro__[1].project_back


def autocorrelation(kind, eta, lam, strength, terms=TERMS):
    """Per direction (lam: array (r,)): stationary variance per unit noise variance and rho_j, shape (terms+1, r)."""
    eta_eff = eta/(1+strength*eta*lam)
    rho = np.empty((terms+1, len(lam)))
    rho[0] = 1.
    if kind == 'gd':
        a = 1-eta_eff*lam
        var = eta_eff/(lam*(2-eta_eff*lam))
        for j in range(1, terms+1):
            rho[j] = rho[j-1]*a
    else:
        alpha = eta_eff*(1-BETA1)
        phi1, phi2 = 1+BETA1-alpha*lam, -BETA1
        var = alpha*(1+BETA1)/(lam*(1-BETA1)*(2*(1+BETA1)-alpha*lam))
        rho[1] = phi1/(1-phi2)
        for j in range(2, terms+1):
            rho[j] = phi1*rho[j-1]+phi2*rho[j-2]
    return var, rho


def ema_factor(rho):
    """Protocol's factor Var(x - EMA_d(x)) / Var(x) for a stationary process with autocorrelation rho (per column)."""
    s = (DECAY**np.arange(len(rho)))@rho
    return 1-2*(1-DECAY)*s+(1-DECAY)*(2*s-1)/(1+DECAY)


def residual_filter_autocorrelation():
    """A(m) = sum_i h_i h_{i+m}, m = -WINDOW..WINDOW, for r_t = x_t - EMA_d(x)_t = sum_i h_i x_{t-i}."""
    h = -(1-DECAY)*DECAY**np.arange(WINDOW+1)
    h[0] = DECAY
    return np.correlate(h, h, 'full')


A = residual_filter_autocorrelation()
M = np.arange(-WINDOW, WINDOW+1)


def residual_covariance(rho, lag):
    """Cov(r_t, r_{t+lag}) / Var(x) per column."""
    return A@rho[np.abs(lag-M)]


def whitening(st, spec, eps):
    """Dense F_hat = (1 - beta2^t)^(1/4) Q diag(d^-1/2) (columns = whitened coordinates) and its inverse action."""
    d = st['exp_avg_sq'].sqrt()+eps
    s = (1-BETA2**st['step'])**.25
    basis = torch.eye(spec.count).reshape((spec.count,)+tuple(spec.shape))/d.sqrt()
    f = s*rotate(basis, st['Q'], transpose=False).reshape(spec.count, spec.count).T
    # Upstream SOAP computes Q in float32, so Q^T is not Q^-1 (|Q^TQ - I| ~ 4e-7): invert F_hat exactly (LU).
    factors = torch.linalg.lu_factor(f)
    inverse = lambda u: torch.linalg.lu_solve(*factors, u.reshape(-1, 1)).reshape(-1)
    v = torch.randn(spec.shape, generator=torch.Generator().manual_seed(spec.count), dtype=torch.float64)
    upstream = s*SOAP_PROJECT_BACK(None, v/d.sqrt(), st).reshape(-1)
    error = float((f@v.reshape(-1)-upstream).norm()/upstream.norm())
    residual = float((f@inverse(v.reshape(-1))-v.reshape(-1)).norm()/v.norm())
    if error > 1e-12 or residual > 1e-12:
        raise RuntimeError(f'{spec.name}: whitening check failed ({error:.2e}, inverse residual {residual:.2e})')
    return f, inverse, max(error, residual)


def empty():
    return {'n': 0, 'y2': 0., 'y2_lag': 0., 'yy': 0., 'pred': 0., 'pred_cov10': 0., 'rho10_raw': 0., 'unstable': 0}


def summarize(acc):
    rows = {}
    for (lo, hi), a in acc.items():
        if a['n'] == 0:
            continue
        rows[f'{lo:g}-{hi:g}'] = {
            'n': a['n'], 'unstable_excluded': a['unstable'],
            'spread_ratio': a['y2']/a['pred'],
            'lag10_measured': a['yy']/math.sqrt(a['y2']*a['y2_lag']),
            'lag10_predicted_residual': a['pred_cov10']/a['pred'],
            'lag10_predicted_raw_mean': a['rho10_raw']/a['n']}
    return rows


def adequate(rows, key):
    top = [r for k, r in rows.items() if float(k.split('-')[0]) >= .1]
    return bool(top) and all(.5 <= r['spread_ratio'] <= 2 and abs(r['lag10_measured']-r[key]) <= .3 for r in top)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = train.unit_config(SEED)
    x, y, manifest = dataset(c)
    ids = manifest['analysis'][:512]
    xb, yb = x[ids], y[ids]
    out, checks = {}, 0.
    for arm, unit in UNITS.items():
        kind, strength = DYNAMICS[arm]
        acc = {subset: {b: empty() for b in BINS} for subset in ['all', 't>=100']}
        per_step = {}
        for t in STEPS:
            ck = torch.load(unit/'checkpoints'/f'step-{t:04d}.pt', map_location='cpu', weights_only=False)
            lag = torch.load(unit/'checkpoints'/f'step-{t+LAG:04d}.pt', map_location='cpu', weights_only=False)
            model = make_model(c)
            registry = Registry(model)
            commit(model, registry, ck['parameters'])
            flat = ck['parameters']
            delta, delta_lag = flat-ck['ema'][DECAY], lag['parameters']-lag['ema'][DECAY]
            blocks, _ = streamed_blocks(model, registry, flat, xb, 8, 768)
            gradients = per_example_gradients(model, xb, yb)
            params = dict(model.named_parameters())
            eta, eps = ck['lr'], ck['optimizer']['param_groups'][0]['eps']
            states = ck['optimizer']['state']
            top_y2 = top_pred = 0.
            for index, (spec, block) in enumerate(zip(registry.specs, blocks)):
                f, inverse, error = whitening(states[index], spec, eps)
                checks = max(checks, error)
                eig, vectors = torch.linalg.eigh(f.T@block@f)
                r = min(RANK, spec.count)
                lams, w = eig[-r:].numpy(), vectors[:, -r:]
                rows = gradients[params[spec.name]].reshape(len(ids), -1)@f
                sigma2 = (((rows-rows.mean(0))@w).square().sum(0)/(len(ids)-1)/len(ids)).numpy()
                ycoord = (w.T@inverse(delta[spec.slice])).numpy()
                ylag = (w.T@inverse(delta_lag[spec.slice])).numpy()
                keep = lams > 1e-12
                if not keep.any():
                    continue
                lams, sigma2, ycoord, ylag = lams[keep], sigma2[keep], ycoord[keep], ylag[keep]
                var, rho = autocorrelation(kind, eta, lams, strength)
                stable = np.isfinite(var) & (var > 0)
                pred = var*sigma2*ema_factor(rho)
                cov10 = var*sigma2*residual_covariance(rho, LAG)
                for k in range(len(lams)):
                    b = next(bb for bb in BINS if bb[0] <= eta*lams[k] < bb[1])
                    for subset in ['all', 't>=100'] if t >= 100 else ['all']:
                        a = acc[subset][b]
                        if not stable[k]:
                            a['unstable'] += 1
                            continue
                        a['n'] += 1
                        a['y2'] += float(ycoord[k])**2
                        a['y2_lag'] += float(ylag[k])**2
                        a['yy'] += float(ycoord[k]*ylag[k])
                        a['pred'] += float(pred[k])
                        a['pred_cov10'] += float(cov10[k])
                        a['rho10_raw'] += float(rho[LAG, k])
                    if stable[k] and eta*lams[k] >= .1:
                        top_y2 += float(ycoord[k])**2
                        top_pred += float(pred[k])
            per_step[t] = {'y2': top_y2, 'pred': top_pred, 'ratio': top_y2/top_pred if top_pred > 0 else None}
            print(f'{arm} step {t}: spread ratio (eta*lambda >= 0.1) {per_step[t]["ratio"]}', flush=True)
        out[arm] = {subset: summarize(a) for subset, a in acc.items()}
        out[arm]['per_step_top'] = per_step
        print(arm, json.dumps(out[arm]['all'], indent=1), flush=True)
    verdict = {}
    for subset in ['all', 't>=100']:
        for key, label in [('lag10_predicted_residual', 'primary'), ('lag10_predicted_raw_mean', 'literal')]:
            per_arm = {arm: adequate(out[arm][subset], key) for arm in UNITS}
            verdict[f'{label}|{subset}'] = {'per_arm': per_arm, 'O1': 'adequate' if all(per_arm.values()) else 'not adequate'}
    result = {'O1': verdict['primary|all']['O1'], 'verdicts': verdict, 'arms': out, 'whitening_check_max': checks,
              'units': {k: str(v) for k, v in UNITS.items()}}
    (STUDY/'results').mkdir(exist_ok=True)
    (STUDY/'results'/'ou_check.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({'O1': result['O1'], 'verdicts': verdict, 'whitening_check_max': checks}, indent=1))


if __name__ == '__main__':
    main()
