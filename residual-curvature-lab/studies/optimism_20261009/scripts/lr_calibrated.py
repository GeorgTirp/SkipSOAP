"""Exploratory (not pre-declared): PARSEC-L with learning-rate calibration beta = eta_t against damping calibration.

In SOAP's whitened coordinates its step -eta z is the proximal step for (1/(2 eta)) |zeta|^2. Adding the
true curvature gives zeta = -eta (I + eta F0^T C F0)^-1 z, i.e. PARSEC-L with beta = eta_t, where
kappa_k = eta lambda_k is the edge-of-stability ratio along whitened direction k. This script measures
fixed steps (native SOAP step size, no search) at the six states. Writes results/lr_calibrated.json.
"""
import json
import statistics
import sys
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY/'scripts'))
from optimism import PARENT, STATES  # noqa: E402
from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.curvature.operators import loss  # noqa: E402
from residual_curvature.data import dataset  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry, make_model  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402
from residual_curvature.optim import capacity  # noqa: E402
from residual_curvature.optim.soap import SOAPMetric  # noqa: E402

RANK = 16


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, _ = dataset(c)
    out = []
    for trajectory, step in STATES:
        source = PARENT/f'{trajectory}-{step:04d}'
        arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
        info = json.loads((source/'diagnostics.json').read_text())
        flat, g, momentum, contexts, ggn, native = [arrays[k] for k in ['parameters', 'gradient', 'momentum_rhs', 'contexts', 'ggn_blocks', 'raw_soap']]
        model = make_model(c)
        registry = Registry(model)
        commit(model, registry, flat)
        construction, panel_ids = info['construction_ids'], info['panel_ids']
        fn = functional_logits(model, registry, x[construction])
        own_loss = lambda w: float(loss(fn(w), y[construction]))
        panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: float(loss(f(w), t)) for ids in panel_ids]
        base, panel_base = own_loss(flat), [p(flat) for p in panels]
        metric = SOAPMetric(registry, contexts)
        mu = max(.01*sum(float(b.trace()) for b in ggn)/registry.size, c['damping']['floor'])
        factors = [capacity.factor(torch.eye(s.count).reshape(s.shape+(s.count,)), contexts[s.name]).reshape(s.count, s.count) for s in registry.specs]
        pu = torch.cat([f@f.T@momentum[s.slice] for s, f in zip(registry.specs, factors)])
        eta = float(native.norm()/pu.norm())
        row = {'state': f'{trajectory}-{step}', 'eta': eta, 'beta_damping': 1/(mu*metric.scale), 'beta_ratio_lr_over_damping': eta*mu*metric.scale}
        for label, beta in [('damping', 1/(mu*metric.scale)), ('lr', eta)]:
            ops, kmax, above1 = [], 0., 0
            for f, block in zip(factors, ggn):
                eig, q = torch.linalg.eigh(beta*f.T@block@f)
                r = min(RANK, len(eig))
                w, kappa = q[:, -r:], eig[-r:].clamp_min(0)
                ops.append(f@(torch.eye(len(eig))-w@torch.diag(kappa/(1+kappa))@w.T)@f.T)
                kmax = max(kmax, float(kappa.max()))
                above1 += int((eig > 1).sum())
            row[f'kappa_max_{label}'] = kmax
            row[f'directions_kappa_above_1_{label}'] = above1
            for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
                step_vec = -eta*torch.cat([a@rhs[s.slice] for s, a in zip(registry.specs, ops)])
                held = [b-p(flat+step_vec) for p, b in zip(panels, panel_base)]
                row[f'{label}_{rhs_name}'] = {'own': base-own_loss(flat+step_vec), 'held': statistics.mean(held),
                                              'norm_ratio_to_soap': float(step_vec.norm()/(eta*torch.cat([f@f.T@rhs[s.slice] for s, f in zip(registry.specs, factors)]).norm()))}
        for rhs_name, rhs in [('gradient', g), ('momentum', momentum)]:
            soap = -eta*torch.cat([f@f.T@rhs[s.slice] for s, f in zip(registry.specs, factors)])
            row[f'soap_{rhs_name}'] = {'own': base-own_loss(flat+soap), 'held': statistics.mean(b-p(flat+soap) for p, b in zip(panels, panel_base))}
        out.append(row)
        print(f"{row['state']}: eta {eta:.2e}, beta_lr/beta_damping {row['beta_ratio_lr_over_damping']:.3f}, "
              f"kappa_max damping {row['kappa_max_damping']:.0f} lr {row['kappa_max_lr']:.2f} (dirs >1: {row['directions_kappa_above_1_lr']}) | "
              + ' | '.join(f"{rhs}: held SOAP {row[f'soap_{rhs}']['held']:+.4f} damp {row[f'damping_{rhs}']['held']:+.4f} lr {row[f'lr_{rhs}']['held']:+.4f} "
                           f"(norm lr/soap {row[f'lr_{rhs}']['norm_ratio_to_soap']:.3f})" for rhs in ['gradient', 'momentum']), flush=True)
    (STUDY/'results'/'lr_calibrated.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()
