"""Smoke run (not an evaluation): native SOAP versus PARSEC-L on the main ViT, one seed, untuned.

Checks that PARSEC-L trains end to end with sampled-label probes and measures its overhead. Writes
logs/smoke.json. Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1):
  smoke.py [--steps 100] [--seed 1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))

from parsec_l import PARSECL, ggn_probes  # noqa: E402
from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.data import DataOrder, dataset  # noqa: E402
from residual_curvature.models import make_model  # noqa: E402

ARMS = {'soap': {'rank': 0},
        'parsec_l_extract': {'rank': 16, 'refresh': 'extract', 'probe_every': 10, 'norm_blend': 0., 'batched': True},
        'parsec_l_track': {'rank': 16, 'refresh': 'track', 'probe_every': 10, 'norm_blend': 0., 'batched': True}}


def evaluate(model, x, y, ids):
    with torch.no_grad():
        z = model(x[ids])
        return float(torch.nn.functional.cross_entropy(z, y[ids])), float((z.argmax(-1) == y[ids]).double().mean())


def run(arm, settings, steps, seed, c, x, y, manifest):
    torch.manual_seed(0)
    c = dict(c, seed=seed)
    model = make_model(c)
    optimizer = PARSECL(model.parameters(), **dict(c['soap'], lr=.01), gamma=.01, seed=seed, **settings)
    order = DataOrder(manifest['training'], 102+1000*seed)
    generator = torch.Generator().manual_seed(1000+seed)
    rows, seconds, probe_seconds = [], 0., 0.
    for step in range(1, steps+1):
        ids = order.take(512)
        started = time.perf_counter()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x[ids]), y[ids])
        loss.backward()
        probes = None
        if optimizer.wants_probes():
            begun = time.perf_counter()
            probes, _ = ggn_probes(model, x[ids], samples=4, generator=generator)
            probe_seconds += time.perf_counter()-begun
        optimizer.step(probes=probes)
        seconds += time.perf_counter()-started
        info = optimizer.last_info
        row = {'step': step, 'train_loss': float(loss.detach()), 'correction_energy': info['correction_energy']}
        if info['probe']:
            tensors = info['probe']['tensors'].values()
            row.update(mu=info['probe']['mu'], kappa_max=max(t['kappa_max'] for t in tensors),
                       residual_held=max((t.get('residual_held', 0.) for t in tensors), default=0.),
                       modes=sorted({t['mode'] for t in tensors}))
        if step in [10, 25, 50, 100, 200, steps]:
            row['validation_loss'], row['validation_accuracy'] = evaluate(model, x, y, manifest['validation'])
            print(f"{arm} step {step}: val={row['validation_loss']:.4f} c={info['correction_energy']:.3f} s={seconds:.1f}", flush=True)
        rows.append(row)
    return {'settings': settings, 'seconds': seconds, 'probe_seconds': probe_seconds, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, manifest = dataset(c)
    out = {name: run(name, settings, args.steps, args.seed, c, x, y, manifest) for name, settings in ARMS.items()}
    (STUDY/'logs').mkdir(exist_ok=True)
    (STUDY/'logs'/'smoke.json').write_text(json.dumps({'steps': args.steps, 'seed': args.seed, 'lr': .01, 'gamma': .01,
                                                       'probe_samples': 4, 'arms': out}, indent=1)+'\n')


if __name__ == '__main__':
    main()
