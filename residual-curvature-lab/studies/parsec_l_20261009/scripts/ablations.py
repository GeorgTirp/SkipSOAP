"""Smoke ablations (not an evaluation): one seed, untuned, 100 steps on the main ViT.

Part 1: PARSEC-L switches (batched extraction, rho signal off, hand-over, norm blend) against the
baseline and native SOAP. Part 2: branch scaling at depth 2 and 4, with the block Jacobian deviation
measured on 64 validation examples before and after training. Writes logs/ablations.json.
Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): ablations.py [--steps 100] [--seed 1]
"""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))

from parsec_l import PARSECL, block_jacobian_deviation, ggn_probes, with_branch_scaling  # noqa: E402
from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.data import DataOrder, dataset  # noqa: E402
from residual_curvature.models import make_model  # noqa: E402

BASE = {'rank': 16, 'refresh': 'extract', 'probe_every': 10, 'norm_blend': 0., 'batched': True}  # as logged
SWITCHES = {'soap': {'rank': 0}, 'parsec_l': {}, 'per_tensor_extraction': {'batched': False},
            'rho_signal_off': {'signals': False}, 'handover_30': {'probe_until': 30}, 'handover_50': {'probe_until': 50},
            'norm_blend_0.5': {'norm_blend': .5}, 'norm_blend_1': {'norm_blend': 1.}}
DEPTHS, EXPONENTS = [2, 4], [0., 1.]


def evaluate(model, x, y, ids):
    with torch.no_grad():
        z = model(x[ids])
        return float(torch.nn.functional.cross_entropy(z, y[ids]))


def run(c, settings, steps, seed, x, y, manifest, jacobian=False):
    c = dict(c, seed=seed)
    model = make_model(c)
    optimizer = PARSECL(model.parameters(), **dict(c['soap'], lr=.01), gamma=.01, seed=seed, **dict(BASE, **settings))
    order = DataOrder(manifest['training'], 102+1000*seed)
    generator = torch.Generator().manual_seed(1000+seed)
    panel = manifest['validation'][:64]
    out = {'settings': settings, 'validation': {}, 'probe_step_seconds': [], 'other_step_seconds': [],
           'probe_seconds': [], 'rescale': [], 'energy': []}
    if jacobian:
        out['jacobian'] = {0: block_jacobian_deviation(model, x[panel], probes=8, power_steps=20,
                                                       generator=torch.Generator().manual_seed(7))}
    for step in range(1, steps+1):
        ids = order.take(512)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x[ids]), y[ids]).backward()
        probes = None
        if optimizer.wants_probes():
            started = time.perf_counter()
            probes, _ = ggn_probes(model, x[ids], samples=4, generator=generator)
            out['probe_seconds'].append(time.perf_counter()-started)
        started = time.perf_counter()
        optimizer.step(probes=probes)
        (out['probe_step_seconds'] if probes is not None else out['other_step_seconds']).append(time.perf_counter()-started)
        info = optimizer.last_info
        if info['rescale'] is not None:
            out['rescale'].append(info['rescale'])
        out['energy'].append(info['correction_energy'])
        if step in [25, 50, 100, steps]:
            out['validation'][step] = evaluate(model, x, y, manifest['validation'])
    if jacobian:
        out['jacobian'][steps] = block_jacobian_deviation(model, x[panel], probes=8, power_steps=20,
                                                          generator=torch.Generator().manual_seed(7))
    return out


def summary(name, r):
    mean = lambda v: statistics.mean(v) if v else 0.
    line = (f"{name:24s} val25 {r['validation'][25]:.4f} val50 {r['validation'][50]:.4f} val100 {r['validation'][100]:.4f} "
            f"| optimizer ms: probe-step {1e3*mean(r['probe_step_seconds']):6.1f} other {1e3*mean(r['other_step_seconds']):5.1f} "
            f"| probes ms {1e3*mean(r['probe_seconds']):6.1f}")
    if r['rescale']:
        line += f" | rescale mean {mean(r['rescale']):.3f} max {max(r['rescale']):.3f}"
    return line


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
    switches = {}
    for name, settings in SWITCHES.items():
        switches[name] = run(c, settings, args.steps, args.seed, x, y, manifest)
        print(summary(name, switches[name]), flush=True)
    depth = {}
    for blocks in DEPTHS:
        for exponent in EXPONENTS:
            cd = with_branch_scaling(dict(c, model=dict(c['model'], blocks=blocks)), exponent=exponent)
            name = f'depth{blocks}_exp{exponent:g}'
            depth[name] = run(cd, {}, args.steps, args.seed, x, y, manifest, jacobian=True)
            depth[name]['branch_scale'] = cd['model']['branch_scale']
            j = depth[name]['jacobian']
            print(f"{name:24s} scale {cd['model']['branch_scale']:.3f} val100 {depth[name]['validation'][args.steps]:.4f} "
                  f"| rms per block step0 {[round(b['rms'], 3) for b in j[0]]} step{args.steps} {[round(b['rms'], 3) for b in j[args.steps]]} "
                  f"| spectral step{args.steps} {[round(b['spectral'], 2) for b in j[args.steps]]}", flush=True)
    (STUDY/'logs').mkdir(exist_ok=True)
    (STUDY/'logs'/'ablations.json').write_text(json.dumps({'steps': args.steps, 'seed': args.seed, 'lr': .01, 'gamma': .01,
                                                           'base': BASE, 'switches': switches, 'depth': depth}, indent=1)+'\n')


if __name__ == '__main__':
    main()
