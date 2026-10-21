"""Diagnosis for DEVIATIONS.md D1, not a protocol reading: which side of the block-against-action audit carries the
float32 error.

From a unit's checkpoint at step s, it rebuilds the batch of step s+1 (same data order and augmentation draws as
train.run), the gradient and the shadow SOAP's contexts. It then runs, for the same random u per tensor:
  1. the audit as written (ggn.tensor_ggn_step with the float32 action), for every tensor, recorded without raising;
  2. float64 recomputations, for the named tensors, of both sides: the whitened block K u (from per-example
     Jacobians) and the reference F0^T C F0 u (from the microbatched GGN action), with the same F0 (upcast).
The errors are given relative to the damped action |F0^T C F0 u + mu' u| in float64, as in the gate.

Usage (from residual-curvature-lab/): diagnose_audit.py --run DIR --unit NAME --step S [--tensors NAME ...]
"""
import argparse
import json
import sys
from pathlib import Path

import common
from common import read

import torch  # noqa: E402

import cifar  # noqa: E402
import ggn  # noqa: E402
import micro  # noqa: E402
import train  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.optim.soap import SOAPMetric, soap_propose  # noqa: E402

DEFAULT_TENSORS = ['head.weight', 'head.bias', 'norm.weight', 'blocks.5.attention.output.weight',
                   'blocks.5.attention.query.weight', 'blocks.0.attention.query.weight', 'input.projection.weight']


def side(model, registry, flat, x, spec, context, u, chunk=128):
    """(K u from the whitened block, F0^T C F0 u from the action) for tensor `spec`, in the dtype of `flat`."""
    blocks, _ = ggn.accumulate(model, registry, flat, x, [spec], chunk, all_parameters=False, contexts={spec.name: context})
    block_u = blocks[spec.name]@u
    v = torch.zeros_like(flat)
    v[spec.slice] = ggn.unwhiten(u.reshape(spec.shape), context).reshape(-1)
    action = micro.ggn_action(model, registry, flat, x, train.MICRO)
    reference = ggn.whiten(action(v)[spec.slice].reshape((1,)+spec.shape), context).reshape(-1)
    return block_u, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--tensors', nargs='+', default=DEFAULT_TENSORS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda')
    spec = read(args.run/'protocol.json')['units'][args.unit]
    if spec['arm'] != 'ggn':
        raise SystemExit('only the plain ggn arm is supported')
    data = train.Data(device)
    order = DataOrder(list(range(len(data.manifest['training']))), train.SEEDS['order']+1000*spec['seed'])
    augment = torch.Generator().manual_seed(train.SEEDS['augment']+spec['seed'])
    for _ in range(args.step+1):
        x, y = data.batch(order.take(train.BATCH), augment)
    checkpoint = torch.load(args.run/args.unit/'checkpoints'/f'step-{args.step:04d}.pt', weights_only=False)

    model = train.build_model(spec, device)
    registry = Registry(model)
    flat = checkpoint['parameters'].to(device)
    commit(model, registry, flat)
    _, g = micro.gradient(model, registry, flat, x, y, train.MICRO)
    _, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), checkpoint['shadow'], train.shadow_config())
    metric = SOAPMetric(registry, contexts)

    # 1. The audit values for every tensor (audit generator seeded as at step s+1). The diagnosis records them
    #    instead of raising (tolerance inf); the gate itself is train.TOLERANCE and is reported against it.
    _, info = ggn.tensor_ggn_step(model, registry, flat, x, g, metric, spec['gamma'], train.FLOOR, float('inf'),
                                  train.GGN_BUDGET, train.GGN_CHUNK, action=micro.ggn_action(model, registry, flat, x, train.MICRO),
                                  audit_generator=torch.Generator().manual_seed(303+args.step+1))
    damping = info['mu']*info['scale']
    out = {'run': str(args.run), 'unit': args.unit, 'checkpoint_step': args.step, 'batch_step': args.step+1,
           'gate': train.TOLERANCE, 'mu': info['mu'], 'scale': info['scale'], 'damping': damping,
           'action_checks': info['action_checks'], 'kappa_max': info['kappa_max'],
           'whitened_residuals': info['whitened_residuals'], 'max_action_check': max(info['action_checks'].values()),
           'over_gate': {k: v for k, v in info['action_checks'].items() if v > train.TOLERANCE},
           'worst': sorted(info['action_checks'].items(), key=lambda kv: -kv[1])[:8]}
    print('mu', info['mu'], 'over gate:', out['over_gate'], flush=True)

    # 2. Both sides in float32 and float64, same u and same (upcast) F0.
    model64 = cifar.make_cifar_model(train.MODEL, spec['seed'], device, dtype=torch.float64)
    registry64 = Registry(model64)
    flat64 = flat.double()
    commit(model64, registry64, flat64)
    x64 = x.double()
    specs, specs64 = {s.name: s for s in registry.specs}, {s.name: s for s in registry64.specs}
    out['tensors'] = {}
    for name in args.tensors:
        context = contexts[name]
        context64 = {'basis': [q.double() if torch.is_tensor(q) else q for q in context['basis']], 'denominator': context['denominator'].double()}
        u64 = torch.randn(specs[name].count, generator=torch.Generator().manual_seed(7), dtype=torch.float64).to(device)
        block32, reference32 = side(model, registry, flat, x, specs[name], context, u64.float())
        block64, reference64 = side(model64, registry64, flat64, x64, specs64[name], context64, u64)
        scale = float((reference64+damping*u64).norm())
        row = {
            'gate_as_written': float((block32-reference32).norm())/float((reference32+damping*u64.float()).norm()),
            'block32_error': float((block32.double()-block64).norm())/scale,
            'reference32_error': float((reference32.double()-reference64).norm())/scale,
            'float64_agreement': float((block64-reference64).norm())/scale,
            'curvature_over_damping': float(reference64.norm())/float(damping*u64.norm()),
            'denominator_decades': float(context['denominator'].max().log10()-context['denominator'].clamp_min(1e-30).min().log10()),
        }
        out['tensors'][name] = row
        print(name, json.dumps(row), flush=True)
    out['note'] = 'Errors relative to |F0^T C F0 u + mu\' u| (float64), as in the gate; see DEVIATIONS.md D1.'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print('max gate value over all tensors at this step:', out['max_action_check'])
    print('worst:', out['worst'])


if __name__ == '__main__':
    sys.exit(main())
