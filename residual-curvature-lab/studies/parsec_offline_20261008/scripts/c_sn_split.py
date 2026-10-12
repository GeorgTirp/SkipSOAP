"""C (SN-1, partial): Kronecker errors of stream writers versus first-in-branch layers at the SN states.

Usage: c_sn_split.py create | work --run DIR --unit NAME | pending --run DIR | finalize --run DIR
"""
import argparse
import sys
from pathlib import Path

import torch

import common
from common import read
import families as fam
import sn
from residual_curvature.curvature.layerwise import streamed_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import relative_error
from residual_curvature.data import dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry
from residual_curvature.models.mlp import functional_logits
from residual_curvature.provenance import write_json

STEPS = [10, 50, 200]
WRITERS = ('attention.output.weight', 'w2.weight')
FIRST = ('attention.query.weight', 'attention.key.weight', 'attention.value.weight', 'w1.weight')


def role(name):
    name = name.replace('.parametrizations.weight.original', '.weight')
    if name.endswith(WRITERS):
        return 'stream_writer'
    if name.endswith(FIRST):
        return 'first_in_branch'
    if name in ['input.projection.weight', 'input.weight']:
        return 'input'
    if name == 'head.weight':
        return 'head'
    return None


def group_errors(registry, blocks):
    out = {}
    for group in ['stream_writer', 'first_in_branch', 'input', 'head']:
        exact, ks1, kdiag = [], [], []
        for spec, block in zip(registry.specs, blocks):
            if len(spec.shape) == 2 and role(spec.name) == group:
                exact.append(block)
                ks1.append(fam.kron_sum_block(block, spec, 1)[0])
                kdiag.append(fam.kron_basis_diag_block(block, spec)[0])
        energy = sum(float(e.square().sum()) for e in exact)
        out[group] = {'tensors': len(exact),
                      'ks1_error': (sum(float((a-e).square().sum()) for a, e in zip(ks1, exact))/energy)**.5,
                      'kron_basis_diag_error': (sum(float((a-e).square().sum()) for a, e in zip(kdiag, exact))/energy)**.5}
    return out


def blocks_with_checks(model, registry, flat, xb, yb, step):
    blocks, _ = streamed_blocks(model, registry, flat, xb, 8, 3072)
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    gen = torch.Generator().manual_seed(303+step)
    worst = 0.
    for spec, block in zip(registry.specs, blocks):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
        worst = max(worst, relative_error(block@v[spec.slice], op.action(v)[spec.slice]))
    if worst > 1e-8:
        raise RuntimeError(f'Action check failed: {worst}')
    return blocks, worst


def work(directory, name):
    common.verify_archive(directory)
    unit = directory/name
    if common.unit_complete(unit):
        print(f'{name}: verified complete; skipped', flush=True)
        return
    common.prepare_unit(unit)
    resolved = read(common.SN_RUN/name/'resolved_arm.json')
    config, bound = resolved['config'], resolved['spec']['bound']
    x, y, manifest = dataset(config)
    construction = manifest['analysis'][:512]
    xb, yb = x[construction], y[construction]
    rows = {}
    for step in STEPS:
        arrays = torch.load(common.SN_RUN/name/f'state-{step:04d}'/'arrays.pt', map_location='cpu', weights_only=False)
        model = sn.make_sn_model(config, bound)
        registry = Registry(model)
        commit(model, registry, arrays['parameters'])
        flat = registry.flatten(dict(model.named_parameters())).detach()
        raw, raw_check = blocks_with_checks(model, registry, flat, xb, yb, step)
        twin, twin_registry = sn.effective_twin(model, config)
        twin_flat = twin_registry.flatten(dict(twin.named_parameters())).detach()
        effective, eff_check = blocks_with_checks(twin, twin_registry, twin_flat, xb, yb, step)
        rows[step] = {'raw': group_errors(registry, raw), 'effective': group_errors(twin_registry, effective),
                      'max_action_error': max(raw_check, eff_check),
                      'active_sn_layers': sum(r['active'] for r in sn.spectral_report(model, bound).values())}
        if name == 'vit__nosn__s0' and step in [10, 50]:
            stored = torch.load(common.FAMILY/'b2_w16__s0'/f'state-{step:04d}'/'arrays.pt', map_location='cpu', weights_only=False)['ggn_blocks']
            difference = (sum(float((a-b).square().sum()) for a, b in zip(effective, stored))/sum(float(b.square().sum()) for b in stored))**.5
            rows[step]['gate_relative_frobenius'] = difference
            if difference > 1e-12:
                raise RuntimeError(f'Block gate failed at step {step}: {difference}')
        print(f"{name} step {step}: writers ks1 {rows[step]['effective']['stream_writer']['ks1_error']:.3f} "
              f"first {rows[step]['effective']['first_in_branch']['ks1_error']:.3f}", flush=True)
    write_json(unit/'groups.json', rows)
    common.finish_unit(unit, {'steps': STEPS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['create', 'work', 'pending', 'finalize'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--unit')
    args = parser.parse_args()
    common.guard()
    common.setup_torch()
    if args.command == 'create':
        c = common.study_config()
        units = {n: {} for n in read(common.SN_RUN/'protocol.json')['units']}
        print(common.create_run(c, {'study': 'parsec_offline_20261008', 'part': 'C (SN-1 partial)', 'document': str(common.PROTOCOL),
                                    'units': units, 'source': str(common.SN_RUN)},
                                [str(common.STUDY_REL/'scripts'/'queue_runner.sh')]))
    elif args.command == 'work':
        try:
            work(args.run, args.unit)
        except Exception as error:
            write_json(args.run/'failure.json', {'unit': args.unit, 'error': repr(error)})
            raise
    elif args.command == 'pending':
        print('\n'.join(n for n in read(args.run/'protocol.json')['units'] if not common.unit_complete(args.run/n)))
    else:
        names = list(read(args.run/'protocol.json')['units'])
        missing = [n for n in names if not common.unit_complete(args.run/n)]
        if missing or (args.run/'failure.json').exists():
            raise RuntimeError(f'Incomplete or failed: {missing}')
        write_json(args.run/'complete.json', {'completed': True, 'units': len(names)})
        print(common.mirror(args.run))


if __name__ == '__main__':
    sys.exit(main())
