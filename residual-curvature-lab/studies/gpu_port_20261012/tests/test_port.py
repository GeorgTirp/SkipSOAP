"""GPU-port tests (PROTOCOL.md): unit plan, copy = original on the CPU, bitwise CPU replay, one GPU step against the
CPU, CUDA repeatability, CIFAR-10 split and ViT."""
import io

import pytest
import torch

import cifar
import common
import gpu_parsec
import parsec_l
import train
from residual_curvature.data import DataOrder
from residual_curvature.models import Registry


def test_unit_plan_and_settings():
    units = train.units()
    assert len(units) == 24
    assert sorted(n for n, s in units.items() if s['windows']) == ['soap__cpu64__s0', 'topfree__cpu64__s0']
    assert {n: s['repeat_of'] for n, s in units.items() if s['repeat_of']} == {
        'soap__gpu64rep__s0': 'soap__gpu64__s0', 'soap__gpu32rep__s0': 'soap__gpu32__s0',
        'topfree__gpu64rep__s0': 'topfree__gpu64__s0', 'topfree__gpu32rep__s0': 'topfree__gpu32__s0'}
    assert {s['seed'] for s in units.values() if s['configuration'] == 'gpu32'} == {0, 4, 5, 6}
    c = train.unit_config(0)
    model = torch.nn.Linear(2, 2)
    topfree = train.make_optimizer(units['topfree__gpu32__s0'], model, c)
    assert isinstance(topfree, gpu_parsec.PARSECL)
    assert {k: topfree.parsec[k] for k in ['rank', 'probe_every', 'calibration', 'strength', 'top_rhs', 'norm_blend',
                                           'batched', 'refresh', 'seed']} == {
        'rank': 16, 'probe_every': 10, 'calibration': 'lr', 'strength': 64., 'top_rhs': 'gradient', 'norm_blend': 0.,
        'batched': False, 'refresh': 'extract', 'seed': 0}
    assert topfree.param_groups[0]['lr'] == .01 and topfree.param_groups[0]['betas'] == [.95, .95]
    original = train.make_optimizer(units['topfree__cpu64orig__s0'], model, c)
    assert type(original) is parsec_l.PARSECL and original.parsec == topfree.parsec
    assert train.make_optimizer(units['soap__cpu64__s0'], model, c).parsec['rank'] == 0


def run_steps(spec, steps, save=()):
    """A short unit on its device; returns final flat parameters and the states saved after the listed steps."""
    c, x, y, manifest, model, _ = train.build(spec)
    registry = Registry(model)
    order = DataOrder(manifest['training'], common.data_order_seed(spec['seed']))
    optimizer = train.make_optimizer(spec, model, c)
    probe_generator = torch.Generator().manual_seed(train.PROBE_SEED+spec['seed'])
    probe_fn = train.implementation(spec['implementation']).ggn_probes
    ema, states = {}, {}
    for step in range(1, steps+1):
        train.train_step(model, registry, optimizer, probe_fn, probe_generator, x, y, order.take(train.BATCH))
        if step in save:
            buffer = io.BytesIO()
            train.save_state(buffer, step, model, registry, optimizer, ema, order, probe_generator)
            buffer.seek(0)
            states[step] = torch.load(buffer, weights_only=False)
    return train.flat_cpu(model, registry), states, (c, x, y, model, registry)


def test_copy_equals_original_bitwise_on_cpu():
    units = train.units()
    for arm in train.ARMS:
        copy, _, _ = run_steps(units[f'{arm}__cpu64__s0'], 14)
        original, _, _ = run_steps(units[f'{arm}__cpu64orig__s0'], 14)
        assert torch.equal(copy, original)


def test_cpu_replay_is_bitwise_and_gpu_step_agrees():
    spec = train.units()['topfree__cpu64__s0']
    _, states, (c, x, y, model, registry) = run_steps(spec, 13, save=(11, 12, 13))
    rows = [train.replay_step(spec, c, model, registry, x, y, states[t], states[t+1]) for t in (11, 12)]
    assert all(r['relative_update_error'] == 0.0 and {v for k, v in r['state_max_abs_difference'].items()
                                                       if not k.endswith('_tensor')} == {0.0} for r in rows)
    assert [r['probe'] for r in rows] == [True, False]
    gpu = dict(spec, device='cuda')
    gc, gx, gy, _, gmodel, _ = train.build(gpu)
    gregistry = Registry(gmodel)
    for t in (11, 12):
        row = train.replay_step(gpu, gc, gmodel, gregistry, gx, gy, states[t], states[t+1])
        assert row['relative_update_error'] <= 1e-8


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_cuda_runs_repeat_bitwise(dtype):
    spec = dict(train.units()['topfree__gpu32__s0'], dtype=dtype)
    first, _, _ = run_steps(spec, 13)
    second, _, _ = run_steps(spec, 13)
    assert torch.isfinite(first).all() and torch.equal(first, second)


def test_cifar_split_and_models():
    raw, y, manifest = cifar.cifar10()
    assert raw.shape == (50000, 3, 32, 32) and raw.dtype == torch.uint8
    train_ids, validation_ids = set(manifest['training']), set(manifest['validation'])
    assert len(train_ids) == 45000 and len(validation_ids) == 5000 and not train_ids & validation_ids
    assert manifest['official_test'] == 'reserved, not accessed' and manifest == cifar.cifar10()[2]
    x = cifar.normalize(raw[manifest['training'][:1000]], manifest, torch.float64)
    assert x.mean((0, 2, 3)).abs().max() < .1 and (x.std((0, 2, 3))-1).abs().max() < .1
    sizes = {name: Registry(cifar.make_cifar_model(name, 0, 'cpu', torch.float64)).size for name in cifar.MODELS}
    assert sizes == {'M1': 208074, 'M2': 1205898}
    model, twin = (cifar.make_cifar_model('M1', 0, 'cpu', torch.float64) for _ in range(2))
    assert torch.equal(Registry(model).flatten(dict(model.named_parameters())),
                       Registry(twin).flatten(dict(twin.named_parameters())))
    assert {s.role for s in Registry(model).specs} == {'embedding', 'linear_weight', 'bias', 'normalization'}
    assert model(x[:5]).shape == (5, 10)
    probes, _ = gpu_parsec.ggn_probes(model, x[:3], samples=2, generator=torch.Generator().manual_seed(0))
    assert all(v.shape == (6, *p.shape) for p, v in probes.items())
