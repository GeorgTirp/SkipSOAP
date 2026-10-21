"""OT probes against the lab's dense Wasserstein blocks, the cross-batch stream, and the top-free arm against
cifar_ggn's (bitwise on the CPU)."""
import importlib.util
import sys

import pytest
import torch

import common  # first: it puts the earlier studies' modules on the path
import probes_ot  # noqa: E402
import train  # noqa: E402
from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.curvature.wasserstein import streamed_wasserstein_blocks  # noqa: E402
from residual_curvature.data import dataset  # noqa: E402
from residual_curvature.models import Registry, make_model  # noqa: E402


def ggn_train():
    if 'ggn_train' not in sys.modules:
        spec = importlib.util.spec_from_file_location('ggn_train', common.LIB/'train.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['ggn_train'] = module
        spec.loader.exec_module(module)
    return sys.modules['ggn_train']


def test_exact_ot_probes_reproduce_the_labs_wasserstein_blocks():
    torch.set_default_dtype(torch.float64)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, manifest = dataset(c)
    model = make_model(c)
    registry = Registry(model)
    xs = x[manifest['training'][:9]]
    flat = registry.flatten(dict(model.named_parameters())).detach()
    blocks, _ = streamed_wasserstein_blocks(model, registry, flat, xs, microbatch=3)
    rows = probes_ot.ot_probes(model, xs, exact=True, chunk=5)
    params = dict(model.named_parameters())
    for spec, block in zip(registry.specs, blocks):
        u = rows[params[spec.name]].reshape(len(xs)*10, -1)
        assert torch.allclose(u.T@u, block, rtol=1e-9, atol=1e-12*float(block.abs().max())+1e-15), spec.name


def test_gaussian_ot_probes_are_unbiased():
    torch.set_default_dtype(torch.float64)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, _, manifest = dataset(c)
    model = make_model(c)
    registry = Registry(model)
    xs = x[manifest['training'][:3]]
    exact = probes_ot.ot_probes(model, xs, exact=True)
    sampled = probes_ot.ot_probes(model, xs, samples=4000, generator=torch.Generator().manual_seed(1), chunk=1000)
    p = model.head.weight
    e, s = exact[p].reshape(len(exact[p]), -1), sampled[p].reshape(len(sampled[p]), -1)
    reference = e.T@e
    assert (s.T@s-reference).norm() <= .1*reference.norm()


def batch(n=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=gen), torch.randint(0, 10, (n,), generator=gen)


def test_topfree_arm_equals_cifar_ggn_bitwise():
    spec = {'arm': 'topfree', 'lr': 3e-2, 'seed': 1, 'measure': False}
    ours, theirs = train.Arm(spec, 'cpu'), ggn_train().Arm(spec, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        assert ours.step(t, x, y, None) == theirs.step(t, x, y, None)
    assert torch.equal(ours.flat(), theirs.flat())


@pytest.mark.parametrize('arm', ['topfree_ot', 'topfree_xbatch'])
def test_variant_arms_probe_at_step_2_only(arm):
    spec = train.spec_of(arm, 1)
    model_arm = train.Arm(spec, 'cpu')
    x, y = batch()
    calls = []

    def probe_batch():
        calls.append(len(calls))
        return batch(seed=99)

    for t in range(1, 4):
        info = model_arm.step(t, x, y, probe_batch)
        assert info['probe'] == (t == 2)
    assert calls == ([0] if arm == 'topfree_xbatch' else [])
    assert torch.isfinite(model_arm.flat()).all()
    assert model_arm.optimizer.parsec['strength'] == spec.get('strength', train.STRENGTH)


def test_unit_plan():
    assert sorted(s['strength'] for s in train.tune_units().values()) == train.C_GRID
    chosen = {'topfree_ot': {'strength': 1024., 'ema_decay': .98}, 'topfree': {'strength': 64.}, 'topfree_xbatch': {'strength': 64.}}
    units = train.eval_units(chosen)
    assert len(units) == 9 and 'topfree__lr0.03__s1' in units and 'topfree_ot__lr0.03__c1024__s2' in units
    assert train.extension_units({'topfree_ot': {'edge': 'high', 'grid': train.C_GRID}}).popitem()[1]['strength'] == 16384.
    assert train.extension_units({'topfree_ot': {'edge': 'low', 'grid': train.C_GRID}}).popitem()[1]['strength'] == 16.
