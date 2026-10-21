"""Schedule, units, and the arms against cifar_ggn's and probe_variants' (bitwise on the CPU under the same schedule)."""
import importlib.util
import sys

import pytest
import torch

import common  # first: it puts the earlier studies' modules on the path
import train  # noqa: E402


def load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def batch():
    gen = torch.Generator().manual_seed(0)
    return torch.randn(6, 3, 32, 32, generator=gen), torch.randint(0, 10, (6,), generator=gen)


def test_schedule_and_units():
    assert train.STEPS == 1000 and train.lr_factor(1) == 1/50 and train.lr_factor(800) == 1.
    assert abs(train.lr_factor(801)-1.) <= 1e-15 and abs(train.lr_factor(1000)-.01) <= 1e-15
    units = train.units()
    assert sorted(units) == ['soap__lr0.03__s1', 'topfree__lr0.03__s1', 'topfree_ot__lr0.03__s1']
    assert units['topfree_ot__lr0.03__s1']['strength'] == 4096. and units['topfree__lr0.03__s1']['strength'] == 64.


@pytest.mark.parametrize('arm', ['soap', 'topfree'])
def test_arms_equal_cifar_ggn(arm, monkeypatch):
    gt = load('ggn_train', common.LIB/'train.py')
    monkeypatch.setattr(train, 'lr_factor', gt.lr_factor)
    spec = {'arm': arm, 'lr': 3e-2, 'seed': 1, 'measure': False}
    ours, theirs = train.Arm(spec, 'cpu'), gt.Arm(spec, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        assert ours.step(t, x, y) == theirs.step(t, x, y, None)
    assert torch.equal(ours.flat(), theirs.flat())


def test_topfree_ot_equals_probe_variants_arm(monkeypatch):
    pv = load('pv_train', common.LAB/'studies'/'probe_variants_20261013'/'scripts'/'train.py')
    monkeypatch.setattr(train, 'lr_factor', pv.lr_factor)
    spec = {'arm': 'topfree_ot', 'lr': 3e-2, 'seed': 1, 'measure': False, 'strength': 4096.}
    ours, theirs = train.Arm(spec, 'cpu'), pv.Arm(spec, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        a, b = ours.step(t, x, y), theirs.step(t, x, y, None)
        assert {k: v for k, v in b.items() if k != 'eta_lambda_max'} == a
    assert torch.equal(ours.flat(), theirs.flat())
