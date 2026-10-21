"""Unit plan, topfree_c at c = 64 equal to cifar_ggn's topfree, and topfree_ot equal to probe_variants' arm (CPU)."""
import importlib.util
import sys

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


def test_unit_plan():
    units = train.tune_units()
    assert sorted(s['strength'] for s in units.values()) == train.C_GRID and {s['arm'] for s in units.values()} == {'topfree_c'}
    assert 'topfree_c__lr0.03__c64__s0' in units
    chosen = {'topfree_c': {'strength': 1024.}, 'topfree_ot': {'strength': 4096.}}
    evaluation = train.eval_units(chosen)
    assert sorted(evaluation) == sorted([f'topfree_c__lr0.03__c1024__s{s}' for s in (1, 2, 3)]+[f'topfree_ot__lr0.03__c4096__s{s}' for s in (1, 2, 3)])


def test_topfree_c_at_64_equals_cifar_ggn_topfree():
    gt = load('ggn_train', common.LIB/'train.py')
    ours = train.Arm(train.spec_of('topfree_c', 1, 64.), 'cpu')
    theirs = gt.Arm({'arm': 'topfree', 'lr': 3e-2, 'seed': 1, 'measure': False}, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        a, b = ours.step(t, x, y, None), theirs.step(t, x, y, None)
        assert {k: v for k, v in a.items() if k != 'eta_lambda_max'} == b
    assert torch.equal(ours.flat(), theirs.flat())


def test_topfree_ot_equals_probe_variants_arm():
    pv = load('pv_train', common.LAB/'studies'/'probe_variants_20261013'/'scripts'/'train.py')
    spec = train.spec_of('topfree_ot', 1, 4096.)
    ours, theirs = train.Arm(spec, 'cpu'), pv.Arm(spec, 'cpu')
    x, y = batch()
    for t in range(1, 4):
        assert ours.step(t, x, y, None) == theirs.step(t, x, y, None)
    assert torch.equal(ours.flat(), theirs.flat())
