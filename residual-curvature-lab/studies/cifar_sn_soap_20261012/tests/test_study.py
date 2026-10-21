"""cifar_sn_soap study tests (PROTOCOL.md, section 6): the unit plan, the arms' models, the soap arm against
cifar_ggn_20261012's soap arm, SN inactive equals plain SOAP, and the G1 and reading helpers."""
import importlib.util

import torch

import common  # first: it puts the cifar_ggn_20261012 modules on the path
import cifar  # noqa: E402
import g1  # noqa: E402
import measure  # noqa: E402
import summarize  # noqa: E402
import train  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402


def ggn_study_train():
    """cifar_ggn_20261012's train.py under another module name (its `common` resolves to this study's; unused here)."""
    spec = importlib.util.spec_from_file_location('cifar_ggn_train', common.LIB/'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch(n=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=gen), torch.randint(0, 10, (n,), generator=gen)


def steps(arm, count, *extra):
    x, y = batch()
    for t in range(1, count+1):
        arm.step(t, x, y, *extra)
    return arm.flat()


def test_unit_plan_and_schedule():
    units = train.tune_units()
    assert sorted((s['arm'], s['lr'], s['seed']) for s in units.values()) == [('soap_sn', lr, 0) for lr in train.LR_GRID]
    assert sorted(s['lr'] for s in train.extension_units({'soap_sn': {'edge': 'high', 'grid': train.LR_GRID}}).values()) == [9e-2]
    assert sorted(s['lr'] for s in train.extension_units({'soap_sn': {'edge': 'low', 'grid': train.LR_GRID}}).values()) == [1e-3/3]
    assert train.extension_units({'soap_sn': {'edge': None, 'grid': train.LR_GRID}}) == {}
    same = train.eval_units({'soap_sn': {'lr': 1e-2}})
    assert sorted((s['arm'], s['lr'], s['seed']) for s in same.values()) == sorted(
        [('soap', 1e-2, s) for s in (1, 2, 3)]+[('soap_sn', 1e-2, s) for s in (1, 2, 3)])
    other = train.eval_units({'soap_sn': {'lr': 3e-3}})
    assert len(other) == 9 and all(s['measure'] for s in other.values())
    assert {(s['arm'], s['lr']) for s in other.values()} == {('soap', 1e-2), ('soap_sn', 3e-3), ('soap_sn', 1e-2)}
    assert train.lr_factor(1) == 1/25 and train.lr_factor(400) == 1. and train.lr_factor(500) == .01
    assert train.SOAP_LR == 1e-2 and train.SOAP_EMA_DECAY == .98 and train.SN_BOUND == 1.


def test_arm_models():
    plain = train.build_model({'arm': 'soap', 'seed': 1}, 'cpu')
    sn = train.build_model({'arm': 'soap_sn', 'seed': 1}, 'cpu')
    assert type(plain) is cifar.CifarViT and type(sn) is cifar.CifarViTSN and sn.bound == 1.
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), sn.parameters()))
    assert Registry(plain).size == Registry(sn).size == 308042
    active = [r['active'] for r in measure.spectral_report(sn).values()]
    assert len(active) == 36 and sum(active) >= 24  # c = 1 clips most branch weights already at initialization
    opt = train.make_optimizer({'arm': 'soap_sn', 'lr': 3e-3, 'seed': 1}, sn)
    assert opt.param_groups[0]['weight_decay'] == .05 and opt.param_groups[0]['betas'] == (.95, .95)


def test_soap_arm_equals_cifar_ggn_soap_arm_bitwise():
    """Three CPU steps of this study's soap arm equal three of cifar_ggn_20261012's (G1 on the CPU)."""
    spec = {'arm': 'soap', 'lr': 1e-2, 'seed': 1}
    ours = steps(train.Arm(spec, 'cpu'), 3)
    theirs = steps(ggn_study_train().Arm(spec, 'cpu'), 3, None)
    assert torch.equal(ours, theirs)


def test_soap_sn_with_an_inactive_bound_is_plain_soap():
    old = train.SN_BOUND
    try:
        train.SN_BOUND = 1e9
        loose = steps(train.Arm({'arm': 'soap_sn', 'lr': 1e-2, 'seed': 1}, 'cpu'), 3)
    finally:
        train.SN_BOUND = old
    plain = steps(train.Arm({'arm': 'soap', 'lr': 1e-2, 'seed': 1}, 'cpu'), 3)
    assert (loose-plain).norm() <= 1e-6*plain.norm()
    active = steps(train.Arm({'arm': 'soap_sn', 'lr': 1e-2, 'seed': 1}, 'cpu'), 3)
    assert (active-plain).norm() > 1e-3*plain.norm() and torch.isfinite(active).all()


def test_g1_and_reading_helpers():
    a = [{'step': 1, 'validation_loss': 1.5, 'training_seconds': 3.}, {'step': 2, 'nested': {'seconds': 1., 'x': [1, 2]}}]
    b = [{'step': 1, 'validation_loss': 1.5, 'training_seconds': 9.}, {'step': 2, 'nested': {'seconds': 7., 'x': [1, 2]}}]
    assert g1.first_difference(g1.strip(a), g1.strip(b)) is None
    b[1]['nested']['x'][1] = 2.0000001
    assert g1.first_difference(g1.strip(a), g1.strip(b)) == '[1]/nested/x[1]'
    assert summarize.compare([1, 2, 3], [2, 3, 4]) == 'better' and summarize.compare([3, 4, 5], [2, 3, 4]) == 'worse'
    assert summarize.compare([1, 4, 3], [2, 3, 4]) == 'mixed' and summarize.compare([None, 1, 1], [2, 2, 2]) == 'mixed'
