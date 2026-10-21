"""Measures A-D (scripts/nonlinear.py) against independent references, in float64 on the CPU, for the plain and the
SN model."""
import pytest
import torch

import common  # first: it puts the cifar_ggn_20261012 modules on the path
import micro  # noqa: E402
import nonlinear  # noqa: E402
import train  # noqa: E402
from residual_curvature.curvature.operators import loss  # noqa: E402
from residual_curvature.models import Registry  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402


def state(arm, n=6, scale=1e-3, seed=0):
    model = train.build_model({'arm': arm, 'seed': 1}, 'cpu').double()
    registry = Registry(model)
    flat = registry.flatten(dict(model.named_parameters())).detach()
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=gen, dtype=torch.float64)
    y = torch.randint(0, 10, (n,), generator=gen)
    d = scale*torch.randn(registry.size, generator=gen, dtype=torch.float64)
    return model, registry, flat, d, x, y


@pytest.mark.parametrize('arm', ['soap', 'soap_sn'])
def test_error_split_against_references(arm):
    model, registry, flat, d, x, y = state(arm)
    out = nonlinear.error_split(model, registry, flat, d, x, y, 4, targets=(0.05,))
    step = out['step']
    objective = lambda w: float(loss(functional_logits(model, registry, x)(w), y))
    assert abs(step['actual']-(objective(flat+d)-objective(flat))) <= 1e-12
    _, g = micro.gradient(model, registry, flat, x, y, 4)
    assert abs(step['gd']-float(g@d)) <= 1e-10*abs(float(g@d))
    full = float(d@micro.ggn_action(model, registry, flat, x, 4)(d))
    assert abs(step['full']-full) <= 1e-10*full
    # Jd by central differences of the logits, then the linearized loss.
    fn, h = functional_logits(model, registry, x), 1e-4
    jd = (fn(flat+h*d)-fn(flat-h*d)).detach()/(2*h)
    assert abs(step['linearized']-(float(loss(fn(flat).detach()+jd, y))-objective(flat))) <= 1e-7*abs(step['linearized'])
    assert abs(step['network_error']+step['taylor_error']-(step['actual']-step['predicted_full'])) <= 1e-15
    # The residual term is d'Hd - d'Cd to second order: the relative gap shrinks linearly with the step.
    gaps = []
    for factor in (1., .1):
        e = factor*d
        hessian = micro.hessian_quadratic(model, registry, flat, e, x, y, 4)
        reference = hessian-float(e@micro.ggn_action(model, registry, flat, x, 4)(e))
        term = nonlinear.error_split(model, registry, flat, e, x, y, 4)['step']['residual_term']
        gaps.append(abs(term-reference)/abs(reference))
    assert gaps[0] <= .05 and .05 <= gaps[1]/gaps[0] <= .2
    # Linearization error is first order in the step; B scales the step to the target RMS logit change.
    half = nonlinear.error_split(model, registry, flat, d/2, x, y, 4)['step']['linearization_error']
    assert abs(half/step['linearization_error']-.5) <= .05
    matched = out['matched']['0.05']
    assert abs(matched['scale']*out['jd_rms']-.05) <= 1e-12
    assert abs((matched['full']/step['full'])**.5-matched['scale']) <= 1e-10*matched['scale']


@pytest.mark.parametrize('arm', ['soap', 'soap_sn'])
def test_persistence_against_references(arm):
    model, registry, flat, d, x, _ = state(arm)
    now = nonlinear.directional_ggn(model, registry, flat, d, x, 4)
    assert abs(now-float(d@micro.ggn_action(model, registry, flat, x, 4)(d))) <= 1e-10*now
    out = nonlinear.persistence(model, registry, flat, d, x, 4, now, old_flat=flat)
    assert out['ten_step_ratio'] == 1.
    after = float(d@micro.ggn_action(model, registry, flat+d, x, 4)(d))
    assert abs(out['after_step']-after) <= 1e-10*after
    assert nonlinear.persistence(model, registry, flat, d, x, 4, now)['ten_step_ratio'] is None


@pytest.mark.parametrize('arm', ['soap', 'soap_sn'])
def test_representation_measures(arm):
    model, registry, flat, d, x, _ = state(arm, n=8)
    out, after = nonlinear.representation(model, registry, flat, d, x)
    assert list(out) == [f'depth{k}_stream' for k in range(7)]+['head'] and 'drift' not in out['head']
    half, _ = nonlinear.representation(model, registry, flat, d/2, x)
    for name in out:
        if name != 'depth0_stream':  # the embedding is linear in its parameters: no remainder beyond rounding
            assert abs(half[name]['linearization_error']/out[name]['linearization_error']-.5) <= .05
    assert out['depth0_stream']['linearization_error'] <= 1e-9
    again, _ = nonlinear.representation(model, registry, flat, d, x, previous=after)
    assert all(r['drift'] == 0 and r['distance_change'] == 0 and abs(r['cka']-1) <= 1e-12 for r in again.values())
    moved, _ = nonlinear.representation(model, registry, flat+50*d, d, x, previous=after)
    assert all(r['drift'] > 0 and r['cka'] < 1 for r in moved.values())


def test_cka_and_distance_change_are_rotation_invariant():
    gen = torch.Generator().manual_seed(3)
    a = torch.randn(40, 12, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(12, 12, generator=gen, dtype=torch.float64))
    assert abs(nonlinear.linear_cka(a@q, a)-1) <= 1e-12 and nonlinear.distance_change(a@q, a) <= 1e-12
    assert abs(nonlinear.linear_cka(3*a, a)-1) <= 1e-12 and abs(nonlinear.distance_change(3*a, a)-2) <= 1e-12
    b = a+torch.randn(40, 12, generator=gen, dtype=torch.float64)
    assert nonlinear.linear_cka(b, a) < 1 and nonlinear.distance_change(b, a) > 0
