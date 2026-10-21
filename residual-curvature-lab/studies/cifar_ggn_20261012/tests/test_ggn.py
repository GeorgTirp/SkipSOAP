"""The GPU exact per-tensor GGN step equals the lab's streamed_blocks + solve_blocks (CPU float64, MNIST ViT)."""
import torch

import ggn
from residual_curvature.curvature.layerwise import solve_blocks, streamed_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.soap import SOAPMetric, frozen_apply, soap_propose


def soap_metric_state(c, model, registry, x, y, steps=3):
    """A SOAP metric after a few shadow SOAP steps on the batch (as the GGN arm builds it)."""
    flat = registry.flatten(dict(model.named_parameters())).detach()
    state = None
    for _ in range(steps):
        g = GGN(functional_logits(model, registry, x), flat, y).gradient
        _, state, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), state, c['soap'])
    return flat, SOAPMetric(registry, contexts)


def test_blocks_and_solve_equal_the_lab_reference(mnist):
    c, x, y, manifest = mnist
    model = make_model(c)
    registry = Registry(model)
    ids = manifest['training'][:24]
    flat, metric = soap_metric_state(c, model, registry, x[ids], y[ids])
    op = GGN(functional_logits(model, registry, x[ids]), flat, y[ids])
    reference, construction = streamed_blocks(model, registry, flat, x[ids], 8, 768)
    gamma, floor = .001, 1e-8
    mu = max(gamma*construction['trace']/registry.size, floor)
    expected = solve_blocks(reference, registry, op.gradient, metric, [mu], 1e-8)[0][mu]
    systems = {}
    for spec, block in zip(registry.specs, reference):
        eye = torch.eye(spec.count, dtype=flat.dtype)
        omega = metric.scale*frozen_apply(eye.reshape(spec.shape+(spec.count,)), metric.contexts[spec.name],
                                              inverse=True).reshape(spec.count, spec.count)
        systems[spec.name] = ((block+mu*omega)+(block+mu*omega).T)/2
    for budget in (2**40, 300_000*8):  # one group, and several groups
        delta, info = ggn.tensor_ggn_step(model, registry, flat, x[ids], op.gradient, metric, gamma, floor, 1e-8, budget,
                                          chunk=7, action=op.action, audit_generator=torch.Generator().manual_seed(0))
        assert abs(info['trace']-construction['trace']) <= 1e-12*construction['trace'] and info['mu'] == mu or \
            abs(info['mu']-mu) <= 1e-12*mu
        # Backward error: the step solves the reference's own damped systems (solutions of these ill-conditioned
        # early systems differ from the reference's at rounding times the condition number).
        for spec in registry.specs:
            g = op.gradient[spec.slice]
            assert (systems[spec.name]@delta[spec.slice]+g).norm() <= 1e-10*max(float(g.norm()), 1e-300) or \
                float(g.norm()) <= 1e-12*float(op.gradient.norm())
        assert (delta-expected).norm() <= 1e-6*expected.norm()
        # The whitened residual is gated at the lab's float64 tolerance (it is 5.9e-10 here); the backward error in
        # the reference's own systems above is 1e-10.
        assert max(info['action_checks'].values()) <= 1e-10 and info['residual'] <= 1e-8
    assert info['groups'] > 1


def test_accumulated_blocks_equal_reference_blocks(mnist):
    c, x, y, manifest = mnist
    model = make_model(c)
    registry = Registry(model)
    ids = manifest['training'][:16]
    flat = registry.flatten(dict(model.named_parameters())).detach()
    reference, construction = streamed_blocks(model, registry, flat, x[ids], 8, 768)
    blocks, trace = ggn.accumulate(model, registry, flat, x[ids], registry.specs, 5, all_parameters=True)
    assert abs(trace-construction['trace']) <= 1e-12*construction['trace']
    scale = sum(float(b.norm()) for b in reference)
    for spec, block in zip(registry.specs, reference):
        # Relative to the scale of all blocks: the attention key biases have zero gradient, so their blocks are
        # rounding noise (about 1e-34) in both computations.
        assert (blocks[spec.name]-block).norm() <= 1e-12*scale


def test_whitened_blocks_equal_whitened_reference(mnist):
    c, x, y, manifest = mnist
    model = make_model(c)
    registry = Registry(model)
    ids = manifest['training'][:16]
    flat, metric = soap_metric_state(c, model, registry, x[ids], y[ids])
    reference, _ = streamed_blocks(model, registry, flat, x[ids], 8, 768)
    blocks, _ = ggn.accumulate(model, registry, flat, x[ids], registry.specs, 5, all_parameters=False, contexts=metric.contexts)
    scale = 0.
    expected = {}
    for spec, block in zip(registry.specs, reference):
        columns = torch.stack([ggn.unwhiten(e.reshape(spec.shape), metric.contexts[spec.name]).reshape(-1)
                               for e in torch.eye(spec.count, dtype=flat.dtype)], 1)  # F0
        expected[spec.name] = columns.T@block@columns
        scale += float(expected[spec.name].norm())
    for spec in registry.specs:
        assert (blocks[spec.name]-expected[spec.name]).norm() <= 1e-10*scale
