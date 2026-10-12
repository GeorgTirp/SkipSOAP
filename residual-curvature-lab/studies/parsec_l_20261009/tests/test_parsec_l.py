"""PARSEC-L implementation tests: SOAP identity, exact-operator equivalence, probes, refresh modes, triggers."""
import copy

import pytest
import torch

from parsec_l import PARSECL, depth_branch_scale, ggn_probes, with_branch_scaling
from parsec_l.optimizer import default_dtype, invariance_residual, offdiagonal_ratio, randomized_top, track_step
from third_party.soap.soap import SOAP
from residual_curvature.curvature.layerwise import block_action, solve_blocks, streamed_blocks
from residual_curvature.curvature.operators import GGN
from residual_curvature.curvature.solves import relative_error
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity
from residual_curvature.optim.soap import SOAPMetric, soap_propose


def flat_params(model, registry):
    return registry.flatten(dict(model.named_parameters())).detach().clone()


def set_grads(model, registry, x, y):
    flat = flat_params(model, registry)
    gradient = registry.unflatten(GGN(functional_logits(model, registry, x), flat, y).gradient)
    for name, p in model.named_parameters():
        p.grad = gradient[name].detach().clone()
    return flat, gradient


def lab_chain(c, x, y, manifest, steps, settings):
    """Lab SOAP (soap_propose) and PARSECL on identical batches; returns both models, registry, optimizer, lab state."""
    model, reference = make_model(c), make_model(c)
    registry = Registry(model)
    optimizer = PARSECL(model.parameters(), **c['soap'], **settings)
    state = None
    for step in range(steps):
        ids = manifest['training'][64*step:64*(step+1)]
        rflat = flat_params(reference, registry)
        g = GGN(functional_logits(reference, registry, x[ids]), rflat, y[ids]).gradient
        delta, state, _ = soap_propose(registry.unflatten(rflat), registry.unflatten(g), state, c['soap'])
        with torch.no_grad():
            for name, p in reference.named_parameters():
                p.add_(delta[name])
        set_grads(model, registry, x[ids], y[ids])
        optimizer.step()
    return model, reference, registry, optimizer, state


def test_without_curvature_equals_upstream_soap_bitwise_and_lab_soap(mnist):
    """Bitwise against persistent upstream SOAP. The lab's soap_propose chain applies theta + (theta_new - theta)
    and reloads state every step, so it agrees only to rounding (about 3e-12 after 12 steps)."""
    c, x, y, manifest = mnist
    c = copy.deepcopy(c)
    c['soap']['lr'] = .01
    for settings in [{'rank': 16}, {'rank': 0}]:
        model, reference, registry, optimizer, _ = lab_chain(c, x, y, manifest, 12, settings)
        upstream_model = make_model(c)
        upstream = SOAP(upstream_model.parameters(), **c['soap'])
        for step in range(12):
            ids = manifest['training'][64*step:64*(step+1)]
            set_grads(upstream_model, registry, x[ids], y[ids])
            with default_dtype(torch.float64):
                upstream.step()
        assert torch.equal(flat_params(model, registry), flat_params(upstream_model, registry))
        assert relative_error(flat_params(model, registry), flat_params(reference, registry)) <= 1e-10
        assert optimizer.last_info['correction_energy'] == 0.


def upstream_state(optimizer):
    """The optimizer's SOAP state without the PARSEC-L entries (loadable by the lab's soap_propose)."""
    state = {k: v for k, v in copy.deepcopy(optimizer.state_dict()).items() if k != 'parsec'}
    for value in state['state'].values():
        value.pop('parsec', None)
    return state


def exact_state(mnist, settings, steps=4):
    """Several SOAP steps, then the next step's lab context (same SOAP state), exact blocks and exact probes."""
    c, x, y, manifest = mnist
    c = copy.deepcopy(c)
    c['soap']['lr'] = .01
    model, _, registry, optimizer, _ = lab_chain(c, x, y, manifest, steps, settings)
    ids = manifest['training'][64*steps:64*(steps+1)]
    probe_x = x[manifest['analysis'][:16]]
    before = flat_params(model, registry)
    g = GGN(functional_logits(model, registry, x[ids]), before, y[ids]).gradient
    native, _, contexts = soap_propose(registry.unflatten(before), registry.unflatten(g), upstream_state(optimizer), c['soap'])
    blocks, _ = streamed_blocks(model, registry, before, probe_x, 8, 768)
    probes, _ = ggn_probes(model, probe_x, exact=True)
    set_grads(model, registry, x[ids], y[ids])
    return c, model, registry, optimizer, before, registry.flatten(native), contexts, blocks, probes


def test_full_rank_equals_damped_ggn_step_in_soap_metric(mnist):
    c, model, registry, optimizer, before, native, contexts, blocks, probes = exact_state(mnist, {'rank': 10**6, 'refresh': 'exact', 'norm_blend': 0.})
    metric = SOAPMetric(registry, contexts)
    context = next(iter(contexts.values()))
    size = context['lr']*context['bias_correction']
    u = metric.apply(-native/size)/metric.scale
    trace = sum(float(b.trace()) for b in blocks)
    mu = max(.01*trace/registry.size, 1e-8)
    values, _ = solve_blocks(blocks, registry, u, metric, [mu], 1e-8)
    expected = size*mu*metric.scale*values[mu]
    optimizer.step(probes=probes, trace=trace)
    actual = flat_params(model, registry)-before
    # Gate on the residual in the lab's damped system (backward error). Some tensor systems have
    # condition numbers near 1e9, so the forward difference to the lab's own solve is about 3e-9.
    solution = actual/(size*mu*metric.scale)
    residual = relative_error(block_action(blocks, registry, solution)+mu*metric.apply(solution), -u)
    assert residual <= 1e-10
    assert relative_error(actual, expected) <= 1e-8
    assert abs(optimizer.last_info['probe']['probe_trace']/trace-1) <= 1e-12
    assert relative_error(native, expected) > .1  # the correction is not negligible here


def test_truncated_rank_equals_exact_top_r_shrinkage(mnist):
    c, model, registry, optimizer, before, native, contexts, blocks, probes = exact_state(mnist, {'rank': 4, 'refresh': 'exact', 'norm_blend': 0.})
    metric = SOAPMetric(registry, contexts)
    context = next(iter(contexts.values()))
    size = context['lr']*context['bias_correction']
    u = metric.apply(-native/size)/metric.scale
    trace = sum(float(b.trace()) for b in blocks)
    beta = 1/(max(.01*trace/registry.size, 1e-8)*metric.scale)
    expected = torch.zeros_like(native)
    for spec, block in zip(registry.specs, blocks):
        ctx = contexts[spec.name]
        f = capacity.factor(torch.eye(spec.count).reshape(spec.shape+(spec.count,)), ctx).reshape(spec.count, spec.count)
        eig, q = torch.linalg.eigh(beta*f.T@block@f)
        r = min(4, spec.count, len(probes[next(p for n, p in model.named_parameters() if n == spec.name)]))
        w, kappa = q[:, -r:], eig[-r:].clamp_min(0)
        z = capacity.factor(u[spec.slice].reshape(spec.shape), ctx, transpose=True).reshape(-1)
        z = z-w@((kappa/(1+kappa))*(w.T@z))
        expected[spec.slice] = -size*capacity.factor(z.reshape(spec.shape), ctx).reshape(-1)
    optimizer.step(probes=probes, trace=trace)
    assert relative_error(flat_params(model, registry)-before, expected) <= 1e-8


def test_exact_probes_reproduce_ggn_blocks_and_sampled_rows_are_per_example_gradients(mnist):
    c, x, y, manifest = mnist
    model = make_model(c)
    registry = Registry(model)
    xb = x[manifest['analysis'][:8]]
    blocks, _ = streamed_blocks(model, registry, flat_params(model, registry), xb, 8, 768)
    exact, _ = ggn_probes(model, xb, exact=True)
    params = dict(model.named_parameters())
    for spec, block in zip(registry.specs, blocks):
        u = exact[params[spec.name]].reshape(-1, spec.count)
        assert relative_error(u.T@u, block) <= 1e-12
    sampled, labels = ggn_probes(model, xb, samples=3, generator=torch.Generator().manual_seed(1))
    for row in [0, 4, 23]:
        loss = torch.nn.functional.cross_entropy(model(xb[row//3][None]), labels[row])
        grads = torch.autograd.grad(loss, list(params.values()))
        for p, g in zip(params.values(), grads):
            assert relative_error(sampled[p][row]*(8*3)**.5, g) <= 1e-12


def test_track_mode_and_residual_trigger(mnist):
    c, x, y, manifest = mnist
    c = copy.deepcopy(c)
    c['soap']['lr'] = .01
    for threshold, second_mode in [(None, 'track'), (0., 'extract')]:
        model = make_model(c)
        registry = Registry(model)
        optimizer = PARSECL(model.parameters(), **c['soap'], rank=4, refresh='track', probe_every=2,
                            residual_threshold=threshold)
        modes = []
        for step in range(5):
            ids = manifest['training'][64*step:64*(step+1)]
            set_grads(model, registry, x[ids], y[ids])
            probes = ggn_probes(model, x[ids][:32], samples=2, generator=torch.Generator().manual_seed(step))[0] if optimizer.wants_probes() else None
            optimizer.step(probes=probes)
            if optimizer.last_info['probe']:
                modes.append({row['mode'] for row in optimizer.last_info['probe']['tensors'].values()})
        assert modes[0] == {'extract'} and second_mode in modes[1]


def test_tracking_an_exact_invariant_subspace_keeps_it():
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(40, 25, generator=gen)
    w, kappa = randomized_top(z, 5, gen, 30, 4)
    assert invariance_residual(z, w) <= 1e-10
    tracked, tracked_kappa = track_step(z, w@torch.linalg.qr(torch.randn(5, 5, generator=gen))[0])
    assert invariance_residual(z, tracked) <= 1e-10
    assert torch.allclose(tracked_kappa, kappa, rtol=1e-10)


def test_rho_is_zero_for_probes_diagonal_in_the_basis():
    y = torch.zeros(6, 3, 4)
    for j in range(6):
        y[j, j % 3, j % 4] = 1.+j
    assert offdiagonal_ratio(y, 0) == 0. and offdiagonal_ratio(y, 1) == 0.
    assert offdiagonal_ratio(torch.randn(6, 3, 4, generator=torch.Generator().manual_seed(0)), 0) > 0


def test_side_refresh_matches_upstream_qr_for_that_mode(mnist):
    c, x, y, manifest = mnist
    model, _, registry, optimizer, _ = lab_chain(c, x, y, manifest, 3, {'rank': 0})
    p = dict(model.named_parameters())['blocks.0.w1.weight']
    state = optimizer.state[p]
    reference = copy.deepcopy(state)
    upstream = optimizer.get_orthogonal_matrix_QR(reference, 10000, False)
    old_bases = state['Q']
    momentum = optimizer.project_back(state['exp_avg'], state)
    optimizer._refresh_side(p, 0)
    assert torch.equal(state['Q'][0], upstream[0]) and state['Q'][1] is old_bases[1]
    assert relative_error(optimizer.project_back(state['exp_avg'], state), momentum) <= 1e-5


def test_triggered_side_refresh_and_stop(mnist):
    c, x, y, manifest = mnist
    c = copy.deepcopy(c)
    c['soap']['lr'] = .01
    model = make_model(c)
    registry = Registry(model)
    optimizer = PARSECL(model.parameters(), **c['soap'], rank=4, probe_every=1, soap_refresh='triggered',
                        rho_threshold=-1., stop_energy=1., stop_patience=2)
    p = dict(model.named_parameters())['blocks.0.w1.weight']
    seen = []
    for step in range(5):
        ids = manifest['training'][64*step:64*(step+1)]
        set_grads(model, registry, x[ids], y[ids])
        bases = optimizer.state[p].get('Q')
        probes = ggn_probes(model, x[ids][:16], samples=2, generator=torch.Generator().manual_seed(step))[0] if optimizer.wants_probes() else None
        optimizer.step(probes=probes)
        seen.append((optimizer.last_info, bases, optimizer.state[p]['Q']))
    first_probe = seen[1][0]['probe']
    assert first_probe['side_refreshes'] and seen[1][2][0] is not seen[1][1][0]
    assert optimizer._stopped and not optimizer.wants_probes() and seen[-1][0]['correction_energy'] == 0.


def test_state_dict_round_trip_continues_bitwise(mnist):
    c, x, y, manifest = mnist
    c = copy.deepcopy(c)
    c['soap']['lr'] = .01
    model = make_model(c)
    registry = Registry(model)
    optimizer = PARSECL(model.parameters(), **c['soap'], rank=4, probe_every=2)
    for step in range(4):
        ids = manifest['training'][64*step:64*(step+1)]
        set_grads(model, registry, x[ids], y[ids])
        optimizer.step(probes=ggn_probes(model, x[ids][:16], samples=2, generator=torch.Generator().manual_seed(step))[0]
                       if optimizer.wants_probes() else None)
    twin = copy.deepcopy(model)
    clone = PARSECL(twin.parameters(), **c['soap'], rank=4, probe_every=2)
    clone.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    ids = manifest['training'][256:320]
    for m, o in [(model, optimizer), (twin, clone)]:
        set_grads(m, registry, x[ids], y[ids])
        o.step()
    assert torch.equal(flat_params(model, registry), flat_params(twin, registry))
    assert optimizer.last_info['correction_energy'] > 0


def test_branch_scaling_zero_exponent_is_the_plain_model(mnist):
    c, x, y, manifest = mnist
    assert with_branch_scaling(c, exponent=0.) == c and depth_branch_scale(4, .5) == .5 and depth_branch_scale(8, 1., 2.) == .25
    scaled = with_branch_scaling(c)  # standard: 1/depth
    assert scaled['model']['branch_scale'] == .5 and c['model']['branch_scale'] == 1. and depth_branch_scale(4) == .25
    xb = x[manifest['analysis'][:4]]
    assert torch.equal(make_model(with_branch_scaling(c, exponent=0.))(xb), make_model(c)(xb))
    with pytest.raises(ValueError):
        depth_branch_scale(0)


def run_steps(c, x, y, manifest, settings, steps, probe_examples=16, samples=2, model=None):
    model = make_model(c) if model is None else model
    registry = Registry(model)
    optimizer = PARSECL(model.parameters(), **c['soap'], **settings)
    for step in range(steps):
        ids = manifest['training'][64*step:64*(step+1)]
        set_grads(model, registry, x[ids], y[ids])
        probes = (ggn_probes(model, x[ids][:probe_examples], samples=samples, generator=torch.Generator().manual_seed(step))[0]
                  if optimizer.wants_probes() else None)
        optimizer.step(probes=probes)
    return model, registry, optimizer


def soap_twin_step(c, model, registry, optimizer, ids, x, y):
    """Upstream SOAP from the same state on a copy of the model: returns its parameter step."""
    twin = copy.deepcopy(model)
    soap = SOAP(twin.parameters(), **c['soap'])
    soap.load_state_dict(upstream_state(optimizer))
    before = flat_params(twin, registry)
    set_grads(twin, registry, x[ids], y[ids])
    with default_dtype(torch.float64):
        soap.step()
    return flat_params(twin, registry)-before


def lr01(mnist):
    c, x, y, manifest = mnist
    c = copy.deepcopy(c)
    c['soap']['lr'] = .01
    return c, x, y, manifest


def test_handover_is_plain_soap_afterwards(mnist):
    c, x, y, manifest = lr01(mnist)
    model, registry, optimizer = run_steps(c, x, y, manifest, {'rank': 4, 'probe_every': 1, 'probe_until': 4}, 6)
    assert optimizer._stopped and not optimizer.wants_probes() and optimizer.last_info['correction_energy'] == 0.
    ids = manifest['training'][384:448]
    expected = soap_twin_step(c, model, registry, optimizer, ids, x, y)
    before = flat_params(model, registry)
    set_grads(model, registry, x[ids], y[ids])
    optimizer.step()
    assert torch.equal(flat_params(model, registry)-before, expected)


def test_norm_blend_restores_soap_norm_globally(mnist):
    c, x, y, manifest = lr01(mnist)
    assert PARSECL(make_model(c).parameters(), **c['soap']).parsec['norm_blend'] == 1.  # standard setting
    model, registry, optimizer = run_steps(c, x, y, manifest, {'rank': 8, 'probe_every': 2, 'norm_blend': 0.}, 4)
    ids = manifest['training'][256:320]
    native = soap_twin_step(c, model, registry, optimizer, ids, x, y)
    steps = {}
    for blend in [0., .5, 1.]:
        twin = copy.deepcopy(model)
        clone = PARSECL(twin.parameters(), **c['soap'], rank=8, probe_every=2, norm_blend=blend)
        clone.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        before = flat_params(twin, registry)
        set_grads(twin, registry, x[ids], y[ids])
        clone.step()
        steps[blend] = flat_params(twin, registry)-before
    n, m = float(native.norm()), float(steps[0.].norm())
    # The shrinkage acts in whitened coordinates, so the physical norm can end up on either side of SOAP's.
    assert abs(m/n-1) > 1e-3
    assert abs(float(steps[1.].norm())/n-1) <= 1e-10
    assert abs(float(steps[.5].norm())/(m*n)**.5-1) <= 1e-10
    for blend in [.5, 1.]:
        cosine = float(steps[blend]@steps[0.])/float(steps[blend].norm()*steps[0.].norm())
        assert abs(cosine-1) <= 1e-12


def test_batched_extraction_equals_per_tensor_and_signals_do_not_change_the_step(mnist):
    c, x, y, manifest = lr01(mnist)
    # rank 8 + oversampling 8 = 16 = probe rows (8 examples x 2 draws): the range finder is exact.
    settings = {'rank': 8, 'probe_every': 2}
    batched = run_steps(c, x, y, manifest, dict(settings, batched=True), 5, probe_examples=8)
    single = run_steps(c, x, y, manifest, dict(settings, batched=False), 5, probe_examples=8)
    quiet = run_steps(c, x, y, manifest, dict(settings, batched=True, signals=False), 5, probe_examples=8)
    assert batched[2].last_info['correction_energy'] > 0
    assert relative_error(flat_params(batched[0], batched[1]), flat_params(single[0], single[1])) <= 1e-10
    assert torch.equal(flat_params(batched[0], batched[1]), flat_params(quiet[0], quiet[1]))


def test_block_jacobian_deviation(mnist):
    from parsec_l import block_jacobian_deviation
    c, x, y, manifest = mnist
    xb = x[manifest['analysis'][:1]]
    zero = make_model(dict(c, model=dict(c['model'], branch_scale=0.)))
    assert all(r['rms'] == 0 and r['spectral'] == 0 for r in block_jacobian_deviation(zero, xb, probes=2, power_steps=3))
    model = make_model(c)
    inputs = []
    handle = model.blocks[1].register_forward_hook(lambda m, a, o: inputs.append(a[0].detach()))
    with torch.no_grad():
        model(xb)
    handle.remove()
    h = inputs[0]
    jac = torch.func.jacrev(lambda t: (model.blocks[1](t.reshape(h.shape))-t.reshape(h.shape)).reshape(-1))(h.reshape(-1))
    rows = block_jacobian_deviation(model, xb, probes=400, power_steps=300, generator=torch.Generator().manual_seed(0))
    exact_rms = float(jac.norm())/jac.shape[0]**.5
    assert abs(rows[1]['rms']/exact_rms-1) <= .1
    assert abs(rows[1]['spectral']/float(torch.linalg.matrix_norm(jac, ord=2))-1) <= 1e-6


def dense_factors(registry, contexts):
    return [capacity.factor(torch.eye(s.count).reshape(s.shape+(s.count,)), contexts[s.name]).reshape(s.count, s.count)
            for s in registry.specs]


@pytest.mark.parametrize('strength', [1., 4.])
def test_lr_calibration_is_the_proximal_step_of_soaps_metric(mnist, strength):
    settings = {'rank': 10**6, 'refresh': 'exact', 'norm_blend': 0., 'calibration': 'lr', 'strength': strength}
    c, model, registry, optimizer, before, native, contexts, blocks, probes = exact_state(mnist, settings)
    context = next(iter(contexts.values()))
    size = context['lr']*context['bias_correction']
    metric = SOAPMetric(registry, contexts)
    u = metric.apply(-native/size)/metric.scale
    beta = strength*.01*(1-.95**context['step'])**.5
    expected = torch.zeros_like(native)
    for spec, block, f in zip(registry.specs, blocks, dense_factors(registry, contexts)):
        k = beta*f.T@block@f
        z = f.T@u[spec.slice]
        expected[spec.slice] = -size*f@torch.linalg.solve(torch.eye(spec.count)+k, z)
    optimizer.step(probes=probes, trace=sum(float(b.trace()) for b in blocks))
    assert relative_error(flat_params(model, registry)-before, expected) <= 1e-9
    info = optimizer.last_info['probe']
    assert abs(info['eta_lambda_max']/(max(t['kappa_max'] for t in info['tensors'].values())/strength)-1) <= 1e-12


def test_gradient_rhs_without_curvature_is_the_soap_map_of_the_gradient(mnist):
    c, model, registry, optimizer, before, native, contexts, blocks, probes = exact_state(mnist, {'rank': 0, 'rhs': 'gradient', 'norm_blend': 0.})
    c, x, y, manifest = mnist
    ids = manifest['training'][256:320]
    g = GGN(functional_logits(model, registry, x[ids]), before, y[ids]).gradient
    context = next(iter(contexts.values()))
    size = context['lr']*context['bias_correction']
    expected = torch.cat([-size*f@(f.T@g[s.slice]) for s, f in zip(registry.specs, dense_factors(registry, contexts))])
    optimizer.step()
    assert relative_error(flat_params(model, registry)-before, expected) <= 1e-10


def test_selection_scales_the_step_and_minimizes_the_selection_loss(mnist):
    c, x, y, manifest = lr01(mnist)
    model, registry, optimizer = run_steps(c, x, y, manifest, {'rank': 8, 'probe_every': 2, 'norm_blend': 0.}, 4)
    ids, chosen = manifest['training'][256:320], manifest['analysis'][:128]
    selection_loss = lambda: float(torch.nn.functional.cross_entropy(model(x[chosen]), y[chosen]))
    twin = copy.deepcopy(model)
    plain = PARSECL(twin.parameters(), **c['soap'], rank=8, probe_every=2, norm_blend=0.)
    plain.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    before = flat_params(model, registry)
    set_grads(twin, registry, x[ids], y[ids])
    plain.step()
    unselected = flat_params(twin, registry)-before
    set_grads(model, registry, x[ids], y[ids])
    optimizer.step(selection=selection_loss)
    info = optimizer.last_info['select']
    assert info['winner'] == 'parsec' and info['alpha'] > 0
    assert relative_error(flat_params(model, registry)-before, info['alpha']*unselected) <= 1e-12
    chosen_value = selection_loss()
    for alpha in [0., .5, 1., 2.]:
        commit(model, registry, before+alpha*unselected)
        assert chosen_value <= selection_loss()+1e-15


def test_selection_hands_over_to_soap_when_soap_wins(mnist):
    c, x, y, manifest = lr01(mnist)
    settings = {'rank': 8, 'probe_every': 2, 'norm_blend': 0., 'select_compare_soap': True, 'select_handover': 1}
    model, registry, optimizer = run_steps(c, x, y, manifest, settings, 4)
    ids = manifest['training'][256:320]
    twin = copy.deepcopy(model)
    soap = SOAP(twin.parameters(), **c['soap'])
    soap.load_state_dict(upstream_state(optimizer))
    set_grads(twin, registry, x[ids], y[ids])
    with default_dtype(torch.float64):
        soap.step()
    target = flat_params(twin, registry)
    set_grads(model, registry, x[ids], y[ids])
    optimizer.step(selection=lambda: float((flat_params(model, registry)-target).square().sum()))
    info = optimizer.last_info['select']
    assert info['winner'] == 'soap' and info['handover'] and info['soap_value'] == 0.
    assert torch.equal(flat_params(model, registry), target)
    assert optimizer._stopped and not optimizer.wants_selection() and not optimizer.wants_probes()


def test_optimism_ratio_matches_dense_computation(mnist):
    from parsec_l import per_example_gradients
    c, model, registry, optimizer, before, native, contexts, blocks, probes = exact_state(mnist, {'rank': 4, 'refresh': 'exact', 'norm_blend': 0.})
    c, x, y, manifest = mnist
    ids = manifest['training'][256:320]
    gradients = per_example_gradients(model, x[ids], y[ids])
    params = dict(model.named_parameters())
    rows = torch.cat([gradients[params[s.name]].reshape(len(ids), -1) for s in registry.specs], 1)
    g = GGN(functional_logits(model, registry, x[ids]), before, y[ids]).gradient
    assert relative_error(rows.mean(0), g) <= 1e-12
    trace = sum(float(b.trace()) for b in blocks)
    metric = SOAPMetric(registry, contexts)
    beta = 1/(max(.01*trace/registry.size, 1e-8)*metric.scale)
    expected = {}
    for name in ['soap', 'parsec_l']:
        optimism = signal = 0.
        for spec, block, f in zip(registry.specs, blocks, dense_factors(registry, contexts)):
            a = f@f.T
            if name == 'parsec_l':
                eig, q = torch.linalg.eigh(beta*f.T@block@f)
                r = min(4, spec.count, len(probes[params[spec.name]]))
                w, kappa = q[:, -r:], eig[-r:].clamp_min(0)
                a = f@(torch.eye(spec.count)-w@torch.diag(kappa/(1+kappa))@w.T)@f.T
            block_rows = rows[:, spec.slice]
            centered = block_rows-block_rows.mean(0)
            optimism += float((centered@a*centered).sum())/(len(ids)-1)/len(ids)
            signal += float(block_rows[:32].mean(0)@a@block_rows[32:].mean(0))
        expected[name] = (optimism, signal)
    optimizer.step(probes=probes, trace=trace, true_gradients=gradients)
    info = optimizer.last_info['optimism']
    for name, tolerance in [('soap', 1e-10), ('parsec_l', 1e-8)]:
        assert abs(info[f'optimism_{name}']/expected[name][0]-1) <= tolerance
        assert abs(info[f'signal_{name}']/expected[name][1]-1) <= tolerance
        # r = optimism/signal, and infinite (noise-dominated) when the cross-fitted signal is not positive.
        if expected[name][1] > 0:
            assert abs(info[f'r_{name}']/(expected[name][0]/expected[name][1])-1) <= 10*tolerance
        else:
            assert info[f'r_{name}'] == float('inf')


def test_momentum_free_top_space_matches_dense_computation(mnist):
    settings = {'rank': 4, 'refresh': 'exact', 'norm_blend': 0., 'calibration': 'lr', 'strength': 4., 'top_rhs': 'gradient'}
    c, model, registry, optimizer, before, native, contexts, blocks, probes = exact_state(mnist, settings)
    c0, x, y, manifest = mnist
    ids = manifest['training'][256:320]
    g = GGN(functional_logits(model, registry, x[ids]), before, y[ids]).gradient
    context = next(iter(contexts.values()))
    size, t = context['lr']*context['bias_correction'], context['step']
    metric = SOAPMetric(registry, contexts)
    u = metric.apply(-native/size)/metric.scale
    beta, bc1 = 4.*.01*(1-.95**t)**.5, 1-.95**t
    params = dict(model.named_parameters())
    expected = torch.zeros_like(native)
    for spec, block, f in zip(registry.specs, blocks, dense_factors(registry, contexts)):
        eig, q = torch.linalg.eigh(beta*f.T@block@f)
        r = min(4, spec.count, len(probes[params[spec.name]]))
        w, kappa = q[:, -r:], eig[-r:].clamp_min(0)
        zm, zg = f.T@u[spec.slice], bc1*(f.T@g[spec.slice])
        mixed = zm+w@(w.T@(zg-zm))
        new = mixed-w@((kappa/(1+kappa))*(w.T@mixed))
        expected[spec.slice] = -size*f@new
    optimizer.step(probes=probes, trace=sum(float(b.trace()) for b in blocks))
    assert relative_error(flat_params(model, registry)-before, expected) <= 1e-8
