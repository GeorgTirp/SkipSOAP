"""Lanczos quadrature and matrix functions against exact eigendecompositions, breakdown handling, the noise-scale
formulas against sampling without replacement, B_crit and the toy optimum, and the non-consuming batch peek
(CPU; PROTOCOL.md, section 5, G0)."""
import itertools
import math

import torch

import common  # first: it puts the earlier studies' modules on the path
import lanczos  # noqa: E402
import noise  # noqa: E402
import states  # noqa: E402
import summarize  # noqa: E402

D = torch.float64
POWERS = [-1., -.75, -.5, -.25, 0., .5, 1.]


def power_law(dim, gen, alpha=1.2, floor=1e-4):
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen, dtype=D))
    values = torch.arange(1, dim+1, dtype=D)**-alpha+floor
    return (q*values)@q.T, q, values


def exact(values, basis, vectors, q):
    return ((values**q)[:, None]*(basis.T@vectors).square()).sum(0)


def test_full_length_lanczos_reproduces_exact_forms_and_functions():
    gen = torch.Generator().manual_seed(1)
    a, basis, values = power_law(120, gen)
    v = torch.randn(120, 4, generator=gen, dtype=D)
    run = lanczos.lanczos(lambda x: a@x, v, 120)
    forms = run.forms(POWERS)
    for i, q in enumerate(POWERS):
        assert torch.allclose(forms[i], exact(values, basis, v, q), rtol=1e-8)
    for nu in [.25, .5, 1.]:
        target = (basis*values**-nu)@(basis.T@v[:, 2])
        assert torch.allclose(run.apply(2, -nu), target, rtol=1e-7, atol=1e-10)


def test_three_hundred_steps_converge_on_a_conditioned_power_law():
    gen = torch.Generator().manual_seed(2)
    a, basis, values = power_law(1500, gen, alpha=1.5, floor=1e-4)  # condition number about 1e4
    v = torch.randn(1500, 3, generator=gen, dtype=D)
    forms = lanczos.lanczos(lambda x: a@x, v, 300).forms(POWERS)
    for i, q in enumerate(POWERS):
        reference = exact(values, basis, v, q)
        assert float(((forms[i]-reference).abs()/reference).max()) <= 1e-6


def test_breakdown_keeps_the_leading_block_and_zero_columns_stay_zero():
    gen = torch.Generator().manual_seed(3)
    a, basis, values = power_law(50, gen)
    v = torch.zeros(50, 2, dtype=D)
    v[:, 0] = basis[:, :3]@torch.tensor([1., -2., .5], dtype=D)
    run = lanczos.lanczos(lambda x: a@x, v, 20)
    forms = run.forms(POWERS)
    assert torch.isfinite(forms).all() and torch.all(forms[:, 1] == 0)
    for i, q in enumerate(POWERS):
        assert abs(float(forms[i, 0]-exact(values, basis, v[:, :1], q)[0])) <= 1e-9*float(exact(values, basis, v[:, :1], q)[0])
    assert torch.allclose(run.apply(0, -1.), (basis*values**-1.)@(basis.T@v[:, 0]), rtol=1e-8)


def test_without_replacement_variance_factor_is_exact():
    gen = torch.Generator().manual_seed(4)
    population = torch.randn(8, 3, generator=gen, dtype=D)
    mean = population.mean(0)
    s2 = (population-mean).T@(population-mean)/(8-1)
    for size in [2, 3, 5]:
        draws = [population[list(c)].mean(0)-mean for c in itertools.combinations(range(8), size)]
        exact_cov = sum(torch.outer(d, d) for d in draws)/len(draws)
        assert torch.allclose(exact_cov, (1/size-1/8)*s2, atol=1e-12)


def test_expected_progress_formula_matches_sampling_without_replacement():
    gen = torch.Generator().manual_seed(5)
    dim, corpus, batch = 4, 60, 12
    c, basis, values = power_law(dim, gen, alpha=1., floor=.05)
    examples = torch.randn(corpus, dim, generator=gen, dtype=D)+torch.tensor([1., -.5, .3, .2], dtype=D)
    g = examples.mean(0)
    sigma = (examples-g).T@(examples-g)/(corpus-1)
    for nu in [0., .5, 1.]:
        p = (basis*values**-nu)@basis.T
        pcp = p@c@p
        phi = float(g@p@g)**2/(2*float(g@pcp@g))
        k = float(torch.trace(pcp@sigma))/float(g@pcp@g)
        psi = phi/(1+(1/batch-1/corpus)*k)
        second = 0.
        draws = 20000
        for _ in range(draws):
            idx = torch.randperm(corpus, generator=gen)[:batch]
            gb = examples[idx].mean(0)
            second += float(gb@pcp@gb)/draws
        alpha = float(g@p@g)/second
        assert abs(.5*alpha*float(g@p@g)-psi) <= 2e-2*psi


def test_bcrit_is_the_crossing_point_and_the_toy_optimum_matches_a_search():
    phi1, phi2, k1, k2, n = 1., 1.6, 50., 400., 45000
    g = phi2/phi1
    crit = 1/(1/n+(g-1)/(k2-g*k1))
    psi = lambda phi, k, batch: phi/(1+(1/batch-1/n)*k)
    assert abs(psi(phi1, k1, crit)-psi(phi2, k2, crit)) <= 1e-12
    assert psi(phi2, k2, 2*crit) > psi(phi1, k1, 2*crit) and psi(phi2, k2, crit/2) < psi(phi1, k1, crit/2)
    for ell, s in [(1e3, .5), (1e4, 3.), (50., 10.)]:
        nus = [i/10000 for i in range(10001)]
        value = lambda nu: (1+ell**-nu)**2/(1+s+ell**(1-2*nu))
        assert abs(max(nus, key=value)-max(0., 1-math.log(1+s)/math.log(ell))) <= 2e-4


def test_summary_theory_matches_the_definitions_on_a_synthetic_state():
    corpus = [2.**q for q in summarize.POWERS]
    noise_forms = [3.*1.5**q for q in summarize.POWERS]
    state = {'tensors': {'w': {'corpus': corpus, 'noise_mean': noise_forms}}}
    th = summarize.theory(state)
    nu = .5
    z_neg, z_metric, noise_metric = 2.**-nu, 2.**(1-2*nu), 3.*1.5**(1-2*nu)
    factor = (1/2048-1/45000)/(1/1024-1/45000)
    assert abs(th[nu]['phi']-z_neg**2/(2*z_metric)) <= 1e-12
    assert abs(th[nu]['N']-factor*noise_metric/z_metric) <= 1e-12
    assert abs(th[nu]['K']-noise_metric/(z_metric*(1/1024-1/45000))) <= 1e-9


def test_peek_returns_the_next_batch_without_consuming():
    data = states.GGN_TRAIN.Data('cpu')
    replay = states.Replay('topfree', 1, 'cpu', data)
    before = (replay.order.state_dict()['position'], replay.augment.get_state().clone())
    x_peek, y_peek = noise.peek_step_batch(replay)
    following = noise.peek(replay, 1)
    assert replay.order.state_dict()['position'] == before[0] and torch.equal(replay.augment.get_state(), before[1])
    x, y = replay.next_batch()
    assert torch.equal(x, x_peek) and torch.equal(y, y_peek)
    twin = states.DataOrder([0], 0)
    twin.load_state_dict(replay.order.state_dict())
    assert twin.take(states.BATCH) == following
