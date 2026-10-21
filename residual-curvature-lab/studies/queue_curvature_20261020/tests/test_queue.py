"""Probe rows against micro.lean_probes and explicit Jacobians, the Rademacher square-root factor, CG with Gram
products, capture terms, per-example statistics, groups and age bookkeeping (CPU; PROTOCOL.md, section 6, G0)."""
import torch

import common  # first: it puts the earlier studies' modules on the path
import ggn  # noqa: E402
import micro  # noqa: E402
import probes  # noqa: E402
import queue_analysis  # noqa: E402
import spectral  # noqa: E402
import states  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402

D = torch.float64
NAMES = ['head.weight', 'blocks.0.w1.weight']


def tiny():
    data = states.GGN_TRAIN.Data('cpu')
    replay = states.Replay('topfree', 1, 'cpu', data)
    model, registry = replay.arm.model, replay.arm.registry
    model.double()
    flat = replay.flat().double()
    x = torch.randn(3, 3, 32, 32, generator=torch.Generator().manual_seed(4), dtype=D)
    return model, registry, flat, x


def test_label_rows_match_lean_probes():
    model, registry, flat, x = tiny()
    rows = probes.probe_rows(model, registry, flat, x, NAMES, 'label', torch.Generator().manual_seed(5), 3**-.5)
    lean = micro.lean_probes(model, x, samples=1, generator=torch.Generator().manual_seed(5))
    params = dict(model.named_parameters())
    for name in NAMES:
        assert torch.allclose(rows[name], lean[params[name]], rtol=1e-10, atol=1e-14)


def test_rademacher_factor_and_rows_against_explicit_jacobians():
    gen = torch.Generator().manual_seed(6)
    p = torch.rand(10, generator=gen, dtype=D)+.05
    p = p/p.sum()
    s = probes.square_root_factor(p)
    assert torch.allclose(s@s.T, torch.diag(p)-torch.outer(p, p), atol=1e-15)
    model, registry, flat, x = tiny()
    rows = probes.probe_rows(model, registry, flat, x, NAMES, 'rademacher', torch.Generator().manual_seed(7), 1.)
    logits, jacobians = next(ggn.example_jacobians(model, registry, flat, x, NAMES, 3))
    probability = logits.softmax(-1)
    xi = (torch.randint(0, 2, tuple(probability.shape), generator=torch.Generator().manual_seed(7))*2-1).to(D)
    for i in range(3):
        v = probes.square_root_factor(probability[i])@xi[i]
        for name in NAMES:
            expected = torch.einsum('k,k...->...', v, jacobians[name][i])
            assert torch.allclose(rows[name][i], expected, rtol=1e-10, atol=1e-14)


def test_example_statistics_use_the_drawn_label_and_average_to_the_ggn_trace():
    model, registry, flat, x = tiny()
    y = torch.tensor([1, 4, 7])
    losses, traces = probes.example_statistics(model, registry, flat, x, y, torch.Generator().manual_seed(8))
    logits, jacobians = next(ggn.example_jacobians(model, registry, flat, x, [s.name for s in registry.specs], 3))
    assert torch.allclose(losses, torch.nn.functional.cross_entropy(logits, y, reduction='none'), rtol=1e-12)
    p = logits.softmax(-1)
    drawn = torch.multinomial(p.cpu(), 1, replacement=True, generator=torch.Generator().manual_seed(8))[:, 0]
    for i in range(3):
        j = torch.cat([jacobians[s.name][i].reshape(10, -1) for s in registry.specs], 1)
        v = p[i]-torch.nn.functional.one_hot(drawn[i], 10).to(D)
        assert abs(float(traces[i])-float((j.T@v).square().sum())) <= 1e-10*float(traces[i])
        expectation = sum(float(p[i, c])*float((j.T@(p[i]-torch.nn.functional.one_hot(torch.tensor(c), 10).to(D))).square().sum())
                          for c in range(10))
        exact = float(torch.trace((torch.diag(p[i])-torch.outer(p[i], p[i]))@j@j.T))
        assert abs(expectation-exact) <= 1e-10*exact


def test_cg_with_gram_products_matches_the_dense_solve():
    gen = torch.Generator().manual_seed(9)
    z, b = torch.randn(20, 7, generator=gen, dtype=D), torch.randn(20, generator=gen, dtype=D)
    y = spectral.conjugate_gradient(lambda v: z@(z.T@v)+.3*v, b, 20)
    assert torch.allclose(y, torch.linalg.solve(z@z.T+.3*torch.eye(20, dtype=D), b), rtol=1e-8, atol=1e-10)


def test_capture_terms_are_projected_traces_and_maximal_on_the_top_eigenspace():
    gen = torch.Generator().manual_seed(10)
    rows = torch.randn(30, 12, generator=gen, dtype=D)*torch.linspace(3, .1, 12, dtype=D)
    gram = rows.T@rows
    u = torch.linalg.qr(torch.randn(12, 6, generator=gen, dtype=D))[0]
    terms = queue_analysis.capture_terms(rows, u, [3, 6])
    for k in [3, 6]:
        assert abs(terms[k]-float(torch.trace(u[:, :k].T@gram@u[:, :k]))) <= 1e-10
    top = torch.linalg.eigh(gram)[1].flip(1)[:, :6]
    assert queue_analysis.capture_terms(rows, top, [3])[3] >= terms[3]-1e-12


def test_groups_respect_the_column_budget():
    class Spec:
        def __init__(self, name, count):
            self.name, self.count = name, count
    specs = [Spec(str(i), c) for i, c in enumerate([16384, 16384, 4096, 4096, 640, 16384])]
    out = queue_analysis.groups(specs, 16384, 2*2**30)
    assert [s.name for g in out for s in g] == [s.name for s in specs]
    assert all(sum(s.count for s in g)*16384*4 <= 2*2**30 or len(g) == 1 for g in out)


def test_age_bookkeeping_and_band_statistics():
    total, batch = 100, 32
    order = DataOrder(list(range(total)), 3)
    last = torch.full((total,), -1, dtype=torch.long)
    for t in range(1, 21):
        last[torch.tensor(order.take(batch))] = t
    ages = torch.where(last >= 0, 20-last, last)
    assert int((ages < 0).sum()) == 0 and int(ages.max()) <= 2*((total+batch-1)//batch)
    assert int((ages == 0).sum()) == len(set(range(total)) & set(torch.nonzero(last == 20)[:, 0].tolist()))
    values = torch.arange(total, dtype=D)
    profile = queue_analysis.age_profile(ages, values)
    assert sum(n for n, _ in profile.values()) == total
    synthetic = torch.tensor([0, 1, 1, 20, 25, -1])
    assert abs(queue_analysis.band_difference(synthetic, torch.tensor([1., 2., 3., 10., 20., 99.], dtype=D))-13.) <= 1e-12


def test_squared_norm_matches_the_direct_sum():
    z = torch.randn(37, 2500, generator=torch.Generator().manual_seed(11))
    assert abs(queue_analysis.squared_norm(z)-float(z.double().square().sum())) <= 1e-10*float(z.double().square().sum())
