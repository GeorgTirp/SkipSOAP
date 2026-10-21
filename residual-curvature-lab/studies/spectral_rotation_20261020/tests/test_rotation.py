"""Capture and overlap identities and bounds, scale invariance of the shape distance, spec grouping, the raw dense
GGN block against matrix-free products, and the spectrum record (CPU; PROTOCOL.md, section 5, G0)."""
import math

import torch

import common  # first: it puts the earlier studies' modules on the path
import ggn  # noqa: E402
import micro  # noqa: E402
import rotation  # noqa: E402
import states  # noqa: E402
import summarize  # noqa: E402

D = torch.float64
KS = [1, 4, 16]


def psd(n, gen, decay=1.):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=D))
    values = torch.arange(1, n+1, dtype=D)**-decay
    return (q*values)@q.T, q, values


def test_capture_is_one_on_the_own_eigenspace_and_at_most_one_otherwise():
    gen = torch.Generator().manual_seed(1)
    c, q, values = psd(50, gen)
    own = rotation.capture(q[:, :20], c, values, KS)
    assert all(abs(own[k]-1) <= 1e-12 for k in KS)
    for _ in range(10):
        u, _ = torch.linalg.qr(torch.randn(50, 20, generator=gen, dtype=D))
        other = rotation.capture(u, c, values, KS)
        assert all(0 <= other[k] <= 1+1e-12 for k in KS)
    small = q[:, :20]@torch.linalg.matrix_exp(1e-3*(lambda a: a-a.T)(torch.randn(20, 20, generator=gen, dtype=D)))
    tilted = torch.linalg.qr(small+1e-2*torch.randn(50, 20, generator=gen, dtype=D))[0]
    near = rotation.capture(tilted, c, values, KS)
    assert all(.9 < near[k] <= 1+1e-12 for k in KS)


def test_overlap_is_one_for_the_same_subspace_and_zero_for_orthogonal_ones():
    gen = torch.Generator().manual_seed(2)
    q, _ = torch.linalg.qr(torch.randn(40, 40, generator=gen, dtype=D))
    rotated = q[:, :20]@torch.linalg.qr(torch.randn(20, 20, generator=gen, dtype=D))[0]
    assert abs(rotation.overlap(q[:, :20], q[:, :20], KS)[16]-1) <= 1e-12
    assert abs(rotation.overlap(q[:, :16], rotated[:, :16]@torch.eye(16, dtype=D), [16])[16]
               - float((q[:, :16].T@rotated[:, :16]).square().sum())/16) <= 1e-12
    assert rotation.overlap(q[:, :20], q[:, 20:40], KS)[16] <= 1e-24


def test_shape_distance_ignores_scale_and_counts_excluded_ranks():
    values = [3., 2., 1., .5]
    assert summarize.shape_distance(values, 7., [7*v for v in values], 49.)[0] <= 1e-12
    d, excluded = summarize.shape_distance(values, 7., [3., 2., 0., .5], 7.)
    assert excluded == 1 and d <= 1e-12
    d, _ = summarize.shape_distance(values, 7., [3., 2., 1., 1.], 7.)
    assert abs(d-(math.log(2.)**2/4)**.5) <= 1e-12


def test_groups_cover_specs_once_within_budget():
    class Spec:
        def __init__(self, name, count):
            self.name, self.count = name, count
    specs = [Spec(str(i), c) for i, c in enumerate([100, 300, 50, 400, 10])]
    out = rotation.groups(specs, 200**2*4, 4)
    assert [s.name for g in out for s in g] == [s.name for s in specs]
    assert all(sum(s.count**2*4 for s in g) <= 200**2*4 or len(g) == 1 for g in out)


def test_raw_dense_block_matches_matrix_free_ggn_products():
    data = states.GGN_TRAIN.Data('cpu')
    replay = states.Replay('soap', 1, 'cpu', data)
    model, registry = replay.arm.model, replay.arm.registry
    model.double()
    flat = replay.flat().double()
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(3, 3, 32, 32, generator=gen, dtype=D)
    spec = next(s for s in rotation.matrix_specs(registry) if s.name == 'head.weight')
    blocks, _ = ggn.accumulate(model, registry, flat, x, [spec], 2, all_parameters=False, contexts=None)
    action = micro.ggn_action(model, registry, flat, x, states.MICRO)
    v = torch.zeros_like(flat)
    v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=D)
    dense = blocks[spec.name]@v[spec.slice]
    assert float((action(v)[spec.slice]-dense).norm()/dense.norm()) <= 1e-8
    assert len(rotation.matrix_specs(registry)) == 39


def test_tensor_record_on_a_known_spectrum():
    gen = torch.Generator().manual_seed(4)
    c, _, values = psd(300, gen, decay=1.3)
    record, vectors, top = rotation.tensor_record(c, gen)
    assert torch.allclose(top, values[:200], rtol=1e-10)
    assert abs(record['fits']['all']['alpha']-1.3) <= 1e-8 and abs(record['trace']-float(values.sum())) <= 1e-10
    assert abs(record['top16_trace_share']-float(values[:16].sum()/values.sum())) <= 1e-10 and vectors.shape == (300, 200)
