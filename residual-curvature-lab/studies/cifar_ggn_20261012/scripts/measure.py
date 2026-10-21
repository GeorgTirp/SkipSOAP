"""Spectral normalization and the diagnostics of the CIFAR GGN study (PROTOCOL.md, sections 3 and 5).

SN (cifar.CifarViTSN): W_eff = W min(1, c / sigma(W)), with sigma the exact largest singular value, on the
residual-branch linear weights (attention query/key/value/output, MLP w1/w2), as studies/sn_curvature_20261008; the
optimizer acts on the raw weights.

Quadratic-model fidelity along the step d actually taken, on a batch (x, y) with mean cross-entropy L:
    actual = L(theta + d) - L(theta),   predicted_q = g'd + q/2,
    q in {full: d'Cd (full GGN), block: sum_b d_b'C_bb d_b (per-tensor GGN), hessian: d'Hd}.
`full` comes from the forward-mode product J d, `block` from per-example Jacobians (J_ib d_b for every tensor in
one pass, as the GGN arm builds its blocks), `hessian` by double backward (microbatches of at most 256); by microbatches.

Distance preservation (as studies/sn_curvature_20261008, extended over depth): ratios
r = |phi_k(x) - phi_k(x')| / |x - x'| for the residual stream after the embedding (k = 0) and after each block
(k = 1..depth), as the CLS row (k >= 1; at k = 0 it does not depend on the input) and as the whole stream, and for
the head input. Global pairs are random pairs of
validation images; local pairs are x' = x + eps |x| xi. Reported: quantiles 1/5/50/95/99 and distortion q95/q5.
"""
import torch
from torch.func import functional_call, jvp

from residual_curvature.curvature.operators import metric_action
from residual_curvature.models.mlp import functional_logits

import cifar
import ggn
import micro

QUANTILES = [.01, .05, .5, .95, .99]


def sn_names(model):
    return [f'blocks.{i}.{layer}.weight' for i in range(len(model.blocks)) for layer in cifar.SN_LAYERS]


def effective(model):
    """{name: effective weight} of an SN model, or {} for a plain model."""
    return model.effective_weights() if hasattr(model, 'effective_weights') else {}


def spectral_report(model):
    """Per residual-branch weight: sigma of the raw and the effective weight (SVD), and whether SN clips it."""
    rows = {}
    with torch.no_grad():
        weights = effective(model)
        bound = getattr(model, 'bound', None)
        for name in sn_names(model):
            raw = model.get_parameter(name)
            sigma = float(torch.linalg.matrix_norm(raw, ord=2))
            rows[name] = {'sigma_raw': sigma, 'sigma_effective': float(torch.linalg.matrix_norm(weights.get(name, raw), ord=2)),
                          'active': bound is not None and sigma > bound}
    return rows


def block_quadratic(model, registry, flat, d, x, chunk=64):
    """sum_b d_b' C_bb d_b from per-example Jacobians (one pass, ggn.example_jacobians): J_ib d_b for every tensor."""
    total, out = len(x), 0.
    names = [s.name for s in registry.specs]
    pieces = {s.name: d[s.slice] for s in registry.specs}
    for z, jacobians in ggn.example_jacobians(model, registry, flat, x, names, chunk):
        p = z.softmax(-1)
        n, classes = p.shape
        for name in names:
            jd = jacobians[name].reshape(n, classes, -1)@pieces[name]
            out += float(((p*jd.square()).sum(1)-(p*jd).sum(1).square()).sum())/total
        del jacobians
    return out


def quadratic_check(model, registry, flat, d, x, y, microbatch):
    """Actual loss change along d versus the full-GGN, per-tensor-GGN and Hessian quadratic models on (x, y)."""
    before = micro.batch_loss(model, registry, flat, x, y, microbatch)
    after = micro.batch_loss(model, registry, flat+d, x, y, microbatch)
    _, g = micro.gradient(model, registry, flat, x, y, microbatch)
    total, full = len(x), 0.
    for part in micro.parts(total, microbatch):
        weight = (part.stop-part.start)/total
        fn = functional_logits(model, registry, x[part])
        logits, jd = jvp(fn, (flat,), (d,))
        logits, jd = logits.detach(), jd.detach()
        full += weight*float((jd*metric_action(logits, jd)).sum())
    block = block_quadratic(model, registry, flat, d, x)
    hessian = micro.hessian_quadratic(model, registry, flat, d, x, y, min(microbatch, 256))
    gd = float(g@d)
    out = {'loss_before': before, 'actual': after-before, 'gd': gd, 'full': full, 'block': block, 'hessian': hessian,
           'step_norm': float(d.norm())}
    for kind in ('full', 'block', 'hessian'):
        predicted = gd+out[kind]/2
        out[f'predicted_{kind}'] = predicted
        out[f'ratio_{kind}'] = (after-before)/predicted if predicted != 0 else None
    out['coupling'] = full/block if block > 0 else None
    out['gauss_newton_gap'] = hessian/full if full > 0 else None
    return out


def stream_states(model, x):
    """Residual stream after the embedding and after each block (n, T, W), and the head input (n, W)."""
    with torch.no_grad():
        return model(x, states=True)


def ratio_summary(ratio):
    q = torch.quantile(ratio.double(), torch.tensor(QUANTILES, dtype=torch.float64, device=ratio.device))
    return {'quantiles_1_5_50_95_99': q.tolist(), 'distortion_q95_q5': float(q[3]/q[1]) if float(q[1]) > 0 else None,
            'min': float(ratio.min()), 'max': float(ratio.max())}


def distance_preservation(model, pairs):
    """{kind: {representation: summary}} for kind in pairs ('global', 'local'), pairs[kind] = (x, x')."""
    out = {}
    for kind, (xa, xb) in pairs.items():
        sa, ha = stream_states(model, xa)
        sb, hb = stream_states(model, xb)
        denominator = (xa-xb).flatten(1).norm(dim=1)
        rows = {}
        for k, (u, w) in enumerate(zip(sa, sb)):
            if k > 0:  # after the embedding the CLS row is a learned constant, independent of the input
                rows[f'depth{k}_cls'] = ratio_summary((u[:, 0]-w[:, 0]).norm(dim=1)/denominator)
            rows[f'depth{k}_stream'] = ratio_summary((u-w).flatten(1).norm(dim=1)/denominator)
        rows['head'] = ratio_summary((ha-hb).norm(dim=1)/denominator)
        out[kind] = rows
    return out


def block_jacobian_deviation(model, x, generator, probes=4):
    """RMS singular value of A_k = R_k - I per block at the actual block inputs (parsec_l.diagnostics, with the
    Gaussian probes drawn on the CPU)."""
    inputs = []
    handles = [block.register_forward_hook(lambda module, args, output: inputs.append(args[0].detach())) for block in model.blocks]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    weights = {k: v.detach() for k, v in effective(model).items()}
    out = []
    for index, (block, h) in enumerate(zip(model.blocks, inputs)):
        prefix = f'blocks.{index}.'
        local = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        image = norm = 0.
        for _ in range(probes):
            v = torch.randn(h.shape, generator=generator, dtype=h.dtype).to(h.device)
            image += float(jvp(lambda t: functional_call(block, local, (t,), strict=False)-t, (h,), (v,))[1].detach().square().sum())
            norm += float(v.square().sum())
        out.append({'block': len(out)+1, 'rms': (image/norm)**.5})
    return out
