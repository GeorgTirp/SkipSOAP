"""OT-metric probes for PARSEC-L (PROTOCOL.md, section 3): rows u_i with sum_i E[u_i u_i^T] equal to the lab's dynamic
graph-Wasserstein pullback M = (1/N) sum_i J_p,i^T (L(p_i) + 11^T/K)^-1 J_p,i (residual_curvature.curvature.wasserstein).

With (L + 11^T/K) = R R^T (Cholesky) and H = diag(p) - p p^T (so J_p = H J_z), the row of example i is the gradient of
z_i^T v_i with v_i = H R^-T xi_i / sqrt(N s), xi_i ~ N(0, I_K) drawn on the CPU from `generator`. exact=True replaces the
draws by the K unit vectors (K rows per example), which reproduces M exactly (tests). Memory layout as micro.lean_probes.
"""
import torch
from torch.func import functional_call, grad, vmap

from residual_curvature.curvature.wasserstein import graph_laplacian


def ot_probes(model, x, samples=1, generator=None, chunk=256, exact=False):
    named = {k: v for k, v in model.named_parameters() if v.requires_grad}
    detached = {k: v.detach() for k, v in named.items()}
    with torch.no_grad():
        logits = functional_call(model, detached, (x,))
    if logits.ndim != 2:
        raise ValueError('OT probes expect one output row per example')
    n, classes = logits.shape
    p = logits.softmax(-1)
    lap = graph_laplacian(p)
    chol = torch.linalg.cholesky(lap+torch.ones_like(lap)/classes)
    if exact:
        draws = torch.eye(classes, dtype=logits.dtype, device=logits.device).repeat(n, 1)
        per, scale = classes, n**-.5
    else:
        draws = torch.randn((n*samples, classes), generator=generator, dtype=logits.dtype).to(logits.device)
        per, scale = samples, (n*samples)**-.5
    index = torch.arange(n, device=logits.device).repeat_interleave(per)
    y = torch.linalg.solve_triangular(chol[index].transpose(-1, -2), draws[..., None], upper=True).squeeze(-1)
    q = p[index]
    v = scale*q*(y-(q*y).sum(-1, keepdim=True))

    def projected(params, xi, vi):
        return (functional_call(model, params, (xi[None],)).reshape(-1)*vi).sum()

    per_example = vmap(grad(projected), in_dims=(None, 0, 0))
    out = {k: torch.empty((len(index),)+w.shape, dtype=w.dtype, device=w.device) for k, w in named.items()}
    for start in range(0, len(index), chunk):
        part = slice(start, start+chunk)
        gradients = per_example(detached, x[index[part]], v[part])
        for k in named:
            out[k][part] = gradients[k]
        del gradients
    return {named[k]: rows for k, rows in out.items()}
