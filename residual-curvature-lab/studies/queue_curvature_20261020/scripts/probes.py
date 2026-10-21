"""Probe rows at arbitrary weights, restricted to named tensors (PROTOCOL.md, sections 3-4).

A probe row for example i is r_i = J_i^T v_i with J_i the logit Jacobian and v_i a random output vector with
E[v_i v_i^T] = H_i = diag(p_i) - p_i p_i^T, so E[r_i r_i^T] is example i's GGN:
- 'label': v_i = p_i - e_y with y ~ p_i (the gradient of the cross-entropy at a sampled label; as micro.lean_probes,
  with the same CPU draw);
- 'rademacher': v_i = S_i xi_i with S_i = diag(sqrt p_i) - p_i sqrt(p_i)^T (S_i S_i^T = H_i) and xi_i in {-1, +1}^K.
Rows are multiplied by `weight` (e.g. (number of pooled examples)^-1/2, so that sum_i r_i r_i^T is the average GGN).
"""
import torch
from torch.func import functional_call, grad, vmap


def output_vectors(p, kind, generator):
    """v_i per example (rows of p are probabilities); the draws are made on the CPU."""
    if kind == 'label':
        y = torch.multinomial(p.cpu(), 1, replacement=True, generator=generator)[:, 0].to(p.device)
        return p-torch.nn.functional.one_hot(y, p.shape[-1]).to(p.dtype)
    if kind == 'rademacher':
        xi = (torch.randint(0, 2, tuple(p.shape), generator=generator)*2-1).to(device=p.device, dtype=p.dtype)
        root = p.sqrt()
        return root*xi-p*(root*xi).sum(-1, keepdim=True)
    raise ValueError(f'unknown probe kind {kind}')


def square_root_factor(p):
    """S = diag(sqrt p) - p sqrt(p)^T for one probability vector (S S^T = diag(p) - p p^T)."""
    root = p.sqrt()
    return torch.diag(root)-torch.outer(p, root)


def probe_rows(model, registry, flat, x, names, kind, generator, weight, chunk=256):
    """{name: rows (n, *shape)} of the named tensors at weights `flat` (all other tensors fixed there)."""
    params = {k: v.detach() for k, v in registry.unflatten(flat).items()}
    with torch.no_grad():
        logits = functional_call(model, params, (x,))
    if logits[0].numel() != logits.shape[-1]:
        raise ValueError('one output row per example expected')
    v = output_vectors(logits.softmax(-1), kind, generator)
    fixed = {k: t for k, t in params.items() if k not in names}
    sub = {k: params[k] for k in names}

    def contraction(part, xi, vi):
        return (functional_call(model, {**fixed, **part}, (xi[None],))[0]*vi).sum()

    per_example = vmap(grad(contraction), in_dims=(None, 0, 0))
    out = {k: torch.empty((len(x),)+tuple(sub[k].shape), dtype=sub[k].dtype, device=sub[k].device) for k in names}
    for start in range(0, len(x), chunk):
        part = slice(start, start+chunk)
        rows = per_example(sub, x[part], v[part])
        for k in names:
            out[k][part] = rows[k]*weight
        del rows
    return out


def example_statistics(model, registry, flat, x, y, generator, chunk=256):
    """(per-example cross-entropy with the true labels, per-example ||r_i||^2 over all parameters with a sampled-label
    probe, unweighted: an unbiased estimate of tr(J_i^T H_i J_i))."""
    params = {k: v.detach() for k, v in registry.unflatten(flat).items()}
    with torch.no_grad():
        logits = functional_call(model, params, (x,))
    losses = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    v = output_vectors(logits.softmax(-1), 'label', generator)

    def contraction(part, xi, vi):
        return (functional_call(model, part, (xi[None],))[0]*vi).sum()

    per_example = vmap(grad(contraction), in_dims=(None, 0, 0))
    traces = torch.empty(len(x), dtype=logits.dtype, device=logits.device)
    for start in range(0, len(x), chunk):
        part = slice(start, start+chunk)
        rows = per_example(params, x[part], v[part])
        traces[part] = sum(r.reshape(r.shape[0], -1).square().sum(1) for r in rows.values())
        del rows
    return losses, traces
