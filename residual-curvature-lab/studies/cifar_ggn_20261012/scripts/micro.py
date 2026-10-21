"""Microbatched mean-loss quantities and memory-lean GGN probes (PROTOCOL.md, section 3).

A backward pass of the depth-6 ViT at batch 2048 keeps about 6.6 GB of activations, so every batch quantity is a
weighted sum over microbatches. With microbatches m of sizes n_m and the mean loss L = sum_m (n_m/N) L_m:
    gradient   sum_m (n_m/N) grad L_m
    GGN action sum_m (n_m/N) J_m^T H_m J_m v / n_m-normalized as in operators.GGN
    HVP        sum_m (n_m/N) Hess L_m v;  d' Hess L d by double backward
These equal the full-batch quantities up to summation order.

`lean_probes` computes the same probe rows as gpu_parsec.ggn_probes (bitwise, study test), but writes each chunk into
preallocated output tensors instead of concatenating a list of chunks. That avoids holding the probes twice.
"""
import torch
from torch.func import functional_call, grad, jvp, vjp, vmap

from residual_curvature.curvature.operators import loss, metric_action
from residual_curvature.models.mlp import functional_logits


def parts(n, micro):
    return [slice(start, min(start+micro, n)) for start in range(0, n, micro)]


def gradient(model, registry, flat, x, y, micro):
    """(mean loss, gradient) of the batch, by microbatches."""
    total, value, g = len(x), 0., torch.zeros_like(flat)
    for part in parts(total, micro):
        fn = functional_logits(model, registry, x[part])
        weight = (part.stop-part.start)/total
        part_loss, part_grad = torch.func.grad_and_value(lambda w: loss(fn(w), y[part]))(flat)[::-1]
        value += weight*float(part_loss)
        g += weight*part_grad.detach()
    return value, g


def batch_loss(model, registry, flat, x, y, micro):
    total, value = len(x), 0.
    with torch.no_grad():
        for part in parts(total, micro):
            value += (part.stop-part.start)/total*float(loss(functional_logits(model, registry, x[part])(flat), y[part]))
    return value


def ggn_action(model, registry, flat, x, micro):
    """v -> C v for the mean loss of the batch (C the full GGN), by microbatches."""
    total = len(x)

    def action(v):
        out = torch.zeros_like(flat)
        for part in parts(total, micro):
            fn = functional_logits(model, registry, x[part])
            logits, pullback = vjp(fn, flat)
            tangent = jvp(fn, (flat,), (v,))[1]
            out += (part.stop-part.start)/total*pullback(metric_action(logits, tangent))[0].detach()
        return out
    return action


def hessian_action(model, registry, flat, x, y, micro):
    """v -> Hess L v for the mean loss of the batch, by microbatches (forward over reverse)."""
    total = len(x)

    def action(v):
        out = torch.zeros_like(flat)
        for part in parts(total, micro):
            fn = functional_logits(model, registry, x[part])
            out += (part.stop-part.start)/total*jvp(grad(lambda w: loss(fn(w), y[part])), (flat,), (v,))[1].detach()
        return out
    return action


def hessian_quadratic(model, registry, flat, d, x, y, micro):
    """d' Hess L d for the mean loss of the batch by double backward, d . grad(grad L . d), by microbatches.

    Not forward over forward: in torch 2.14.1 LayerNorm's second-order forward mode is wrong (12% off finite
    differences). Not forward over reverse: through SN's SVD it fails with a dtype mismatch in float32 on CUDA.
    Double backward matches finite differences and the float64 forward-over-reverse HVP (pre-run checks, study test)."""
    total, out = len(x), 0.
    for part in parts(total, micro):
        fn = functional_logits(model, registry, x[part])
        w = flat.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(loss(fn(w), y[part]), w, create_graph=True)[0]
        out += (part.stop-part.start)/total*float(torch.autograd.grad(g@d, w)[0]@d)
    return out


def lean_probes(model, x, samples=1, generator=None, chunk=256):
    """gpu_parsec.ggn_probes (sampled labels drawn on the CPU) with preallocated outputs; returns {parameter: rows}."""
    named = {k: v for k, v in model.named_parameters() if v.requires_grad}
    detached = {k: v.detach() for k, v in named.items()}
    with torch.no_grad():
        logits = functional_call(model, detached, (x,))
    classes = logits.shape[-1]
    rows = logits[0].numel()//classes
    total = len(x)*rows
    probability = logits.softmax(-1).reshape(len(x), rows, classes)

    def weighted_loss(params, xi, yi, wi):
        z = functional_call(model, params, (xi[None],)).reshape(-1, classes)
        return wi*torch.nn.functional.cross_entropy(z, yi.reshape(-1), reduction='sum')

    per_example = vmap(grad(weighted_loss), in_dims=(None, 0, 0, 0))
    draws = torch.multinomial(probability.reshape(-1, classes).cpu(), samples, replacement=True,
                              generator=generator).to(probability.device)
    labels = draws.reshape(len(x), rows, samples).permute(0, 2, 1).reshape(len(x)*samples, rows)
    index = torch.arange(len(x)).repeat_interleave(samples)
    weights = torch.full((len(index),), (total*samples)**-.5, dtype=logits.dtype, device=logits.device)
    out = {k: torch.empty((len(index),)+v.shape, dtype=v.dtype, device=v.device) for k, v in named.items()}
    for start in range(0, len(index), chunk):
        part = slice(start, start+chunk)
        gradients = per_example(detached, x[index[part]], labels[part], weights[part])
        for k in named:
            out[k][part] = gradients[k]
        del gradients
    return {named[k]: v for k, v in out.items()}
