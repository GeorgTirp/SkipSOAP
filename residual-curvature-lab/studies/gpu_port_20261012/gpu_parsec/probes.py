"""GGN probes for PARSEC-L (device-ready copy: labels drawn on the CPU) from per-example gradients with model-sampled labels (cross-entropy).

The loss is the mean cross-entropy over all output rows (logits of shape (B, K) or (B, T, K)). For
example i and draw j, labels y ~ softmax(z) are drawn independently for each of its rows and

    u_ij = grad_theta sum_t CE(z_it, y_ijt) / sqrt(N s),     N = B T rows, s draws,

so that sum_ij E[u_ij u_ij^T] = (1/N) sum_rows J^T (diag(p) - p p^T) J, the GGN of the mean loss.
A probe row is an ordinary per-example gradient, so token-shared weights (transformers) get their
full per-example gradient sum_t delta_t a_t^T, not a rank-one row.

exact=True replaces draws by all K classes weighted by sqrt(p_c) (one output row per example):
then sum_rows u u^T equals the GGN exactly.
"""
import torch
from torch.func import functional_call, grad, vmap


def ggn_probes(model, x, samples=4, generator=None, exact=False, chunk=256):
    """Return ({parameter: tensor (n, *parameter.shape)}, labels) for the trainable parameters."""
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
    if exact:
        if rows != 1:
            raise ValueError('Exact probes need one output row per example')
        index = torch.arange(len(x)).repeat_interleave(classes)
        labels = torch.arange(classes).repeat(len(x))
        weights = probability.reshape(-1).sqrt()/total**.5
    else:
        # Labels are drawn on the CPU (CPU generator) and moved, so every device draws the same labels.
        draws = torch.multinomial(probability.reshape(-1, classes).cpu(), samples, replacement=True,
                                  generator=generator).to(probability.device)
        labels = draws.reshape(len(x), rows, samples).permute(0, 2, 1).reshape(len(x)*samples, rows)
        index = torch.arange(len(x)).repeat_interleave(samples)
        weights = torch.full((len(index),), (total*samples)**-.5, dtype=logits.dtype, device=logits.device)
    out = {k: [] for k in named}
    for start in range(0, len(index), chunk):
        part = slice(start, start+chunk)
        gradients = per_example(detached, x[index[part]], labels[part], weights[part])
        for k in named:
            out[k].append(gradients[k].detach())
    return {named[k]: torch.cat(v) for k, v in out.items()}, labels


def per_example_gradients(model, x, y, chunk=256):
    """True-label per-example gradients {parameter: (B, *shape)} of the mean cross-entropy over each example's
    output rows; their mean is the batch gradient. Used for the optimism diagnostic r (PARSECL.step(true_gradients=...))."""
    named = {k: v for k, v in model.named_parameters() if v.requires_grad}
    detached = {k: v.detach() for k, v in named.items()}

    def example_loss(params, xi, yi):
        z = functional_call(model, params, (xi[None],))
        return torch.nn.functional.cross_entropy(z.reshape(-1, z.shape[-1]), yi.reshape(-1))

    per_example = vmap(grad(example_loss), in_dims=(None, 0, 0))
    out = {k: [] for k in named}
    for start in range(0, len(x), chunk):
        gradients = per_example(detached, x[start:start+chunk], y[start:start+chunk])
        for k in named:
            out[k].append(gradients[k].detach())
    return {named[k]: torch.cat(v) for k, v in out.items()}
