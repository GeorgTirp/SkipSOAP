import math

import torch
from torch.func import vjp

from .operators import output_weights


def categorical_probes(logits, rank, seed, weights=None):
    p = logits.detach().softmax(-1).cpu()
    gen = torch.Generator().manual_seed(seed)
    # Probe-major draws ensure nested prefixes do not change with requested rank.
    draws = [torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=gen).reshape(p.shape[:-1]) for _ in range(rank)]
    xi = torch.stack([torch.nn.functional.one_hot(d, p.shape[-1]) - p for d in draws], -1).to(logits)
    return (xi * output_weights(logits, weights).sqrt()[..., None, None] / math.sqrt(rank)).detach()


def factors_from_probes(fn, flat, xi):
    _, pullback = vjp(fn, flat)
    return torch.stack([pullback(xi[..., a])[0].detach() for a in range(xi.shape[-1])], 1)


def microbatch_factors(model, registry, flat, x, xi, microbatch):
    from ..models.mlp import functional_logits
    result = flat.new_zeros((flat.numel(), xi.shape[-1]))
    for start in range(0, len(x), microbatch):
        result += factors_from_probes(functional_logits(model, registry, x[start:start + microbatch]),
                                      flat, xi[start:start + microbatch])
    return result
