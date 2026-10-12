"""Exact reduced GGN from directional JVPs; no parameter-space factor."""
import time

import torch
from torch.func import jvp

from .operators import metric_action, output_weights


def reduced_ggn(operator, directions):
    if directions.ndim != 2 or directions.shape[0] != operator.params.numel():
        raise ValueError('Directions must have shape D x number_of_directions')
    started = time.perf_counter()
    rank = directions.shape[1]
    if rank == 0:
        return directions.new_empty(0, 0), {'jvps': 0, 'seconds': 0., 'response_bytes': 0}
    tangents = torch.stack([jvp(operator.fn, (operator.params,), (directions[:, i],))[1].detach()
                            for i in range(rank)], -1)
    operator.counts['jvp'] += rank
    p = operator.logits.detach().softmax(-1)[..., None]
    weighted = p * (tangents - (p*tangents).sum(-2, keepdim=True))
    weighted *= output_weights(operator.logits, operator.weights)[..., None, None]
    response, weighted = tangents.reshape(-1, rank), weighted.reshape(-1, rank)
    matrix = response.T @ weighted
    return (matrix + matrix.T)/2, {'jvps': rank, 'seconds': time.perf_counter()-started,
             'response_bytes': response.numel()*response.element_size(),
             'normalization': 'same output weights and mean loss as GGN operator', 'exact': True}


def trace_estimate(operator, rank, seed):
    """Candidate-independent Rademacher Hutchinson trace, using JVPs only."""
    if rank <= 0:
        raise ValueError('Trace probe count must be positive')
    generator = torch.Generator(device=operator.params.device).manual_seed(seed)
    values = []
    started = time.perf_counter()
    for _ in range(rank):
        v = torch.randint(2, operator.params.shape, generator=generator, device=operator.params.device).to(operator.params)*2-1
        tangent = jvp(operator.fn, (operator.params,), (v,))[1].detach()
        values.append((tangent*metric_action(operator.logits, tangent, operator.weights)).sum())
    operator.counts['jvp'] += rank
    values = torch.stack(values)
    if not torch.isfinite(values).all() or float(values.min()) < -1e-10:
        raise RuntimeError('Invalid PSD trace estimate')
    return float(values.mean()), {'source': 'Rademacher Hutchinson estimate', 'rank': rank, 'seed': seed,
        'samples': values.tolist(), 'seconds': time.perf_counter()-started,
        'sample_standard_error': float(values.std(unbiased=True)/rank**.5) if rank > 1 else None}
