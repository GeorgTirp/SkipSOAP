"""Explicit residual-stream reference, intentionally limited to guarded tiny MLPs."""
import torch
from torch.func import functional_call, jacrev


def residual_stages(model, registry, flat, x):
    params = registry.unflatten(flat)
    h = functional_call(model.input, {n[6:]: p for n, p in params.items() if n.startswith('input.')}, (x,)).detach()
    rs, bs, hs, slices = [], [], [h], []
    for i, block in enumerate(model.blocks):
        prefix = f'blocks.{i}.'
        specs = [s for s in registry.specs if s.name.startswith(prefix)]
        sl = slice(specs[0].start, specs[-1].stop)
        local = flat[sl]
        def call(v, state, block=block, specs=specs, sl=sl, prefix=prefix):
            mapping = {s.name[len(prefix):]: v[s.start-sl.start:s.stop-sl.start].reshape(s.shape) for s in specs}
            return functional_call(block, mapping, (state,))
        rs.append(jacrev(lambda state: call(local, state))(h).reshape(h.numel(), h.numel()).detach())
        bs.append(jacrev(lambda v: call(v, h))(local).reshape(h.numel(), -1).detach())
        h = call(local, h).detach()
        slices.append(sl)
        hs.append(h)
    head_params = {n[5:]: p for n, p in params.items() if n.startswith('head.')}
    terminal = jacrev(lambda state: functional_call(model.head, head_params, (state,)))(h).reshape(-1, h.numel()).detach()
    ts = [terminal]
    for r in reversed(rs):
        ts.insert(0, ts[0] @ r)
    return {'R': rs, 'B': bs, 'T': ts, 'h': hs, 'slices': slices}


def transported_metrics(stages, hz):
    metrics = [stages['T'][-1].T @ hz @ stages['T'][-1]]
    for r in reversed(stages['R']):
        metrics.insert(0, r.T @ metrics[0] @ r)
    return metrics


def stagewise_factors(model, registry, flat, x, xi, stages=None):
    from ..models.mlp import functional_logits
    from .sketch import factors_from_probes
    stages = residual_stages(model, registry, flat, x) if stages is None else stages
    y = torch.zeros((registry.size, xi.shape[-1]), dtype=flat.dtype, device=flat.device)
    message = stages['T'][-1].T @ xi.reshape(-1, xi.shape[-1])
    for r, b, sl in reversed(list(zip(stages['R'], stages['B'], stages['slices']))):
        y[sl] = b.T @ message
        message = r.T @ message
    params = registry.unflatten(flat)
    first = [s for s in registry.specs if s.stage == -1]
    sl = slice(first[0].start, first[-1].stop)
    def input_call(v):
        p = {s.name[6:]: v[s.start-sl.start:s.stop-sl.start].reshape(s.shape) for s in first}
        return functional_call(model.input, p, (x,))
    input_b = jacrev(input_call)(flat[sl]).reshape(-1, sl.stop-sl.start)
    y[sl] = input_b.T @ message
    # Terminal head has no downstream residual transport.
    heads = [s for s in registry.specs if s.stage == len(model.blocks)]
    sl = slice(heads[0].start, heads[-1].stop)
    def head_call(v):
        p = {s.name[5:]: v[s.start-sl.start:s.stop-sl.start].reshape(s.shape) for s in heads}
        return functional_call(model.head, p, (stages['h'][-1],))
    y[sl] = factors_from_probes(head_call, flat[sl], xi)
    return y.detach()


def reverse_message_sweep(stages, hz, g, mu, omega, terminal_delta=None):
    """One hidden-block sweep from zero; head may supply a fixed output change."""
    metrics = transported_metrics(stages, hz)
    message = torch.zeros_like(stages['h'][-1]).reshape(-1)
    if terminal_delta is not None:
        message = stages['T'][-1].T @ hz @ terminal_delta.reshape(-1)
    delta = torch.zeros_like(g)
    for i in reversed(range(len(stages['R']))):
        b, r, sl = stages['B'][i], stages['R'][i], stages['slices'][i]
        local = b.T @ metrics[i+1] @ b + mu * omega[sl, sl]
        delta[sl] = torch.linalg.solve(local, -(g[sl] + b.T @ message))
        message = r.T @ (message + metrics[i+1] @ b @ delta[sl])
    return delta
