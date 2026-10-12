"""Instrumentation of unmodified, pinned SOAP; never a replacement optimizer."""
import copy
from contextlib import contextmanager

import torch
from third_party.soap.soap import SOAP

UPSTREAM_COMMIT = 'a1e553530fde97d0e6b307d7c82ac6d38b072340'


@contextmanager
def default_dtype(dtype):
    # Upstream creates GG without a dtype. Keep that behavior, with an explicit
    # scoped default shared by reference and adapter (single-threaded runner).
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


class InstrumentedSOAP(SOAP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts = {}

    def project_back(self, grad, state, **kwargs):
        # First project_back per initialized tensor in a step is the update.
        # Later calls transport momentum during post-update basis maintenance.
        key = id(state)
        if state.get('step', 0) > 0 and key not in self.contexts:
            group = self.param_groups[0]
            b1, b2 = group['betas']
            correction = (1 - b2 ** state['step']) ** .5 / (1 - b1 ** state['step']) if group['correct_bias'] else 1.0
            self.contexts[key] = {
                'basis': copy.deepcopy(state['Q']),
                'denominator': state['exp_avg_sq'].sqrt().add(group['eps']).clone(),
                'momentum': state['exp_avg'].clone(),
                'bias_correction': correction, 'lr': group['lr'], 'step': state['step'],
            }
        return super().project_back(grad, state, **kwargs)


def soap_propose(params, grads, cloned_optimizer_state, config):
    if config['normalize_grads'] or config['weight_decay'] or config['merge_dims']:
        raise ValueError('Core frozen-map audit requires no normalization, weight decay or dimension merging')
    names = list(params)
    variables = [torch.nn.Parameter(params[n].detach().clone()) for n in names]
    opt = InstrumentedSOAP(variables, **config)
    if cloned_optimizer_state is not None:
        opt.load_state_dict(copy.deepcopy(cloned_optimizer_state))
    for n, p in zip(names, variables):
        p.grad = grads[n].detach().clone()
    with default_dtype(variables[0].dtype):
        opt.step()
    context = {n: copy.deepcopy(opt.contexts.get(id(opt.state[p]))) for n, p in zip(names, variables)}
    delta = {n: p.detach() - params[n].detach() for n, p in zip(names, variables)}
    return delta, copy.deepcopy(opt.state_dict()), context


def mode(v, q, axis, operation):
    if len(q) == 0:
        return v
    x = v.movedim(axis, 0)
    shape = x.shape
    x = x.reshape(shape[0], -1)
    if operation == 'forward':
        x = q @ x
    elif operation == 'transpose':
        x = q.T @ x
    elif operation == 'solve':
        x = torch.linalg.solve(q, x)
    else:
        x = torch.linalg.solve(q.T, x)
    return x.reshape(shape).movedim(0, axis)


def frozen_apply(v, context, inverse=False):
    if context is None:
        raise ValueError('SOAP initialization has no used update context')
    q, d = context['basis'], context['denominator']
    for axis, basis in enumerate(q):
        v = mode(v, basis, axis, 'solve' if inverse else 'transpose')
    d = d.reshape(d.shape + (1,) * (v.ndim - d.ndim))
    v = v * d if inverse else v / d
    for axis, basis in enumerate(q):
        v = mode(v, basis, axis, 'solve_transpose' if inverse else 'forward')
    return v


class SOAPMetric:
    """Omega = scale P^-1, with trace(Omega)/D = 1.

    Inverse transforms use solves because upstream float32 bases are only
    approximately orthogonal even when parameters are float64.
    """
    def __init__(self, registry, contexts):
        self.registry, self.contexts = registry, contexts
        trace = 0.0
        for spec in registry.specs:
            c = contexts[spec.name]
            if c is None:
                raise ValueError('Metric unavailable during SOAP initialization')
            weights = torch.ones_like(c['denominator'])
            for axis, q in enumerate(c['basis']):
                if len(q):
                    qinv = torch.linalg.solve(q, torch.eye(q.shape[0], dtype=q.dtype, device=q.device))
                    shape = [1] * weights.ndim
                    shape[axis] = q.shape[0]
                    weights *= qinv.square().sum(1).reshape(shape)
            trace += float((weights * c['denominator']).sum())
        self.scale = registry.size / trace

    def map(self, flat, inverse=False):
        out = torch.empty_like(flat)
        for s in self.registry.specs:
            value = flat[s.slice].reshape(s.shape + flat.shape[1:])
            out[s.slice] = frozen_apply(value, self.contexts[s.name], inverse).reshape((s.count,) + flat.shape[1:])
        return out

    def apply(self, v):
        return self.scale * self.map(v, inverse=True)

    def inverse(self, v):
        return self.map(v) / self.scale
