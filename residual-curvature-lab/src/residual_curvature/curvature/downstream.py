"""Shared GGN probes evaluated by explicit residual-stream VJP transport.

This is the existing Algorithm A estimator, not a different curvature model.
For supplied output probes Xi, this module evaluates J.T @ Xi without dense
hidden-state Jacobians or metrics. Xi must already contain the global loss and
rank normalization; microbatches are summed without additional normalization.
The supported topology is a deterministic input -> residual blocks -> head MLP.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import time

import torch
from torch.func import functional_call, vjp


@dataclass(frozen=True)
class DownstreamFactors:
    factors: torch.Tensor
    metadata: dict
    counts: dict

    def by_tensor(self, registry):
        if _registry_hash(registry) != self.metadata['parameter_registry_hash']:
            raise ValueError('The supplied registry differs from the factor registry')
        return {spec.name: self.factors[spec.slice] for spec in registry.specs}


def _registry_hash(registry):
    return hashlib.sha256(json.dumps(registry.json(), sort_keys=True).encode()).hexdigest()


def nested_probe_prefix(xi, rank):
    """Select a nested prefix of probes already divided by sqrt(xi.shape[-1])."""
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= xi.shape[-1]:
        raise ValueError('Probe rank must be a positive integer no larger than the supplied rank')
    return (xi[..., :rank].detach() * math.sqrt(xi.shape[-1] / rank)).clone()


def _stage(module, prefix, registry, flat):
    specs = [spec for spec in registry.specs if spec.name.startswith(prefix)]
    local = torch.cat([flat[spec.slice] for spec in specs]) if specs else flat.new_empty(0)

    def call(parameters, state):
        mapping = {}
        offset = 0
        for spec in specs:
            mapping[spec.name[len(prefix):]] = parameters[offset:offset + spec.count].reshape(spec.shape)
            offset += spec.count
        return functional_call(module, mapping, (state,))

    return local, call, specs


def propagate_metric_probes(model, registry, flat, x, xi, *, microbatch=None,
                            batch_id=None, state_step=None, probe_seed=None,
                            loss_normalization='provided_globally_normalized_probes',
                            token_mask=None, max_factor_bytes=256 * 2**20):
    """Return D x k parameter responses to shared B x C x k output probes.

    A residual parameter probe is B_i.T @ Psi_{i+1}; its incoming hidden probe
    is R_i.T @ Psi_{i+1}. Thus J_i = T_{i+1} B_i, including the identity path
    and normalization derivatives in R_i. Input, output, bias and normalization
    tensors all receive their registered responses. The returned factor is
    exact for the supplied probes; its use as a GGN estimator is approximate.

    The model, parameters, input probes and RNG are not advanced. A cloned model
    protects buffers; gradients are taken only with respect to the cloned flat
    parameters and detached local hidden states. Tied parameters across stages
    are rejected explicitly rather than silently dropping an alias response.
    """
    if flat.ndim != 1 or flat.numel() != registry.size:
        raise ValueError('Incorrect registered parameter dimension')
    if x.ndim != 2 or xi.ndim != 3 or len(x) == 0 or xi.shape[0] != len(x) or xi.shape[-1] < 1:
        raise ValueError('Expected nonempty MLP inputs B x I and probes B x C x k')
    if x.device != flat.device or xi.device != flat.device or x.dtype != flat.dtype or xi.dtype != flat.dtype:
        raise ValueError('Parameters, inputs and probes must share dtype and device')
    if not torch.isfinite(xi).all():
        raise ValueError('Output probes must be finite')
    microbatch = len(x) if microbatch is None else microbatch
    if not isinstance(microbatch, int) or isinstance(microbatch, bool) or microbatch < 1:
        raise ValueError('Microbatch must be a positive integer')
    named = dict(model.named_parameters())
    if set(named_name for named_name, value in named.items() if value.requires_grad) != {s.name for s in registry.specs}:
        raise ValueError('Registry must cover every trainable parameter exactly once')
    aliases = list(model.named_parameters(remove_duplicate=False))
    if len({id(value) for _, value in aliases}) != len(aliases):
        raise NotImplementedError('Cross-stage tied parameters require alias-aware accumulation')
    for module in model.modules():
        if module.training and isinstance(module, (torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._BatchNorm)):
            raise NotImplementedError('Transport requires deterministic, sample-independent stages')
    rank = xi.shape[-1]
    factor_bytes = registry.size * rank * flat.element_size()
    if max_factor_bytes is not None and factor_bytes > max_factor_bytes:
        raise MemoryError('Downstream probe factor exceeds the explicit factor storage budget')
    if token_mask is not None:
        token_mask = torch.as_tensor(token_mask, dtype=torch.bool, device=xi.device)
        if token_mask.shape != x.shape[:1]:
            raise ValueError('MLP token mask must have shape B')
        if xi[~token_mask].count_nonzero():
            raise ValueError('Masked output probes must already be zero with global normalization')

    started = time.perf_counter()
    frozen = copy.deepcopy(model)
    parameters, probes = flat.detach().clone(), xi.detach().clone()
    stages = [_stage(frozen.input, 'input.', registry, parameters)]
    stages.extend(_stage(block, f'blocks.{i}.', registry, parameters) for i, block in enumerate(frozen.blocks))
    stages.append(_stage(frozen.head, 'head.', registry, parameters))
    covered = [spec.name for _, _, specs in stages for spec in specs]
    if set(covered) != {s.name for s in registry.specs} or len(covered) != len(registry.specs):
        raise ValueError('Transport topology must cover every registered parameter exactly once')
    factors = flat.new_zeros((registry.size, rank))
    counts = {'forward': 0, 'vjp': 0, 'jvp': 0, 'microbatches': 0}
    largest_activations = 0
    largest_hidden_message = 0
    devices = [flat.device.index if flat.device.index is not None else torch.cuda.current_device()] if flat.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        for start in range(0, len(x), microbatch):
            counts['microbatches'] += 1
            hidden = [x[start:start + microbatch].detach().clone()]
            # Keep only detached hidden states, then recompute one stage's graph
            # at a time for the backward transport.
            with torch.no_grad():
                for local, call, _ in stages[:-1]:
                    hidden.append(call(local, hidden[-1]).detach())
                    counts['forward'] += 1
            largest_activations = max(largest_activations, sum(h.numel() for h in hidden))
            message = probes[start:start + microbatch]
            for index in reversed(range(len(stages))):
                local, call, specs = stages[index]
                output, pullback = vjp(call, local, hidden[index])
                counts['forward'] += 1
                if output.shape != message.shape[:-1]:
                    raise ValueError('Probe shape does not match the stage output')
                incoming = []
                for column in range(rank):
                    response, hidden_response = pullback(message[..., column])
                    counts['vjp'] += 1
                    offset = 0
                    for spec in specs:
                        factors[spec.slice, column] += response[offset:offset + spec.count].detach()
                        offset += spec.count
                    if index:
                        incoming.append(hidden_response.detach())
                if index:
                    message = torch.stack(incoming, dim=-1)
                    largest_hidden_message = max(largest_hidden_message, message.numel())
                # Do not retain another stage's autograd graph during recomputation.
                del pullback, output

    metadata = {
        'estimator': 'shared_output_GGN_probes_existing_Algorithm_A',
        'transport': 'terminal_VJP_then_residual_stage_VJPs',
        'exact_for_supplied_probes': True,
        'curvature_representation': 'approximate_shared_GGN_sketch',
        'batch_id': batch_id, 'state_step': state_step, 'probe_seed': probe_seed,
        'probe_rank': rank, 'batch_size': len(x), 'microbatch': microbatch,
        'loss_normalization': loss_normalization,
        'probe_normalization': 'supplied_globally_once_no_microbatch_rescaling',
        'token_mask': None if token_mask is None else token_mask.cpu().tolist(),
        'parameter_registry_hash': _registry_hash(registry),
        'dtype': str(flat.dtype), 'device': str(flat.device),
        'factor_bytes': factor_bytes,
        'factor_storage_budget_bytes': max_factor_bytes,
        'probe_input_bytes': probes.numel() * flat.element_size(),
        'max_detached_activation_bytes': largest_activations * flat.element_size(),
        'max_hidden_message_bytes': largest_hidden_message * flat.element_size(),
        'workspace_estimate_excludes_autograd_internals': True,
        'forward_count_unit': 'individual_stage_invocation_including_recomputation',
        'vjp_count_unit': 'joint_stage_parameter_and_hidden_pullback_per_probe',
        'wall_seconds': time.perf_counter() - started,
    }
    return DownstreamFactors(factors.detach(), metadata, counts)
