"""SNGP spectral normalization as a deterministic parametrization, and model helpers.

W_eff = W * min(1, c / sigma(W)), sigma the exact largest singular value (SVD). The gradient flows
through sigma (equal to the converged power-iteration gradient u v^T). Applied to the linear
weights inside residual branches only (PROTOCOL.md).
"""
import torch
from torch import nn
from torch.nn.utils import parametrize

from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model

VIT_LAYERS = ['attention.query', 'attention.key', 'attention.value', 'attention.output', 'w1', 'w2']
MLP_LAYERS = ['w1', 'w2']


class SpectralNorm(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = float(bound)

    def forward(self, weight):
        return weight*torch.clamp(self.bound/torch.linalg.matrix_norm(weight, ord=2), max=1.)


def sn_module_names(model):
    kind = 'vit' if hasattr(model.blocks[0], 'attention') else 'mlp'
    layers = VIT_LAYERS if kind == 'vit' else MLP_LAYERS
    return [f'blocks.{i}.{layer}' for i in range(len(model.blocks)) for layer in layers]


def make_sn_model(config, bound):
    """make_model(config) (same initialization), with SN registered on residual-branch weights."""
    model = make_model(config)
    if bound is None:
        return model
    modules = dict(model.named_modules())
    for name in sn_module_names(model):
        parametrize.register_parametrization(modules[name], 'weight', SpectralNorm(bound))
    return model


def spectral_report(model, bound):
    """Per SN layer: sigma of the raw and effective weights and whether SN is active."""
    modules = dict(model.named_modules())
    rows = {}
    with torch.no_grad():
        for name in sn_module_names(model):
            module = modules[name]
            raw = module.parametrizations.weight.original if bound is not None else module.weight
            effective = module.weight
            sigma = float(torch.linalg.matrix_norm(raw, ord=2))
            rows[name] = {'sigma_raw': sigma, 'sigma_effective': float(torch.linalg.matrix_norm(effective, ord=2)),
                          'active': bound is not None and sigma > bound}
    return rows


def effective_twin(model, config):
    """Plain model (no parametrization) whose weights equal this model's effective weights."""
    twin = make_model(config)
    registry = Registry(twin)
    with torch.no_grad():
        current = {}
        modules = dict(model.named_modules())
        for spec in registry.specs:
            module_name, _, attribute = spec.name.rpartition('.')
            current[spec.name] = getattr(modules[module_name], attribute).detach().clone()
    commit(twin, registry, registry.flatten(current))
    return twin, registry


def representations(model, x):
    """phi_res (residual stream after the last block; ViT: CLS token) and phi_head (input to the head)."""
    with torch.no_grad():
        if hasattr(model, 'norm'):
            h = model.input(x)
            for block in model.blocks:
                h = block(h)
            res = h[:, 0]
            return res, model.norm(res)
        h = model.input(x)
        for block in model.blocks:
            h = block(h)
        return h, h
