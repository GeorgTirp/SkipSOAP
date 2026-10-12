"""Fixed depth-dependent residual-branch scaling (option 2 of the SN follow-up discussion).

Each residual block computes h + s * branch(h). The lab's ViT and residual MLP already take a fixed,
non-learned `model.branch_scale` s (default 1). This sets

    s = alpha / blocks**exponent,

e.g. exponent 0.5 (alpha / sqrt(depth)) or 1 (alpha / depth). Standard setting: exponent 1 (alpha / depth).
exponent = 0 with alpha = 1 is the plain network, bit for bit. The scale multiplies an existing `branch_scale`.
"""
import copy


def depth_branch_scale(blocks, exponent=1.0, alpha=1.0):
    if blocks < 1:
        raise ValueError('Positive block count required')
    return alpha/blocks**exponent


def with_branch_scaling(config, exponent=1.0, alpha=1.0):
    """Copy of a lab config with model.branch_scale multiplied by alpha / blocks**exponent."""
    c = copy.deepcopy(config)
    c['model']['branch_scale'] = c['model'].get('branch_scale', 1.0)*depth_branch_scale(c['model']['blocks'], exponent, alpha)
    return c
