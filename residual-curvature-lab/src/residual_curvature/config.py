import copy
import math
from pathlib import Path

import yaml

DEFAULT = {
    'seed': 0, 'data_seed': 101, 'probe_seed': 202, 'analysis_seed': 303,
    'device': 'cpu', 'dtype': 'float64', 'threads': 1, 'output': 'runs',
    'model': {'kind': 'mlp', 'input_dim': 6, 'width': 4, 'expansion': 8, 'blocks': 2,
              'classes': 3, 'epsilon': 1e-5, 'branch_scale': 1.0},
    'vit': {'image_size': 28, 'patch_size': 7, 'heads': 2},
    'data': {'dataset': 'synthetic', 'root': 'datasets', 'samples': 128,
             'resolution': 14, 'batch_size': 4, 'panel_batches': 4,
             'train_size': 50000, 'validation_size': 5000, 'analysis_size': 5000},
    'soap': {'lr': 0.003, 'betas': [0.95, 0.95], 'shampoo_beta': -1,
             'eps': 1e-8, 'weight_decay': 0.0, 'precondition_frequency': 10,
             'max_precond_dim': 10000, 'merge_dims': False, 'precondition_1d': False,
             'normalize_grads': False, 'data_format': 'channels_first', 'correct_bias': True},
    'damping': {'gammas': [0.001, 0.01, 0.1], 'floor': 1e-8, 'metric': 'soap'},
    'sketch': {'ranks': [4, 8, 16, 32], 'repeats': 2, 'microbatch': 2},
    'resources': {'dense_cap': 2048, 'dual_cap': 2048, 'budget_mib': 256},
    'solver': {'tolerance64': 1e-8, 'tolerance32': 1e-4, 'max_iterations': 1000},
    'training': {'steps': 3, 'snapshots': [3], 'method': 'soap', 'policy': 'native',
                 'gamma': 0.01, 'rank': 8, 'checkpoint_every': 1},
    'search': {'grid_exponents': list(range(-8, 4)), 'initial': 1.0,
               'armijo': 1e-4, 'max_evaluations': 16, 'minimum': 1e-8},
    'partition': 'tensor',
}


def merge_strict(default, incoming, prefix=''):
    if not isinstance(incoming, dict):
        raise ValueError(f'{prefix or "configuration"} must be a mapping')
    extra = set(incoming) - set(default)
    if extra:
        raise ValueError(f'Unknown configuration keys in {prefix}: {sorted(extra)}')
    result = copy.deepcopy(default)
    for key, value in incoming.items():
        result[key] = merge_strict(default[key], value, prefix + key + '.') if isinstance(default[key], dict) else value
    return result


def validate(c):
    def check_types(template, values, prefix=''):
        for key, original in template.items():
            value, name = values[key], prefix + key
            if isinstance(original, dict):
                if not isinstance(value, dict):
                    raise ValueError(f'{name} must be a mapping')
                check_types(original, value, name + '.')
            elif isinstance(original, bool) and type(value) is not bool:
                raise ValueError(f'{name} must be a boolean')
            elif isinstance(original, str) and not isinstance(value, str):
                raise ValueError(f'{name} must be a string')
            elif isinstance(original, list) and not isinstance(value, list):
                raise ValueError(f'{name} must be a list')
    check_types(DEFAULT, c)
    def positive(v, name, integer=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 or (integer and not isinstance(v, int)):
            raise ValueError(f'{name} must be a positive {"integer" if integer else "number"}')
    for section, keys in [('model', ['input_dim', 'width', 'expansion', 'blocks', 'classes']),
                          ('vit', ['image_size', 'patch_size', 'heads']),
                          ('data', ['samples', 'resolution', 'batch_size', 'panel_batches', 'train_size', 'validation_size', 'analysis_size']),
                          ('sketch', ['repeats', 'microbatch']), ('resources', ['dense_cap', 'dual_cap']),
                          ('training', ['steps', 'checkpoint_every', 'rank']),
                          ('solver', ['max_iterations']), ('search', ['max_evaluations']),
                          ('soap', ['precondition_frequency', 'max_precond_dim'])]:
        for k in keys:
            positive(c[section][k], section + '.' + k, True)
    for section, keys in [('model', ['epsilon']), ('damping', ['floor']), ('resources', ['budget_mib']),
                          ('soap', ['eps', 'lr']), ('solver', ['tolerance64', 'tolerance32']),
                          ('search', ['initial', 'minimum', 'armijo'])]:
        for k in keys:
            positive(c[section][k], section + '.' + k)
    positive(c['threads'], 'threads', True)
    for k in ['seed', 'data_seed', 'probe_seed', 'analysis_seed']:
        if type(c[k]) is not int or c[k] < 0:
            raise ValueError(f'{k} must be a nonnegative integer')
    if c['dtype'] not in ['float64', 'float32'] or c['device'] not in ['cpu', 'cuda']:
        raise ValueError('Supported dtype/device: float64 or float32; cpu or cuda')
    if c['model']['kind'] not in ['mlp', 'vit', 'transformer'] or c['data']['dataset'] not in ['synthetic', 'mnist']:
        raise ValueError('Unknown model or dataset')
    if c['model']['kind'] == 'vit':
        v, m = c['vit'], c['model']
        if v['image_size'] % v['patch_size'] or m['input_dim'] != v['image_size'] ** 2:
            raise ValueError('ViT patch size must divide image size, and input dimension must equal image size squared')
        if m['width'] % v['heads']:
            raise ValueError('ViT heads must divide model width')
        if c['data']['dataset'] == 'mnist' and c['data']['resolution'] != v['image_size']:
            raise ValueError('ViT MNIST resolution must equal image size')
    if c['partition'] not in ['tensor', 'block'] or c['damping']['metric'] not in ['identity', 'soap']:
        raise ValueError('Unknown partition or damping metric')
    if not c['damping']['gammas'] or not c['sketch']['ranks']:
        raise ValueError('Damping and rank grids cannot be empty')
    for v in c['damping']['gammas']:
        positive(v, 'gamma')
    for v in c['sketch']['ranks']:
        positive(v, 'rank', True)
    if sorted(set(c['sketch']['ranks'])) != c['sketch']['ranks']:
        raise ValueError('Sketch ranks must be sorted and unique')
    if len(c['soap']['betas']) != 2 or any(not 0 <= b < 1 for b in c['soap']['betas']):
        raise ValueError('SOAP betas must lie in [0,1)')
    if not (c['soap']['shampoo_beta'] == -1 or 0 <= c['soap']['shampoo_beta'] < 1):
        raise ValueError('shampoo_beta must be -1 or in [0,1)')
    if c['soap']['weight_decay'] != 0 or c['soap']['normalize_grads'] or c['soap']['merge_dims']:
        raise ValueError('Core comparisons disable weight decay, normalization and dimension merging')
    if c['soap']['data_format'] != 'channels_first':
        raise ValueError('SOAP adapter uses channels_first')
    if not 0 < c['search']['armijo'] < 1 or c['search']['minimum'] > c['search']['initial']:
        raise ValueError('Invalid backtracking settings')
    exponents = c['search']['grid_exponents']
    if not exponents or any(type(v) is not int for v in exponents) or sorted(set(exponents)) != exponents or 0 not in exponents:
        raise ValueError('Direction grid needs sorted unique integer exponents including zero')
    steps = c['training']['snapshots']
    if sorted(set(steps)) != steps or any(type(s) is not int or s < 1 or s > c['training']['steps'] for s in steps):
        raise ValueError('Snapshots must be unique sorted consumed iterations within the run')
    if c['training']['method'] not in ['soap', 'sketch_diagonal', 'sketch_coupled']:
        raise ValueError('Unknown training method')
    if c['training']['policy'] not in ['native', 'training_backtracking']:
        raise ValueError('Unknown training policy')
    if c['training']['method'] != 'soap' and c['training']['policy'] != 'training_backtracking':
        raise ValueError('Algorithm A training requires training_backtracking')
    if c['training']['gamma'] not in c['damping']['gammas'] or c['training']['rank'] not in c['sketch']['ranks']:
        raise ValueError('Training gamma/rank must be declared in the grids')
    if c['data']['dataset'] == 'mnist':
        if c['model']['input_dim'] != c['data']['resolution'] ** 2 or c['model']['classes'] != 10:
            raise ValueError('MNIST input dimension must equal resolution squared and classes must be ten')
        if sum(c['data'][k] for k in ['train_size', 'validation_size', 'analysis_size']) != 60000:
            raise ValueError('MNIST pools must partition all 60000 official training examples')
        if c['data']['analysis_size'] < (c['data']['panel_batches'] + 1) * c['data']['batch_size']:
            raise ValueError('Analysis pool cannot supply disjoint construction and panel batches')
    if not math.isfinite(c['model']['branch_scale']):
        raise ValueError('branch_scale must be finite')
    return c


def load_config(path):
    return validate(merge_strict(DEFAULT, yaml.safe_load(Path(path).read_text()) or {}))
