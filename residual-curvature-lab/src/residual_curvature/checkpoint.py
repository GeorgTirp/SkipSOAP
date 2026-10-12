import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def capture_rng():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng):
    torch.set_rng_state(rng['torch'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    if rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])


def save(path, model, optimizer_state, context, step, accepted, order, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'version': 1, 'model': model.state_dict(), 'optimizer': optimizer_state,
             'used_context': context, 'step': step, 'accepted': accepted,
             'data_order': order.state_dict(), 'rng': capture_rng(), 'config': copy.deepcopy(config),
             'probe_stream': {'seed': config['probe_seed'], 'next_iteration': step + 1},
             'analysis_stream': {'seed': config['analysis_seed'], 'policy': 'derived from snapshot and repeat'}}
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load(path, model, order, config):
    # Only load checkpoints created by this trusted local experiment.
    # RNG byte tensors and DataOrder stay on CPU; model/optimizer adapters move
    # their own state to the requested device on restoration/proposal.
    state = torch.load(path, map_location='cpu', weights_only=False)
    saved_config, requested_config = copy.deepcopy(state['config']), copy.deepcopy(config)
    if saved_config['model']['kind'] == requested_config['model']['kind'] == 'mlp':
        # Adding inactive ViT defaults must not invalidate earlier MLP snapshots.
        # All explicitly supplied settings and all active settings still match.
        from .config import DEFAULT
        saved_config.setdefault('vit', copy.deepcopy(DEFAULT['vit']))
        requested_config.setdefault('vit', copy.deepcopy(DEFAULT['vit']))
    if state['version'] != 1 or saved_config != requested_config:
        raise ValueError('Checkpoint version/configuration mismatch')
    model.load_state_dict(state['model'])
    order.load_state_dict(state['data_order'])
    restore_rng(state['rng'])
    return state
