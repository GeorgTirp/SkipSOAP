"""(Copied unchanged from studies/ladder_mechanism_20261013/scripts/states.py, via bulk_krylov_20261014.) Replays of cifar_ggn_20261012's soap and topfree units (PROTOCOL.md, section 2), snapshots and SOAP contexts.

The replay steps cifar_ggn's own Arm (`ggn_train`, loaded under its own name) with the run's data order, augmentation
and probe generator, exactly as that study's run loop does for these arms. A snapshot after step t holds the
parameters, the optimizer state (PARSEC-L's CPU generator state kept on the CPU), the arm's probe-generator state, and
the data-order and augmentation states, so that restoring it and taking the next batch gives batch t + 1.
"""
import copy
import importlib.util
import sys

import common

import torch  # noqa: E402

from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GGN_TRAIN = load_module('ggn_train', common.LIB/'train.py')
BATCH, MICRO = GGN_TRAIN.BATCH, GGN_TRAIN.MICRO
EVAL = common.Path.home()/'skipsoap-study-data'/'cifar_ggn_20261012'/'runs'/'20261012T110529-d747f9fc'
LRS = {'soap': 1e-2, 'topfree': 3e-2}


def to(value, device):
    """Tensors inside nested containers copied to `device`, except PARSEC-L's generator state (must stay on the CPU)."""
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {k: (copy.deepcopy(v) if k == 'generator' else to(v, device)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to(v, device) for v in value)
    return copy.deepcopy(value)


class Replay:
    def __init__(self, arm, seed, device, data=None):
        t = GGN_TRAIN
        self.arm_name, self.seed, self.device, self.t = arm, seed, device, 0
        self.data = data or t.Data(device)
        self.spec = {'arm': arm, 'seed': seed, 'lr': LRS[arm], 'measure': False}
        self.arm = t.Arm(self.spec, device)
        self.order = DataOrder(list(range(len(self.data.manifest['training']))), t.SEEDS['order']+1000*seed)
        self.augment = torch.Generator().manual_seed(t.SEEDS['augment']+seed)

    def flat(self):
        return self.arm.flat()

    def next_batch(self):
        return self.data.batch(self.order.take(BATCH), self.augment)

    def following(self, k):
        """The batch k + 1 places after the last one taken (k = 0: the next one), unaugmented, without consuming."""
        twin = DataOrder([0], 0)
        twin.load_state_dict(copy.deepcopy(self.order.state_dict()))
        for _ in range(k):
            twin.take(BATCH)
        return self.data.batch(twin.take(BATCH))

    def step(self, x, y):
        self.t += 1
        return self.arm.step(self.t, x, y, None)

    def snapshot(self):
        return {'arm': self.arm_name, 'seed': self.seed, 't': self.t, 'parameters': self.flat().cpu(),
                'optimizer': to(self.arm.optimizer.state_dict(), 'cpu'), 'probe_generator': self.arm.probe_generator.get_state(),
                'order': copy.deepcopy(self.order.state_dict()), 'augment': self.augment.get_state()}

    def restore(self, snapshot):
        if (snapshot['arm'], snapshot['seed']) != (self.arm_name, self.seed):
            raise ValueError('snapshot of another unit')
        self.t = snapshot['t']
        commit(self.arm.model, self.arm.registry, snapshot['parameters'].to(self.device))
        self.arm.optimizer.load_state_dict(to(snapshot['optimizer'], self.device))
        self.arm.probe_generator.set_state(snapshot['probe_generator'])
        self.order.load_state_dict(copy.deepcopy(snapshot['order']))
        self.augment.set_state(snapshot['augment'])


def names(optimizer, registry):
    params = [p for g in optimizer.param_groups for p in g['params']]
    if len(params) != len(registry.specs) or any(tuple(p.shape) != tuple(s.shape) for p, s in zip(params, registry.specs)):
        raise ValueError('optimizer and registry parameters differ')
    return {id(p): s.name for p, s in zip(params, registry.specs)}


def contexts(optimizer, registry):
    """{name: {'basis': Q, 'denominator': sqrt(v) + eps}} of the state's SOAP."""
    by_id = names(optimizer, registry)
    return {by_id[id(p)]: {'basis': optimizer.state[p]['Q'], 'denominator': optimizer.state[p]['exp_avg_sq'].sqrt().add(g['eps'])}
            for g in optimizer.param_groups for p in g['params']}
