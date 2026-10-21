"""States for the offline tests (PROTOCOL.md, section 2): deterministic SOAP replays with snapshots, the stored
top-free checkpoints, SOAP's contexts for the damping metric, and SOAP's refresh applied now.

The replay is cifar_sn_soap_20261012's runner (`sns_train`): its soap arm is cifar_ggn_20261012's soap path, and its
units reproduced that study's bitwise. A snapshot after step t holds the parameters, the optimizer state, and the
data-order and augmentation-generator states, so that restoring it and taking the next batch gives batch t + 1.
"""
import copy
import importlib.util
import sys

import common

import torch  # noqa: E402

from gpu_parsec.optimizer import NEVER  # noqa: E402
from residual_curvature.data import DataOrder  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402

GGN_EVAL = common.Path.home()/'skipsoap-study-data'/'cifar_ggn_20261012'/'runs'/'20261012T110529-d747f9fc'


def load_module(name, path):
    """A study runner under its own module name (its `common` import resolves to this study's; only its arms and
    data are used)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SNS_TRAIN = load_module('sns_train', common.SNS/'train.py')
GGN_TRAIN = load_module('ggn_train', common.LIB/'train.py')
BATCH, MICRO = SNS_TRAIN.BATCH, SNS_TRAIN.MICRO


def to(value, device):
    """Tensors inside nested lists, tuples and dicts moved to `device` (copies)."""
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {k: to(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to(v, device) for v in value)
    return copy.deepcopy(value)


class Replay:
    """One SOAP unit (arm `soap`, lr 1e-2) of the evaluation runs, stepped with the run's own batches."""

    def __init__(self, seed, device, data=None):
        t = SNS_TRAIN
        self.seed, self.device, self.t = seed, device, 0
        self.data = data or t.Data(device)
        self.arm = t.Arm(t.spec_of('soap', seed, t.SOAP_LR), device)
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
        return self.arm.step(self.t, x, y)

    def snapshot(self):
        return {'seed': self.seed, 't': self.t, 'parameters': self.flat().cpu(),
                'optimizer': to(self.arm.optimizer.state_dict(), 'cpu'),
                'order': copy.deepcopy(self.order.state_dict()), 'augment': self.augment.get_state()}

    def restore(self, snapshot):
        if snapshot['seed'] != self.seed:
            raise ValueError('snapshot of another seed')
        self.t = snapshot['t']
        commit(self.arm.model, self.arm.registry, snapshot['parameters'].to(self.device))
        self.arm.optimizer.load_state_dict(to(snapshot['optimizer'], self.device))
        self.order.load_state_dict(copy.deepcopy(snapshot['order']))
        self.augment.set_state(snapshot['augment'])

    def freeze_bases(self):
        """Switch SOAP's periodic refresh off (Part B windows)."""
        for state in self.arm.optimizer.state.values():
            if 'precondition_frequency' in state:
                state['precondition_frequency'] = NEVER


def soap_contexts(optimizer, registry):
    """{name: {'basis': Q list, 'denominator': sqrt(v) + eps}} of a SOAP state, for SOAPMetric and the whitening of
    ggn.py (the preconditioner P = Q diag(1/den) Q^T that SOAP applies, up to its step size)."""
    names = optimizer_names(optimizer, registry)
    return {names[id(p)]: {'basis': optimizer.state[p]['Q'], 'denominator': optimizer.state[p]['exp_avg_sq'].sqrt().add(group['eps'])}
            for group in optimizer.param_groups for p in group['params']}


def optimizer_names(optimizer, registry):
    """{id(parameter): registry name}: the optimizer holds the model's parameters in the registry's order."""
    params = [p for g in optimizer.param_groups for p in g['params']]
    if len(params) != len(registry.specs) or any(tuple(p.shape) != tuple(s.shape) for p, s in zip(params, registry.specs)):
        raise ValueError('optimizer and registry parameters differ')
    return {id(p): spec.name for p, spec in zip(params, registry.specs)}


def refresh_now(optimizer):
    """SOAP's scheduled refresh applied immediately, without the factor update (upstream update_preconditioner,
    lines 273-307: momentum back to original coordinates, get_orthogonal_matrix_QR, which also reorders the second
    moments, momentum into the new bases)."""
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if state.get('Q') is None:
                continue
            state['exp_avg'] = optimizer.project_back(state['exp_avg'], state)
            state['Q'] = optimizer.get_orthogonal_matrix_QR(state, group['max_precond_dim'], group['merge_dims'])
            state['exp_avg'] = optimizer.project(state['exp_avg'], state)


def topfree(seed, step, device):
    """(model, registry, PARSEC-L optimizer, checkpoint) of cifar_ggn's top-free unit at a stored checkpoint."""
    t = GGN_TRAIN
    spec = {'arm': 'topfree', 'lr': 3e-2, 'seed': seed}
    checkpoint = torch.load(GGN_EVAL/f'topfree__lr0.03__s{seed}'/'checkpoints'/f'step-{step:04d}.pt', weights_only=False)
    model = t.build_model(spec, device)
    from residual_curvature.models import Registry
    registry = Registry(model)
    commit(model, registry, checkpoint['parameters'].to(device))
    optimizer = t.make_optimizer(spec, model)
    state = to({k: v for k, v in checkpoint['optimizer'].items() if k != 'parsec'}, device)
    state['parsec'] = copy.deepcopy(checkpoint['optimizer']['parsec'])  # the generator state must stay on the CPU
    optimizer.load_state_dict(state)
    return model, registry, optimizer, checkpoint
