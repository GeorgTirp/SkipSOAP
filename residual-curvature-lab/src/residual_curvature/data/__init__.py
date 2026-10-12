import hashlib

import torch


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def synthetic(c):
    gen = torch.Generator().manual_seed(c['data_seed'])
    m, d = c['model'], c['data']
    dtype = getattr(torch, c['dtype'])
    teacher = torch.randn(m['input_dim'], m['classes'], generator=gen, dtype=dtype)
    n = d['samples'] + (d['panel_batches'] + 1) * d['batch_size']
    x = torch.randn(n, m['input_dim'], generator=gen, dtype=dtype)
    y = (x @ teacher + .1 * torch.randn(n, m['classes'], generator=gen, dtype=dtype)).argmax(-1)
    ids = torch.arange(n)
    return x, y, {'training': ids[:d['samples']].tolist(), 'validation': [],
                  'analysis': ids[d['samples']:].tolist(), 'data_sha256': fingerprint(x, y),
                  'generator_seed': c['data_seed'], 'transformation': 'Gaussian teacher classification'}


def mnist(c, download=False):
    try:
        from torchvision.datasets import MNIST
    except ImportError as e:
        raise RuntimeError('Install the mnist extra before using MNIST') from e
    dataset = MNIST(c['data']['root'], train=True, download=download)
    raw, y = dataset.data, dataset.targets
    x = torch.nn.functional.interpolate(raw[:, None].to(getattr(torch, c['dtype'])) / 255,
                                        size=(c['data']['resolution'],) * 2, mode='area').flatten(1)
    gen = torch.Generator().manual_seed(c['data_seed'])
    order = torch.randperm(len(y), generator=gen)
    a, b = c['data']['train_size'], c['data']['validation_size']
    return x, y, {'training': order[:a].tolist(), 'validation': order[a:a+b].tolist(),
                  'analysis': order[a+b:].tolist(), 'data_sha256': fingerprint(raw, y),
                  'generator_seed': c['data_seed'], 'official_test': 'reserved, not accessed',
                  'transformation': {'resize': 'area', 'resolution': c['data']['resolution'], 'scale': 'divide by 255'}}


def dataset(c):
    return synthetic(c) if c['data']['dataset'] == 'synthetic' else mnist(c, download=False)


class DataOrder:
    def __init__(self, ids, seed):
        self.ids = torch.tensor(ids)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = self.ids[torch.randperm(len(ids), generator=self.generator)]
        self.position = 0

    def take(self, batch_size):
        selected = []
        while len(selected) < batch_size:
            remaining = min(batch_size - len(selected), len(self.order) - self.position)
            selected.extend(self.order[self.position:self.position+remaining].tolist())
            self.position += remaining
            if self.position == len(self.order):
                self.order = self.ids[torch.randperm(len(self.ids), generator=self.generator)]
                self.position = 0
        return selected

    def state_dict(self):
        return {'ids': self.ids, 'order': self.order, 'position': self.position, 'rng': self.generator.get_state()}

    def load_state_dict(self, state):
        self.ids, self.order, self.position = state['ids'], state['order'], state['position']
        self.generator.set_state(state['rng'])
