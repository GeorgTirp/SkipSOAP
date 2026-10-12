from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.func import functional_call


class ResidualBlock(nn.Module):
    def __init__(self, width, expansion, epsilon, branch_scale):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=epsilon)
        self.w1 = nn.Linear(width, expansion)
        self.w2 = nn.Linear(expansion, width)
        self.branch_scale = branch_scale

    def forward(self, h):
        return h + self.branch_scale * self.w2(torch.nn.functional.gelu(self.w1(self.norm(h))))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=6, width=4, expansion=8, blocks=2, classes=3,
                 epsilon=1e-5, branch_scale=1.0):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResidualBlock(width, expansion, epsilon, branch_scale)
                                     for _ in range(blocks)])
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple
    count: int
    start: int
    stop: int
    role: str
    stage: int
    orientation: str

    @property
    def slice(self):
        return slice(self.start, self.stop)


class Registry:
    def __init__(self, model):
        self.specs = []
        offset = 0
        normalizations = {f'{name}.{parameter}' for name, module in model.named_modules()
                          if isinstance(module, nn.LayerNorm)
                          for parameter, _ in module.named_parameters(recurse=False)}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            stage = int(name.split('.')[1]) if name.startswith('blocks.') else (-1 if name.startswith('input.') else len(model.blocks))
            embedding = name.endswith(('cls_token', 'position_embedding'))
            role = ('normalization' if name in normalizations else 'embedding' if embedding
                    else 'bias' if name.endswith('bias') else 'linear_weight')
            self.specs.append(TensorSpec(name, tuple(p.shape), p.numel(), offset,
                                        offset + p.numel(), role, stage,
                                        'token,feature' if embedding else 'out,in' if p.ndim == 2 else 'feature'))
            offset += p.numel()
        self.size = offset

    def flatten(self, params):
        return torch.cat([params[s.name].reshape(-1) for s in self.specs])

    def unflatten(self, flat):
        if flat.numel() != self.size:
            raise ValueError('Incorrect registered parameter dimension')
        return {s.name: flat[s.slice].reshape(s.shape) for s in self.specs}

    def groups(self, partition='tensor'):
        if partition == 'tensor':
            return [s.slice for s in self.specs]
        if partition != 'block':
            raise ValueError('partition must be tensor or block')
        stages = sorted({s.stage for s in self.specs})
        groups = []
        for stage in stages:
            specs = [s for s in self.specs if s.stage == stage]
            start, stop = specs[0].start, specs[-1].stop
            if stop - start != sum(s.count for s in specs):
                raise ValueError('Block partition requires contiguous stage parameters')
            groups.append(slice(start, stop))
        return groups

    def json(self):
        return [asdict(s) for s in self.specs]


def make_model(config):
    m = config['model'].copy()
    kind = m.pop('kind')
    if kind == 'vit':
        from .vit import SmallViT
        model_type = SmallViT
        m.update(config.get('vit', {}))
    elif kind == 'mlp':
        model_type = ResidualMLP
    else:
        raise NotImplementedError('Stage 4 decoder is not implemented; see STATUS.md')
    # Isolate initialization from the caller's training RNG.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.manual_seed(config['seed'])
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(getattr(torch, config['dtype']))
            model = model_type(**m).to(device=config['device'])
        finally:
            torch.set_default_dtype(previous_dtype)
    return model


def functional_logits(model, registry, x):
    return lambda flat: functional_call(model, registry.unflatten(flat), (x,))
