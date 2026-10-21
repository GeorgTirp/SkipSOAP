"""CIFAR-10 split and ViT for the cost measurement C1 (PROTOCOL.md, section 3).

The split draws 45,000 training and 5,000 validation examples from the official training set with a seeded
permutation. The official test split is reserved and never accessed. The ViT takes 3 x 32 x 32 images, reuses the
lab's TransformerBlock and keeps the lab's parameter names (input.*, blocks.<i>.*, norm, head), so
residual_curvature.models.Registry works unchanged.
"""
import torch
from torch import nn

from residual_curvature.data import fingerprint
from residual_curvature.models.vit import TransformerBlock

SPLIT_SEED, TRAIN_SIZE, VALIDATION_SIZE = 101, 45000, 5000
MODELS = {
    'M1': {'width': 64, 'expansion': 256, 'blocks': 4, 'heads': 4},
    'M2': {'width': 128, 'expansion': 512, 'blocks': 6, 'heads': 4},
}


def cifar10(root='datasets'):
    """Official CIFAR-10 training set as (x uint8 (N, 3, 32, 32), y int64 (N,), manifest); no download."""
    from torchvision.datasets import CIFAR10
    data = CIFAR10(root, train=True, download=False)
    raw = torch.from_numpy(data.data).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(data.targets, dtype=torch.int64)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(SPLIT_SEED))
    training, validation = order[:TRAIN_SIZE], order[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
    scaled = raw[training].double()/255
    mean, std = scaled.mean((0, 2, 3)), scaled.std((0, 2, 3))
    manifest = {'training': training.tolist(), 'validation': validation.tolist(), 'data_sha256': fingerprint(raw, y),
                'generator_seed': SPLIT_SEED, 'official_test': 'reserved, not accessed',
                'normalization': {'from': 'training split', 'mean': mean.tolist(), 'std': std.tolist()}}
    return raw, y, manifest


def normalize(raw, manifest, dtype=torch.float32):
    """(raw / 255 - mean) / std per channel, with the training-split statistics in the manifest."""
    n = manifest['normalization']
    mean = torch.tensor(n['mean'], dtype=torch.float64).reshape(1, 3, 1, 1)
    std = torch.tensor(n['std'], dtype=torch.float64).reshape(1, 3, 1, 1)
    return ((raw.double()/255-mean)/std).to(dtype)


class CifarPatchInput(nn.Module):
    def __init__(self, image_size, patch_size, channels, width):
        super().__init__()
        self.image_size, self.patch_size, self.channels = image_size, patch_size, channels
        self.patches_per_side = image_size // patch_size
        self.token_count = self.patches_per_side ** 2 + 1
        self.cls_token = nn.Parameter(torch.zeros(1, width))
        self.position_embedding = nn.Parameter(torch.empty(self.token_count, width))
        self.projection = nn.Linear(channels * patch_size ** 2, width)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.position_embedding, std=.02)

    def forward(self, x):
        if x.shape[1:] != (self.channels, self.image_size, self.image_size):
            raise ValueError('CifarViT requires images of shape (batch, channels, size, size)')
        batch, side, patch = x.shape[0], self.patches_per_side, self.patch_size
        patches = x.reshape(batch, self.channels, side, patch, side, patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, side ** 2, self.channels * patch ** 2)
        h = self.projection(patches)
        cls = self.cls_token.unsqueeze(0).expand(batch, -1, -1)
        return torch.cat((cls, h), dim=1) + self.position_embedding.unsqueeze(0)


class CifarViT(nn.Module):
    def __init__(self, width, expansion, blocks, heads, classes=10, epsilon=1e-5, branch_scale=1.0,
                 image_size=32, patch_size=4, channels=3):
        super().__init__()
        if image_size % patch_size or width % heads:
            raise ValueError('patch size must divide the image size and heads must divide the width')
        self.input = CifarPatchInput(image_size, patch_size, channels, width)
        self.blocks = nn.ModuleList([TransformerBlock(width, expansion, heads, epsilon, branch_scale)
                                     for _ in range(blocks)])
        self.norm = nn.LayerNorm(width, eps=epsilon)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h[:, 0]))


def make_cifar_model(name, seed, device, dtype=torch.float32):
    """Initialize in float64 on the CPU under `seed`, then round and move (as the MNIST units)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = CifarViT(**MODELS[name])
        finally:
            torch.set_default_dtype(previous)
    return model.to(device=device, dtype=dtype)
