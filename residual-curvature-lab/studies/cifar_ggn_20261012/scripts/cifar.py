"""CIFAR-10 split, augmentation and ViT (PROTOCOL.md, section 2). Copied from studies/gpu_port_20261012/scripts/cifar.py
(same split, normalization and ViT class); adds the depth-6 model M1d6 and seeded crop/flip augmentation.

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
    'M1d6': {'width': 64, 'expansion': 256, 'blocks': 6, 'heads': 4},
    'M2': {'width': 128, 'expansion': 512, 'blocks': 6, 'heads': 4},
}
PAD = 4


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

    def forward(self, x, states=False):
        """Logits; with states=True the residual stream after the embedding and after each block, and the head input."""
        h = self.input(x)
        stream = [h]
        for block in self.blocks:
            h = block(h)
            stream.append(h)
        if states:
            return stream, self.norm(h[:, 0])
        return self.head(self.norm(h[:, 0]))


SN_LAYERS = ['attention.query', 'attention.key', 'attention.value', 'attention.output', 'w1', 'w2']


def spectral_norms(weights):
    """Exact largest singular values (torch.linalg.matrix_norm, ord 2: an SVD per matrix), with exact first and second
    derivatives. A batched float64 Gram-eigenvalue variant was 15x faster alone but slower inside the GGN step, whose
    Jacobian passes run its float64 backward vectorized over 640 output directions (pre-run dry run: 80 s against
    56 s per step)."""
    return [torch.linalg.matrix_norm(w, ord=2) for w in weights]


class CifarViTSN(CifarViT):
    """CifarViT with spectral normalization W_eff = W min(1, c / sigma(W)) on the residual-branch weights (attention
    query/key/value/output, MLP w1/w2), as studies/sn_curvature_20261008 (exact sigma). The parameters are the raw
    weights under the plain model's names; every forward computes the effective weights (spectral_norms) and runs the
    plain network with them, so functional_call over the raw parameters differentiates through SN."""

    def __init__(self, bound, **kwargs):
        super().__init__(**kwargs)
        self.bound = float(bound)
        self.sn_names = [f'blocks.{i}.{layer}.weight' for i in range(len(self.blocks)) for layer in SN_LAYERS]
        self._effective = False

    def raw(self, name):
        """The current tensor under a parameter name (under functional_call it is a plain tensor, not a Parameter)."""
        module, attribute = name.rsplit('.', 1)
        return getattr(self.get_submodule(module), attribute)

    def effective_weights(self):
        raws = [self.raw(n) for n in self.sn_names]
        return {n: w*torch.clamp(self.bound/s, max=1.) for n, w, s in zip(self.sn_names, raws, spectral_norms(raws))}

    def forward(self, x, states=False):
        if self._effective:
            return super().forward(x, states)
        weights = self.effective_weights()
        self._effective = True
        try:
            return torch.func.functional_call(self, weights, (x,), {'states': states}, strict=False)
        finally:
            self._effective = False


def make_cifar_model(name, seed, device, dtype=torch.float32, sn_bound=None):
    """Initialize in float64 on the CPU under `seed`, then round and move (as the MNIST units). With sn_bound, the
    same initialization as CifarViTSN."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = CifarViT(**MODELS[name]) if sn_bound is None else CifarViTSN(sn_bound, **MODELS[name])
        finally:
            torch.set_default_dtype(previous)
    return model.to(device=device, dtype=dtype)


def augment(images, generator):
    """Random crop with zero padding PAD and random horizontal flip, per image, on the images' device.

    Offsets and flips are drawn on the CPU from `generator`, so every device draws the same augmentation.
    `images` are normalized (zero padding is the training-split mean)."""
    n, _, size, _ = images.shape
    offsets = torch.randint(0, 2*PAD+1, (n, 2), generator=generator)
    flips = torch.rand(n, generator=generator) < .5
    padded = torch.nn.functional.pad(images, (PAD,)*4)
    rows = (offsets[:, 0, None]+torch.arange(size)).to(images.device)
    cols = (offsets[:, 1, None]+torch.arange(size)).to(images.device)
    out = padded[torch.arange(n, device=images.device)[:, None, None, None], torch.arange(images.shape[1], device=images.device)[None, :, None, None],
                 rows[:, None, :, None], cols[:, None, None, :]]
    return torch.where(flips.to(images.device)[:, None, None, None], out.flip(-1), out)
