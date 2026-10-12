"""Small pre-normalized vision transformer with explicit attention operations.

All patch projections share parameters across tokens. Ordinary matrix products
and softmax keep forward/reverse differentiation available for exact GGN audits.
"""
import math

import torch
from torch import nn


class PatchInput(nn.Module):
    def __init__(self, image_size, patch_size, width):
        super().__init__()
        self.image_size, self.patch_size = image_size, patch_size
        self.patches_per_side = image_size // patch_size
        self.token_count = self.patches_per_side ** 2 + 1
        # Two-dimensional tensors avoid special embedding reshaping in SOAP.
        self.cls_token = nn.Parameter(torch.zeros(1, width))
        self.position_embedding = nn.Parameter(torch.empty(self.token_count, width))
        self.projection = nn.Linear(patch_size ** 2, width)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.position_embedding, std=.02)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.image_size ** 2:
            raise ValueError('ViT requires flattened single-channel images')
        batch, side, patch = x.shape[0], self.patches_per_side, self.patch_size
        patches = x.reshape(batch, side, patch, side, patch)
        patches = patches.permute(0, 1, 3, 2, 4).reshape(batch, side ** 2, patch ** 2)
        h = self.projection(patches)
        cls = self.cls_token.unsqueeze(0).expand(batch, -1, -1)
        return torch.cat((cls, h), dim=1) + self.position_embedding.unsqueeze(0)


class ExplicitAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        def split(projection):
            return projection(x).reshape(batch, tokens, self.heads, self.head_width).transpose(1, 2)
        q, k, v = split(self.query), split(self.key), split(self.value)
        probabilities = ((q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)).softmax(-1)
        mixed = (probabilities @ v).transpose(1, 2).reshape(batch, tokens, width)
        return self.output(mixed)


class TransformerBlock(nn.Module):
    def __init__(self, width, expansion, heads, epsilon, branch_scale):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=epsilon)
        self.attention = ExplicitAttention(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=epsilon)
        self.w1 = nn.Linear(width, expansion)
        self.w2 = nn.Linear(expansion, width)
        self.branch_scale = branch_scale

    def forward(self, h):
        h = h + self.branch_scale * self.attention(self.norm1(h))
        return h + self.branch_scale * self.w2(torch.nn.functional.gelu(self.w1(self.norm2(h))))


class SmallViT(nn.Module):
    def __init__(self, input_dim=784, width=16, expansion=32, blocks=2, classes=10,
                 epsilon=1e-5, branch_scale=1.0, image_size=28, patch_size=7, heads=2):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError('ViT patch size must divide its positive image size')
        if input_dim != image_size ** 2:
            raise ValueError('ViT input dimension must equal image size squared')
        if heads <= 0 or width % heads:
            raise ValueError('ViT head count must divide its positive width')
        self.input = PatchInput(image_size, patch_size, width)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, expansion, heads, epsilon, branch_scale)
            for _ in range(blocks)])
        self.norm = nn.LayerNorm(width, eps=epsilon)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h[:, 0]))
