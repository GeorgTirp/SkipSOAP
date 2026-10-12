# Third-party provenance

SOAP is vendored without edits from the authors' official repository:
https://github.com/nikhilvyas/SOAP/tree/a1e553530fde97d0e6b307d7c82ac6d38b072340

Pinned commit: `a1e553530fde97d0e6b307d7c82ac6d38b072340`.
Files `third_party/soap/soap.py` and `third_party/soap/LICENSE` are byte-identical to
that checkout. The MIT license preserves Copyright (c) 2024 Nikhil Vyas.
The upstream source also acknowledges PyTorch AdamW and GaLore modifications.

The adapter subclasses SOAP and captures the first update-time `project_back` call
for each tensor. It calls upstream `step` unchanged on cloned parameters/state.
Upstream's denominator is `sqrt(exp_avg_sq) + eps`; bias correction is the separate
scalar `sqrt(1-beta2**step)/(1-beta1**step)`. The first call sets bases and skips the
parameter update. Vectors use no Shampoo basis by default, but retain Adam moments.
Moments and bases change during the post-update maintenance call.

Upstream allocates second-moment factors using the default dtype and computes
eigen/QR bases in float32. The adapter and reference audit use the same scoped default
dtype as the parameters. Stored float64 bases thus have float32 orthogonality error.
The frozen positive map uses the actual saved bases; its inverse uses per-axis linear
solves instead of assuming exact orthogonality. This changes neither baseline SOAP
nor its proposed displacement. The map is self-adjoint/SPD and its inverse is tested.

PyTorch, NumPy, SciPy, Matplotlib, PyYAML, pytest and optional torchvision are installed
dependencies, not vendored source. Their exact tested versions are recorded separately.

Primary references inspected during implementation:

- PyTorch 2.14 `torch.func`: https://docs.pytorch.org/docs/2.14/func.html
- Torchvision MNIST and explicit download flag: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
- SOAP Algorithm 3: https://arxiv.org/html/2409.11321v2
- Modular curvature backpropagation: https://proceedings.mlr.press/v108/dangel20a.html
- Full-GN language-model study: https://arxiv.org/html/2510.09378v2
- K-FAC: https://proceedings.mlr.press/v37/martens15.html

These establish ingredients and context, not novelty of this prototype. In particular,
transport identities, Fisher/GGN probes, Woodbury, Gauss–Seidel and scalar searches
are established operations.
