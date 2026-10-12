"""How far each residual block is from the identity, measured at the data (for the depth test, N14).

For block k with input h_k (captured from a forward pass on x) and Jacobian R_k = d block / d h, the
transport argument needs A_k = R_k - I small. Two estimates per block, both at the actual block inputs:

  rms       sqrt(sum_v |A_k v|^2 / sum_v |v|^2) over `probes` Gaussian tangents v: an unbiased-ratio
            estimate of the RMS singular value of A_k (|A_k|_F^2 / dimension). Forward-mode only.
  spectral  largest singular value of A_k over the batch (A_k is block diagonal across examples), by
            `power_steps` power iterations with one JVP and one VJP each; omitted if power_steps = 0.

Both use the stream dimension of one example (tokens x width for the ViT). Works for any modules in
`blocks` that map the stream to itself (default: model.blocks).
"""
import torch
from torch.func import jvp, vjp


def block_jacobian_deviation(model, x, blocks=None, probes=8, power_steps=0, generator=None):
    blocks = list(model.blocks if blocks is None else blocks)
    inputs = []
    handles = [block.register_forward_hook(lambda module, args, output: inputs.append(args[0].detach())) for block in blocks]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    out = []
    for block, h in zip(blocks, inputs):
        def deviation(t, block=block):
            return block(t)-t
        image = norm = 0.
        for _ in range(probes):
            v = torch.randn(h.shape, generator=generator, dtype=h.dtype, device=h.device)
            image += float(jvp(deviation, (h,), (v,))[1].detach().square().sum())
            norm += float(v.square().sum())
        row = {'block': len(out), 'rms': (image/norm)**.5}
        if power_steps:
            v = torch.randn(h.shape, generator=generator, dtype=h.dtype, device=h.device)
            v = v/v.norm()
            _, pullback = vjp(deviation, h)
            for _ in range(power_steps):
                w = pullback(jvp(deviation, (h,), (v,))[1])[0].detach()
                norm_w = w.norm()
                if float(norm_w) == 0:
                    break
                v = w/norm_w
            row['spectral'] = float(jvp(deviation, (h,), (v,))[1].detach().norm())
        out.append(row)
    return out
