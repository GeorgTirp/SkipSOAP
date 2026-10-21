"""Times one float64 GGN action of M1d6 at batch 2048 (random inputs), with and without SN; DEVIATIONS.md D1."""
import sys, time
sys.path.insert(0, 'studies/cifar_ggn_20261012/scripts')
import common, torch, cifar, micro
from residual_curvature.models import Registry
dev = torch.device('cuda')
x = torch.randn(2048, 3, 32, 32, generator=torch.Generator().manual_seed(0), dtype=torch.float64).to(dev)
for sn in (None, 1.):
    for m in (128, 256):
        model = cifar.make_cifar_model('M1d6', 1, dev, dtype=torch.float64, sn_bound=sn)
        reg = Registry(model)
        flat = reg.flatten(dict(model.named_parameters())).detach()
        act = micro.ggn_action(model, reg, flat, x, m)
        v = torch.randn_like(flat)
        torch.cuda.reset_peak_memory_stats(); act(v); torch.cuda.synchronize()
        t = time.perf_counter(); act(v); torch.cuda.synchronize()
        print('sn', sn, 'micro', m, 'seconds/action %.2f' % (time.perf_counter()-t), 'peak GiB %.2f' % (torch.cuda.max_memory_allocated()/2**30), flush=True)
        del model, act
