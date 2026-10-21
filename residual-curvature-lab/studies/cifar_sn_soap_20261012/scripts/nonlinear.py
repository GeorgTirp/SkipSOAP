"""Nonlinearity, curvature-persistence and representation-drift measures along the step taken (PROTOCOL.md, section 5,
measures A-D). Added in this study; cifar_ggn_20261012's measure.py is imported unchanged.

A. Error split along d on a batch (x, y), mean cross-entropy, computed from logits only. With f = f(theta),
   Jd (forward mode), f+ = f(theta + d) and the logit-linearization remainder r = f+ - f - Jd:
       actual = L(f+) - L(f),   linearized = L(f + Jd) - L(f)  (linearized network, exact loss),
       gd = mean (dl/df) . Jd = g'd,   full = mean Jd' (diag p - p p') Jd = d'Cd,
       actual - (gd + full/2) = [actual - linearized] + [linearized - gd - full/2]
                                 network nonlinearity    loss Taylor remainder (cross-entropy beyond 2nd order).
   Also the linearization error |r| / |Jd| (Frobenius over the batch) and the residual term 2 mean (dl/df) . r,
   which equals d'Hd - d'Cd to second order in d.
B. The same at fixed function-space step sizes: d scaled so that the RMS over examples of |J_i d| equals each target.
   This compares arms at equal logit movement, a size that does not depend on the parametrization (raw or effective).
C. Curvature persistence along d: d'C(theta + d)d / d'C(theta)d (one step later) and d'C(theta_old)d / d'C(theta)d
   (theta_old: the state ten steps earlier).
D. Per depth (stream after the embedding and after each block, head input), on fixed inputs: the representation's
   linearization error along d, and its temporal drift since the previous measurement: relative change, relative
   change of the pairwise-distance matrix (first 256 inputs), and linear CKA (invariant to rotations and isotropic
   scale; 1 - CKA measures deformation).
"""
import torch
from torch.func import functional_call, jvp

from residual_curvature.curvature.operators import loss, metric_action, residual
from residual_curvature.models.mlp import functional_logits

import micro


def _linear(model, registry, flat, d, x, microbatch):
    """Per microbatch: (part, logits f, Jd), detached."""
    out = []
    for part in micro.parts(len(x), microbatch):
        f, jd = jvp(functional_logits(model, registry, x[part]), (flat,), (d,))
        out.append((part, f.detach(), jd.detach()))
    return out


def _split(model, registry, flat, d, x, y, linear, scale):
    """Error split (A) for the step scale*d, from stored logits and Jd."""
    total = len(x)
    sums = dict.fromkeys(('actual', 'linearized', 'gd', 'full', 'residual_term', 'remainder2', 'linear2'), 0.)
    for part, f, jd in linear:
        weight = (part.stop-part.start)/total
        yp, sjd = y[part], scale*jd
        with torch.no_grad():
            after = functional_logits(model, registry, x[part])(flat+scale*d)
        base = float(loss(f, yp))
        r = after-f-sjd
        dl = residual(f, yp)
        sums['actual'] += weight*(float(loss(after, yp))-base)
        sums['linearized'] += weight*(float(loss(f+sjd, yp))-base)
        sums['gd'] += weight*float((dl*sjd).sum())
        sums['full'] += weight*float((sjd*metric_action(f, sjd)).sum())
        sums['residual_term'] += weight*2*float((dl*r).sum())
        sums['remainder2'] += float(r.square().sum())
        sums['linear2'] += float(sjd.square().sum())
    predicted = sums['gd']+sums['full']/2
    network, taylor = sums['actual']-sums['linearized'], sums['linearized']-predicted
    return {'scale': scale, 'actual': sums['actual'], 'linearized': sums['linearized'], 'gd': sums['gd'], 'full': sums['full'],
            'predicted_full': predicted, 'ratio_full': sums['actual']/predicted if predicted != 0 else None,
            'network_error': network, 'taylor_error': taylor,
            'network_share': abs(network)/(abs(network)+abs(taylor)) if network or taylor else None,
            'linearization_error': (sums['remainder2']/sums['linear2'])**.5 if sums['linear2'] > 0 else None,
            'residual_term': sums['residual_term']}


def error_split(model, registry, flat, d, x, y, microbatch, targets=()):
    """A at the step itself and B at each target RMS logit change: {'jd_rms', 'step', 'matched': {target: split}}."""
    linear = _linear(model, registry, flat, d, x, microbatch)
    rms = (sum(float(jd.square().sum()) for _, _, jd in linear)/len(x))**.5
    out = {'jd_rms': rms, 'step': _split(model, registry, flat, d, x, y, linear, 1.)}
    out['matched'] = {f'{target:g}': _split(model, registry, flat, d, x, y, linear, target/rms) if rms > 0 else None
                      for target in targets}
    return out


def directional_ggn(model, registry, flat, d, x, microbatch):
    """d'C d at parameters `flat` for the mean loss of the batch (forward mode)."""
    return sum((part.stop-part.start)/len(x)*float((jd*metric_action(f, jd)).sum())
               for part, f, jd in _linear(model, registry, flat, d, x, microbatch))


def persistence(model, registry, flat, d, x, microbatch, now, old_flat=None):
    """C: one-step and ten-step ratios of the directional GGN along d (now = d'C(flat)d on the same batch)."""
    after = directional_ggn(model, registry, flat+d, d, x, microbatch)
    old = directional_ggn(model, registry, old_flat, d, x, microbatch) if old_flat is not None else None
    return {'now': now, 'after_step': after, 'ten_steps_earlier': old,
            'one_step_ratio': after/now if now > 0 else None,
            'ten_step_ratio': old/now if old is not None and now > 0 else None}


def _states(model, registry, x):
    return lambda w: functional_call(model, registry.unflatten(w), (x,), {'states': True})


def _flat_states(states):
    streams, head = states
    return [s.detach().flatten(1) for s in streams]+[head.detach()]


def representation_names(count):
    return [f'depth{k}_stream' for k in range(count-1)]+['head']


def linear_cka(a, b):
    """Linear CKA of two representation matrices (n, p) and (n, q), in float64."""
    a, b = a.double(), b.double()
    a, b = a-a.mean(0), b-b.mean(0)
    k, l = a@a.T, b@b.T
    return float((k*l).sum()/(k.norm()*l.norm()))


def distance_change(a, b, count=256):
    """|D(a) - D(b)| / |D(b)| for the pairwise-distance matrices of the first `count` rows, in float64."""
    da = torch.cdist(a[:count].double(), a[:count].double(), compute_mode='donot_use_mm_for_euclid_dist')
    db = torch.cdist(b[:count].double(), b[:count].double(), compute_mode='donot_use_mm_for_euclid_dist')
    return float((da-db).norm()/db.norm())


def representation(model, registry, flat, d, x, previous=None):
    """D: ({representation: measures}, the representations at flat + d for the next call's `previous`)."""
    fn = _states(model, registry, x)
    now, tangent = jvp(fn, (flat,), (d,))
    with torch.no_grad():
        after = fn(flat+d)
    now, tangent, after = _flat_states(now), _flat_states(tangent), _flat_states(after)
    out = {}
    for index, name in enumerate(representation_names(len(now))):
        r = after[index]-now[index]-tangent[index]
        row = {'linearization_error': float(r.norm()/tangent[index].norm()) if float(tangent[index].norm()) > 0 else None,
               'step_change': float((after[index]-now[index]).norm()/now[index].norm())}
        if previous is not None:
            old = previous[index]
            row.update(drift=float((after[index]-old).norm()/old.norm()), distance_change=distance_change(after[index], old),
                       cka=linear_cka(after[index], old))
        out[name] = row
    return out, after
