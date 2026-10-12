import math

import torch


def noise_learning_rate_search(flat, drift, noise, loss_fn, base_lr):
    """Search eta/base_lr with one fixed noise draw and frozen optimizer state.

    delta(a) = a*drift + sqrt(a)*noise. This is a curved path, not a ray.
    The selected conditional noise distribution is altered by loss-based selection.
    """
    def displacement(a):
        return a*drift+math.sqrt(a)*noise

    scalar = flat.new_zeros(1)
    def along_path(point):
        return loss_fn(flat+displacement(float(point[0])))
    _, info = broad_direction_search(scalar, torch.ones_like(scalar), 1., along_path)
    if not info['bracketed'] and not info['rejected']:
        raise RuntimeError('Positive noise learning-rate search reached the boundary')
    multiplier = info['selected_alpha']
    info.update(policy='fixed_draw_learning_rate_grid_then_golden_refinement',
                base_lr=base_lr, selected_lr=base_lr*multiplier,
                path='a*drift + sqrt(a)*noise', fixed_draw=True)
    info.pop('radius')
    return displacement(multiplier), info


def broad_direction_search(flat, delta, radius, loss_fn, exponents=range(-16, 17), refinement_evaluations=18):
    """Common radius grid followed by bounded golden-section refinement.

    The grid is always evaluated in full. Refinement uses construction loss only.
    Boundary and zero-step selections stay explicit; no global optimality claim.
    """
    if radius <= 0 or float(delta.norm()) == 0:
        raise ValueError('A positive native radius and nonzero direction are required')
    direction = delta * (radius / float(delta.norm()))
    multipliers = [0.0] + [2.0 ** j for j in exponents]
    trials = []
    def evaluate(alpha):
        value = float(loss_fn(flat + alpha * direction))
        trials.append({'alpha': alpha, 'loss': value if math.isfinite(value) else None})
        return value if math.isfinite(value) else math.inf
    values = [evaluate(a) for a in multipliers]
    index = min(range(len(values)), key=values.__getitem__)
    boundary = index == len(values) - 1
    bracket = None
    if 0 < index < len(values) - 1:
        low, high = multipliers[index-1], multipliers[index+1]
        bracket = [low, high]
        ratio = (5**.5 - 1) / 2
        left, right = high - ratio * (high-low), low + ratio * (high-low)
        fl, fr = evaluate(left), evaluate(right)
        for _ in range(max(0, refinement_evaluations - 2)):
            if fl <= fr:
                high, right, fr = right, left, fl
                left = high - ratio * (high-low)
                fl = evaluate(left)
            else:
                low, left, fl = left, right, fr
                right = low + ratio * (high-low)
                fr = evaluate(right)
    selected = min((t for t in trials if t['loss'] is not None), key=lambda t: t['loss'])
    return direction * selected['alpha'], {
        'policy': 'broad_common_radius_grid_then_golden_refinement', 'radius': radius,
        'trials': trials, 'selected_alpha': selected['alpha'], 'evaluations': len(trials),
        'evaluation_budget': len(multipliers) + refinement_evaluations,
        'grid_best_alpha': multipliers[index], 'bracket': bracket,
        'bracketed': bracket is not None, 'boundary': boundary,
        'rejected': selected['alpha'] == 0,
    }


def local_direction_grid(flat, delta, radius, loss_fn, exponents):
    if radius <= 0 or float(delta.norm()) == 0:
        return delta * 0, {'available': False, 'reason': 'zero native radius or zero candidate', 'evaluations': 0}
    direction = delta * (radius / float(delta.norm()))
    multipliers = [0.0] + [2.0 ** j for j in exponents]
    trials = [{'alpha': a, 'loss': float(loss_fn(flat + a * direction))} for a in multipliers]
    selected = min(trials, key=lambda t: t['loss'])
    return direction * selected['alpha'], {'available': True, 'policy': 'local_direction_grid', 'radius': radius,
             'trials': trials, 'selected_alpha': selected['alpha'], 'evaluations': len(trials),
             'boundary': selected['alpha'] in [multipliers[1], multipliers[-1]], 'rejected': selected['alpha'] == 0}


def training_backtracking(flat, delta, g, loss_fn, settings):
    slope = float(g @ delta)
    if slope >= 0:
        return delta * 0, {'policy': 'training_backtracking', 'accepted': False,
                          'reason': 'non-descent true-gradient slope', 'trials': [], 'evaluations': 0}
    base = float(loss_fn(flat))
    alpha, trials = settings['initial'], []
    for _ in range(settings['max_evaluations']):
        if alpha < settings['minimum']:
            break
        value = float(loss_fn(flat + alpha * delta))
        trials.append({'alpha': alpha, 'loss': value})
        if value <= base + settings['armijo'] * alpha * slope:
            return alpha * delta, {'policy': 'training_backtracking', 'accepted': True,
                                  'selected_alpha': alpha, 'trials': trials, 'evaluations': len(trials)}
        alpha /= 2
    return delta * 0, {'policy': 'training_backtracking', 'accepted': False, 'reason': 'exhausted search',
                      'trials': trials, 'evaluations': len(trials)}
