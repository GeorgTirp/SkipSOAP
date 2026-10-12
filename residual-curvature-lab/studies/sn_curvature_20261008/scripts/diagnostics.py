"""Per-state diagnostics A-C (PROTOCOL.md): distance preservation, curvature structure, estimation.

Copied pieces: `full_ggn` follows studies/geometry_followup_20261008/scripts/arms.py `_streamed_factor`
(GGN kind); `families.py` is copied from studies/gap_audit_20261007/scripts/families.py.
"""
import copy
import math
import time

import torch
from torch.func import jacrev

import common  # noqa: F401
import families as fam
import sn
from residual_curvature.curvature.information import information_blocks
from residual_curvature.curvature.layerwise import solve_blocks, workspace_estimate
from residual_curvature.curvature.operators import GGN, loss
from residual_curvature.curvature.solves import relative_error
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim import capacity
from residual_curvature.optim.soap import SOAPMetric, soap_propose
from run_large_batch_oracle import measurement, state_equal
from run_mini_vit_allocation import search_candidate

BUDGET_MIB = 3072
TOL = 1e-8


def full_ggn(model, registry, flat, x, microbatch=8):
    rows = []
    for start in range(0, len(x), microbatch):
        fn = functional_logits(model, registry, x[start:start+microbatch])
        p = fn(flat).detach().softmax(-1)
        j = jacrev(fn, chunk_size=16)(flat).detach()
        centered = j-(p[..., None]*j).sum(1, keepdim=True)
        rows.append((centered*(p/len(x)).sqrt()[..., None]).reshape(-1, registry.size))
    factor = torch.cat(rows)
    return factor.T@factor


def pooled_error(approx, exact, specs=None, registry=None):
    pairs = list(zip(approx, exact)) if specs is None else [(a, e) for a, e, s in zip(approx, exact, registry.specs) if s in specs]
    return (sum(float((a-e).square().sum()) for a, e in pairs)/sum(float(e.square().sum()) for _, e in pairs))**.5


def spectrum(full):
    eig = torch.linalg.eigvalsh((full+full.T)/2).flip(0).clamp_min(0)
    total = float(eig.sum())
    p = eig/eig.sum()
    p = p[p > 0]
    return {'top': float(eig[0]), 'trace': total, 'top16_share': float(eig[:16].sum())/total,
            'effective_rank_entropy': float(torch.exp(-(p*p.log()).sum())),
            'numerical_rank': int((eig > 1e-10*eig[0]).sum()),
            'rank16_truncation_error': float((eig[16:].square().sum()/eig.square().sum()).sqrt()),
            'top10': eig[:10].tolist()}


def curvature_structure(model, registry, flat, x, y, blocks, tolerance=TOL):
    """B for one parameterization: full GGN checks, spectrum, cross-tensor energy, Kronecker errors."""
    op = GGN(functional_logits(model, registry, x), flat, y)
    full = full_ggn(model, registry, flat, x)
    gen = torch.Generator().manual_seed(4242)
    v = torch.randn(registry.size, generator=gen, dtype=flat.dtype)
    checks = {'full_action': relative_error(full@v, op.action(v)),
              'full_diagonal_blocks': pooled_error([full[s.slice, s.slice] for s in registry.specs], blocks)}
    if max(checks.values()) > tolerance:
        raise RuntimeError(f'Full GGN checks failed: {checks}')
    total = float(full.square().sum())
    out = {'checks': checks, 'spectrum': spectrum(full),
           'cross_tensor_energy': (total-sum(float(b.square().sum()) for b in blocks))/total}
    matrices = [s for s in registry.specs if len(s.shape) == 2]
    kron, kdiag = [], []
    for spec, block in zip(registry.specs, blocks):
        if len(spec.shape) == 2:
            kron.append(fam.kron_sum_block(block, spec, 1)[0])
            kdiag.append(fam.kron_basis_diag_block(block, spec)[0])
    exact2 = [b for s, b in zip(registry.specs, blocks) if len(s.shape) == 2]
    out['nearest_kronecker_error'] = pooled_error(kron, exact2)
    out['kron_basis_diag_error'] = pooled_error(kdiag, exact2)
    out['diagonal_error'] = pooled_error([torch.diag(b.diag()) for b in blocks], blocks)
    hutchinson = torch.randint(0, 2, (registry.size, 128), generator=gen, dtype=torch.int64).to(flat.dtype)*2-1
    quadratic = (hutchinson*(full@hutchinson)).sum(0).reshape(16, 8).mean(1)
    trace = float(full.trace())
    out['hutchinson8_relative_rms'] = float(((quadratic-trace).square().mean()).sqrt())/trace
    out['tensor_count_matrix'] = len(matrices)
    return out, full, op


def distance_preservation(model, x, manifest):
    """A: global and local representation/input distance ratios."""
    pool = manifest['analysis'][512:2560]
    gen = torch.Generator().manual_seed(9001)
    i = torch.randint(0, len(pool), (2000,), generator=gen)
    j = (i+1+torch.randint(0, len(pool)-1, (2000,), generator=gen)) % len(pool)
    xa, xb = x[[pool[k] for k in i.tolist()]], x[[pool[k] for k in j.tolist()]]
    construction = x[manifest['analysis'][:512]]
    xi = torch.randn(construction.shape, generator=torch.Generator().manual_seed(9002), dtype=construction.dtype)
    xi = xi/xi.norm(dim=1, keepdim=True)
    local = construction+1e-3*construction.norm(dim=1, keepdim=True)*xi
    out = {}
    for kind, (u, w) in {'global': (xa, xb), 'local': (construction, local)}.items():
        ru, rw = sn.representations(model, u), sn.representations(model, w)
        denominator = (u-w).norm(dim=1)
        for index, name in enumerate(['phi_res', 'phi_head']):
            ratio = (ru[index]-rw[index]).norm(dim=1)/denominator
            q = torch.quantile(ratio, torch.tensor([.01, .05, .5, .95, .99], dtype=ratio.dtype))
            out[f'{kind}_{name}'] = {'quantiles_1_5_50_95_99': q.tolist(), 'distortion_q95_q5': float(q[3]/q[1]),
                                     'min': float(ratio.min()), 'max': float(ratio.max())}
    return out


def headroom_and_estimation(model, registry, flat, state, x, y, construction, panel_ids, c, step, blocks, info_blocks):
    """B (local headroom) and C (A-energy errors), raw parameterization, damping as in the gap audit."""
    xb, yb = x[construction], y[construction]
    fn = functional_logits(model, registry, xb)
    objective = lambda w: loss(fn(w), yb)
    op = GGN(fn, flat, yb)
    g = op.gradient
    proposal, _, contexts = soap_propose(registry.unflatten(flat), registry.unflatten(g), state, c['soap'])
    native = registry.flatten(proposal)
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in blocks)/registry.size, c['damping']['floor'])
    teacher = fam.DampedBlocks(blocks, registry, metric, mu, TOL)
    gen = torch.Generator().manual_seed(270100+step)
    zs = torch.randn(registry.size, 16, dtype=flat.dtype, generator=gen)
    factor_rhs = torch.empty_like(zs)
    for spec in registry.specs:
        factor_rhs[spec.slice] = capacity.factor(zs[spec.slice].reshape(spec.shape+(16,)), contexts[spec.name],
                                                 transpose=True, inverse=True).reshape(spec.count, 16)
    groups = {'gradient': g[:, None], 'factor': factor_rhs}
    exact_values = {k: teacher.solve(v) for k, v in groups.items()}
    approximations = {'diagonal': [torch.diag(b.diag()) for b in blocks],
                      'kron_basis_diag': fam.family_blocks('kron_basis_diag', registry, blocks, None, contexts)[0],
                      'empirical_fisher': info_blocks['empirical_fisher'],
                      'mc_fisher_1': info_blocks['sampled_1_repeat_0']}
    energy = {}
    for name, approx in approximations.items():
        damped = fam.DampedBlocks(approx, registry, metric, mu, TOL)
        energy[name] = {k: capacity.energy_errors(v, damped.solve(v), exact_values[k], teacher.systems, registry) for k, v in groups.items()}
        energy[name] = {k: {'pooled_ray': e['ray_numerator_sum']/e['denominator_sum'], 'pooled_raw': e['raw_numerator_sum']/e['denominator_sum']}
                        for k, e in energy[name].items()}
    panels = [lambda w, f=functional_logits(model, registry, x[ids]), t=y[ids]: loss(f(w), t) for ids in panel_ids]
    base, panel_base = float(objective(flat)), [float(f(flat)) for f in panels]
    rows = {}
    for name in ['identity', 'exact', 'soap_basis_diag', 'kron_basis_diag']:
        candidate_blocks = fam.family_blocks(name, registry, blocks, None, contexts)[0]
        values, solve = solve_blocks(candidate_blocks, registry, g, metric, [mu], TOL)
        selected, search = search_candidate(flat, values[mu], native, objective)
        m = measurement(selected, op, metric, mu, objective, base, panels, panel_base, registry)
        m.pop('tensor_update_shares')
        rows[name] = {'independent_mean_decrease': m['independent_mean_decrease'], 'measurement': m,
                      'solve_residual': solve[mu]['residual'],
                      'search': {k: search[k] for k in ['selected_alpha', 'evaluations', 'bracketed', 'rejected']}}
    headroom = rows['exact']['independent_mean_decrease']-rows['identity']['independent_mean_decrease']
    recovery = {n: (rows[n]['independent_mean_decrease']-rows['identity']['independent_mean_decrease'])/headroom if abs(headroom) > 1e-4 else None
                for n in ['soap_basis_diag', 'kron_basis_diag']}
    return {'mu': mu, 'energy_errors': energy, 'candidates': rows, 'headroom': headroom, 'recovery': recovery}, native, contexts


def diagnose(model, config, bound, registry, flat, state, x, y, manifest, panel_ids, step):
    """All of A-C at one saved state; verifies that model, optimizer state and RNG are not mutated."""
    started = time.perf_counter()
    before, optimizer, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(state), torch.get_rng_state().clone()
    construction = manifest['analysis'][:512]
    xb, yb = x[construction], y[construction]
    out = {'step': step, 'bound': bound, 'spectral': sn.spectral_report(model, bound),
           'distance': distance_preservation(model, x, manifest)}
    info_blocks, info_gradient, _, info = information_blocks(model, registry, flat, xb, yb, ranks=(1, 4, 16), repeats=2,
                                                             seed=config['analysis_seed']+7919*step, budget_mib=BUDGET_MIB)
    blocks = info_blocks['ggn']
    op = GGN(functional_logits(model, registry, xb), flat, yb)
    gen = torch.Generator().manual_seed(config['analysis_seed']+step)
    actions = []
    for spec, block in zip(registry.specs, blocks):
        v = torch.zeros_like(flat)
        v[spec.slice] = torch.randn(spec.count, generator=gen, dtype=flat.dtype)
        actions.append(relative_error(block@v[spec.slice], op.action(v)[spec.slice]))
    checks = {'max_block_action_error': max(actions), 'information_gradient_error': relative_error(info_gradient, op.gradient)}
    raw, full, _ = curvature_structure(model, registry, flat, xb, yb, blocks)
    raw['estimation_frobenius'] = {
        'empirical_fisher': pooled_error(info_blocks['empirical_fisher'], blocks),
        **{f'mc_fisher_{r}': sum(pooled_error(info_blocks[f'sampled_{r}_repeat_{k}'], blocks) for k in range(2))/2 for r in [1, 4, 16]}}
    null = {}
    for spec, block in zip(registry.specs, blocks):
        module_name = spec.name.split('.parametrizations')[0]
        if '.parametrizations.' in spec.name and out['spectral'][module_name]['active']:
            w = flat[spec.slice]
            null[module_name] = float((block@w).norm()/(torch.linalg.eigvalsh(block)[-1]*w.norm()))
    raw['sn_null_direction'] = null
    del full
    estimation, native, contexts = headroom_and_estimation(model, registry, flat, state, x, y, construction, panel_ids,
                                                           config, step, blocks, info_blocks)
    twin, twin_registry = sn.effective_twin(model, config)
    twin_flat = twin_registry.flatten(dict(twin.named_parameters())).detach()
    checks['effective_twin_logits'] = relative_error(functional_logits(twin, twin_registry, xb)(twin_flat),
                                                     functional_logits(model, registry, xb)(flat))
    from residual_curvature.curvature.layerwise import streamed_blocks
    twin_blocks, _ = streamed_blocks(twin, twin_registry, twin_flat, xb, 8, BUDGET_MIB)
    effective, twin_full, _ = curvature_structure(twin, twin_registry, twin_flat, xb, yb, twin_blocks)
    del twin_full
    checks.update({f'raw_{k}': v for k, v in raw['checks'].items()})
    checks.update({f'effective_{k}': v for k, v in effective['checks'].items()})
    if max(checks.values()) > TOL:
        raise RuntimeError(f'State checks failed: {checks}')
    if not (state_equal(before, model.state_dict()) and state_equal(optimizer, state) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('Diagnostics mutated the training state')
    out.update(checks=checks, raw=raw, effective=effective, estimation=estimation,
               information=info, seconds=time.perf_counter()-started, nonmutation_passed=True)
    arrays = {'parameters': flat, 'gradient': op.gradient, 'optimizer': state, 'contexts': contexts, 'raw_soap': native}
    return out, arrays
