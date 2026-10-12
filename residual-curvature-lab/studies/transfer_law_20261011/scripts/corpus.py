"""Part A of PROTOCOL.md: transfer law on a finite corpus (the 5,000-example analysis split). No training.

Usage (from residual-curvature-lab/, PYTHONDONTWRITEBYTECODE=1): corpus.py STATE_INDEX [--smoke]
Writes results/corpus_<state>.json (not with --smoke, which only times one small and one large subset).
"""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch

STUDY = Path(__file__).resolve().parents[1]
LAB = STUDY.parents[1]
for _path in [LAB/'scripts', LAB/'studies'/'parsec_l_20261009']:
    sys.path.insert(0, str(_path))

from parsec_l import per_example_gradients  # noqa: E402
from residual_curvature.config import load_config  # noqa: E402
from residual_curvature.curvature.layerwise import streamed_blocks  # noqa: E402
from residual_curvature.curvature.operators import GGN, loss  # noqa: E402
from residual_curvature.data import dataset  # noqa: E402
from residual_curvature.experiments.training import commit  # noqa: E402
from residual_curvature.models import Registry, make_model  # noqa: E402
from residual_curvature.models.mlp import functional_logits  # noqa: E402
from residual_curvature.optim.soap import SOAPMetric, frozen_apply  # noqa: E402
from run_large_batch_oracle import state_equal  # noqa: E402
from run_mini_vit_allocation import search_candidate  # noqa: E402

PARENT = Path('runs/20261005T143029-9b582ed5')
STATES = [(t, s) for t in ['searched_soap', 'exact_tensor_ggn'] for s in [10, 25, 50]]
SIZES, SUBSETS, SEED = [64, 128, 256, 512, 1024, 2048, 4096], 8, 20261011
OPERATORS = ['sgd', 'soap_map', 'exact_ggn_N']


def predict(theory, size, n):
    """Shares predicted by the transfer law for a same-batch-searched step on `size` of the n corpus examples."""
    f, o, s = size/n, theory['trace']/size, theory['signal']
    r_train, r_pop = (1-f)*o/s, o/s
    return {'f': f, 'r_train': r_train, 'r_pop': r_pop, 'corpus': (1-r_train)/(1+r_train),
            'validation_conditional': (2*theory['signal_val']-s-(1-f)*o)/(s+(1-f)*o),
            'validation_population': (1-r_pop)/(1+r_pop)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('state', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = load_config('configs/mini_vit_allocation.yaml')
    x, y, manifest = dataset(c)
    trajectory, step = STATES[args.state]
    name = f'{trajectory}-{step}'
    source = PARENT/f'{trajectory}-{step:04d}'
    arrays = torch.load(source/'arrays.pt', map_location='cpu', weights_only=False)
    info = json.loads((source/'diagnostics.json').read_text())
    flat, g, contexts, native = arrays['parameters'], arrays['gradient'], arrays['contexts'], arrays['raw_soap']
    model = make_model(c)
    registry = Registry(model)
    commit(model, registry, flat)
    model_before, contexts_before, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(contexts), torch.get_rng_state().clone()
    corpus, validation = manifest['analysis'], manifest['validation']
    n = len(corpus)
    started = time.perf_counter()

    # Exact corpus statistics: per-example gradients, mean, per-tensor covariance (N - 1), validation gradient, GGN.
    per_example = per_example_gradients(model, x[corpus], y[corpus])
    params = dict(model.named_parameters())
    gradients = torch.cat([per_example[params[s.name]].reshape(n, -1) for s in registry.specs], 1)
    del per_example
    position = {i: k for k, i in enumerate(corpus)}
    checks = {'construction_gradient': float((gradients[[position[i] for i in info['construction_ids']]].mean(0)-g).norm()/g.norm())}
    if checks['construction_gradient'] > 1e-12:
        raise RuntimeError(f'Gate failed: {checks}')
    mean = gradients.mean(0)
    sigma = []
    for s in registry.specs:
        centered = gradients[:, s.slice]-mean[s.slice]
        sigma.append(centered.T@centered/(n-1))
    g_val = GGN(functional_logits(model, registry, x[validation]), flat, y[validation]).gradient
    blocks, _ = streamed_blocks(model, registry, flat, x[corpus], 8, c['resources']['budget_mib'])
    metric = SOAPMetric(registry, contexts)
    mu = max(.01*sum(float(b.trace()) for b in blocks)/registry.size, c['damping']['floor'])
    eye = lambda s: torch.eye(s.count).reshape(s.shape+(s.count,))
    omega = [metric.scale*frozen_apply(eye(s), contexts[s.name], inverse=True).reshape(s.count, s.count) for s in registry.specs]
    matrices = {'sgd': None, 'soap_map': omega, 'exact_ggn_N': [b+mu*o for b, o in zip(blocks, omega)]}
    factors = {k: None if v is None else [torch.linalg.cholesky((m+m.T)/2) for m in v] for k, v in matrices.items()}
    statistics_seconds = time.perf_counter()-started

    def solve(op, v):
        if factors[op] is None:
            return v.clone()
        out = torch.empty_like(v)
        for s, l in zip(registry.specs, factors[op]):
            out[s.slice] = torch.cholesky_solve(v[s.slice, None], l)[:, 0]
        return out

    def trace(op):
        if factors[op] is None:
            return sum(float(m.trace()) for m in sigma)
        return sum(float(torch.cholesky_solve(m, l).diagonal().sum()) for m, l in zip(sigma, factors[op]))

    theory = {}
    for op in OPERATORS:
        a_mean = solve(op, mean)
        signal, tr = float(mean@a_mean), trace(op)
        theory[op] = {'signal': signal, 'trace': tr, 'signal_val': float(g_val@a_mean), 'noise_scale': tr/signal,
                      'f_star': (tr/signal)/(n+tr/signal)}

    corpus_fn, val_fn = functional_logits(model, registry, x[corpus]), functional_logits(model, registry, x[validation])
    corpus_loss = lambda w: float(loss(corpus_fn(w), y[corpus]))
    val_loss = lambda w: float(loss(val_fn(w), y[validation]))
    base_corpus, base_val = corpus_loss(flat), val_loss(flat)
    generator = torch.Generator().manual_seed(SEED+args.state)
    draws = [(size, k, torch.randperm(n, generator=generator)[:size].tolist()) for size in SIZES for k in range(SUBSETS)]
    if args.smoke:
        draws = [d for d in draws if d[1] == 0 and d[0] in (64, 4096)]
    rows = []
    for size, k, positions in draws:
        began = time.perf_counter()
        ids = [corpus[i] for i in positions]
        g_sub = gradients[positions].mean(0)
        fn = functional_logits(model, registry, x[ids])
        objective = lambda w, fn=fn, t=y[ids]: loss(fn(w), t)
        base = float(objective(flat))
        for op in OPERATORS:
            d = -solve(op, g_sub)
            row = {'size': size, 'subset': k, 'operator': op, 'q': -float(g_sub@d), 'p_corpus': -float(mean@d), 'p_val': -float(g_val@d)}
            try:
                selected, search = search_candidate(flat, d, native, objective)
            except RuntimeError as error:
                if 'boundary' not in str(error):
                    raise
                row.update(boundary=True, error=str(error))
                rows.append(row)
                continue
            new = flat+selected
            row.update(boundary=False, own=base-float(objective(new)), corpus=base_corpus-corpus_loss(new),
                       validation=base_val-val_loss(new), alpha=search['selected_alpha'], rejected=search['rejected'],
                       bracketed=search['bracketed'], step_norm=float(selected.norm()))
            rows.append(row)
        if args.smoke:
            print(f'size {size}: {time.perf_counter()-began:.1f} s for {len(OPERATORS)} searches', flush=True)
    if not (state_equal(model_before, model.state_dict()) and state_equal(contexts_before, contexts) and torch.equal(rng, torch.get_rng_state())):
        raise RuntimeError('State mutated')
    if args.smoke:
        print(json.dumps({'state': name, 'checks': checks, 'statistics_seconds': statistics_seconds, 'theory': theory, 'rows': rows}, indent=1))
        return

    summary = {}
    for op in OPERATORS:
        summary[op] = {}
        for size in SIZES:
            kept = [r for r in rows if r['operator'] == op and r['size'] == size and not r['boundary']]
            own = statistics.mean(r['own'] for r in kept)
            entry = {'subsets': len(kept), 'boundary_excluded': SUBSETS-len(kept), 'own': own,
                     'corpus': statistics.mean(r['corpus'] for r in kept), 'validation': statistics.mean(r['validation'] for r in kept),
                     'gamma_realized': statistics.mean(r['p_corpus']/r['q'] for r in kept), 'rejected': sum(r['rejected'] for r in kept)}
            entry['share_corpus'] = entry['corpus']/own if own > 0 else None
            entry['share_validation'] = entry['validation']/own if own > 0 else None
            entry['predicted'] = predict(theory[op], size, n)
            summary[op][size] = entry
            print(f"{name} {op:11s} B={size:4d} f={size/n:.3f}: corpus share {entry['share_corpus']:+.3f} "
                  f"(pred {entry['predicted']['corpus']:+.3f}) | validation {entry['share_validation']:+.3f} "
                  f"(cond {entry['predicted']['validation_conditional']:+.3f}, pop {entry['predicted']['validation_population']:+.3f})", flush=True)
    out = {'state': name, 'n': n, 'mu_N': mu, 'checks': checks, 'theory': theory, 'summary': summary, 'rows': rows,
           'seconds': time.perf_counter()-started, 'statistics_seconds': statistics_seconds}
    (STUDY/'results').mkdir(exist_ok=True)
    (STUDY/'results'/f'corpus_{name}.json').write_text(json.dumps(out, indent=1, default=str)+'\n')
    print(f'{name}: done in {out["seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    main()
