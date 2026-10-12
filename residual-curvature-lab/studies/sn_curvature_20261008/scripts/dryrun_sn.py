"""Cost dry run: three searched-SOAP steps per model and SN condition (seed 0), then one complete
A-C diagnostic state per model (no-SN after 3 steps); writes measured seconds and CPU-h projection."""
import json
import time

import torch

import common
import diagnostics
import run
import sn
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry
from residual_curvature.provenance import write_json
from run_mini_vit_allocation import propose as mini_propose


def main():
    common.guard()
    common.setup_torch()
    directory = common.create_run(run.unit_config({'model': 'vit', 'seed': 0}),
                                  {'study': 'sn_curvature_20261008', 'phase': 'cost dry run', 'document': str(common.PROTOCOL)})
    print(directory, flush=True)
    result = {'steps': {}, 'diagnostics': {}}
    panel_ids = common.read(run.PANEL_SOURCE)['panel_ids']
    for model_kind in ['vit', 'mlp']:
        c = run.unit_config({'model': model_kind, 'seed': 0})
        x, y, manifest = dataset(c)
        for bound in run.BOUNDS:
            model = sn.make_sn_model(c, bound)
            registry = Registry(model)
            order = DataOrder(manifest['training'], common.data_order_seed(0))
            state, times = None, []
            for step in range(1, 4):
                ids = order.take(512)
                flat = registry.flatten(dict(model.named_parameters())).detach()
                started = time.perf_counter()
                delta, state, _, _ = mini_propose('searched_soap', model, registry, flat, state, x[ids], y[ids], c, step)
                commit(model, registry, flat+delta)
                times.append(time.perf_counter()-started)
            result['steps'][f'{model_kind}_{run.condition(bound)}'] = {'step_seconds': times, 'per_step_after_init': sum(times[1:])/2}
            print(model_kind, run.condition(bound), [round(t, 3) for t in times], flush=True)
            if bound is None:
                flat = registry.flatten(dict(model.named_parameters())).detach()
                started = time.perf_counter()
                out, _ = diagnostics.diagnose(model, c, bound, registry, flat, state, x, y, manifest, panel_ids, 3)
                result['diagnostics'][model_kind] = {'seconds': time.perf_counter()-started, 'checks': out['checks'],
                                                     'peak_rss_bytes': common.peak_rss_bytes()}
                print(model_kind, 'diagnostic state', round(result['diagnostics'][model_kind]['seconds'], 1), 's', flush=True)
            write_json(directory/'dryrun.json', result)
    per_unit = {k: 200*v['per_step_after_init'] for k, v in result['steps'].items()}
    projection = sum(per_unit[f'{m}_{run.condition(b)}']*len(run.SEEDS) for m in ['vit', 'mlp'] for b in run.BOUNDS)
    projection += sum(3*len(run.BOUNDS)*len(run.SEEDS)*result['diagnostics'][m]['seconds'] for m in ['vit', 'mlp'])
    result['projected_cpu_hours_sequential'] = projection/3600
    result['projected_cpu_hours_with_1.6x_load'] = 1.6*projection/3600
    write_json(directory/'dryrun.json', result)
    write_json(directory/'complete.json', {'completed': True})
    print(json.dumps({k: v for k, v in result.items() if k.startswith('projected')}), flush=True)
    print(common.mirror(directory), flush=True)


if __name__ == '__main__':
    main()
