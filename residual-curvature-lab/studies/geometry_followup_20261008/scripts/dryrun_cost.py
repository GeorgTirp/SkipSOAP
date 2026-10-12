"""Cost dry run: three steps of every arm (seed 0) plus one PARSEC Lanczos refresh at ranks 8/16/32."""
import json
import time

import common
import arms
from residual_curvature.curvature.operators import GGN
from residual_curvature.data import DataOrder, dataset
from residual_curvature.experiments.training import commit
from residual_curvature.models import Registry, make_model
from residual_curvature.models.mlp import functional_logits
from residual_curvature.optim.parsec import lanczos_ritz
from residual_curvature.provenance import write_json

SETTINGS = {'parsec_default': {'lr': .003, 'refresh_every': 10, 'rank': 16},
            'parsec_split': {'lr_perp': .003, 'lr_top': .003, 'refresh_every': 10, 'rank': 16},
            'ot_tensor': {'gamma': .01}, 'ggn_full': {'gamma': .001}, 'ot_full': {'gamma': .01}}


def main():
    common.guard()
    common.setup_torch()
    c = common.study_config()
    directory = common.create_run(c, {'study': 'geometry_followup_20261008', 'phase': 'cost dry run', 'document': str(common.PROTOCOL)})
    print(directory, flush=True)
    x, y, manifest = dataset(c)
    result = {'arms': {}, 'lanczos': {}}
    for arm, settings in SETTINGS.items():
        model = make_model(c)
        registry = Registry(model)
        order = DataOrder(manifest['training'], common.data_order_seed(0))
        state, times = None, []
        for step in range(1, 4):
            ids = order.take(512)
            flat = registry.flatten(dict(model.named_parameters())).detach()
            started = time.perf_counter()
            delta, state, _, row = arms.propose_step(arm, settings, 'study', model, registry, flat, state, x[ids], y[ids], c, step)
            commit(model, registry, flat+delta)
            times.append(time.perf_counter()-started-row.get('audit_seconds', 0.))
        result['arms'][arm] = {'step_seconds': times, 'per_step_after_init': sum(times[1:])/2}
        print(arm, json.dumps(result['arms'][arm]), flush=True)
        if arm == 'parsec_default':
            op = GGN(functional_logits(model, registry, x[ids]), flat, y[ids])
            for rank in [8, 16, 32]:
                started = time.perf_counter()
                lanczos_ritz(op.action, op.gradient, rank, 2*rank, 0)
                result['lanczos'][rank] = time.perf_counter()-started
            print('lanczos', result['lanczos'], flush=True)
        write_json(directory/'dryrun.json', result)
    write_json(directory/'complete.json', {'completed': True})
    print(common.mirror(directory), flush=True)


if __name__ == '__main__':
    main()
