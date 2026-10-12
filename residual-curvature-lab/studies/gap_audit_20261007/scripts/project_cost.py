"""Project CPU-hours per check from the section 4.5 dry run (measured seconds).

Single-threaded processes: CPU-hours = summed process seconds / 3600. Counts follow
the protocol; worst cases include every edge extension and both RHS where scored.
"""
import json
import sys
from pathlib import Path

import common
from common import read
from residual_curvature.provenance import write_json

STARTUP = 6.  # dataset load + imports per process, measured order of magnitude


def project(dry):
    a = {k: v['per_step_after_init'] for k, v in dry['arms'].items()}
    val = dry['validation_seconds']
    ev = dry['fixed_state']['candidate_evaluation_seconds']
    action = dry['fixed_state']['ggn_action_seconds']
    searched = (a['searched_soap_tuned']+a['searched_soap_pf10_b9-99'])/2
    gate = 50*(a['native_soap']+a['searched_soap']+a['adam_native']+a['ggn_exact_trace'])+4*(4*val+STARTUP)
    tune = (5*500*a['native_soap']+5*500*a['adam_native']+4*500*searched+3*200*a['ggn_exact_trace']
            + 17*(22*val+STARTUP))
    extension = 500*a['native_soap']+500*a['adam_native']+500*searched+200*a['ggn_exact_trace']+4*(22*val+STARTUP)
    per_seed = {steps: steps*sum(a[k] for k in ['native_soap', 'adam_native', 'searched_soap', 'searched_soap_tuned',
                                                'ggn_exact_trace', 'dense_ef', 'soap_gradient_map'])
                for steps in [500, 300]}
    evaluation = {steps: 3*value+21*(len(range(25, steps+1, 25))+2)*val+21*STARTUP for steps, value in per_seed.items()}
    check1 = {'gate': gate, 'tuning': tune, 'tuning_edge_extensions_worst_case': extension,
              'evaluation_500': evaluation[500], 'evaluation_300_fallback': evaluation[300]}
    check1['total_500_worst_case'] = gate+tune+extension+evaluation[500]
    # 2c: 17 families on the gradient and up to 17 on momentum; PCG energy errors on 18 RHS.
    pcg_full_matvecs = 18*(2+3+5+10)*2
    check2c = 6*(34*ev+pcg_full_matvecs*action+30.)
    # 3: 12 state-targets x 35 searches, 6 OT block constructions, fitting/audit as in the screen.
    check3 = 12*35*ev+6*dry['fixed_state']['ot_block_seconds']+12*30.
    # 4a/4b: 22 trajectories x 50 searched-SOAP steps; 66 snapshots with blocks + per-tensor action checks;
    # 4b scores 6 candidates x 2 RHS per state with the search cost scaled by forward cost.
    arch = dry['architectures']
    main_step = arch['b2_w16']['per_step_after_init']
    seeds = {f'b{b}_w{w}': (3 if b in [1, 2] and w in [8, 16] else 2) for b in [1, 2, 3] for w in [8, 16, 24]}
    tensors = {f'b{b}_w{w}': 8+16*b for b in [1, 2, 3] for w in [8, 16, 24]}
    check4a = sum(n*(50*arch[k]['per_step_after_init']+3*(arch[k]['ggn_block_seconds']
                  + tensors[k]*arch[k]['ggn_action_seconds']+5.)+STARTUP) for k, n in seeds.items())
    check4b = sum(n*3*(12*ev*arch[k]['per_step_after_init']/main_step+2*(2+5)*arch[k]['ggn_action_seconds']*2+20.)
                  for k, n in seeds.items())
    check4c = 3*3600.+0.5*3600.
    check4 = {'4a_dataset': check4a, '4b_headroom': check4b, '4c_predictor_cap_plus_evaluation': check4c,
              'total': check4a+check4b+check4c, 'total_without_4c': check4a+check4b,
              'fallback_without_step25_snapshots': check4a-sum(n*(arch[k]['ggn_block_seconds']+tensors[k]*arch[k]['ggn_action_seconds']+5.)
                                                                for k, n in seeds.items())+check4b*2/3+check4c}
    hours = lambda s: s/3600
    return {'seconds': {'check1': check1, 'check2c': check2c, 'check3': check3, 'check4': check4},
            'cpu_hours': {'check1_with_2a_2b': hours(check1['total_500_worst_case']),
                          'check1_300_step_fallback': hours(gate+tune+extension+evaluation[300]),
                          'check2c': hours(check2c), 'check3': hours(check3), 'check4': hours(check4['total']),
                          'check4_without_4c': hours(check4['total_without_4c'])},
            'budgets_cpu_hours': {'check1_with_2a_2b': 12, 'check2c': 1, 'check3': 1, 'check4': 6},
            'inputs': {'per_step_seconds': a, 'validation_seconds': val, 'candidate_evaluation_seconds': ev,
                       'ggn_action_seconds': action, 'startup_seconds_assumed': STARTUP}}


if __name__ == '__main__':
    common.guard()
    run = Path(sys.argv[1])
    result = project(read(run/'dryrun.json'))
    write_json(common.STUDY/'logs'/'cost_projection.json', result)
    print(json.dumps(result['cpu_hours'], indent=2))
    print(json.dumps(result['seconds'], indent=2))
