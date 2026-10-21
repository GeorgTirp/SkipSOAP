#!/bin/sh
# Queue-curvature study (from residual-curvature-lab/; PROTOCOL.md): waits for spectral_rotation_20261020's
# orchestration to finish (one GPU process at a time; Part F reads its summary), then before-manifest and suites, the
# replays and Parts V and S (gate G1 inside run.py), the summary, after-manifest and suites. Launched detached
# (setsid nohup). Stops at the first failing phase.
set -e
PY=.venv/bin/python
S=studies/queue_curvature_20261020
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # DEVIATIONS.md D1
suites() {
    for t in tests studies/parsec_l_20261009/tests studies/cifar_ggn_20261012/tests studies/cifar_sn_soap_20261012/tests \
             studies/basis_refresh_snr_20261013/tests studies/n9_training_20261013/tests studies/probe_variants_20261013/tests \
             studies/ladder_mechanism_20261013/tests studies/long_run_20261013/tests studies/ggn_c_control_20261013/tests \
             studies/ot_1000_20261013/tests studies/ot_tuned_1000_20261014/tests studies/ggn_ot_fairness_20261014/tests \
             studies/bulk_krylov_20261014/tests studies/spectral_rotation_20261020/tests studies/queue_curvature_20261020/tests; do
        echo "$t: $($PY -m pytest -p no:cacheprovider -q $t 2>&1 | tail -1)"
    done
}
echo "$(date -u +%FT%TZ) waiting for spectral_rotation_20261020" >> $S/logs/orchestrate.log
while pgrep -f 'studies/spectral_rotation_20261020/scripts/orchestrate.sh' > /dev/null; do
    sleep 60
done
echo "$(date -u +%FT%TZ) start" >> $S/logs/orchestrate.log
date -u +%FT%TZ > $S/logs/isolation_manifest_before.time
sh $S/scripts/isolation_manifest.sh > $S/logs/isolation_manifest_before.txt
suites > $S/logs/baseline_suite.log
$PY $S/scripts/run.py > $S/logs/run_stdout.txt 2>&1
$PY $S/scripts/summarize.py > $S/logs/summary_stdout.txt 2>&1
date -u +%FT%TZ > $S/logs/isolation_manifest_after.time
sh $S/scripts/isolation_manifest.sh > $S/logs/isolation_manifest_after.txt
suites > $S/logs/final_suite.log
echo "orchestration complete $(date -u +%FT%TZ)" >> $S/logs/orchestrate.log
