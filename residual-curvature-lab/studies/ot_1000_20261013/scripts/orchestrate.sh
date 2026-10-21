#!/bin/sh
# Exploratory 1,000-step check (from residual-curvature-lab/): before-manifest and suites, the three units, the report,
# after-manifest and suites. Runs detached (setsid nohup).
PY=.venv/bin/python
S=studies/ot_1000_20261013
T=$S/scripts/train.py
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
suites() {
    for t in tests studies/parsec_l_20261009/tests studies/cifar_ggn_20261012/tests studies/cifar_sn_soap_20261012/tests \
             studies/basis_refresh_snr_20261013/tests studies/n9_training_20261013/tests studies/probe_variants_20261013/tests \
             studies/ladder_mechanism_20261013/tests studies/long_run_20261013/tests studies/ggn_c_control_20261013/tests \
             studies/ot_1000_20261013/tests; do
        echo "$t: $($PY -m pytest -p no:cacheprovider -q $t 2>&1 | tail -1)"
    done
}
date -u +%FT%TZ > $S/logs/isolation_manifest_before.time
sh $S/scripts/isolation_manifest.sh > $S/logs/isolation_manifest_before.txt
suites > $S/logs/baseline_suite.log
RUN=$($PY $T create | tail -1)
echo "$RUN" > $S/logs/run.txt
sh $S/scripts/run_queue.sh "$RUN" || echo "$(date -u +%FT%TZ) queue FAILED" >> $S/logs/orchestrate.log
$PY $T report --run "$RUN" > $S/logs/report.txt 2>&1
date -u +%FT%TZ > $S/logs/isolation_manifest_after.time
sh $S/scripts/isolation_manifest.sh > $S/logs/isolation_manifest_after.txt
suites > $S/logs/final_suite.log
echo "orchestration complete $(date -u +%FT%TZ)" >> $S/logs/orchestrate.log
