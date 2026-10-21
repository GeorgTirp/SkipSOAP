#!/bin/sh
# Tuned 1,000-step comparison (from residual-curvature-lab/; PROTOCOL.md): before-manifest and suites, seed-0 tuning,
# the extension run for any selected lr at a grid edge, the evaluation on seeds 1-3, the summary, after-manifest and
# suites. Launched detached (setsid nohup). Stops at the first failing phase.
set -e
PY=.venv/bin/python
S=studies/ot_tuned_1000_20261014
T=$S/scripts/train.py
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
suites() {
    for t in tests studies/parsec_l_20261009/tests studies/cifar_ggn_20261012/tests studies/cifar_sn_soap_20261012/tests \
             studies/basis_refresh_snr_20261013/tests studies/n9_training_20261013/tests studies/probe_variants_20261013/tests \
             studies/ladder_mechanism_20261013/tests studies/long_run_20261013/tests studies/ggn_c_control_20261013/tests \
             studies/ot_1000_20261013/tests studies/ot_tuned_1000_20261014/tests; do
        echo "$t: $($PY -m pytest -p no:cacheprovider -q $t 2>&1 | tail -1)"
    done
}
echo "$(date -u +%FT%TZ) start" >> $S/logs/orchestrate.log
date -u +%FT%TZ > $S/logs/isolation_manifest_before.time
sh $S/scripts/isolation_manifest.sh > $S/logs/isolation_manifest_before.txt
suites > $S/logs/baseline_suite.log
TUNE=$($PY $T create --phase tune | tail -1)
echo "$TUNE" > $S/logs/run_tune.txt
sh $S/scripts/run_queue.sh "$TUNE"
EXT=$($PY $T create --phase extend --tuning "$TUNE" | tail -1)
TUNINGS="$TUNE"
if [ "$EXT" != "none" ]; then
    echo "$EXT" > $S/logs/run_extend.txt
    sh $S/scripts/run_queue.sh "$EXT"
    TUNINGS="$TUNE $EXT"
fi
EVAL=$($PY $T create --phase eval --tuning $TUNINGS | tail -1)
echo "$EVAL" > $S/logs/run_eval.txt
sh $S/scripts/run_queue.sh "$EVAL"
$PY $S/scripts/summarize.py --tuning $TUNINGS --eval "$EVAL" > $S/logs/summary_stdout.txt 2>&1
date -u +%FT%TZ > $S/logs/isolation_manifest_after.time
sh $S/scripts/isolation_manifest.sh > $S/logs/isolation_manifest_after.txt
suites > $S/logs/final_suite.log
echo "orchestration complete $(date -u +%FT%TZ)" >> $S/logs/orchestrate.log
