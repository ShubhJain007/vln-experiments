#!/usr/bin/env bash
# Evaluate everything trained overnight. NOTE: no grep filter on the python
# output -- the previous runner filtered to metric lines only, which silently
# swallowed a TypeError and made four evals look like they produced nothing.
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1
for ck in pointing_6scan/final pointing_hist/final; do
  for mode in "--strict" ""; do
    echo "######## $ck  ${mode:-diagnostic}  n=150 thr=0.05 ########"
    $PY -u scripts/eval_pointing.py --checkpoint "checkpoints/$ck" \
        --split val_unseen --limit 150 --stop-threshold 0.05 $mode 2>&1 \
      | grep -v "Loading weights\|tied weights"
    echo "  (exit ${PIPESTATUS[0]})"
  done
done
echo "######## ALL EVALS DONE $(date) ########"
