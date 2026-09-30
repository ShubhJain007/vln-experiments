#!/usr/bin/env bash
# Full-corpus model FIRST (61 scans / 10,819 episodes), then the 6-scan ablation.
# No grep on the metric output -- the previous runner filtered stdout to metric
# lines only and silently swallowed a TypeError, making four evals look empty.
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1
for ck in pointing_hist/final pointing_6scan/final; do
  for mode in "--strict" ""; do
    echo "######## $ck  ${mode:-diagnostic}  n=150 thr=0.05 ########"
    $PY -u scripts/eval_pointing.py --checkpoint "checkpoints/$ck" \
        --split val_unseen --limit 150 --stop-threshold 0.05 $mode 2>&1 \
      | grep -v "Loading weights\|tied weights"
  done
done
echo "######## ALL EVALS DONE $(date) ########"
