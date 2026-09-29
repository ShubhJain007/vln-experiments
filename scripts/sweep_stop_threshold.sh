#!/usr/bin/env bash
# The point of a separate binary STOP head: the operating point is chosen
# AFTER training. Measured p(stop) on step10000: mean 0.013, p95 0.044,
# max 0.253 -- so the default 0.5 never fires. Sweep around the observed range.
PY=/home/kneepolean/miniconda3/envs/latentpilot/bin/python
cd /home/kneepolean/shubhj/latentpilot || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1
for thr in 0.10 0.05 0.03; do
  echo "######## stop_threshold=$thr ########"
  $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing/step10000 \
      --split val_unseen --limit 60 --stop-threshold $thr 2>&1 \
    | grep -v "Loading weights\|tied weights" \
    | grep -E "SR |SPL |OS |nDTW |NE |model_stop|within_radius|timeout|p\(stop\)|would fire"
done
echo "######## SWEEP DONE ########"
