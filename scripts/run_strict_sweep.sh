#!/usr/bin/env bash
# TRUE success rate: no proximity break, so an episode ends only on the MODEL's
# own STOP or on timeout. Every SR number we have reported so far is really
# oracle-success -- with the proximity break in place a model_stop can only
# fire when the agent is already >3 m from the goal, i.e. it is structurally
# guaranteed to be a failure. This is the first measurement that can show a
# non-zero agent-initiated SR.
PY=/home/kneepolean/miniconda3/envs/latentpilot/bin/python
cd /home/kneepolean/shubhj/latentpilot || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

# Wait for the threshold sweep to finish so they do not fight over the GPU.
while pgrep -f "sweep_stop_threshold.sh" >/dev/null; do sleep 20; done

for thr in 0.10 0.05 0.03; do
  echo "######## STRICT stop_threshold=$thr ########"
  $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing/step10000 \
      --split val_unseen --limit 60 --stop-threshold $thr --strict 2>&1 \
    | grep -v "Loading weights\|tied weights" \
    | grep -E "SR |SPL |OS |nDTW |NE |model_stop|within_radius|timeout|p\(stop\)|would fire"
done
echo "######## STRICT SWEEP DONE ########"
