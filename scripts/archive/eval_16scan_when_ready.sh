#!/usr/bin/env bash
# Closes the gap in run_step1_then_step2.sh: that script TRAINS
# stage1_16scan but never evaluates it. Without this the 2.4 h run ends with
# no closed-loop number, which is the only thing it was run to produce.
#
# Kept as a SEPARATE process rather than an edit, because bash reads a script
# incrementally -- editing the file while it runs can make it execute garbage.

PY_LP=${PY:-python}         # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

CK=checkpoints/stage1_16scan/final
N=150

echo "waiting for $CK ..."
# Wait for the adapter to exist AND training to have exited, so we never load
# a checkpoint that is still being written.
while [ ! -f "$CK/adapter.pt" ] || pgrep -f "train_stage1.py --steps" | grep -qv "^$$\$"; do
  sleep 60
done
sleep 30   # let the final write settle

echo "######## 16-scan eval (n=$N) started $(date) ########"
$PY_LP -u scripts/eval_stage0_prime.py --checkpoint "$CK" \
    --split val_unseen --limit $N 2>&1 \
  | grep -v "Loading weights\|tied weights" | tee logs/evaln150_stage1_16scan.log
echo "######## 16-scan eval finished $(date) ########"

echo "######## recording sample rollouts $(date) ########"
$PY_LP -u scripts/record_episode.py --checkpoint "$CK" \
    --split val_unseen --per-scan 1 --fps 4 \
    --out recordings/16scan_multiscene 2>&1 \
  | grep -v "Loading weights\|tied weights" | grep -E "^\[|->|recordings"
echo "######## ALL COMPLETE $(date) ########"
