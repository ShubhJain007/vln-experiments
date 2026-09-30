#!/usr/bin/env bash
# DATA SCALING ABLATION — is the ceiling data, or architecture?
#
# Everything is held fixed except the data pool: same 20,000 steps, same seed,
# same LR schedule, same architecture. Fixed COMPUTE, varied DATA -- the
# standard form for a scaling question. (The 6-scan arm therefore sees each
# sample ~2.5x more often; that is the point, not a confound.)
#
#   6-scan  : 1,665 episodes /  69,606 steps  <- LatentPilot's EXACT corpus
#   16-scan : 4,203 episodes / 171,281 steps  <- already trained
#
# This also un-confounds an earlier claim: "pointing beats LatentPilot
# (p=0.0055)" compared a 4,203-episode pointing run against 1,665-episode
# LatentPilot runs, so architecture and data scale were entangled. The 6-scan
# pointing run is the matched comparison LatentPilot deserves.
#
# Big gap  -> data-limited; scaling to the full 61 scans is the priority.
# Flat     -> data is not binding at this scale; the ceiling is elsewhere.
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

while pgrep -f "train_pointing.py --steps 20000 --batch-size 8 --out checkpoints/pointing$" >/dev/null; do sleep 60; done
while pgrep -f "run_strict_sweep.sh" >/dev/null; do sleep 30; done

echo "######## 6-SCAN pointing train started $(date) ########"
$PY -u src/train/train_pointing.py --steps 20000 --batch-size 8 \
    --scans 8WUmhLawc2A JeFG25nYj2p Vvot9Ly1tCj ac26ZMwG7aT r47D5H71a5s ur6pFq6Qu1A \
    --out checkpoints/pointing_6scan 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## 6-SCAN train finished $(date) ########"

for mode in "" "--strict"; do
  echo "######## 6-SCAN eval ${mode:-diagnostic} ########"
  $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing_6scan/final \
      --split val_unseen --limit 150 --stop-threshold 0.10 $mode 2>&1 \
    | grep -v "Loading weights\|tied weights" \
    | grep -E "SR |SPL |OS |nDTW |NE |model_stop|within_radius|timeout|p\(stop\)"
done
echo "######## DATA ABLATION DONE $(date) ########"
