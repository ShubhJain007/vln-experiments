#!/usr/bin/env bash
# Re-collect all 16 train scans WITH per-step rotations + camera intrinsics.
#
# WHY RE-COLLECT rather than reconstruct yaw from the action list: turns are
# deterministic, so reconstruction is exact 93.8% of the time -- but the
# remaining 6% land ~15 deg off (one full turn) where collisions desynchronise
# the action/frame alignment. Pointing targets are projected through that yaw,
# so a 15 deg error puts the target waypoint in the wrong half of the image.
# Silently-wrong labels are the most expensive kind; 20 minutes of rendering is
# cheaper than debugging them later.
PY_HAB=${HABITAT_PYTHON:-python}   # python of the habitat_render env (environment/habitat_render.yml)
cd "$(dirname "$0")/.." || exit 1

echo "######## RECOLLECT train (16 scans) started $(date) ########"
PYTHONPATH=src $PY_HAB -u src/data/collect_rollouts.py --split train --overwrite 2>&1 \
  | grep --line-buffered -viE "^\[|warning|nv-|Lighting Layout"
echo "######## RECOLLECT train finished $(date) ########"

echo "######## RECOLLECT val_unseen started $(date) ########"
PYTHONPATH=src $PY_HAB -u src/data/collect_rollouts.py --split val_unseen --overwrite 2>&1 \
  | grep --line-buffered -viE "^\[|warning|nv-|Lighting Layout"
echo "######## RECOLLECT val_unseen finished $(date) ########"
echo "######## DONE $(date) ########"
