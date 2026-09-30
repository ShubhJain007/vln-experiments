#!/usr/bin/env bash
# Render rollouts for the full 61-scan R2R train split (10,819 episodes).
# Habitat rendering is CPU/GPU-light, so it overlaps with GPU training.
# Existing episodes are skipped (no --overwrite), so the 16 scans already
# collected with rotations are left untouched.
PY_HAB=${HABITAT_PYTHON:-python}   # python of the habitat_render env (environment/habitat_render.yml)
cd "$(dirname "$0")/.." || exit 1
echo "######## FULL collect started $(date) ########"
PYTHONPATH=src $PY_HAB -u src/data/collect_rollouts.py --split train 2>&1 \
  | grep --line-buffered -viE "^\[|warning|nv-|Lighting Layout|no-layout"
echo "######## FULL collect finished $(date) ########"
