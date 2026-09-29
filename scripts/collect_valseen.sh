#!/usr/bin/env bash
# Render val_seen (778 episodes, 53 scans) so the STOP threshold can be chosen
# on a VALIDATION set and frozen before scoring val_unseen.
#
# Every threshold-swept number reported so far picked the best value ON
# val_unseen -- that is test-set tuning and is optimistic. val_seen uses
# TRAINING scenes with held-out episodes, so tuning there never touches the
# test scenes.
PY_HAB=/home/kneepolean/miniconda3/envs/habitat_render/bin/python
cd /home/kneepolean/shubhj/latentpilot || exit 1
echo "######## val_seen collect started $(date) ########"
PYTHONPATH=src $PY_HAB -u src/data/collect_rollouts.py --split val_seen 2>&1 \
  | grep --line-buffered -viE "^\[|warning|nv-|Lighting Layout|no-layout"
echo "######## val_seen collect finished $(date) ########"
