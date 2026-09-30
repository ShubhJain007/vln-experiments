#!/usr/bin/env bash
# Step 2 — collect rollouts + cache v_bar for the 10 newly downloaded train
# scans, taking us from 1,665 episodes / 6 scans to 4,203 / 16.
#
# Runs alongside the D26 lambda experiment on purpose: collection is habitat
# RENDERING (CPU-bound) while training is GPU-bound, so they contend far less
# than two training jobs would.
#
# NOT used by the running lambda experiment -- that stays on the 6-scan data so
# the comparison against the failed baseline holds exactly one variable
# (Sec. 0.5). This data is for the run AFTER lambda is settled.

# TWO ENVIRONMENTS, on purpose (see src/eval/habitat_worker.py):
#   habitat_render (py3.9)  has habitat_sim, cannot run transformers
#   latentpilot    (py3.10) has torch/transformers, cannot import habitat_sim
# Rendering rollouts needs the former; caching v_bar needs the latter.
PY_HAB=${HABITAT_PYTHON:-python}   # python of the habitat_render env (environment/habitat_render.yml)
PY_LP=${PY:-python}         # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

NEW_SCANS="1pXnuDYAj8r 2n8kARJN3HM 5LpN3gDmAk7 B6ByNegPMKs E9uDoFAP3SH \
EDJbREhghzL PX4nDJXEHrG S9hNv5qa7GM mJXqzFtmKg4 sT4fr6TAbpF"

echo "######## COLLECT rollouts started $(date) ########"
PYTHONPATH=src $PY_HAB -u src/data/collect_rollouts.py --split train --scans $NEW_SCANS 2>&1 \
  | grep --line-buffered -v "it/s\]"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "######## COLLECT FAILED (rc=$rc) -- not caching v_bar $(date) ########"
  exit "$rc"
fi
echo "######## COLLECT rollouts finished $(date) ########"

# v_bar must be cached in fp32 (D7) or the Eq.14 targets are batch-dependent.
echo "######## CACHE v_bar started $(date) ########"
$PY_LP -u src/data/cache_vbar.py --split train 2>&1 \
  | grep --line-buffered -v "it/s\]"
echo "######## CACHE v_bar finished $(date) ########"

echo "######## STEP 2 DATA READY $(date) ########"
