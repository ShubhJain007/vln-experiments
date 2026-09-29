#!/usr/bin/env bash
# Download the FULL R2R corpus: 61 train + 11 val_unseen scans (10,819 train
# episodes, 2.6x our current 4,203).
#
# The ToS prompt is auto-confirmed. This proceeds under the Matterport3D terms
# the user ALREADY accepted when downloading the first 27 scans; it is the same
# dataset and the same agreement, and the user explicitly requested the full
# download. MP3D remains RESEARCH-ONLY -- see D-notes on licensing.
PY=/home/kneepolean/miniconda3/envs/latentpilot/bin/python
cd /home/kneepolean/shubhj/latentpilot || exit 1
export PYTHONPATH=""
echo "######## MP3D full download started $(date) ########"
yes "" | $PY -u scripts/download_mp3d.py --episodes 99999 2>&1 \
  | grep --line-buffered -vE "^\s*$"
echo "######## MP3D full download finished $(date) ########"
