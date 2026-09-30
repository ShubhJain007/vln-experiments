#!/usr/bin/env bash
# D27 pointing training on the full 16-scan corpus (4,203 eps / 171,281 steps).
# 20,000 steps x batch 8 = 160k samples ~= 0.93 epochs.
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1
echo "######## POINTING train started $(date) ########"
$PY -u src/train/train_pointing.py --steps 20000 --batch-size 8 \
    --out checkpoints/pointing 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## POINTING train finished $(date) ########"
