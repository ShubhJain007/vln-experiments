#!/usr/bin/env bash
# Longer Stage 2: does 3x the scheduled-sampling fine-tune (7,800 steps, 75 % own latents) remove the Pilot-slot
# shortcut? Then the held-out slot test and the closed-loop eval (OS, n=150). ~2 h on one 16 GB GPU.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}
export HABITAT_PYTHON=${HABITAT_PYTHON:-python} PYTHONPATH="" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGE1_SCANS="8WUmhLawc2A JeFG25nYj2p Vvot9Ly1tCj ac26ZMwG7aT r47D5H71a5s ur6pFq6Qu1A"
filter() { grep --line-buffered -v "Loading weights\|it/s\]"; }
set -o pipefail
echo "######## Stage 2 long: 7,800 steps, p_final=0.75  $(date)"
$PY -u src/train/train_stage2.py --scans $STAGE1_SCANS --p-final 0.75 --steps 7800 --out checkpoints/stage2_p75_long 2>&1 | filter || exit 1
echo "######## slot test  $(date)"
$PY -u scripts/test_slot_shortcut.py --checkpoint checkpoints/stage2_p75_long/final --name stage2_p75_long --per-scan 10 2>&1 | filter
echo "######## closed loop, n=150 (diagnostic, OS)  $(date)"
$PY -u scripts/eval_stage0_prime.py --checkpoint checkpoints/stage2_p75_long/final --limit 150 2>&1 | filter
echo "######## done  $(date)"
