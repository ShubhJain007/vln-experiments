#!/usr/bin/env bash
# Follow-up to the slot-shortcut test: does Stage 2 (scheduled sampling on the Pilot slot) remove the shortcut?
# Fine-tunes Stage 1 (learned G_psi) on the SAME 6 train scans, mixing the slot input between the true next frame and
# the model's own z_{t-1}, then re-runs the held-out slot test (scripts/test_slot_shortcut.py). ~1.5 h, one 16 GB GPU.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
export PYTHONPATH="" HF_HUB_OFFLINE=1
STAGE1_SCANS="8WUmhLawc2A JeFG25nYj2p Vvot9Ly1tCj ac26ZMwG7aT r47D5H71a5s ur6pFq6Qu1A"   # 1,665 episodes, as Stage 1
filter() { grep --line-buffered -v "Loading weights\|it/s\]"; }
for p in ${P_FINALS:-0.5 0.75}; do   # Sec. 5.4 of Agents.md: p_final above 0.75 needs explicit approval
  tag=p$(printf '%.0f' "$(echo "$p*100" | bc)")
  echo "######## Stage 2, p_final=$p  $(date)"
  $PY -u src/train/train_stage2.py --scans $STAGE1_SCANS --p-final $p --out checkpoints/stage2_$tag 2>&1 | filter
  echo "######## slot test, stage2_$tag  $(date)"
  $PY -u scripts/test_slot_shortcut.py --checkpoint checkpoints/stage2_$tag/final --name stage2_$tag --per-scan 10 2>&1 | filter
done
echo "######## done  $(date)"
