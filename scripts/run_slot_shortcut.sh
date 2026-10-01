#!/usr/bin/env bash
# Offline test of the LatentPilot slot shortcut (see scripts/test_slot_shortcut.py). ~1 h on one 16 GB GPU.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
export PYTHONPATH="" HF_HUB_OFFLINE=1
run() { $PY -u scripts/test_slot_shortcut.py "$@" 2>&1 | grep --line-buffered -v "Loading weights\|it/s\]"; }
echo "######## main comparison, 10 episodes per scan  $(date)"
run --checkpoint checkpoints/stage1_learned/final   --name stage1_learned   --per-scan 10
run --checkpoint checkpoints/stage1_identity/final  --name stage1_identity  --per-scan 10
run --checkpoint checkpoints/stage0prime_B/final    --name stage0prime_B    --per-scan 10
run --checkpoint checkpoints/stage0/final           --name stage0           --per-scan 10
echo "######## learned G_psi over training, 5 episodes per scan  $(date)"
for s in 2000 4000 6000 8000 10000 12000 14000 16000; do
  run --checkpoint checkpoints/stage1_learned/step$s --name stage1_learned_step$s --per-scan 5 \
      --out results/slot_shortcut_over_training.json
done
run --checkpoint checkpoints/stage1_learned/final --name stage1_learned_step17400 --per-scan 5 \
    --out results/slot_shortcut_over_training.json
echo "######## done  $(date)"
