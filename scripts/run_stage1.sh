#!/usr/bin/env bash
# Stage 1 — AGENTS.md Sec. 4.2 requires BOTH runs for the decisive gate.
#
#   learned   L = L_act + 0.1*L_pil, G_psi trainable        (the claim)
#   identity  G_psi frozen to I, so z_t = h_t^pil           (gate (b), the control)
#
# Sec. 4.2: "If the learned model does not beat identity-passthrough on L_pil
# held-out error and on SR, the model has learned to copy its input, not to
# predict dynamics. Report this either way."
#
# Not chained with && -- if the learned run dies, the control must still run.
# python -u + grep --line-buffered so the log is watchable while it happens.

PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

STEPS=17400          # 2 epochs over 69,606 pairs at batch 8
BS=8

echo "######## STAGE 1 learned  started $(date) ########"
$PY -u src/train/train_stage1.py --steps $STEPS --batch-size $BS \
    --g-psi learned --out checkpoints/stage1_learned 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## STAGE 1 learned  finished $(date) ########"

echo "######## STAGE 1 identity started $(date) ########"
$PY -u src/train/train_stage1.py --steps $STEPS --batch-size $BS \
    --g-psi identity --out checkpoints/stage1_identity 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## STAGE 1 identity finished $(date) ########"

echo "######## ALL DONE $(date) ########"
