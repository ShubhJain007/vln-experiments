#!/usr/bin/env bash
# D26 — test whether D25's lambda drift is what broke Stage 1.
#
# Held fixed vs the failed run: data (6 scans), seed, LoRA, schedule, steps.
# Changed: ONLY the lambda weighting. Sec. 0.5, one thing at a time.
#
#   balanced  lambda_t recomputed each step to hold lambda*L_pil/L_act = 0.9
#             (0.9 = the stable ratio the identity control settled at while
#              navigating 4x better than the learned run)
#   lam 0.02  the simplest fix: a constant closer to the paper's form.
#             0.9 * L_act_final / L_pil_final = 0.9*0.055/3.0 = 0.017 -> 0.02
#
# Baseline to beat (stage1_learned/final): OS 0.10, NE 8.58
# Bar set by the control (stage1_identity/final): OS 0.40, NE 7.40

PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

STEPS=17400
BS=8

echo "######## D26 balanced (ratio 0.9) started $(date) ########"
$PY -u src/train/train_stage1.py --steps $STEPS --batch-size $BS \
    --lambda-mode balanced --target-ratio 0.9 \
    --out checkpoints/stage1_balanced 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## D26 balanced finished $(date) ########"

echo "######## D26 lam=0.02 fixed started $(date) ########"
$PY -u src/train/train_stage1.py --steps $STEPS --batch-size $BS \
    --lambda-mode fixed --lam 0.02 \
    --out checkpoints/stage1_lam002 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## D26 lam=0.02 finished $(date) ########"

echo "######## ALL DONE $(date) ########"
