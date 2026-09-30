#!/usr/bin/env bash
# STEP 1  n=150 evals -- firm up the one robust claim (identity > learned).
#         At n=30 every CI spanned ~0.24; n=150 narrows that to ~0.11, and
#         n=42 is the minimum to call 0.13 vs 0.40 at 80% power.
# STEP 2  cache v_bar for the 10 new scans, then train on all 16 scans
#         (4,203 episodes, 2.5x the data) to test whether "identity beats
#         learned" survives more data -- it may be a small-data artefact.
#
# Serialised on purpose: the fp32 v_bar encoder alone is 9.08 GB and an eval
# is ~6.1 GB, which does not fit together on a 15.56 GB card (D7 forbids
# dropping to bf16 -- it would corrupt the Eq. 14 targets).

PY_LP=${PY:-python}         # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

N=150

echo "######## STEP 1: n=$N evals started $(date) ########"
for ck in stage1_identity/final stage1_learned/final stage1_balanced/step10000; do
  tag=$(echo "$ck" | tr '/' '_')
  echo "---- eval $ck (n=$N) ----"
  $PY_LP -u scripts/eval_stage0_prime.py --checkpoint "checkpoints/$ck" \
      --split val_unseen --limit $N 2>&1 \
    | grep -v "Loading weights\|tied weights" \
    | tee "logs/evaln${N}_${tag}.log" | grep -E "SR|SPL|OS|nDTW|NE|model_stop|within_radius|timeout|\|\|z_t\|\|"
done
echo "######## STEP 1 finished $(date) ########"

echo "######## STEP 2a: cache v_bar started $(date) ########"
$PY_LP -u src/data/cache_vbar.py --split train 2>&1 | grep --line-buffered -v "it/s\]"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "CACHE FAILED rc=$rc -- not training"; exit "$rc"; fi
echo "######## STEP 2a finished $(date) ########"

echo "######## STEP 2b: train on 16 scans started $(date) ########"
# lambda-mode fixed at the paper's 0.1: the lambda axis was shown flat
# (balanced vs learned, p=0.333), so we hold it at the paper's value and
# change ONLY the data -- Sec. 0.5, one variable.
$PY_LP -u src/train/train_stage1.py --steps 17400 --batch-size 8 \
    --lambda-mode fixed --lam 0.1 \
    --out checkpoints/stage1_16scan 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## STEP 2b finished $(date) ########"
echo "######## ALL DONE $(date) ########"
