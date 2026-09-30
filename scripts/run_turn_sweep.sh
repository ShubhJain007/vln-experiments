#!/bin/bash
# Closed-loop sweep over the bearing->action decision boundary.
# Per-step accuracy peaks at 7.5d, but navigation is asymmetric: a missed turn
# is unrecoverable (wrong room), a spurious turn costs ~2 steps. So the
# accuracy-optimal threshold need not be the success-optimal one.
# Full logs kept -- a previous runner grepped stdout to metric lines and
# silently swallowed a TypeError, reporting four "completed" evals with no data.
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
CK=checkpoints/pointing_fusion/step120000
OUT=logs/turnsweep_$(date +%m%d_%H%M%S)
mkdir -p "$OUT"
for T in 3.0 4.0 5.0 7.5 10.0; do
  echo "=== turn threshold ${T} deg ===" | tee -a "$OUT/summary.txt"
  PYTHONPATH="" HF_HUB_OFFLINE=1 $PY -u scripts/eval_pointing.py \
      --checkpoint "$CK" --split val_unseen --limit 150 \
      --stop-threshold 0.20 --turn-threshold-deg "$T" \
      > "$OUT/turn_${T}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=$rc -- tail:" | tee -a "$OUT/summary.txt"
    tail -15 "$OUT/turn_${T}.log" | sed 's/^/    /' | tee -a "$OUT/summary.txt"
  else
    grep -E "^(SR|OS|SPL|nDTW|NE|stopped|self)" "$OUT/turn_${T}.log" \
      | sed 's/^/  /' | tee -a "$OUT/summary.txt"
  fi
done
echo "SWEEP DONE" | tee -a "$OUT/summary.txt"
