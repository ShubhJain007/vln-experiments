#!/bin/bash
# Two ablations under identical conditions -- full-chunk execution, STOP only
# after `stop_patience` consecutive empty chunks:
#   A) the real R2R instruction   -- tests whether it reasons about the task
#   B) a fixed "move forward"     -- tests control with no task knowledge
# If B beats A, the instruction is hurting rather than helping.
PY=/home/kneepolean/miniconda3/envs/latentpilot/bin/python
OUT=logs/edge_ablation_$(date +%m%d_%H%M%S); mkdir -p "$OUT"
COMMON="--limit 40 --max-steps 100 --chunk 24 --steps 20 --guidance 7.5 --stop-patience 2"

echo "=== A: R2R instruction ===" | tee -a "$OUT/summary.txt"
PYTHONPATH="" HF_HUB_OFFLINE=1 $PY -u scripts/eval_edge.py $COMMON > "$OUT/A_r2r.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "FAILED rc=$rc"; tail -15 "$OUT/A_r2r.log"; } | tee -a "$OUT/summary.txt"

echo "=== B: fixed 'move forward' ===" | tee -a "$OUT/summary.txt"
PYTHONPATH="" HF_HUB_OFFLINE=1 $PY -u scripts/eval_edge.py $COMMON \
    --fixed-prompt "The camera moves forward." > "$OUT/B_fixed.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "FAILED rc=$rc"; tail -15 "$OUT/B_fixed.log"; } | tee -a "$OUT/summary.txt"

for f in "$OUT"/A_r2r.log "$OUT"/B_fixed.log; do
  echo "--- $(basename $f) ---" | tee -a "$OUT/summary.txt"
  grep -E "SR |SPL |OS |nDTW |NE |model_stop|timeout|within_radius|ever within|net forward|net \|yaw\||implied|calls/episode" \
    "$f" | tee -a "$OUT/summary.txt"
done
echo "ABLATION DONE" | tee -a "$OUT/summary.txt"
