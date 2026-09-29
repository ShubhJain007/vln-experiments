#!/usr/bin/env bash
# Stage 0' A/B overnight run — D17 Designs A and B.
#
# A: pilot_mode=last          [<vs>; v_t; <ve>; Tok(x); PILOT(z)]
#    h_act == h_pil == H[-1]  -- one hidden state, two heads
# B: pilot_mode=action_query  [... ; PILOT(z); ACTION_QUERY]
#    h_pil = H[-2], h_act = H[-1] -- distinct states
#
# Everything else is identical (data, seed, cosine schedule, LoRA config), so
# any difference is attributable to the Pilot slot layout alone.
#
# Two deliberate choices:
#  * NOT chained with && -- if A fails, B must still run. Waking to two
#    failures is recoverable; waking to one failure that silently blocked the
#    other wastes the whole night.
#  * python -u AND grep --line-buffered -- without both, output sits in a pipe
#    buffer for hours and the run cannot be monitored while it happens.

PY=/home/kneepolean/miniconda3/envs/latentpilot/bin/python
cd /home/kneepolean/shubhj/latentpilot || exit 1
export PYTHONPATH=""

STEPS=17400          # 2 full epochs over 69,606 training pairs
BS=8

echo "############ RUN A (pilot_mode=last) started $(date) ############"
$PY -u src/train/train_stage0_prime.py \
    --pilot-mode last --batch-size $BS --steps $STEPS \
    --out checkpoints/stage0prime_A 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights"
echo "############ RUN A finished $(date) ############"

echo "############ RUN B (pilot_mode=action_query) started $(date) ############"
$PY -u src/train/train_stage0_prime.py \
    --pilot-mode action_query --batch-size $BS --steps $STEPS \
    --out checkpoints/stage0prime_B 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights"
echo "############ RUN B finished $(date) ############"

echo "############ ALL DONE $(date) ############"
