#!/usr/bin/env bash
# OVERNIGHT: pointing + frame history (D28) on the FULL 61-scan corpus.
#
# Waits for the 6-scan ablation and the rollout collection to finish, probes
# the largest batch that fits on the now-free GPU, then trains and evaluates.
#
# WHY THE PROBE: history is quadratic in sequence length. K=2 means 588 visual
# tokens against 196, and the batch that fits with the GPU shared (2) is not
# the batch that fits with it free. Hard-coding a batch size here would either
# waste the card or OOM six hours in.
PY=/home/kneepolean/miniconda3/envs/latentpilot/bin/python
cd /home/kneepolean/shubhj/latentpilot || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

STRIDE=8
K=2                     # 3 frames total. Robostral uses frame history; K=2 is
                        # the most we can afford at this sequence cost.
HOURS=7.0

echo "waiting for ablation + collection ..."
while pgrep -f "train_pointing.py --steps 20000" >/dev/null \
   || pgrep -f "collect_rollouts.py --split train" >/dev/null; do sleep 60; done
sleep 30
echo "######## GPU free $(date) ########"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ---- probe the largest batch that fits --------------------------------
BS=$($PY - <<'PY' 2>/dev/null | tail -1
import sys, torch
sys.path.insert(0,'src')
from data.pointing_dataset import PointingDataset
from model.input_sequence import InputSequenceBuilder
from model.backbone import Backbone
from train.train_pointing import build_model, make_loader, forward_batch
from losses.pointing_loss import pointing_loss
import bitsandbytes as bnb
ds=PointingDataset('train',max_episodes=40,history_frames=K)
enc,head=build_model(grad_checkpointing=True)
b=InputSequenceBuilder(enc.model,enc.processor.tokenizer); bb=Backbone(enc.model)
params=[p for p in enc.model.parameters() if p.requires_grad]+list(head.parameters())
opt=bnb.optim.AdamW8bit(params,lr=1e-4)
best=1
for bs in (2,3,4,6):
    torch.cuda.empty_cache()
    try:
        it=iter(make_loader(ds,bs,workers=0))
        for _ in range(2):
            fr,ins,hist,tgt=next(it)
            tgt={k:v.to('cuda') for k,v in tgt.items()}
            pred=forward_batch(enc,b,bb,head,fr,ins,hist)
            loss,_=pointing_loss(pred,tgt,40.0); loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        best=bs
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); break
print(best)
PY
)
BS=${BS:-2}
echo "######## probed batch size: $BS ########"

# Steps that fit the wall-clock budget, measured at ~1.5 it/s for K=2.
STEPS=$(python3 -c "print(int($HOURS*3600*1.5))")
echo "######## HISTORY train K=$K bs=$BS steps=$STEPS started $(date) ########"
$PY -u src/train/train_pointing.py --steps "$STEPS" --batch-size "$BS" \
    --history-frames $K --history-stride $STRIDE --save-every 2000 \
    --out checkpoints/pointing_hist 2>&1 \
  | grep --line-buffered -v "it/s\]\|tied weights\|Loading weights"
echo "######## HISTORY train finished $(date) ########"

for mode in "--strict" ""; do
  echo "######## HISTORY eval n=150 ${mode:-diagnostic} ########"
  $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing_hist/final \
      --split val_unseen --limit 150 --stop-threshold 0.05 $mode 2>&1 \
    | grep -v "Loading weights\|tied weights" \
    | grep -E "SR    =|SPL   =|OS    =|nDTW  =|NE    =|model_stop|within_radius|timeout|p\(stop\)"
done
echo "######## OVERNIGHT COMPLETE $(date) ########"
