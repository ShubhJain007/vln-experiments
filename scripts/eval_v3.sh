#!/bin/bash
cd /home/kneepolean/shubhj/latentpilot
export PYTHONPATH="" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kneepolean/miniconda3/envs/latentpilot/bin/python
for CK in final step1500; do
  $PY -u scripts/eval_cosmos_nav.py --policy direct --no-think --limit 40 \
      --adapter checkpoints/reasoner_sft_v3/$CK --arm-name v3_${CK}_direct \
      --results-json results/runs/v3_${CK}_direct.json > logs/eval_v3_${CK}.log 2>&1
  echo "done $CK rc=$?"
done
$PY -u scripts/probe_reasoner_nav.py --adapter checkpoints/reasoner_sft_v3/final \
    --variants hier_overall_nothink --trajectories 40 --points 4 \
    --out results/probe_v3_final > logs/probe_v3_final.log 2>&1
echo "done probe rc=$?"
echo ALL_DONE
