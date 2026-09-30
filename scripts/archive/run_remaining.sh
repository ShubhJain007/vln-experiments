#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="" HF_HUB_OFFLINE=1
exec "${PY:-python}" -u scripts/run_ablations.py \
  --limit 40 --chunk 24 --steps 20 --guidance 7.5 --stop-patience 2 \
  --arms reasoner_diffusion_grounded reasoner_direct reasoner_diffusion_bare \
         r2r_av_reasoner av_action_only r2r_av_action fixed_forward r2r_raw
