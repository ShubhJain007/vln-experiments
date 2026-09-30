#!/usr/bin/env bash
# Cache v_bar for the 10 new scans, but ONLY once the GPU is free.
#
# WHY THE WAIT: cache_vbar loads the vision encoder in float32 (D7 -- fp32 is
# batch-invariant to 1.3e-6 where bf16 is 6.8e-2, and these are the Eq. 14
# regression targets). That is 9.08 GB on its own, which does not fit beside a
# training run holding ~5.9 GB on a 15.56 GB card. Dropping to bf16 to make it
# fit would silently corrupt every target, so we wait instead.

PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="" HF_HUB_OFFLINE=1

echo "waiting for training to finish before caching v_bar ..."
while pgrep -f "train_stage1.py --steps" > /dev/null; do sleep 60; done
echo "######## GPU free -- CACHE v_bar started $(date) ########"

$PY -u src/data/cache_vbar.py --split train 2>&1 | grep --line-buffered -v "it/s\]"
rc=${PIPESTATUS[0]}
echo "######## CACHE v_bar finished (rc=$rc) $(date) ########"
exit "$rc"
