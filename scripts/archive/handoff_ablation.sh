#!/bin/bash
# The running driver holds the OLD ARMS list in memory, so it would launch the
# bare-caption diffusion arm next. Wait until reasoner_direct writes its JSON
# (its 40 episodes are then safely recorded), stop that driver, and relaunch
# with the corrected ARMS -- skipping reasoner_direct so it is not redone.
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)

while ! ls results/runs/*_reasoner_direct.json >/dev/null 2>&1; do sleep 20; done
echo "$(date +%H:%M:%S) reasoner_direct JSON written; handing off"

PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
[ -n "$PID" ] && kill -TERM -"$(ps -o pgid= -p "$PID" | tr -d ' ')" 2>/dev/null
sleep 8

exec $PY -u scripts/run_ablations.py --limit 40 --chunk 24 --steps 20 \
     --guidance 7.5 --stop-patience 2 \
     --arms reasoner_diffusion_grounded reasoner_diffusion_bare r2r_av_reasoner \
            av_action_only r2r_av_action fixed_forward r2r_raw
