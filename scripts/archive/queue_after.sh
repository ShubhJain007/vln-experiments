#!/bin/bash
# Requeue reasoner_direct after the current driver finishes all its arms.
# Waits on the driver PROCESS, not on a file glob -- the glob approach matched a
# leftover smoke-test JSON and killed a running arm at 34/40.
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}            # python of the latentpilot env (environment/latentpilot.yml)
while pgrep -f "run_ablations.py --limit" >/dev/null 2>&1; do sleep 60; done
echo "$(date +%H:%M:%S) driver finished; running reasoner_direct"
exec $PY -u scripts/run_ablations.py --limit 40 --chunk 24 --steps 20 \
     --guidance 7.5 --stop-patience 2 --arms reasoner_direct
