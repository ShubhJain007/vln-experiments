#!/usr/bin/env bash
# Unattended runner for the follow-up experiments. Safe to call repeatedly (cron does, every 15 min and at boot):
#   * a lock (flock, taken by the cron line) means only one copy ever runs;
#   * every item leaves a marker in results/followups/state/, so finished work is never redone and a reboot resumes;
#   * it waits (exits, to retry on the next tick) while the GPU is busy with someone else's job;
#   * a failing item is retried twice, then skipped, so one bad run cannot stall the queue.
#
# Items, in order:
#   stage2_p75        Stage 2 fine-tune at p_final=0.75 + held-out slot test   (needs no scenes)
#   scenes            the 19 Matterport3D buildings, fetched with scripts/download_mp3d.py --scans if a link file exists
#   cl_*              closed-loop: Stage 2 navigation, pointing strict SR, val_seen threshold, reasoner strict SR/repeats
#   summary           results/followups/summary.md + regenerated figures, then ALL_DONE
#
# The Matterport3D link is never stored in the repo. To let this script download the scenes itself, put the link you
# were emailed in ~/.config/vln-experiments/mp3d_habitat_url (chmod 600). Creating that file is your confirmation that
# you have agreed to the Matterport3D terms of use.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/miniconda3/envs/latentpilot/bin/python}
export HABITAT_PYTHON=${HABITAT_PYTHON:-$HOME/miniconda3/envs/habitat_render/bin/python}
FIG_PY=${FIG_PY:-$HOME/shubhj/VisionJev/laya-vision/.venv/bin/python}      # any python with matplotlib + numpy
export PYTHONPATH="" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STATE=results/followups/state; LOGS=logs/followups; mkdir -p "$STATE" "$LOGS"
URL_FILE="$HOME/.config/vln-experiments/mp3d_habitat_url"
VAL_UNSEEN="2azQ1b91cZZ 8194nk5LbLH EU6Fwq7SyZv QUCTc6BB5sX TbHJrupSAjP X7HyMhZNoso Z6MFQCViBuw oLBMNvg9in8 pLe4wQe7qrG x8F5xyUWy9e zsNo4HB9uLZ"
VAL_SEEN="Pm6F8kyY3z2 r47D5H71a5s 5LpN3gDmAk7 i5noydFURQK ULsKaCPVFJR JeFG25nYj2p jh4fc5c5qoQ sT4fr6TAbpF"
STAGE1_SCANS="8WUmhLawc2A JeFG25nYj2p Vvot9Ly1tCj ac26ZMwG7aT r47D5H71a5s ur6pFq6Qu1A"

say() { echo "[$(date '+%F %T')] $*"; }
[ -f "$STATE/ALL_DONE" ] && exit 0

gpu_free() {   # true when no process of another job holds more than 3 GB of GPU memory
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-99999}" -lt 3000 ]
}

run_item() {   # run_item NAME CMD... : skip if done/skipped; retry up to 3 attempts in total; marker on success
  local name=$1; shift
  [ -f "$STATE/$name.done" ] || [ -f "$STATE/$name.skipped" ] && return 0
  if ! gpu_free; then say "GPU busy (another job); will retry on the next tick"; exit 0; fi
  local tries; tries=$(cat "$STATE/$name.tries" 2>/dev/null || echo 0)
  if [ "$tries" -ge 3 ]; then say "$name failed 3 times; skipping"; touch "$STATE/$name.skipped"; return 0; fi
  echo $((tries + 1)) > "$STATE/$name.tries"
  say "start $name (attempt $((tries + 1)))"
  if "$@" > "$LOGS/$name.log" 2>&1; then
    touch "$STATE/$name.done"; say "done  $name"
  else
    say "FAILED $name (see $LOGS/$name.log)"; exit 0      # stop this tick; retried on the next one
  fi
}

# ---------------------------------------------------------------- 1. Stage 2 at p_final=0.75 (no scenes needed)
if pgrep -f "run_stage2_followu[p]|train_stage[2].py" >/dev/null; then say "a Stage 2 run is in progress; waiting"; exit 0; fi
if [ ! -f "$STATE/stage2_p75.done" ] && [ -f checkpoints/stage2_p75/final/adapter.pt ] \
   && grep -q '"stage2_p75"' results/slot_shortcut.json 2>/dev/null; then
  touch "$STATE/stage2_p75.done"                  # finished by the earlier manual run
fi
run_item stage2_p75 bash -c "$PY -u src/train/train_stage2.py --scans $STAGE1_SCANS --p-final 0.75 --out checkpoints/stage2_p75 \
  && $PY -u scripts/test_slot_shortcut.py --checkpoint checkpoints/stage2_p75/final --name stage2_p75 --per-scan 10"

# ---------------------------------------------------------------- 2. scenes
scenes_ready() {
  for s in $VAL_UNSEEN $VAL_SEEN; do
    [ -f "data/scene_datasets/mp3d/$s/$s.glb" ] && [ -f "data/scene_datasets/mp3d/$s/$s.navmesh" ] || return 1
  done
}
if ! scenes_ready; then
  if [ -r "$URL_FILE" ]; then
    say "fetching scenes (link read from $URL_FILE)"
    MP3D_HABITAT_URL=$(head -1 "$URL_FILE") $PY -u scripts/download_mp3d.py --accept-terms \
      --scans $VAL_UNSEEN $VAL_SEEN >> "$LOGS/scenes.log" 2>&1
  fi
  scenes_ready || { say "waiting for scenes (run scripts/download_mp3d.py --scans ..., or create $URL_FILE)"; exit 0; }
fi

# ---------------------------------------------------------------- 3. closed-loop evaluations
for ck in stage2_p50 stage2_p75; do
  run_item "cl_$ck" $PY -u scripts/eval_stage0_prime.py --checkpoint checkpoints/$ck/final --limit 150
done
run_item cl_pointing_final  $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing/final --limit 150 --strict --stop-threshold 0.05
run_item cl_pointing_hist   $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing_hist/final --limit 150 --strict --stop-threshold 0.05
run_item cl_pointing_full3_40k $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing_full3/step40000 --limit 150 --strict --stop-threshold 0.10
for thr in 0.10 0.15 0.20 0.30; do
  run_item "cl_valseen_thr$thr" $PY -u scripts/eval_pointing.py --checkpoint checkpoints/pointing_fusion/step120000 \
    --split val_seen --scans $VAL_SEEN --strict --stop-threshold $thr
done
for r in 1 2 3; do
  run_item "cl_reasoner_strict_r$r" $PY -u scripts/eval_cosmos_nav.py --policy direct --no-think --limit 40 --strict \
    --adapter checkpoints/reasoner_sft_v3/step1500 --arm-name v3_step1500_strict_r$r \
    --results-json results/runs/v3_step1500_direct_strict_r$r.json
done
for r in 2 3; do
  run_item "cl_reasoner_diag_r$r" $PY -u scripts/eval_cosmos_nav.py --policy direct --no-think --limit 40 \
    --adapter checkpoints/reasoner_sft_v3/step1500 --arm-name v3_step1500_direct_r$r \
    --results-json results/runs/v3_step1500_direct_r$r.json
done

# ---------------------------------------------------------------- 4. summary + figures
run_item summary $PY -u tools/summarise_followups.py
PYTHONNOUSERSITE=1 "$FIG_PY" tools/make_figures.py >> "$LOGS/figures.log" 2>&1 && cp docs/figures/*.png paper/figures/
touch "$STATE/ALL_DONE"
say "ALL DONE -- see results/followups/summary.md"
# remove our own cron lines (installed as described in the README); nothing is left to schedule
crontab -l 2>/dev/null | grep -v "run_pending_followups.sh" | crontab - 2>/dev/null || true
