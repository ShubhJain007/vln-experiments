# Archived one-off scripts

These shell scripts orchestrated specific runs on the original machine. Each one did one of these jobs:
- waited for the GPU or another process to finish;
- handed an ablation over from one driver to another;
- re-queued a job;
- caught up on evaluations after an overnight run;
- ran one session's steps in order.

None of them is needed to reproduce a result; the scripts in [`../`](../) are. They are kept, unchanged apart from
path fixes, because [`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md) cites them as the record of how specific
runs were launched.

| Script | What it did |
|---|---|
| `cache_vbar_when_free.sh` | cached v̄ targets for 10 new scans once the GPU was free |
| `collect_new_scans.sh` | collected rollouts for 10 extra train scans (6 → 16); superseded by `../collect_full.sh` |
| `recollect_with_poses.sh` | one-time re-collection of 16 scans with per-step poses (fixed a yaw bug) |
| `run_step1_then_step2.sh` | one session's sequence: n = 150 Stage 1 re-evaluations, then a 16-scan Stage 1 (never trained) |
| `eval_16scan_when_ready.sh` | would evaluate that 16-scan Stage 1 once it finished (it never did) |
| `run_all_evals.sh`, `run_evals_ordered.sh` | caught up on pointing evaluations after overnight runs |
| `handoff_ablation.sh`, `queue_after.sh`, `run_remaining.sh` | re-queued arms of the Cosmos prompt/policy ablation (`../run_ablations.py`) |

They run from any directory (they `cd` to the repository root) and read `$PY` / `$HABITAT_PYTHON` like the other
scripts.
