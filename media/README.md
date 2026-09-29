# Rollout videos and demos

Closed-loop rollouts on R2R-CE `val_unseen`, recorded with `scripts/record_episode.py` (Stage 1) and
`scripts/record_pointing.py` (pointing). The loop is the same one the evaluation measures. An overlay shows the
instruction, step, distance to goal, the chosen action and the policy's STOP probability; for the pointing policy it also
shows the predicted point. Recordings are 448 px wide, 538–553 px tall, at 4 fps.

## Comparisons (`comparisons/`)

11 side-by-side videos (1344 × 644), one per `val_unseen` scene: the first episode of each scan, sorted by scan id.
Each shows the same instruction for:

1. the expert following the reference path (its last frame is held after it arrives);
2. Stage 1 LatentPilot (learned Pilot Token) at step 10,000;
3. Stage 1 LatentPilot, final checkpoint.

Built from frames already on disk by [`../tools/make_demo_media.py`](../tools/make_demo_media.py). `comparisons/index.json`
lists the scan, episode id and geodesic distance of each video. Per-episode outcomes are tabulated in the
[main README](../README.md#demos).

**Provenance of the step-10k panel.** It was recorded from an earlier Stage 1 run (09-02, about 09:50–11:32, no training
log). That run's `step10000` checkpoint files were overwritten by the logged run at 13:24, so the step-10k and final panels
come from two runs with the same configuration, not one.

## Rollout sets (`rollouts/`)

| Set | Model | Episodes | Scans |
|---|---|---|---|
| [`final_multiscene/`](rollouts/final_multiscene/) | Stage 1, learned Pilot Token, final checkpoint (`checkpoints/stage1_learned/final`): 0/11 reached 3 m | 11 | 2azQ1b91cZZ, 8194nk5LbLH, EU6Fwq7SyZv, QUCTc6BB5sX, TbHJrupSAjP, X7HyMhZNoso, Z6MFQCViBuw, oLBMNvg9in8, pLe4wQe7qrG, x8F5xyUWy9e, zsNo4HB9uLZ |
| [`step10000/`](rollouts/step10000/) | Stage 1, learned Pilot Token, step 10,000 (earlier run, see above) | 3 | zsNo4HB9uLZ |
| [`step10000_multiscene/`](rollouts/step10000_multiscene/) | Stage 1, learned Pilot Token, step 10,000 (earlier run): 1/11 reached 3 m (x8F5xyUWy9e) | 11 | same 11 scans as `final_multiscene/` |
| [`step100k_failures/`](rollouts/step100k_failures/) | Pointing policy `pointing_full3/step100000`: the 16 failures of 22 recorded episodes (the other 6 stopped within 3 m) | 16 | 8194nk5LbLH, EU6Fwq7SyZv, QUCTc6BB5sX, TbHJrupSAjP, X7HyMhZNoso, Z6MFQCViBuw, oLBMNvg9in8, pLe4wQe7qrG, zsNo4HB9uLZ |

## Previews (`previews/`)

| GIF | Shows |
|---|---|
| `compare_ep09_x8F5xyUWy9e.gif` | Comparison: step 10k comes within 2.8 m; final never gets closer than 9.2 m |
| `compare_ep00_2azQ1b91cZZ.gif` | Comparison: step 10k's closest approach is 3.4 m, final's 6.4 m |
| `compare_ep07_oLBMNvg9in8.gif` | Comparison: neither checkpoint ever gets closer to the goal than its start point |
| `final_multiscene_ep00_2azQ1b91cZZ.gif` | Stage 1 final, single view |
| `step100k_failures_ep02_8194nk5LbLH.gif` | Pointing failure: never approaches the goal (timeout) |
| `step100k_failures_ep04_EU6Fwq7SyZv.gif` | Pointing failure: passes within 0.5 m of the goal and never stops (timeout) |
| `step100k_failures_ep16_pLe4wQe7qrG.gif` | Pointing failure: passes 1.5 m from the goal, then stops 5.4 m away |

## Licence of this media

These frames are rendered in Habitat from **Matterport3D** scenes. The Matterport3D data is provided for **non-commercial
academic use only** under the [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf), which
apply to this media as material derived from the data. Do not use it commercially.

The bulk rendered training data (expert rollouts, training clips) is not published, because distributing a substantial
portion of the data requires a click-wrap agreement under the same terms.
