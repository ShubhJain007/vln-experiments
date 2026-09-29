# Rollout videos

Closed-loop rollouts on R2R-CE `val_unseen`, recorded with `scripts/record_episode.py` — the same loop the evaluation
measures, with an overlay showing the instruction, step, distance to goal, the chosen action and the policy's STOP
probability (and, for the pointing policy, the predicted point). 448 × 538 px, 4 fps.

| Set | Model | Episodes | Scans |
|---|---|---|---|
| [`final_multiscene/`](rollouts/final_multiscene/) | Stage 1, learned Pilot Token — final checkpoint (`checkpoints/stage1_learned/final`) | 11 | 2azQ1b91cZZ, 8194nk5LbLH, EU6Fwq7SyZv, QUCTc6BB5sX, TbHJrupSAjP, X7HyMhZNoso, Z6MFQCViBuw, oLBMNvg9in8, pLe4wQe7qrG, x8F5xyUWy9e, zsNo4HB9uLZ |
| [`step10000/`](rollouts/step10000/) | Stage 1, learned Pilot Token — step 10,000 (`checkpoints/stage1_learned/step10000`) | 3 | zsNo4HB9uLZ |
| [`step10000_multiscene/`](rollouts/step10000_multiscene/) | Stage 1, learned Pilot Token — step 10,000, episodes across scenes | 11 | 2azQ1b91cZZ, 8194nk5LbLH, EU6Fwq7SyZv, QUCTc6BB5sX, TbHJrupSAjP, X7HyMhZNoso, Z6MFQCViBuw, oLBMNvg9in8, pLe4wQe7qrG, x8F5xyUWy9e, zsNo4HB9uLZ |
| [`step100k_failures/`](rollouts/step100k_failures/) | Pointing policy (D27) — failure cases selected for analysis | 16 | 8194nk5LbLH, EU6Fwq7SyZv, QUCTc6BB5sX, TbHJrupSAjP, X7HyMhZNoso, Z6MFQCViBuw, oLBMNvg9in8, pLe4wQe7qrG, zsNo4HB9uLZ |

Previews: [`previews/`](previews/).

## Licence of this media

These frames are rendered in Habitat from **Matterport3D** scenes. The Matterport3D data is provided for **non-commercial
academic use only** under the Matterport3D Terms of Use: http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf — which applies
to this media as material derived from the data. Do not use it commercially. The bulk rendered training data (expert
rollouts, training clips) is not published, because distributing a substantial portion of the data requires a click-wrap
agreement under the same terms.
