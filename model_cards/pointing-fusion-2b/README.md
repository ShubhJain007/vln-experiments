---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license
base_model: nvidia/Cosmos-Reason2-2B
library_name: peft
pipeline_tag: robotics
tags:
  - vision-and-language-navigation
  - vln
  - r2r-ce
  - habitat
  - lora
  - pointing
metrics:
  - success_rate
  - spl
---

# Pointing policy with layer fusion (Cosmos-Reason2-2B + LoRA): VLN on R2R-CE

The best policy in *Evaluating Four Approaches to Vision-and-Language Navigation* ([report](https://github.com/ShubhJain007/vln-experiments/blob/main/paper/main.pdf),
[code](https://github.com/ShubhJain007/vln-experiments)): approach ②, with **success rate 23.3 %** (own STOP within 3 m; STOP threshold chosen on `val_seen`) on
150 unseen R2R-CE episodes.

The idea comes from [Robostral Navigate](https://arxiv.org/abs/2607.20785) by Mistral AI: point at the next waypoint in
the image instead of choosing an action. This model tests that idea's **raw capability**: a 2 B model, 61 buildings,
supervised imitation only, **no RL fine-tuning** (Robostral adds online RL), and a geometric controller instead of a
learned low-level policy. It is evaluated zero-shot in unseen buildings.

> **Weights:** not published yet. Checkpoint `pointing_fusion/step120000` (LoRA adapter + head, 25 MB) is kept by the
> author. This card documents it so that the result can be reproduced and understood.

## Model details

| | |
|---|---|
| Base model | [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) (Qwen-VL family, SigLIP-2 vision encoder, d = 2048) |
| Trainable parameters | LoRA r = 16, α = 32 on q/k/v/o of all 28 decoder layers (6.4 M), plus a linear pointing head (49 k) |
| Frozen | the vision encoder and all base weights |
| Input | the current 448 × 448 RGB frame, 2 history frames (8 steps apart) and the English instruction |
| Output | image point (u, v) of the next waypoint, unit direction, visibility logit, STOP logit |
| Head input | the concatenated hidden states of decoder layers 23, 26 and 28 at the action-query token |
| Controller | turn 15° toward the predicted bearing if it exceeds 7.5°, otherwise move forward 0.25 m; STOP when σ(stop) > 0.20 |

## Intended use

Research on instruction-following navigation in simulation (Habitat, R2R-CE). **Not for deployment on real robots**,
and not for commercial use: it was trained on Matterport3D-derived data (see below).

## Training data

Expert rollouts on R2R-CE `train`: 61 Matterport3D buildings, 10,819 episodes and 435,468 steps. The expert follows the
annotated reference path, not the shortest path. Pointing labels come from that path with two horizons: a 24-step
pointing target, and an 8-step control target at the first position at least 0.20 m away. They agree with the
expert's own actions on 97.2 % of steps.

## Training procedure

- **Loss:** squared error on (u, v) (visible targets only), a cosine direction loss, and weighted BCE on STOP (weight 0.5;
  positive weight up to 8 for the 2.5 % base rate) and visibility (weight 0.2).
- **Optimisation:** 8-bit AdamW, learning rate 1e-4 with cosine decay, batch 4.
- **Schedule:** planned for 163,300 steps (1.5 epochs); this is the step-120,000 checkpoint (1.1 epochs).
- **Hardware:** one RTX 4070 Ti SUPER (16 GB).

## Evaluation

R2R-CE `val_unseen`, first 150 episodes (8 buildings), strict mode, where success requires the model's own STOP
within 3 m:

| STOP threshold | SR (own STOP ≤ 3 m) | end-SR (ended ≤ 3 m) | SPL* | OS | nDTW | NE (m) |
|---|---|---|---|---|---|---|
| 0.10 | 23.3 | 28.7 | 24.3 | 36.7 | 0.363 | 7.03 |
| 0.15 | 23.3 | 29.3 | 24.4 | 40.0 | 0.351 | 7.05 |
| **0.20** | **22.7** | **31.3** | 25.4 | 42.7 | 0.323 | 7.22 |
| 0.30 | 19.3 | 29.3 | 23.6 | 44.7 | 0.287 | 7.59 |

SR is the standard R2R-CE success rate. end-SR also counts episodes that ran out of steps within 3 m; it is what the
evaluation script originally reported as SR. \* SPL is computed with end-SR as the success criterion.

Choosing the threshold on `val_seen` instead (8 training buildings, 159 new episodes; SR there 39.3 / 35.3 / 31.3 /
24.7 % at 0.10 / 0.15 / 0.20 / 0.30) picks **0.10**, so the untuned result is **SR 23.3 %**, within a point of the 22.7 %
tuned on the test split. For comparison, Robostral Navigate
(8 B, 2.4 M trajectories) reports 73.4 % with supervised training alone and 77.4 % after RL.

## Limitations

- **Under-turning:** the predicted bearing is 0.42× the expert's, so many turns are missed and taken as FORWARD.
- **Stopping:** the policy passes near the goal without stopping, or stops several metres short. Only 49 % of its own
  stops are within 3 m, and OS exceeds SR by 20 points.
- **No recovery training:** imitation learning only sees expert states. Robostral's RL phase exists to teach recovery,
  and it was not run.
- **Instruction dependence:** STOP separation collapses when the instruction is swapped. The policy uses language, but
  its stopping behaviour is fragile.
- **Evaluation size:** a single evaluation on 150 episodes, with no seeds repeated.

## How to run

```bash
conda activate latentpilot     # environment/latentpilot.yml
python scripts/eval_pointing.py --checkpoint checkpoints/pointing_fusion/step120000 \
    --limit 150 --strict --stop-threshold 0.20
python scripts/record_pointing.py --checkpoint checkpoints/pointing_fusion/step120000 \
    --out recordings/pointing --keep-all          # videos with the predicted point drawn on each frame
```

## Citation

```bibtex
@techreport{jain2026vlnexperiments,
  title       = {Evaluating Four Approaches to Vision-and-Language Navigation: An Experimental Study on a Single GPU},
  author      = {Jain, Shubh},
  year        = {2026},
  institution = {GitHub},
  url         = {https://github.com/ShubhJain007/vln-experiments}
}
```

Pointing supervision follows Robostral Navigate ([arXiv:2607.20785](https://arxiv.org/abs/2607.20785)).
