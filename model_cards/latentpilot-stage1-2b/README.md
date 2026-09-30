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
  - latentpilot
  - reproduction
---

# LatentPilot Stage 1 at 2 B (reimplementation): learned and identity-control checkpoints

Approach ① of *Evaluating Four Approaches to Vision-and-Language Navigation* ([report](https://github.com/ShubhJain007/vln-experiments/blob/main/paper/main.pdf),
[code](https://github.com/ShubhJain007/vln-experiments)). This is a from-scratch reimplementation of
[LatentPilot](https://arxiv.org/abs/2603.29165) (Hao et al., 2026; the official code was not released) on a 2 B
backbone. It is published as a **negative result**: at this scale the method's Stage 1 gate fails.

**Why this method.** If a VLM could reason in latent space instead of in words or extra frames, it would use far
fewer tokens per step, run faster, and reason visually in its native embedding space. LatentPilot carries a single
Pilot Token forward as the model's only memory, trained to predict the embedding of the frame two steps ahead.

**What was and was not run.** Stages 0, 0′ and 1 with offline expert demonstrations, evaluated zero-shot in unseen
buildings. The paper's **data flywheel** (the model drives, an expert corrects, the model retrains) and Stage 2
(scheduled sampling) were **not** run, so this measures Stage 1's raw capability, not the full method.

> **Weights:** not published yet. `stage1_learned/final` and `stage1_identity/final` (LoRA adapter + Pilot module,
> 33 MB each) are kept by the author.

## Model details

| | |
|---|---|
| Base model | [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B); the paper uses LLaVA-Video-7B |
| Trainable parameters | LoRA r = 16, α = 32 on q/k/v/o (6.4 M), plus the Pilot module `G_ψ` (a linear layer, 2048 × 2048) and a learned `z_0` |
| Input sequence | `[frame tokens ; instruction ; Pilot slot z_{t-1}]` (image-first; the paper's literal order halves zero-shot instruction following), with an action-query token (Design B) |
| Output | FORWARD / LEFT / RIGHT / STOP as vocabulary tokens through the native LM head, plus `z_t = G_ψ(h_pil)` |
| Objective | `L_act + 0.1 · Σ ‖G_ψ(h_pil,t) − v̄_{t+2}‖²`; during training the slot is teacher-forced with `v̄_{t+1}` |
| Checkpoints | **learned**: `G_ψ` trained; **identity**: `G_ψ` frozen to the identity, as a control |

## Training data and procedure

- **Data:** expert rollouts on 6 Matterport3D buildings (1,665 episodes, 69,606 steps).
- **Schedule:** 17,400 steps (2 epochs) at batch 8.
- **Optimisation:** 8-bit AdamW, learning rate 1e-4 (100 warm-up steps, cosine decay).
- **Targets:** cached in fp32, because bf16 encoder output changes by 7 % with batch size.
- **Initialisation:** `G_ψ` starts with zero weights and a mean-embedding bias.

## Evaluation

R2R-CE `val_unseen`, first 150 episodes (8 buildings). The evaluation is diagnostic (it ends when the agent first comes
within 3 m), so the success column is **oracle success**, not SR.

| Checkpoint | final training `L_pil` | OS | nDTW | NE (m) | own STOP |
|---|---|---|---|---|---|
| learned `G_ψ` | 3.02 | 10.7 % | 0.277 | 8.50 | 4.7 % |
| identity `G_ψ` (control) | 6.80 | 17.3 % | 0.268 | 8.17 | 0 % |

The learned Pilot Token predicts future frames 2.25× better, but navigates worse.

**Most plausible cause: a training shortcut.** During training the Pilot slot holds the *true* embedding of the next
frame, which reveals the action just taken; at test time it holds the model's own prediction. Training accuracy jumps
from 78 % without the slot (Stage 0) to 98 % with it, and 78 % vs 92 % at matched steps. The model learns to read the
action off the slot, a shortcut that is gone at test time. Loss-balance drift (`λ·L_pil / L_act` rises from 0.52 to
5.5) contributes, but an adaptive λ that removed it did not recover navigation (OS 6.7 %). The flywheel and scheduled
sampling, which were not run, are the parts of the method designed to close this gap.

## Limitations

- **Scale:** 2 B parameters and 6 buildings, against 7 B and full data in the paper. This is a finding at this scale,
  not a refutation of the paper.
- **Missing stages:** Stage 2 (scheduled sampling) and the paper's data flywheel were not run.
- **Stopping:** the model almost never stops on its own.
- **Use restrictions:** research use in simulation only. It was trained on Matterport3D-derived data, which is limited
  to non-commercial academic use.

## How to run

```bash
conda activate latentpilot
python scripts/eval_stage0_prime.py --checkpoint checkpoints/stage1_learned/final --limit 150
python scripts/record_episode.py --checkpoint checkpoints/stage1_learned/final --per-scan 1 \
    --out recordings/final_multiscene
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

Please also cite LatentPilot, [arXiv:2603.29165](https://arxiv.org/abs/2603.29165).
