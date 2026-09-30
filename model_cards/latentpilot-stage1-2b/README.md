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

Approach ① of *Four Ways to Navigate* ([report](https://github.com/ShubhJain007/vln-four-ways/blob/main/paper/main.pdf),
[code](https://github.com/ShubhJain007/vln-four-ways)). This is a from-scratch reimplementation of
[LatentPilot](https://arxiv.org/abs/2603.29165) (Hao et al., 2026; the official code was not released) on a 2 B
backbone. It is published as a **negative result**: at this scale the method's Stage 1 gate fails.

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

The learned Pilot Token predicts future frames 2.25× better, but navigates worse. The cause is **loss-balance drift**:
`λ·L_pil / L_act` rises from 0.52 at initialisation to 5.5 at the end of training. An adaptive-λ run removed the drift
but did not recover navigation (OS 6.7 %).

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
@techreport{jain2026fourways,
  title       = {Four Ways to Navigate: An Empirical Study of Vision-and-Language Navigation Policies on a Single GPU},
  author      = {Jain, Shubh},
  year        = {2026},
  institution = {GitHub},
  url         = {https://github.com/ShubhJain007/vln-four-ways}
}
```

Please also cite LatentPilot, [arXiv:2603.29165](https://arxiv.org/abs/2603.29165).
