---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Cosmos3-Edge
library_name: peft
pipeline_tag: video-text-to-text
tags:
  - vision-and-language-navigation
  - vln
  - r2r-ce
  - habitat
  - lora
  - cosmos
---

# Cosmos3-Edge reasoner, fine-tuned as a navigation policy (SFT v3)

Approach ④ of *Evaluating Four Approaches to Vision-and-Language Navigation* ([report](https://github.com/ShubhJain007/vln-experiments/blob/main/paper/main.pdf),
[code](https://github.com/ShubhJain007/vln-experiments)). The reasoning tower of
[`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge) is fine-tuned to answer "what should the robot do
next?" from its own recent video.

> **Weights:** not published yet. Checkpoints `reasoner_sft_v3/step1500` (best in closed loop) and `final` are LoRA
> adapters of 27 MB each, kept by the author.

## Model details

| | |
|---|---|
| Base model | `nvidia/Cosmos3-Edge`, reasoner tower (`Cosmos3EdgeForConditionalGeneration`) |
| Trainable parameters | LoRA r = 16, α = 32 on the language model's q/k/v/o and the vision projector (6.9 M) |
| Frozen | the SigLIP-2 vision tower |
| Input | the last 8 s of the agent's own view (120 frames at 15 fps, sampled to 8 at 1 fps), an embodiment prompt and the R2R instruction |
| Output | one of `move forward`, `turn left`, `turn right`, `stop` |
| Execution | each command runs as 2 primitives (0.5 m, or 30°), then the model re-plans; STOP needs 2 consecutive stop answers |
| Decoding | no chain-of-thought; top-p 0.8, top-k 20, temperature 0.7 |

The exact prompt is in the report (Appendix A) and in `src/nav/cosmos_prompts.py`.

## Training data

45,000 decision points sampled uniformly from expert rollouts on R2R-CE `train` (FORWARD 66 %, RIGHT 14 %, LEFT 15 %,
STOP 5 %). Labels come from the expert's motion: a net yaw beyond ±20° over the next 16 video frames (about 1 s) means turn, the
final frames of a walk mean stop, and anything else means move forward.

## Training procedure

One epoch, learning rate 1e-4, gradient accumulation 16, AdamW, on one RTX 4070 Ti SUPER. SFT v1, the previous
version, used onset-balanced sampling, a ~1 s context and no projector LoRA. A v2 run was stopped after 200 steps.

## Evaluation

| | step 1,500 | final |
|---|---|---|
| Closed-loop **oracle success** (n = 40, 6 buildings) | **47.5 %** | 32.5 % |
| nDTW | 0.389 | 0.304 |
| NE (m) | 6.41 | 7.62 |
| Next-action probe accuracy (160 decision points; majority baseline 54.4 %) | – | 63.1 % |

**Important:** these closed-loop numbers come from a diagnostic evaluation that ends the episode when the agent first
comes within 3 m. That measures *reaching* the goal, not *stopping* at it. The model's own STOP never fired within 3 m,
so its strict success rate was not measured and would be far lower. They come from a single run with stochastic
decoding, where ±15 points is roughly the 95 % interval at n = 40.

## Limitations

- **Stopping:** the model almost never says stop (STOP recall 0.10 on the probe).
- **Probe vs. closed loop:** offline probe accuracy did not predict closed-loop success. SFT v1 scored higher on the
  probe (67.5 %) but lower in closed loop (17.5 %).
- **Confounded changes:** v1 → v3 changed sampling, context length and trainable modules at once.
- **Use restrictions:** research use in simulation only. It was trained on Matterport3D-derived data, which is limited to
  non-commercial academic use.

## How to run

```bash
conda activate latentpilot
python scripts/eval_cosmos_nav.py --policy direct --no-think --limit 40 \
    --adapter checkpoints/reasoner_sft_v3/step1500 --results-json results/runs/v3_step1500_direct.json
python scripts/probe_reasoner_nav.py --adapter checkpoints/reasoner_sft_v3/final \
    --variants hier_overall_nothink --trajectories 40 --points 4 --out results/probe_v3_final
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
