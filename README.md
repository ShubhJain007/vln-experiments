# Dreaming Ahead at 2B: Reimplementing LatentPilot for Vision-and-Language Navigation

**A from-scratch reproduction study of LatentPilot ([arXiv:2603.29165](https://arxiv.org/abs/2603.29165)) on R2R-CE, on
one 16 GB GPU. It covers the paper's method, where it broke at this scale, and two follow-up policies.**

<p align="center">
  <img src="media/previews/compare_ep09_x8F5xyUWy9e.gif" width="780"><br>
  <sub>The same unseen <code>val_unseen</code> episode from three agents: the expert following the reference path; a Stage 1
  LatentPilot policy after 10k steps (comes within 2.8 m of the goal); and the fully trained policy (never closer than 9.2 m).
  Training further made the Pilot Token a better predictor of future frames and a worse navigator (finding 1 below).</sub>
</p>

> **Status (Sept 2026): research code, finished and not maintained.** Results are on subsets of R2R-CE `val_unseen`
> (n = 40–150), with thresholds tuned on that split; read [Caveats](#caveats) before quoting a number.

## Abstract

LatentPilot adds a *Pilot Token* to a vision-language navigation policy. The token is a latent trained to predict the
visual embedding two steps ahead and fed back at the next step, so the policy "dreams ahead". The official code was never
released.

We re-derive the method equation by equation on `nvidia/Cosmos-Reason2-2B` (the paper uses LLaVA-Video-7B), with LoRA
on one RTX 4070 Ti SUPER. Measuring every silent assumption first turned up a series of implementation traps that change results
without raising an error, among them lost multimodal RoPE, batch-dependent bf16 targets and a layout that halves
zero-shot instruction following.

The paper's Stage 1 gate **fails** at this scale. The learned Pilot Token predicts future frames 2.25× better than an
identity control, yet reaches the goal less often (oracle success 10.7 % vs 17.3 %, n = 150). We trace this to a loss
balance that drifts 10× during training.

We then replace action classification with **waypoint pointing** (Robostral-style). With label, data, history and
layer-fusion fixes this reaches **strict success 31.3 % / SPL 25.4 %** (n = 150). A **Cosmos3-Edge reasoner** fine-tuned as
a direct policy reaches the goal region in **47.5 %** of 40 episodes, but almost never stops there.

The paper reports SR 51.7–54.0 with a 7 B model on full data.

## Results

![Navigation results](docs/figures/fig_navigation.png)

**Read the two metrics carefully.** *OS* (oracle success) counts an episode if the agent ever came within 3 m of the
goal. *SR* counts it only if the agent **stopped** within 3 m by itself. Most evaluations here ran in a diagnostic mode
that ends the episode on arrival, which makes SR identical to OS. Those numbers are reported as OS. True SR exists only
for the pointing policy, evaluated with `--strict`. Details: [`docs/EXPERIMENTS.md` §0](docs/EXPERIMENTS.md#0-protocol).

| Policy | n (scans) | SR (strict) | SPL | OS | nDTW | NE (m) |
|---|---|---|---|---|---|---|
| **LatentPilot, as specified** (Stage 1, learned `G_ψ`) | 150 (8) | not measured | — | 10.7 | 0.277 | 8.50 |
| LatentPilot, identity `G_ψ` control | 150 (8) | not measured | — | 17.3 | 0.268 | 8.17 |
| **Pointing**, 16 scans, final layer | 150 (8) | 13.3 | 13.0 | 14.0 | 0.350 | 7.34 |
| Pointing + history, 61 scans | 150 (8) | 18.0 | 16.8 | 22.7 | 0.394 | 6.55 |
| Pointing + history, 3-epoch schedule (step 40k) | 150 (8) | 24.7 | 22.3 | 31.3 | 0.345 | 7.10 |
| **Pointing + layer fusion (step 120k)** | 150 (8) | **31.3** | **25.4** | 42.7 | 0.323 | 7.22 |
| Cosmos3-Edge action model, zero-shot | 40 (6) | not measured | — | 7.5 | 0.276 | 8.81 |
| Cosmos reasoner, SFT v1 | 40 (6) | not measured | — | 17.5 | 0.365 | 7.41 |
| **Cosmos reasoner, SFT v3 (step 1,500)** | 40 (6) | not measured¹ | — | **47.5** | 0.389 | 6.41 |
| *Paper, Table 3 NaN row (7 B, full split)* | 1,839 (11) | *51.7* | *47.1* | *57.0* | — | *5.3* |
| *Paper, Stage 1 / flywheel round 1* | 1,839 (11) | *~54.0* | *~48.5* | — | — | — |

¹ None of its own STOP decisions fell within 3 m of the goal, so its strict SR would be far below 47.5 %.

Pointing rows use STOP thresholds between 0.05 and 0.20, chosen on this split. Every row, including the ones not
shown, is in [`docs/EXPERIMENT_LOG.md` §3](docs/EXPERIMENT_LOG.md#3-consolidated-navigation-evaluations-r2r-ce-val_unseen).

## Key findings

**1. The Stage 1 gate fails because the loss balance drifts.** λ = 0.1 balances the action loss and the Pilot loss at
initialisation. The action loss then falls ~51× while the Pilot loss falls ~4.8×, so by the end 5.5× more gradient goes
to predicting frames than to choosing actions. An adaptive λ removed the drift but did not recover navigation (OS 6.7 %),
so at 2 B parameters and 1,665 episodes the Pilot objective did not help. This is a scale-limited finding, not a
refutation of the paper.

<img src="docs/figures/fig_loss_balance.png" width="720">

**2. A zero-shot probe picked the wrong architecture.** Untrained, the Pilot-last design kept 59 % instruction
following and the action-query design 29 %. After training, the first scored 0 % and the second 16.7 % in closed loop. The
same thing happened with the reasoner: the checkpoint with the best next-action probe accuracy (67.5 %) navigated worse
than one at 63.1 %. Rank policies closed-loop.

**3. STOP is decodable, just not from the last layer.** A linear probe finds STOP at AUC 0.72 in layer 15 and 0.44
(below chance) at the output layer, which the head was reading. Fusing layers 23/26/28 gave the best strict SR (31.3 %).
A matched comparison shows little difference at equal training, so the size of the fusion effect is not established.

<img src="docs/figures/fig_layer_probe.png" width="720">

**4. The pointing policy under-turns.** Its predicted bearing is 0.42× the true bearing. Missed turns become FORWARD
(39–43 %) and almost never the wrong direction. Lowering the turn threshold does not fix this in closed loop.

<img src="docs/figures/fig_bearing.png" width="720">

**5. Stopping, not reaching, is the bottleneck.** OS exceeds SR for every policy, and the gap moves with the STOP
threshold. A higher threshold stops less often: OS keeps rising while strict SR peaks at 0.20.

<img src="docs/figures/fig_stop_threshold.png" width="820">

**6. Silent bugs rivalled any modelling change.**
- Qwen3-VL silently drops mRoPE when fed `inputs_embeds` (hidden relative L2 0.42–0.50).
- bf16 targets change 7 % with batch size.
- Eq. 5's literal layout gives 21 % zero-shot instruction following against 78 % for image-first.
- Pointing labels agreed with the expert only 69 % of the time until three bugs were fixed (97 %).
- One run trained the wrong history stride for 7 h.

See [`docs/GOTCHAS.md`](docs/GOTCHAS.md).

The full narrative, one section per experiment covering why it was run, the setup, the result and what was decided, is in
**[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)**.

## Demos

All videos are closed-loop rollouts on unseen `val_unseen` scenes, at 4 fps. The overlay shows the instruction, distance
to goal, chosen action and STOP probability.

### Expert vs Stage 1 LatentPilot: 11 scenes, first episode of each

| Scene | Geodesic | Stage 1, step 10k: closest | Stage 1, final: closest | Video |
|---|---|---|---|---|
| 2azQ1b91cZZ | 7.1 m | 3.4 m | 6.4 m | [mp4](media/comparisons/ep00_2azQ1b91cZZ_expert_vs_stage1.mp4) |
| 8194nk5LbLH | 11.8 m | 11.8 m | 11.8 m | [mp4](media/comparisons/ep01_8194nk5LbLH_expert_vs_stage1.mp4) |
| EU6Fwq7SyZv | 4.6 m | 4.3 m | 3.6 m | [mp4](media/comparisons/ep02_EU6Fwq7SyZv_expert_vs_stage1.mp4) |
| QUCTc6BB5sX | 13.2 m | 9.0 m | 11.1 m | [mp4](media/comparisons/ep03_QUCTc6BB5sX_expert_vs_stage1.mp4) |
| TbHJrupSAjP | 9.7 m | 9.7 m | 9.1 m | [mp4](media/comparisons/ep04_TbHJrupSAjP_expert_vs_stage1.mp4) |
| X7HyMhZNoso | 10.1 m | 8.8 m | 8.0 m | [mp4](media/comparisons/ep05_X7HyMhZNoso_expert_vs_stage1.mp4) |
| Z6MFQCViBuw | 11.5 m | 11.5 m | 11.5 m | [mp4](media/comparisons/ep06_Z6MFQCViBuw_expert_vs_stage1.mp4) |
| oLBMNvg9in8 | 11.0 m | 11.0 m | 11.0 m | [mp4](media/comparisons/ep07_oLBMNvg9in8_expert_vs_stage1.mp4) |
| pLe4wQe7qrG | 4.5 m | 4.5 m | 4.5 m | [mp4](media/comparisons/ep08_pLe4wQe7qrG_expert_vs_stage1.mp4) |
| x8F5xyUWy9e | 10.5 m | **2.8 m ✓** | 9.2 m | [mp4](media/comparisons/ep09_x8F5xyUWy9e_expert_vs_stage1.mp4) |
| zsNo4HB9uLZ | 8.0 m | 8.0 m | 6.3 m | [mp4](media/comparisons/ep10_zsNo4HB9uLZ_expert_vs_stage1.mp4) |

"Closest" is the minimum distance to the goal. A closest distance equal to the geodesic means the agent never got closer
than where it started. The step-10k policy comes from an earlier Stage 1 run whose checkpoint was later overwritten
([details](docs/EXPERIMENTS.md#5-corrections-to-earlier-reports)).

<p>
  <img src="media/previews/compare_ep00_2azQ1b91cZZ.gif" width="780">
</p>

### Pointing policy: success and three failure modes

`pointing_full3` at step 100k succeeded by its own STOP in 6 of 22 recorded episodes. The failures show the three ways it
goes wrong ([all 16](media/rollouts/step100k_failures/)):

| Never approaches the goal | Passes within 0.5 m, never stops | Stops 5.4 m away, after passing 1.5 m from the goal |
|---|---|---|
| <img src="media/previews/step100k_failures_ep02_8194nk5LbLH.gif" width="250"> | <img src="media/previews/step100k_failures_ep04_EU6Fwq7SyZv.gif" width="250"> | <img src="media/previews/step100k_failures_ep16_pLe4wQe7qrG.gif" width="250"> |

The cyan cross-hair is the predicted waypoint; the bottom line shows the action, `p(stop)` and the point.

### Stage 1 LatentPilot, final checkpoint

<img src="media/previews/final_multiscene_ep00_2azQ1b91cZZ.gif" width="300">

All 41 rollouts and 11 comparisons are listed in [`media/README.md`](media/README.md).

## Method at a glance

| Phase | Policy | Output | Training data | Where it is |
|---|---|---|---|---|
| I. LatentPilot | Cosmos-Reason2-2B + LoRA + Pilot Token (`z_t` → `G_ψ` → `v̄_{t+2}`), Stages 0 / 0′ / 1 | 4 actions as vocabulary tokens | 1,665 episodes, 6 scans | `src/model`, `src/losses`, `src/train/train_stage*.py` |
| II. Pointing | same backbone + pointing head (u, v, visibility, STOP); K = 2 history frames; fusion of layers 23/26/28 | a waypoint in the image, then a bearing controller (turn if > 7.5°) | up to 10,819 episodes, 61 scans | `src/train/train_pointing.py`, `scripts/eval_pointing.py` |
| III. Cosmos | Cosmos3-Edge: action diffusion stream (zero-shot); reasoner tower + LoRA SFT | ego-pose chunks / a next-action phrase | 45,000 decision samples | `src/nav`, `scripts/eval_cosmos_nav.py`, `scripts/train_reasoner_sft.py` |

Every deviation from the paper, with the measurement that justified it, is in [`DECISIONS.md`](DECISIONS.md). Every
equation is extracted in [`docs/EQUATIONS.md`](docs/EQUATIONS.md).

## Caveats

- **Thresholds were tuned on the test split.** STOP and turn thresholds were chosen on `val_unseen`. A `val_seen` set
  was collected for selection but never used, so the best pointing numbers are optimistic.
- **Small, unpaired evaluations.** n = 150 (8 scans) for LatentPilot and pointing, and n = 40 (6 scans) for Cosmos.
  Both take the first n episodes of the split, so the sets differ. At n = 40, ±15 points is roughly the 95 % interval.
- **The reasoner result is a single stochastic run** (top-p 0.8, temperature 0.7), and its v1 → v3 change altered three
  things at once.
- **Unfinished runs.** The long pointing runs were stopped at 1.1–1.2 of their planned 1.5–3 epochs, so intermediate
  checkpoints are un-annealed. Stage 2, a 16-scan Stage 1 and most Cosmos ablation arms were never run
  ([list](docs/EXPERIMENTS.md#6-planned-but-not-run)).
- **Different backbone and scale from the paper** (2 B vs 7 B, 6–61 scans vs full data). The results say what happens at
  this scale, not whether the paper is right.

## Reproduction

Two conda environments are needed, because no habitat-sim build supports Python 3.10:

| Env | Python | Contains | Used for |
|---|---|---|---|
| `latentpilot` | 3.10 | torch, transformers, peft | model code, training, evaluation driver |
| `habitat_render` | 3.9 | habitat-sim 0.3.3, habitat-lab | rendering and simulation (separate worker process) |

```bash
PYTHONPATH="" python -m pytest tests/ -q          # 311 tests, one file per equation group
python scripts/verify_env.py                       # backbone loads, d = 2048, N_v = 196, VRAM
conda activate habitat_render && python scripts/verify_habitat.py --scene <scene .glb>

# data: Matterport3D (terms of use required) + R2R-CE, expert rollouts
python scripts/download_mp3d.py && bash scripts/download_full.sh && bash scripts/collect_full.sh

# Phase I
bash scripts/run_ab_overnight.sh                   # Stage 0' designs A and B
bash scripts/run_stage1.sh                         # Stage 1: learned vs identity G_psi

# Phase II: best strict result
bash scripts/run_fusion.sh                         # pointing, K=2 stride 8, layers 23/26/28
python scripts/eval_pointing.py --checkpoint checkpoints/pointing_fusion/step120000 \
    --limit 150 --strict --stop-threshold 0.20

# Phase III
python scripts/train_reasoner_sft.py ...           # see the script header for v3 settings
bash scripts/eval_v3.sh

# figures and demo media, from logs and recordings already on disk (no GPU)
python tools/make_figures.py && python tools/make_demo_media.py
```

Hardware: one RTX 4070 Ti SUPER (16 GB). The base model in bf16 takes ≈ 4.6 GB, so standard LoRA fits (no QLoRA).

## Repository layout

```
src/model/       Eq. 3–8: vision encoder, action space, input sequence, backbone (explicit mRoPE), action head, Pilot Token
src/losses/      Eq. 13–15: L_act, L_pil
src/data/        rollout collection (reference-path follower), v̄ caching, pointing targets, Cosmos datasets
src/eval/        NE / SR / OS / SPL / nDTW; the Habitat worker runs in its own process
src/train/       Stage 0 / 0′ / 1 / 2 and pointing training (LoRA); Cosmos sequence packing
src/nav/         Cosmos reasoner navigator, prompts, reasoner SFT data
scripts/         download, collection, training, evaluation, sweeps, probes (each script's header says why it exists)
tools/           make_figures.py (docs/figures from logs), make_demo_media.py (media/comparisons from recordings)
tests/           one test file per equation group (311 tests)
results/         probe reports and closed-loop run summaries (JSON)
logs/            every training and evaluation log (text)
docs/            EXPERIMENTS.md (narrative), EXPERIMENT_LOG.md (every number + source), EQUATIONS.md, GOTCHAS.md, figures/
media/           rollout videos, comparisons and previews (Matterport3D-derived, non-commercial)
DECISIONS.md     every deviation from the paper with the measurement behind it
HANDOFF.md       session state through Stage 0′;  Agents.md: ground rules the reproduction followed
```

Suggested reading order: this README → [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) → [`DECISIONS.md`](DECISIONS.md) →
[`docs/GOTCHAS.md`](docs/GOTCHAS.md).

## Data and checkpoints (not in this repository)

| What | Size | How to get it |
|---|---|---|
| Matterport3D scenes for Habitat | 32 GB for 17 scans (11 val_unseen + 6 train); more for all 72 | accept the [Matterport3D terms](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf), then `python scripts/download_mp3d.py` (resumable, verifies before extracting) |
| R2R-CE episodes | small | VLN-CE release (`scripts/download_full.sh`) |
| Expert rollouts, rendered frames, resampled video | ~42 GB | regenerate with `scripts/collect_full.sh`, `scripts/render_resampled.py` |
| Base models | — | Hugging Face `nvidia/Cosmos-Reason2-2B`, `nvidia/Cosmos3-Edge` |
| Trained LoRA adapters | ~12.5 GB | not published; kept by the author |

## Citation

```bibtex
@misc{jain2026latentpilotrepro,
  title  = {Dreaming Ahead at 2B: Reimplementing LatentPilot for Vision-and-Language Navigation},
  author = {Jain, Shubh},
  year   = {2026},
  howpublished = {\url{https://github.com/ShubhJain007/latentpilot}}
}
```

The original method, and the works it builds on, should be cited as their authors list them on arXiv:
LatentPilot, [arXiv:2603.29165](https://arxiv.org/abs/2603.29165); Robostral Navigate,
[arXiv:2607.20785](https://arxiv.org/abs/2607.20785); Chain-of-Visual-Thought,
[arXiv:2511.19418](https://arxiv.org/abs/2511.19418).

## Licence

No licence has been chosen for the code yet, so default copyright applies. Third-party models and datasets keep their own
licences (NVIDIA model licences, VLN-CE / R2R).

**Media** in `media/` is rendered from Matterport3D scenes and is for **non-commercial academic use only**, under the
[Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).
