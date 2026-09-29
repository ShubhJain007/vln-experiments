# LatentPilot reimplementation (Vision-and-Language Navigation, R2R-CE)

A from-scratch reimplementation of **LatentPilot: Scene-Aware Vision-and-Language Navigation by Dreaming Ahead with Latent
Visual Reasoning** ([arXiv:2603.29165](https://arxiv.org/abs/2603.29165)). The official code was never released. The project
reproduces the paper equation by equation on a single 16 GB GPU, documents every deviation with its measured reason, and — after
the paper's Stage 1 gate failed at this scale — pivots to pointing supervision and then to a Cosmos reasoner policy.

**Status (Sept 2026): research code, not maintained.** The latest arm (Cosmos-Reason reasoner, "v3", direct policy) reaches
**SR 47.5% / SPL 47.4%** on 40 held-out R2R-CE episodes at training step 1,500, against the paper's reported SR 54.0 / SPL 48.5
(different backbone, far smaller data; n = 40, so roughly ±15 points).

---

## Results

All numbers are on R2R-CE `val_unseen` in Habitat, measured with `src/eval/metrics.py` (SR, SPL, OS, NE, nDTW; success radius
3 m). `n` is the number of episodes. Raw logs are in `logs/`, run summaries in `results/`.

### 1. LatentPilot as specified (Stages 0 → 1): the Stage 1 gate failed

| Checkpoint | n | SR | SPL | OS | nDTW | NE (m) |
|---|---|---|---|---|---|---|
| Stage 1, learned `G_psi` (final) | 150 | 10.7 | 10.7 | 10.7 | 0.277 | 8.50 |
| Stage 1, identity `G_psi` (control, final) | 150 | 17.3 | 17.2 | 17.3 | 0.268 | 8.17 |
| Paper (LLaVA-Video-7B, full data) | — | 54.0 | 48.5 | — | — | — |

The learned Pilot Token is a 2.25× better *predictor* of future frames (held-out `L_pil` 3.02 vs 6.80) and a worse
*navigator*. Mechanism (DECISIONS.md D25): with λ = 0.1 the two losses are balanced only at initialisation — `L_act` converges
~51× and `L_pil` ~4.8×, so by the end the model spends 5.5× more gradient on predicting frames than on choosing actions. The
model also never learned to stop (`model_stop` 0–4.7% in every checkpoint), so SR equals oracle-stop success here. This is a
finding at 2B parameters / 6 scans / 1,665 episodes, not a refutation of the paper.

### 2. Pivot to pointing supervision (Robostral-style, D27)

Actions replaced by waypoint pointing ([Robostral Navigate, arXiv:2607.20785](https://arxiv.org/abs/2607.20785) §2.2) and a
bearing controller (turn if |bearing| > 7.5°, else forward). Label/expert agreement was raised from 0.689 to **0.972** by fixing
three silent label bugs (in-place turns, one shared horizon, furthest-vs-first waypoint).

| Checkpoint | n | SR | SPL | OS | nDTW | NE (m) |
|---|---|---|---|---|---|---|
| `pointing` step 10,000 | 60 | 26.7 | 26.5 | 26.7 | 0.322 | 8.66 |
| `pointing_full3` step 20,000 | 100 | 15.0 | 13.1 | 28.0 | 0.209 | 9.21 |

### 3. Cosmos reasoner policy (latest)

A Cosmos-Reason reasoner fine-tuned to choose the next navigation action from a 16-frame egocentric video context, run as a
direct policy (`scripts/eval_cosmos_nav.py`, `results/runs/*.json`):

| Run | n | SR | SPL | OS | nDTW | NE (m) |
|---|---|---|---|---|---|---|
| **v3, step 1,500** | 40 | **47.5** | **47.4** | 47.5 | 0.389 | 6.41 |
| v3, final | 40 | 32.5 | 32.5 | 32.5 | 0.304 | 7.62 |
| SFT, final | 40 | 17.5 | 17.5 | 17.5 | 0.365 | 7.41 |
| SFT, step 1,250 | 40 | 0.0 | 0.0 | 0.0 | 0.280 | 8.64 |

Next-action probe (160 decision points on 40 trajectories, `results/probe_*.md`; majority-class baseline 54.4%):

| Model | Accuracy |
|---|---|
| SFT, final | **67.5%** |
| SFT, step 1,250 | 65.6% |
| v3, final (120-frame context) | 63.1% |
| Zero-shot (no fine-tuning) | 32.5% |

The early v3 checkpoint beating the final one (47.5 vs 32.5 SR) is consistent with over-fitting to the small rollout corpus;
with n = 40 the difference is suggestive, not conclusive.

---

## What is in this repository

```
src/model/      Eq. 3–8: vision encoder, action space, input sequence, backbone (explicit mRoPE), action head, Pilot Token
src/losses/     Eq. 13–15: L_act, L_pil
src/data/       rollout collection (reference-path follower), v̄ caching, pointing targets, Cosmos datasets
src/eval/       NE / SR / OS / SPL / nDTW, habitat worker in its own process
src/train/      Stage 0 / 0′ / 1 / 2 and pointing training (LoRA), Cosmos sequence packing
src/nav/        Cosmos reasoner navigator, prompts, reasoner SFT data
scripts/        data download, collection, training, evaluation, ablations, probes
tests/          one test file per equation group (311 tests)
results/        probe reports and run summaries
logs/           training and evaluation logs (text)
docs/           EQUATIONS.md (every equation, extracted once), GOTCHAS.md (silent-failure findings, environment facts)
AGENTS.md       ground rules the reproduction followed
HANDOFF.md      session-by-session state (last full update: Stage 0 / 0′)
DECISIONS.md    every deviation from the paper, with the measurement that justified it (D1–D27)
```

Read order: `docs/GOTCHAS.md` → `DECISIONS.md` → `docs/EQUATIONS.md` → `HANDOFF.md`.

## Findings worth knowing before touching the code

Full detail and the tests that pin each one down are in [`docs/GOTCHAS.md`](docs/GOTCHAS.md):

1. **mRoPE is silently lost with `inputs_embeds`.** Qwen3-VL-family models fall back to sequential positions; measured hidden
   relative L2 0.42–0.50 and only 2/5 top-1 tokens agreeing. `Backbone` rebuilds mRoPE ids explicitly (bitwise-checked).
2. **bf16 encoder output depends on batch size** (~7% relative L2); regression targets are cached in fp32.
3. **Image-first layout beats the paper's Eq. 5 order** (77.5% vs 21.2% zero-shot instruction following over 160 trials).
4. Habitat returns RGBA; its default turn is 10°, not the paper's 15°; the vision tower returns a wrapper, not a tensor; pooled
   embeddings are strongly anisotropic (black vs white cosine 0.907).
5. **Loss balance drifts during training** (D25): fix λ at initialisation and it is still wrong by 5.5× at the end.

## Setup

Two conda environments (no habitat-sim build supports Python 3.10):

| Env | Python | Contains | Used for |
|---|---|---|---|
| `latentpilot` | 3.10 | torch, transformers, peft | model code, training, evaluation driver |
| `habitat_render` | 3.9 | habitat-sim 0.3.3, habitat-lab | rendering and simulation (separate process) |

```bash
PYTHONPATH="" python -m pytest tests/ -q        # the PYTHONPATH guard matters if ROS is installed
python scripts/verify_env.py                     # latentpilot env
conda activate habitat_render && python scripts/verify_habitat.py --scene <test scene .glb>
```

Hardware used: one RTX 4070 Ti SUPER (16 GB). Base model in bf16 ≈ 4.6 GB, so standard LoRA (not QLoRA).

## Data and checkpoints (not in this repository)

| What | Size locally | How to get it |
|---|---|---|
| Matterport3D scenes for Habitat (17 scans) | 32 GB | requires accepting the Matterport3D terms of use; `python scripts/download_mp3d.py` (resumable, verifies before extracting) |
| R2R-CE episodes | small | from the VLN-CE release (see `scripts/download_full.sh`) |
| Expert rollouts, rendered frames, videos | 42 GB | regenerate: `scripts/collect_full.sh`, `scripts/render_resampled.py` |
| Base models | — | Hugging Face: `nvidia/Cosmos-Reason2-2B` (backbone), `nvidia/Cosmos3-Edge` (reasoner experiments) |
| Trained LoRA adapters (stage0, stage0′ A/B, stage1, pointing, reasoner SFT / v3) | ~12.5 GB | not published; kept by the author |

The two reference papers this work builds on are linked rather than included:
[LatentPilot (arXiv:2603.29165)](https://arxiv.org/abs/2603.29165),
[Robostral Navigate (arXiv:2607.20785)](https://arxiv.org/abs/2607.20785), and
[Chain-of-Visual-Thought (arXiv:2511.19418)](https://arxiv.org/abs/2511.19418).

## Licence

No licence has been chosen yet; until one is added, the default copyright applies. Third-party models and datasets keep their
own licences (Matterport3D terms, NVIDIA model licences, VLN-CE / R2R).
