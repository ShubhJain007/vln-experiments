# LatentPilot Reimplementation

From-scratch reimplementation of *LatentPilot: Scene-Aware Vision-and-Language
Navigation by Dreaming Ahead with Latent Visual Reasoning* (arXiv:2603.29165).
The official code was never released.

**Read order for any new session:** `AGENTS.md` (ground rules) → `HANDOFF.md`
(current state) → `DECISIONS.md` (every deviation and why) → `docs/EQUATIONS.md`
(all equations). **Never re-read the PDF** — everything was extracted into
`docs/EQUATIONS.md` in Session 1.

---

# ⚠️ CRITICAL GOTCHAS — read before writing any code

These are findings that **fail silently**. Nothing crashes; the numbers are just
quietly wrong. Each was discovered the hard way and cost real debugging time.

## 1. bf16 encoder output depends on BATCH SIZE (~7% relative L2)

The same frame encoded at batch size 3 vs batch size 1 differs by **~7% relative
L2** in bf16 (cosine still 0.998, so it looks fine at a glance).

Verified this is precision, not a bug:
- preprocessing is **bitwise identical** between the two paths
- in **fp32 the same comparison agrees to ~8e-6**

Cause: CUDA selects different kernels/reduction orders per batch shape, and the
error compounds through 24 ViT layers.

**Why it matters:** from Stage 1 onward `v̄_{t+2}` is the Pilot Token's
*regression target* (Eq. 11/14). A target that moves by 7% depending on how
frames were grouped means `L_pil` is partly fitting batching noise.

**The rule — and it differs by tensor. Do NOT just "match the batch sizes".**

| Tensor | Role | Where computed | Rule |
|---|---|---|---|
| `v̄` (pooled, 2048-d) | Pilot **target** (Eq. 14) + Pilot slot input (Eq. 12) | cached offline, once | **compute and store in fp32.** Batch-invariant to ~1e-5, so cache batch size becomes irrelevant. 8 KB/frame — ~800 MB for 100k frames. |
| `v_t` (tokens, 196×2048) | model **input** (Eq. 5) | live, during training | leave in bf16 at whatever batch size. Too large to cache (~1.6 MB/frame fp32 → 160 GB at 100k frames). |

**Why not match cache batch size to train batch size:** it creates a hidden
coupling where changing the training batch size — for VRAM tuning, gradient
accumulation, anything — silently invalidates the entire `v̄` cache with no
error. Caching in fp32 decouples them permanently.

**Why not train at batch size 1:** it costs enormous throughput to fix a problem
that only affects the *target*, which fp32 caching solves for ~800 MB of disk.
The encoder is frozen, so `v̄` is computed once offline and never recomputed
during training — the training batch size has no effect on it at all.

Residual, accepted: `v_t` is computed live, so it carries ~7% bf16 batch
variation, and eval rollouts (batch 1) differ slightly from training (batch N).
This is an *input* perturbation of the same order as ambient bf16 training
noise, never compared against a cached reference. Assumed benign — not measured.
Revisit only if Stage 0 metrics look unstable across batch sizes.

Pinned by `tests/test_vision_encoder.py::test_batch_invariance_is_a_bf16_artifact`,
which fails loudly if this ever stops being a precision issue (i.e. if something
structural breaks in preprocessing or `pooler_output` splitting).

## 2. mRoPE position ids are silently LOST when using `inputs_embeds`

**Largest silent-failure risk found so far.** Qwen3-VL gives image tokens 3D
(temporal, height, width) positions, and an image advances the position counter
by only `max(H,W)/merge`, **not** by `N_v`:

```
448×448 image → 196 tokens, positions t=4, h=4..17, w=4..17
position before image = 3      position after image = 18      (+14, not +196)
```

`compute_3d_position_ids` only takes that path when `input_ids` **and**
`mm_token_type_ids` **and** `image_grid_thw` are all present. Eq. 5 must build
from `inputs_embeds` (PILOT(z) is a continuous latent with no token id), so the
automatic path is unreachable and the model **silently falls back to sequential
positions** — stretching the image across 196 position units and displacing
everything after it by 182.

Measured cost, same sequence, only `position_ids` differing (5 real R2R-CE
instructions × real walk.mov frames):

| Quantity | Sequential vs mRoPE |
|---|---|
| hidden-state cosine | 0.87 – 0.91 |
| hidden-state relative L2 | **0.42 – 0.50** |
| action-position cosine | 0.39 – 0.94 |
| top-1 next token agreed | **2 / 5** |

An order of magnitude worse than the bf16 batch noise (#1), and it changes the
model's actual predictions most of the time.

**`Backbone` reconstructs mRoPE ids explicitly and uses them by default.**
Never call the language model with `inputs_embeds` and no `position_ids`.

Rationale for mRoPE despite the paper using LLaVA-Video (which has no mRoPE):
use each backbone *as it was pretrained*. LLaVA-Video at sequential positions is
in-distribution **for LLaVA-Video**; Cosmos at sequential is out-of-distribution
**for Cosmos**. Fidelity means preserving the relationship, not copying the
mechanism.

Guarded by `test_backbone.py::test_mrope_reconstruction_matches_model_ground_truth`
(bitwise vs the model's own `get_rope_index`) and
`test_image_advances_position_by_grid_side_not_n_v`.

## 3. Sequence layout is image-first, NOT Eq. 5's stated order

**Approved deviation (D10)** — Eq. 5 writes `[Tok(x); v_t]`, we use
`[<vision_start>; v_t; <vision_end>; Tok(x)]`. Measured zero-shot instruction
following over 160 trials (chance 25%): literal Eq. 5 = **21.2%**, ours =
**77.5%**. Ordering and markers interact — half the format is worse than either.

Do NOT "fix" this back to the paper's order without re-reading `DECISIONS.md`.
`layout="eq5_literal"` exists to re-measure after training. Whatever is used
must be identical in train and eval.

## 4. Habitat returns RGBA (4 channels), Eq. 1 wants RGB (3)

`habitat_sim`'s `color_sensor` yields `(H, W, 4)`. Eq. 1 defines
`o_t ∈ R^{H×W×C}` with `C=3`. Always drop alpha: `frame[..., :3]`.

`src/model/vision_encoder.py::to_rgb_pil` handles this centrally so callers
can't forget — pass raw habitat frames straight in. Tested by
`test_rgba_input_matches_rgb_input`.

## 5. Habitat default turn angle is 10°, the paper says 15°

Measured on habitat-sim 0.3.3:

| Primitive | Habitat default | Paper (AGENTS.md Sec. 3.1) |
|---|---|---|
| `move_forward` | 0.25 m | 0.25 m ✅ matches |
| `turn_left` / `turn_right` | **10°** | **15°** ❌ differs |

Only forward motion matches. Building an agent config from habitat's defaults
gives 10° turns — wrong action semantics, no error raised, and it would surface
much later as unexplained metric drift.

`src/model/action_space.py::habitat_agent_action_config()` sets every amount
explicitly. Never construct a bare `habitat_sim.AgentConfiguration()` and use
its `action_space` as-is. Guarded by
`test_action_space.py::test_turn_angle_differs_from_habitat_default`.

## 6. The vision tower does NOT return a tensor

`model.visual(...)` returns a `BaseModelOutputWithDeepstackFeatures` wrapper.
Calling `.mean(dim=0)` on it raises. The merged visual tokens are in
`.pooler_output`, a **tuple** with one `(N_v, d)` tensor per image.

Correct call (verified against transformers 5.16.1):

```python
out = model.model.get_image_features(pixel_values, image_grid_thw)
v_t = out.pooler_output[0]      # (196, 2048) == (N_v, d)   <- Eq. 4
```

## 7. Mean-pooled embeddings are strongly ANISOTROPIC

Raw cosine similarity on this encoder is shifted far from the usual scale:

| Pair | Cosine |
|---|---|
| solid black vs solid white | **0.907** |
| unrelated random images | **0.752** |

A naive "similarity > 0.9 means too similar" threshold would classify
*black vs white* as near-identical. Any cosine threshold must be calibrated
against a measured unrelated-pair floor, not against 0. This is exactly what
invalidated the literal Stage -1 gate thresholds — see `HANDOFF.md`.

## 8. Two conda envs — they are not interchangeable

No `habitat-sim` build supports Python 3.10, but the backbone stack needs it.

| Env | Python | Contains | Used for |
|---|---|---|---|
| `latentpilot` | 3.10 | torch, transformers, peft | model code, training, `verify_env.py` |
| `habitat_render` | 3.9 | habitat-sim, habitat-lab | frame pre-rendering, `verify_habitat.py` |

This also structurally enforces AGENTS.md's rule that habitat rendering and GPU
training never share a process.

## 9. ROS pollutes `sys.path`

ROS Humble injects `/opt/ros/humble/lib/python3.10/site-packages` via
`PYTHONPATH`, which breaks unrelated imports (`launch` → missing `lark`).

- **Scripts** are already guarded — they strip `/opt/ros/*` from `sys.path` at
  import time. Run them normally.
- **pytest is NOT guarded** — its plugin autoloader runs *before* any test file
  executes. Always run tests as:
  ```bash
  PYTHONPATH="" python -m pytest tests/ -q
  ```

---

## Verified environment facts (Session 1–3)

| Fact | Value |
|---|---|
| Backbone | Cosmos-Reason2-2B (Qwen3-VL-2B lineage), approved substitution for LLaVA-Video-7B |
| Hidden size `d` | 2048 |
| `N_v` @ 448×448 | 196 — confirmed 3 independent ways |
| Vision encoder out dim | 2048 (== `d`, no projection mismatch) |
| VRAM, bf16 load + forward | 4.64 GB → **standard LoRA, not QLoRA** |
| GPU | RTX 4070 Ti SUPER, 16 GB |

## Layout

```
src/model/      Eq. 3-7      (encoder, actions, sequence, backbone, head — done)
src/losses/     Eq. 13 done; 14-15 at Stage 1
src/data/       trajectory extraction, v̄ caching, Stage -1 probe
src/eval/       NE, SR, OS, SPL, nDTW
scripts/        agent writes, human runs
tests/          one test file per equation group
```

## Commands

```bash
# tests (note the PYTHONPATH guard)
PYTHONPATH="" python -m pytest tests/ -q

# environment check (latentpilot env)
python scripts/verify_env.py

# habitat check (habitat_render env)
conda activate habitat_render
python scripts/verify_habitat.py --scene data/scene_datasets/versioned_data/habitat_test_scenes/skokloster-castle.glb
```
