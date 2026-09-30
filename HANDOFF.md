# HANDOFF.md — LatentPilot Session State

**Updated:** Session 3 (2026-08-31)
**Current stage:** Stage 0 IN PROGRESS — model code AND metrics complete.
205 tests passing. Everything buildable without MP3D is now done.
**D15 RESOLVED (human, 2026-09-01): Option 1 — build the recurrent Pilot
Token. Phase A machinery is DONE (`src/model/pilot.py`, 27 tests). See D16/D17
for the `z_0` and placement decisions.**

**Original blocker text, kept for context: Stage 0 cannot reach its gate as
specified — see D15 in DECISIONS.md. Table 3's NaN row PROPAGATES the Pilot Token (supplementary
Sec. A.1); our Stage 0 is memoryless. Confirmed empirically: the policy spins
in place, 10/12 episodes zero progress, FWD only 12.7% of rollout actions vs
61.7% in training. Needs a human decision before more work. !!**

**MP3D IS DOWNLOADED AND VALIDATED** (see below) — no longer a blocker.
**DATA COLLECTED + VALIDATED, TRAINING SCRIPT WRITTEN AND PROBED.**
**Next action:** the human runs the Stage 0 training run (command below),
then evaluation against Table 3's NaN row.
**All deviations and their evidence are in `DECISIONS.md` — read it before
changing anything that looks like it contradicts the paper.**

---

## STAGE 0 PROGRESS (Session 3)

| Equation | Module | Tests | Status |
|---|---|---|---|
| Eq. 4 `v_t = E_phi(o_t)` | `src/model/vision_encoder.py` | 21 | DONE |
| Eq. 3 `A = {FWD,LEFT,RIGHT,STOP}` | `src/model/action_space.py` | 28 | DONE |
| Eq. 5 `u_t = [Tok(x); v_t]` | `src/model/input_sequence.py` | 17 | DONE |
| Eq. 6 `H_t = F_theta(u_t)` | `src/model/backbone.py` | 19 | DONE |
| Eq. 7 `pi = Softmax(W_a h^act)` | `src/model/action_head.py` | 24 | DONE |
| Eq. 13 `L_act` | `src/losses/action_loss.py` | (in above) | DONE |
| Metrics NE/SR/OS/SPL/nDTW | `src/eval/metrics.py` | 37 | DONE |
| Data pipeline (follower rollout) | `src/data/collect_rollouts.py` | smoke-tested | DONE |
| Bulk rollout collection | `data/rollouts/` | validated | DONE |
| Training script (LoRA) | `src/train/train_stage0.py` | smoke+probe OK | DONE |
| Stage 0 training run | `checkpoints/stage0/` | done — FAILED (see D15) | — |
| Pilot machinery (Eq. 8, z_0, cache) | `src/model/pilot.py` | 27 | DONE |
| v-bar cache | `src/data/` | — | NEXT |
| Stage 0' (NaN row, recurrence) | — | — | after v-bar |
| Real gate run | — | — | blocked on MP3D |

`PYTHONPATH="" python -m pytest tests/ -q` -> **239 passed**.

Test fixtures are now session-scoped in `tests/conftest.py`. Previously each
module built its own VisionEncoder, holding one ~4.6 GB copy of the backbone
PER MODULE and OOM-ing the 16 GB card on a full run. Do not reintroduce
per-module model fixtures.

### Verified this session
- Eq. 7/13 feasibility CONFIRMED (no STOP-and-ask needed): the 4 actions map to
  EXISTING single vocabulary tokens -- forward=13435, left=2359, right=1291,
  stop=9495. `W_a` IS `model.lm_head`, no new head, no vocab resize. This also
  resolves AGENTS.md Sec. 8 item 5: W_a and W_LM are the same matrix.
- Uniform-model loss = 1.3863 = ln(4) exactly, confirming the softmax is over
  the 4 actions and not the 151669-token vocabulary (which would give 11.93).
- Real R2R-CE sequence lengths: min 204 / median 228 / max 329, matching
  AGENTS.md's ~300-token design point.
- Causal mask verified BITWISE (perturbing a later token leaves every earlier
  hidden state byte-identical), plus a control proving forward flow exists.

### CRITICAL finding this session -- mRoPE (see README gotcha #2)
Building sequences from `inputs_embeds` (required by Eq. 5, since PILOT(z) is a
continuous latent with no token id) makes transformers SILENTLY fall back to
sequential position ids. Measured cost vs correct mRoPE, same sequence, only
position_ids differing: hidden relative L2 **0.42-0.50**, top-1 next token
agreed only **2/5**. An order of magnitude worse than the bf16 batch noise.
`Backbone` now reconstructs mRoPE ids explicitly (verified BITWISE against the
model's own `get_rope_index`) and uses them by default.

Reasoning for mRoPE though the paper's LLaVA-Video has none: use each backbone
as it was PRETRAINED. LLaVA-Video at sequential is in-distribution for
LLaVA-Video; Cosmos at sequential is out-of-distribution for Cosmos.

### Decisions RESOLVED this session (full detail in DECISIONS.md)
1. **Sequence layout -> image-first with vision markers (D10, human-approved).**
   `[<vision_start>; v_t; <vision_end>; Tok(x)]`, deviating from Eq. 5's stated
   `[Tok(x); v_t]`. Measured over 160 trials (chance 25%): literal 21.2% vs
   chosen 77.5%. NO chat template (measured to hurt: 68.8% vs 77.5%).
   `layout="eq5_literal"` retained to re-measure after training.
   METHOD NOTE: a 16-trial first pass gave the opposite ranking. Do not judge
   format questions on small probe sets.
2. **Loss reduction -> sum within trajectory, mean across batch (D9).** The
   paper is explicit (Eq. 13/14 sums, Eq. 15 expectation). The old `mean`
   default silently made lambda = 0.1*T/(T-2) -- a 50% error at T=6. Changed
   now while provably inert (no L_pil in Stage 0).

### Still OPEN
- **Pilot slot position under the D10 layout** (blocks Stage 1 only). Eq. 5 puts
  PILOT last, but the instruction is now last.
- **Where h^act comes from** (Stage 1 only). Two coherent readings: (A) pilot is
  the last input, so h^act and h^pil are the SAME state with two heads on it --
  legal under Eqs. 7/8 and arguably the paper's thesis stated directly; or
  (B) an explicit ACTION_QUERY slot follows the pilot, giving distinct states --
  supported only by Fig. 4 typing A as its own token. Undetermined; see
  DECISIONS.md "OPEN". Stage 0 is unaffected (h^act = H[-1], no pilot).
- LoRA rank, LR, batch size, epochs -- not in paper; set at training time.

### MP3D — DOWNLOADED, EXTRACTED AND VALIDATED (2026-09-01)

**DONE.** 17/17 scans extracted to `data/scene_datasets/mp3d/<scan>/<scan>.glb`
(+ .navmesh), 3.7 GB on disk. The 15 GB `mp3d_habitat.zip` may now be deleted.

END-TO-END VALIDATION PASSED (this is the important part):
- habitat-sim loads a real MP3D scene and steps the agent -- frame (448,448,4)
- every R2R-CE start and goal position tested is `pathfinder.is_navigable`
- stored `info.geodesic_distance` matches habitat's COMPUTED geodesic to
  **0.0000 m** on every episode checked

That last point confirms the meshes, navmeshes and episodes share one
coordinate frame, and that the SPL denominator we use is exactly what habitat
computes. The data stack is consistent.

Re-run / re-extract with:
```bash
python scripts/download_mp3d.py --dry-run     # inspect
python scripts/download_mp3d.py               # resumable; verifies before extracting
```

**Do NOT use the official `download_mp.py` directly.** Three reasons, all
verified by reading it:
1. It is PYTHON 2 ONLY (`print` statements, `raw_input`, `urllib.urlretrieve`)
   and will not parse under Python 3. No Python 2 exists in these envs.
2. `download_task_data` has a re-run bug: the `download_file` call sits INSIDE
   `if not os.path.isdir(localdir):`, so once the output dir exists it silently
   downloads NOTHING and prints nothing. A 15 GB transfer will get interrupted.
3. `--id` does NOT apply to task data, so the habitat bundle cannot be filtered
   by scan at download time.

**Which data:** `mp3d_habitat.zip` (TASK_FILES['habitat']) — contains the .glb
meshes and .navmesh files habitat-sim loads. The raw `matterport_mesh` filetype
is NOT habitat-ready and would need the mesh conversion pipeline.

**Verified live:** HTTP 200, Content-Length 16,085,306,031 (14.98 GB),
`Accept-Ranges: bytes` so resume is supported. Full MP3D release is 1.3 TB.

The archive covers all 90 scenes and cannot be filtered at download time, so
`scripts/download_mp3d.py` extracts ONLY the 17 scans this project needs,
derived from the episode files (not hardcoded, so it cannot drift):

  val_unseen: 11 scans / 1839 episodes  (gate split, must be complete)
  train     :  6 scans / 1665 episodes  (AGENTS.md wants 1000-2000)
```
2azQ1b91cZZ 8194nk5LbLH 8WUmhLawc2A EU6Fwq7SyZv JeFG25nYj2p QUCTc6BB5sX
TbHJrupSAjP Vvot9Ly1tCj X7HyMhZNoso Z6MFQCViBuw ac26ZMwG7aT oLBMNvg9in8
pLe4wQe7qrG r47D5H71a5s ur6pFq6Qu1A x8F5xyUWy9e zsNo4HB9uLZ
```
Zero train/val_unseen overlap, so val_unseen stays genuinely unseen.

R2R_VLNCE_v1-3 is already in `data/` (all 4 splits). Episode schema carries
`reference_path` (nDTW), `info.geodesic_distance` (SPL denominator), and
`goals[].radius = 3.0`. It carries NO action labels — expert actions must be
generated by rolling out habitat's GreedyGeodesicFollower.

### STAGE 0 TRAINING — VRAM MEASURED, READY TO RUN

`python src/train/train_stage0.py --smoke-test` (measured, not assumed):

| batch | peak GB | reserved GB | s/step |
|---|---|---|---|
| 1 | 5.32 | 5.38 | 0.07 |
| 2 | 6.12 | 6.33 | 0.10 |
| 4 | 7.76 | 8.12 | 0.19 |
| **8** | **10.76** | **11.46** | **0.37** |
| 12 | 13.94 | 14.66 | 0.57 |

**Use batch 8** — 10.76 GB peak leaves ~4 GB headroom on the 15.56 GB card.
Batch 12 at 14.66 GB reserved is too tight to be safe.

Trainable: 6,422,528 params = 0.263% of 2.45B. LoRA r=16 on 112 language-model
q/k/v/o projections. The build ASSERTS that count and that no vision module was
adapted — a silent match there would unfreeze the encoder and break D2.

PEFT GOTCHA (cost a debugging cycle): `get_peft_model` returns a wrapper that
adds a nesting level, so `peft.model.model` is NOT the Qwen3VLModel that owns
`get_vision_position_ids`. But peft injects adapters IN PLACE into the base
tree (`peft.get_base_model() is base`, and the base reports the LoRA params as
trainable), so the code keeps using the BASE model everywhere and holds the
wrapper only for `save_pretrained`.

Command:
```bash
conda activate latentpilot
python src/train/train_stage0.py --batch-size 8 --steps 8000
```
~8700 steps is one epoch over the 69606 training pairs; at 0.37 s/step that is
about 54 min/epoch.

### EXPERT ROLLOUTS — COLLECTED AND VALIDATED (2026-09-01)

`scripts/validate_rollouts.py --split <split> --check-frames` -> **PASS** on both.

| | train | val_unseen |
|---|---|---|
| episodes | 1665 | 1839 |
| training pairs (steps) | 69606 | 72024 |
| status | all `stop` | all `stop` |
| SR (real geodesic) | 1.0000 | 1.0000 |
| SPL | 1.0000 | 0.9999 |
| OS | 1.0000 | 1.0000 |
| nDTW | 0.7544 | 0.7529 |
| NE | 2.88 m | 2.87 m |
| episode length min/med/max | 11 / 40 / 95 | 9 / 37 / 90 |
| structural issues | none | none |

Disk: 9.1 GB total (4.2 GB train + 5.0 GB val_unseen), JPEG q95, 448x448.
Every episode: frames == actions == positions, exactly one STOP, at the end.

### ACTION PRIOR — read the training curve against THIS, not ln(4)

| baseline | train | val_unseen |
|---|---|---|
| LOSS uniform, ln(4) | 1.3863 | 1.3863 |
| **LOSS marginal H(actions)** | **0.9744** | **1.0065** |
| ACC majority class (FWD) | 0.639 | 0.617 |

(-ln(p_FWD) is NOT an "always-FWD loss" -- that policy puts probability 0 on
36% of steps, so its cross-entropy is infinite. Majority class is an ACCURACY
baseline only.)

The action distribution is heavily imbalanced (FWD ~62-64%, STOP ~2.4-2.6%):
- A model that learns only the PRIOR sits at loss ~0.97 with accuracy ~0.64.
  CONFIRMED EMPIRICALLY: a 150-step probe run converged to loss 1.02 against a
  marginal of 1.029, accuracy 0.58 against a majority class of 0.596, and
  STOP-recall 0.000. That is the prior and nothing else.
- Do NOT read a falling loss or ~64% accuracy as competence.
- STOP occurs once per ~40 steps but decides SR outright: predict it late or
  never and the episode fails regardless of how good the other 39 actions were.
  Watch STOP precision/recall separately from overall accuracy.

---

## Stage -1 verdict — CONFIRMED by human, CLOSED

Predictability probe on real footage (walk.mov, move2.mp4, move3.mp4), replicated
across all 3 independent clips:

| fps | Forward-retrieval vs chance | Raw gate (Sec. 2 threshold) |
|---|---|---|
| 4 | 2.0x / 2.9x / 8.3x — all above chance | FAILS raw >0.98 in all 3 |
| 2 | ~1.0x in all 3 — exactly chance | MARGINAL/FAIL, inconsistent |

**Finding:** the raw cosine gate (AGENTS.md Sec. 2, thresholds 0.90/0.98) is
miscalibrated for this encoder — mean-pooled SigLIP features are anisotropic
(solid black vs. solid white scores 0.907; unrelated images average 0.752), so
the literal threshold assumes a scale this encoder doesn't have.
The metric that actually discriminates signal from noise — forward-retrieval
(`v̄_t → v̄_{t+2}` vs. a 50-frame pool) — is consistently above chance at fps=4
and collapses to exact chance at fps=2, replicated in all 3 independent clips.
fps=4 also matches the paper's own action cadence (FWD=0.25m at walking speed
≈ 5 fps).

**DECISION: fps≈4 is the validated operating point for future data pipelines
(Stage 0 frame pre-rendering, Pilot Token horizon work in Stage 1+). Proceed
to Stage 0.** Human confirmed 2026-08-31.

Full per-clip numbers, calibration methodology (`compute_random_pair_baseline`,
`normalized_headroom`, `centered_copy_shortcut`), and probe code are in
`src/data/predictability_probe.py`. This section of HANDOFF.md is the
one-paragraph summary — do not re-run the analysis to rediscover it.

---

## Habitat-sim installed — required a separate conda env (Session 2, later)

**Problem:** no `habitat-sim` build on `conda-forge`/`aihabitat` supports
Python 3.10 (max is 3.9); `latentpilot` env is pinned to 3.10 for
torch/transformers.

**Fix applied:** created a second env, `habitat_render` (Python 3.9), for
habitat-sim + habitat-lab only. This also structurally enforces AGENTS.md's
rule that habitat rendering and GPU training never run in the same process.

```bash
conda create -n habitat_render python=3.9 -y
conda activate habitat_render
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
pip install git+https://github.com/facebookresearch/habitat-lab.git
```

**Verified working** — `scripts/verify_habitat.py` (new, split out of
`verify_env.py` because it needs the other Python version) passed against the
free test scenes:

```bash
conda activate habitat_render
python scripts/verify_habitat.py --scene data/scene_datasets/versioned_data/habitat_test_scenes/skokloster-castle.glb
```
Result: scene loads, agent steps 10x, frame shape `(448, 448, 4)`.

**IMPORTANT — flag for Stage 0 data pipeline:** habitat-sim's `color_sensor`
returns **RGBA** (4 channels), but Eq. 1 specifies `o_t ∈ R^{H×W×C}` with
`C=3` (RGB). Any frame-loading code in `src/data/` must drop the alpha
channel (`frame[..., :3]`) before passing to the vision encoder.

Free test scenes (no MP3D agreement needed, already downloaded):
```
data/scene_datasets/versioned_data/habitat_test_scenes/{apartment_1,skokloster-castle,van-gogh-room}.glb
```
These are NOT MP3D and won't match R2R-CE episodes — useful only for
environment smoke-testing, not for the Stage 0 gate itself.

---

## Session 2: script audit — 9 bugs found and fixed

Both scripts were audited against the actually-installed APIs
(transformers 5.16.1, torch 2.5.1+cu121) rather than assumed behaviour.

| # | File | Bug | Severity |
|---|---|---|---|
| 1 | probe | `model.visual(...)` returns `BaseModelOutputWithDeepstackFeatures`, not a tensor — `.mean(dim=0)` would crash | **CRITICAL** |
| 2 | verify_env | `obs.get(a) or obs.get(b)` on numpy arrays raises "truth value ambiguous" | **CRITICAL** (latent, in habitat path) |
| 3 | verify_env | `sim_cfg.scene_id` set *after* `Configuration()` — ignored; no sensor spec defined | high |
| 4 | probe | `{{t+h}}` in non-f-strings printed literal double braces | cosmetic |
| 5 | probe | duplicate dead `copy_score` assignment | correctness risk |
| 6 | probe | `random` unseeded → probe not reproducible | reproducibility |
| 7 | both | ROS `/opt/ros/*` on `PYTHONPATH` breaks imports | **environment** |
| 8 | probe | video reader leaked handles on exception; `imageio` v2 API not pinned | robustness |
| 9 | tests | two tests asserted mathematically wrong premises | test validity |

**Correct vision-encoder API (verified empirically):**
```python
out = model.model.get_image_features(pixel_values, image_grid_thw)
v_t = out.pooler_output[0]      # (196, 2048) == (N_v, d)  <- Eq. 4
vbar = v_t.mean(dim=0)          # (2048,)                  <- Eq. 11
```

**ROS fix is now built into both scripts** (they strip `/opt/ros/*` from
`sys.path` at import time), so `PYTHONPATH=""` is no longer required.

---

## MUST READ — the Sec. 2 gate thresholds are likely miscalibrated

Measured on Cosmos-Reason2-2B mean-pooled features:

| Image pair | Cosine similarity |
|---|---|
| solid black vs solid white | **0.907** |
| random unrelated images (mean) | **0.752** |

The AGENTS.md Sec. 2 thresholds (0.90 = marginal, 0.98 = fail) assume a scale
where unrelated ≈ 0. This encoder's mean-pooled space is strongly anisotropic:
everything sits in a narrow cone. **A raw threshold of 0.90 would classify
"solid black vs solid white" as "too smooth to learn."**

The probe therefore now reports three calibration numbers next to the raw gate:

- **unrelated-pair floor** — mean `cos(v_i, v_j)` for `|i-j| >= 8`, i.e. what
  "unrelated" actually scores on *this* footage.
- **normalized headroom** — `(copy - floor) / (1 - floor)`. 1.0 = indistinguishable
  from a perfect copy; 0.0 = no better than two unrelated frames.
- **mean-centered copy shortcut** — the gate quantity with the shared component
  removed.

The raw gate verdict is still printed unchanged, per AGENTS.md Sec. 2.
**I have not altered the gate.** Whether to judge Stage -1 on the raw value or
the normalized headroom is a decision for the human — see Open Questions.

Demonstration on synthetic smooth data: raw copy = 0.9986 (verdict FAIL), but
the unrelated floor on the same data = 0.9969, normalized headroom = 0.55,
centered = 0.31. The raw number alone gives a false FAIL.

---

## Verification status

```
tests/test_predictability_probe.py .............................. 47 passed
scripts/verify_env.py ......................................... clean run
src/data/predictability_probe.py .............. end-to-end run on synthetic data
```

`verify_env.py` now cross-checks `N_v` three independent ways — all agree at 196:
formula `(448/16)²/2² `, `image_token_id` count, and encoder output rows.

---

## Commands to run NOW

### 1. Tests (no data needed, <1s)
```bash
cd <repo root>
conda activate latentpilot
python -m pytest tests/test_predictability_probe.py -q
```
Expect: `47 passed`.

### 2. The probe on REAL footage

Needs ~20+ sequential frames of real indoor corridor walking. Phone video is
fine — walk down a hallway, turn a corner or two, ~30-60 seconds.

Synthetic/noise data will NOT answer the gate question; the numbers above show
it saturates the metric. It must be real scene footage.

```bash
# video file
python src/data/predictability_probe.py --data /path/to/corridor.mp4 --fps 4

# or a directory of frames (sorted filenames = temporal order)
python src/data/predictability_probe.py --data /path/to/frames_dir/
```

If video reading fails: `python -m pip install "imageio[ffmpeg]"`.

Paste back the full results block.

---

## What counts as pass

- Tests: 47 passed.
- Probe raw gate `< 0.90` → PASS, proceed to Stage 0.
- Probe raw gate `0.90-0.97` → MARGINAL, flag and proceed.
- Probe raw gate `> 0.98` → raw FAIL, but **do not act on it before comparing
  against the unrelated-pair floor and normalized headroom.** If headroom is
  low the target really is too smooth; if headroom is high the raw number is
  mostly anisotropy and the gate needs recalibrating with the human.

---

## Section 1 gate (carried forward, re-verified this session)

| Check | Result |
|---|---|
| d | 2048 |
| N_v at 448×448 | 196 (three independent checks agree) |
| vision_out | 2048 (== d, no mismatch) |
| VRAM bf16 load+fwd | 4.64 GB alloc / 4.89 GB reserved / 4.83 GB peak |
| LoRA strategy | standard LoRA (>11 GB headroom) |

---

## Open questions — need a human decision

1. ~~Gate calibration~~ — RESOLVED: fps≈4, human confirmed 2026-08-31.
2. **retrieval@1 definition (NEW).** AGENTS.md Sec. 2 says "given v̄_{t+2}, rank
   50 frames". Read literally the query is in its own pool, making the metric
   trivially 1.0 for distinct frames. The probe reports both the literal
   (self-retrieval) and predictive (v̄_t → v̄_{t+2}) readings. Not blocking —
   informational for whoever revisits the probe.
3. Attention-mask layout from Fig. 4 — token ordering to verify in Stage 0.
4. LR, schedule, batch size, epochs — NOT in paper; decide during Stage 0.
5. `tie_word_embeddings=False` — set at Stage 0 model loading to silence warning.
6. Episode-boundary Pilot cache reset — presumably `z_0`; not stated in paper
   (relevant from Stage 1 onward, not Stage 0).

---

## Installs status

- ~~habitat-sim + habitat-lab~~ — DONE, in separate `habitat_render` env
  (Python 3.9; see above). Verified against free test scenes.
- Matterport3D scenes — PENDING, human pursuing agreement in parallel with
  Stage 0 model code. NOT required to start Stage 0 (see below).
- R2R-CE episode files — PENDING, same as above.

---

## STAGE 0 PLAN — what Session 3 does

**Goal:** reproduce Table 3's "NaN" ablation row. No Pilot Token. Actions
decoded via the backbone's native LM vocabulary projection, not a new head
(Sec. 3.3) — see AGENTS.md Sec. 3 for full equation list.

**Does NOT need MP3D/R2R-CE to start** — build and unit-test with dummy
data first; swap in real data once it lands:

1. Eq. 4 — frozen vision encoder wrapper (`src/model/vision_encoder.py`).
   Reuse the verified API from Session 2: `model.model.get_image_features(...)
   .pooler_output[0]` → `(N_v, d)`. Test: shape, determinism, `requires_grad=False`.
2. Eq. 3 — action set `{FWD, LEFT, RIGHT, STOP}` mapped to Habitat primitives
   (FWD=0.25m, turns=15°). Test: exactly 4 actions, mapping matches paper.
3. Eq. 5 minus Pilot — `u_t = [Tok(x); v_t]`. Test: seq len = len(instr) + N_v;
   assert NO history frames present.
4. Eq. 6 — backbone forward, `H_t = F_θ(u_t)`. Test: causal mask — changing a
   later token must not change an earlier position's output.
5. Eq. 7/13 — actions as vocab tokens via native `W_LM`, not a new head.
   **STOP and ask if infeasible with Cosmos-Reason2-2B's tokenizer.**
   Test: probabilities sum to 1; loss on uniform-random model ≈ ln(4) ≈ 1.386.
6. L_act (Eq. 13) — cross-entropy. Test: loss → 0 on one-hot logits.
7. Metrics (`src/eval/`) — NE, SR, OS, SPL, nDTW exactly per Sec. 4.1. Test on
   ground-truth reference paths: SR≈1.0, SPL≈1.0. **Fix the metric before
   evaluating any model if this fails.**
8. Dry run: use the free test scenes (`data/scene_datasets/...`) + Habitat's
   shortest-path follower to validate the full pipeline mechanically —
   collection → training loop → metrics — before real R2R-CE/MP3D data
   arrives. Remember: habitat's `color_sensor` returns RGBA — drop the alpha
   channel (`frame[..., :3]`) before the vision encoder.

**Real gate (needs MP3D + R2R-CE, run once data lands):** LoRA r=16-32 on
attention projections, bf16, gradient checkpointing, 8-bit AdamW, ~300-token
sequences, 1000-2000 R2R-CE train episodes, frames pre-rendered to disk
(never render + train concurrently). Target on R2R Val-Unseen: SR≈51.7,
SPL≈47.1, NE≈5.3, OS≈57.0. SR<45 means the harness is wrong — STOP, don't
"fix" by changing the model.

---

## Session 3 kickoff prompt

```
Session 3 — Stage 0 (skeleton, no Pilot Token). Read HANDOFF.md and
docs/EQUATIONS.md first, not the PDF. Start with Eq. 4 (vision encoder
wrapper) per the STAGE 0 PLAN section of HANDOFF.md. One equation at a
time, test each before moving to the next, per AGENTS.md Sec. 0.3/0.9.
```
