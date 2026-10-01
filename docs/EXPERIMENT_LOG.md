# Experiment log (complete, chronological)

The exhaustive record behind [`EXPERIMENTS.md`](EXPERIMENTS.md): every run, probe and evaluation with its exact numbers and the
file it came from. `EXPERIMENTS.md` is the readable narrative; this file is the audit trail.

Compiled read-only from DECISIONS.md, HANDOFF.md, Agents.md, docs/*, README.md, results/, logs/, checkpoints/*/(config|trainlog).json and the script docstrings. All dates are 2026 and times are EDT (from log `date` stamps or file mtimes). Numbers are copied verbatim from the cited source. Anything marked **not recorded** is absent from every source, and anything marked **FLAG** is a contradiction or a claim that could not be verified (all flags are collected in Section 9).

## 0. Conventions to know before reading any number

* **Two eval modes.** In the default *diagnostic* mode the episode ends as soon as the agent comes within 3 m of the goal (`within_radius`, the "proximity break"). That makes **SR equal to OS by construction**, and the model's own STOP can only fire when it is more than 3 m away. `--strict` removes the proximity break, so success then requires the model's own STOP within 3 m. Sources: `scripts/eval_stage0.py` docstring, `scripts/run_strict_sweep.sh` header, D23 note (DECISIONS.md:682-684: "Report them as OS, never as SR").
* **Every LatentPilot eval (Stage 0/0'/1), every Cosmos3-Edge eval and every Cosmos reasoner eval is diagnostic.** `eval_v3.sh` and `run_ablations.py` never pass `--strict`, and `eval_cosmos_nav.py` breaks on `d <= success_radius` (lines 37-38, 69-70). Only pointing evals labelled STRICT are true self-stop SR.
* **Different `n` means different episodes.** Evals take the first N val_unseen episodes. Per the logs, n=30 covers 6 scans, n=40 covers 6, n=60 covers 7, n=100 covers 7 and n=150 covers 8. Numbers at different n are not paired.
* **Split.** Every closed-loop eval is R2R-CE `val_unseen` in Habitat (1,839 episodes, 11 scans). Max steps are 100 primitives (`--max-steps` default; Edge ablation `--max-steps 100`).
* **Paper targets** (docs/EQUATIONS.md:219-235). Stage 0 / "NaN" row: SR 51.7, SPL 47.1, NE 5.3, OS 57.0. Stage 1 flywheel round 1: SR ~54.0, SPL ~48.5.

---

## 1. Timeline at a glance

| Date (2026) | Phase | Main events |
|---|---|---|
| 08-31 | Stage -1 / setup | Env verification, Stage -1 predictability probe (D5), mRoPE (D6), bf16 batch (D7), action tokens (D8), loss reduction (D9), layout probe (D10) |
| 09-01 | Stage 0 | MP3D download and validation, expert rollouts (D11-D14), VRAM smoke test, Stage 0 training 17:02-18:07, Stage 0 closed-loop failure (D15), D16/D17 zero-shot A/B slot probe |
| 09-02 00:31-05:42 | Stage 0' | A/B overnight training (Design A vs B); D19 closed-loop reversal; D23 slot ablation; D24 checkpoint-size fix |
| 09-02 ~09:50-16:50 | Stage 1 | Learned vs identity G_psi runs; gate FAILED (D25) |
| 09-02 17:05-19:37 | Stage 1 follow-ups | D26 adaptive-lambda run (aborted about step 10,750), n=150 re-evals, data expansion to 16 scans, re-collection with poses |
| 09-02 20:14 to 09-05 22:00 | Pointing (D27-D29) | Pointing training (16 scans), stop-threshold / strict sweeps, 6-scan data ablation, full 61-scan corpus, history (D28), 3-epoch run, instruction-dependence, layer probe, multi-layer fusion (D29) |
| 09-09 13:56-15:00 | Pointing analysis | Failure analysis, bearing calibration, turn-threshold sweep |
| 09-09 15:24 to 09-10 01:10 | Cosmos3-Edge (diffusion/action) | Zero-shot inverse dynamics, zero-shot policy, resampled render, scale probe, flow-matching post-train attempt, controllability / CFG sweeps, closed-loop Edge evals |
| 09-10 00:54-15:42 | Cosmos ablations | `run_ablations.py` arms (mostly incomplete: OOM or killed) |
| 09-10 01:10 to 09-11 02:58 | Cosmos reasoner | Reasoner probes (zero-shot), SFT v1 (onset-sampled), uniform (aborted), v3 (uniform, 120-frame, projector LoRA), closed-loop direct-policy evals, next-action probes |

---

## 2. Detailed catalogue (chronological)

Format: **id · date · stage**, then Question / Why / Setup / Result / Conclusion / Source.

### Stage -1 and setup (08-31)

**D1 · 08-31 (pre-existing) · all stages. Backbone substitution**
- Q/Why: The paper uses LLaVA-Video-7B. The approved substitute is Cosmos-Reason2-2B (Qwen3-VL-2B lineage), whose vision encoder is SigLIP-2, the same family as the paper's.
- Result: Not an experiment. Its consequence is that gates are judged on relative movement, not absolute parity.
- Source: DECISIONS.md:15-24.

**D2 · pre-existing · Stage 0-3. Vision encoder frozen**
- Why: v̄_{t+2} (the Pilot target) comes from the same encoder, and a trainable target invites collapse. VRAM is a second reason.
- Source: DECISIONS.md:27-36.

**E-01 · 08-31 · setup. Environment / Section 1 gate**
- Q: What are d, N_v and the VRAM footprint on a 16 GB card?
- Result: d = 2048; N_v = 196 at 448×448, checked 3 independent ways; vision out = 2048; bf16 load + forward uses 4.64 GB allocated, 4.89 GB reserved, 4.83 GB peak. EQUATIONS.md gives 4.66/4.92 GB from Session 1.
- Conclusion: Standard LoRA, not QLoRA (D4). Real R2R-CE sequence lengths are min 204 / median 228 / max 329 (D3).
- Source: HANDOFF.md:426-434; DECISIONS.md:40-63; docs/EQUATIONS.md:205-215; `scripts/verify_env.py`.

**D3 · 08-31 · all. Input 448×448, N_v = 196** (NOT IN PAPER). Source: DECISIONS.md:40-51.

**D4 · 08-31 · all. Standard LoRA.** 4.64 GB allocated leaves more than 11 GB headroom. Source: DECISIONS.md:54-63.

**E-02 · 08-31 · setup. Session 2 script audit**
- Result: 9 bugs found and fixed. Examples: `model.visual()` returns a wrapper and not a tensor; ROS on PYTHONPATH; unseeded probe; two tests with wrong premises.
- Source: HANDOFF.md:307-333.

**E-03 (backs D5) · 08-31 · Stage -1. Encoder anisotropy calibration**
- Q: Are the literal cosine thresholds (0.90 / 0.98) meaningful for this encoder?
- Result: Solid black vs solid white has cosine **0.907**; unrelated random images average **0.752**. On synthetic smooth data, raw copy = 0.9986 (verdict FAIL) against an unrelated floor of 0.9969, normalized headroom 0.55, centered 0.31.
- Conclusion: The raw threshold is miscalibrated.
- Source: HANDOFF.md:336-365; docs/GOTCHAS.md §7.

**D5 · 08-31 (human-approved) · Stage -1. Predictability probe; judged on fps≈4**
- Q: Is v̄_{t+2} predictable enough from v̄_t to be a learnable target?
- Setup: 3 real phone clips (walk.mov, move2.mp4, move3.mp4 in `data/`). Forward retrieval v̄_t → v̄_{t+2} against a 50-frame pool.
- Result: At fps 4, forward retrieval is **2.0× / 2.9× / 8.3×** chance, above chance in all 3 clips; the raw gate FAILS (>0.98) in all 3. At fps 2 it is **≈1.0×**, exactly chance in all 3.
- Conclusion: fps≈4 is the operating point, and Stage -1 passes.
- Source: DECISIONS.md:66-85; HANDOFF.md:235-262; `src/data/predictability_probe.py`.

**D6 · 08-31 · all. mRoPE ids reconstructed explicitly**
- Q: What does the silent fallback to sequential position ids (when using `inputs_embeds`) cost?
- Setup: 5 real R2R-CE instructions × walk.mov frames, identical sequence with only `position_ids` differing.
- Result: Hidden-state relative L2 **0.42-0.50**, hidden cosine 0.87-0.91, action-position cosine 0.39-0.94, top-1 next token agreed **2/5**.
- Conclusion: mRoPE is reconstructed explicitly and verified bitwise against `get_rope_index`.
- Source: DECISIONS.md:88-113; docs/GOTCHAS.md §2; HANDOFF.md:64-75.

**D7 · 08-31 · Stage 1 target cache. v̄ cached in fp32**
- Result: The same frame at batch 3 vs batch 1 differs by **~7% relative L2** in bf16 (cosine 0.998). In fp32 the difference is **8e-6**. `scripts/archive/cache_vbar_when_free.sh` states fp32 is batch-invariant to 1.3e-6 while bf16 is 6.8e-2.
- Accepted residual (not measured): v_t is left in bf16.
- Source: DECISIONS.md:117-137; GOTCHAS §1.

**D8 · 08-31 · Stage 0. Actions as existing vocab tokens via native W_LM**
- Result: forward = 13435, left = 2359, right = 1291, stop = 9495. Uniform loss = **1.3863 = ln(4)**, where the full vocabulary would give 11.93.
- Source: DECISIONS.md:140-155; HANDOFF.md:53-58.

**D9 · 08-31 · all losses. Sum within trajectory, mean across batch**
- Why: The old `mean` default made λ effectively 0.1·T/(T-2), which is **0.15 at T=6** (a 50% error).
- Source: DECISIONS.md:159-176.

**D10 · 08-31 (human-approved) · Stage 0. Sequence layout image-first with markers**
- Q: Which token layout lets the untrained model follow instructions zero-shot?
- Setup: 40 probes × 4 real frames = **160 trials**; chance is 25%; mRoPE correct in all arms.
- Result:

  | Arm | Accuracy | L/R separation | P(correct) |
  |---|---|---|---|
  | A: literal Eq. 5 | **21.2%** | −0.041 | 0.239 |
  | D: instruction first + markers | 41.2% | +0.087 | 0.411 |
  | E: image first, no markers | 33.1% | −0.023 | 0.328 |
  | **F: image first + markers (chosen)** | **77.5%** | **+0.407** | **0.716** |
  | C: full native + chat template | 68.8% | +0.328 | 0.652 |

  A first pass at 16 trials gave E = 87.5% (the wrong conclusion).
- Conclusion: Layout F is adopted, with no chat template.
- Source: DECISIONS.md:180-209.

### Stage 0 (09-01)

**E-04 · 09-01 · data. MP3D download and consistency validation**
- Result: 17/17 scans extracted (11 val_unseen / 1,839 episodes; 6 train / 1,665 episodes), zero train/val overlap. Stored geodesic matches habitat's computed geodesic to **0.0000 m** on every episode checked.
- Source: HANDOFF.md:101-155.

**D11 · 09-01 · metrics. SPL denominator is the geodesic start-to-goal distance**
- Result: Over all 1,839 val_unseen episodes the reference path is a median **1.07×** longer than the geodesic, so a perfect reference-follower scores **SPL ≈ 0.92**. The paper's NaN row gives 47.1/51.7 = 0.911.
- Source: DECISIONS.md:213-238.

**D12 · 09-01 · data. Expert follows reference_path, not the shortest path**
- Why: This was a bug caught by nDTW. The shortest-path expert ignores the language.
- Numbers for the old expert: not recorded.
- Source: DECISIONS.md:241-256.

**D13 · 09-01 · metrics. nDTW resamples both paths to 0.25 m**
- Result: For a perfect expert rollout, the literal formula gives raw DTW 18.76 against a normaliser of 18.0, so nDTW = **0.19**. Resampling both paths gives nDTW = **0.80**.
- Source: DECISIONS.md:260-285.

**D14 · 09-01 · data. Follower goal radius = 3.0 m** (NOT IN PAPER). The expert stops at a mean NE of 2.91 m. Source: DECISIONS.md:289-300.

**E-05 · 09-01 · data. Expert rollout collection and validation**
- Setup: `validate_rollouts.py --check-frames` (PASS).

  | | train | val_unseen |
  |---|---|---|
  | episodes | 1,665 | 1,839 |
  | steps | 69,606 | 72,024 |
  | SR / SPL / OS | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.9999 / 1.0000 |
  | nDTW | 0.7544 | 0.7529 |
  | NE | 2.88 m | 2.87 m |
  | length min/med/max | 11/40/95 | 9/37/90 |

  Action prior: FWD 0.639 (train) and 0.617 (val); marginal action entropy 0.9744 (train) and 1.0065 (val).
- Source: HANDOFF.md:191-231.

**E-06 · 09-01 · Stage 0. VRAM smoke test (batch size)**
- Result (peak GB / s per step): batch 1 = 5.32 / 0.07; batch 2 = 6.12 / 0.10; batch 4 = 7.76 / 0.19; **batch 8 = 10.76 / 0.37**; batch 12 = 13.94 / 0.57.
- Trainable parameters: 6,422,528 (0.263% of 2.45B), LoRA r=16 on 112 q/k/v/o projections.
- Source: HANDOFF.md:157-189.

**E-07 · 09-01 · Stage 0. 150-step probe run (does the model learn beyond the prior?)**
- Result: Loss 1.02 against a marginal of 1.029; accuracy 0.58 against a majority class of 0.596; STOP recall 0.000.
- Conclusion: The model learned only the prior.
- Source: HANDOFF.md:224-227.

**E-08 · 09-01 17:02-18:07 · Stage 0. Memoryless Stage 0 training**
- Setup: `train_stage0.py`, batch 8 (HANDOFF), 8,000 steps; checkpoint every 500. `checkpoints/stage0/` holds step500 … step8000 and final.
- Result: trainlog.json has 160 entries. First entry (step 50): loss 1.8446, acc 0.4475. Last (step 8000): loss 0.6526, acc 0.775, stop_recall 0.0. Mean of the last 20 entries: loss 0.620, acc 0.782, stop_recall 0.000 (computed from trainlog.json).
- Source: `checkpoints/stage0/trainlog.json`. There is no training log in `logs/`.

**D15 / E-09 · 09-01 · Stage 0. Closed-loop eval and "Stage 0 cannot reproduce the NaN row"**
- Setup: Trained Stage 0 checkpoint, **12 episodes on zsNo4HB9uLZ**, diagnostic mode (3 m early exit).
- Result: **10/12** episodes made exactly zero progress. Rollout action mix: LEFT 44.2%, RIGHT 43.0%, FWD 12.7% (the training mix was FWD 61.7%, LEFT 19.2%, RIGHT 16.6%). model_stop fired **0/12**. `eval_stage0.py` also records mean P(STOP) ~0.045 on true-STOP frames, with 0/30 predicted correctly.
- SR / SPL / NE for this eval: not recorded.
- Conclusion: The NaN row is a recurrent policy. The human chose Option 1 on 09-01: build the recurrent Pilot slot (Stage 0').
- Source: DECISIONS.md:304-369; HANDOFF.md:3-14; `scripts/eval_stage0.py` docstring.

**D16 · 09-01 (approved) · Stage 0'/1. z_0 is a learned nn.Parameter(2048)**
- Result: `<|placeholder|>` splits into 5 tokens. The embedding matrix has 311,164,928 params (48× the LoRA). Init norm is about 1.44 (real tokens) vs about 0.36 (spare rows).
- Source: DECISIONS.md:373-395.

**D17 · 09-01 (approved, later reversed by D19) · Stage 0'. Pilot slot placement; zero-shot Design A vs B**
- Setup: 160 trials, untrained pilot.
- Result:

  | Condition | Accuracy | L/R separation | P(correct) | P(STOP) on STOP |
  |---|---|---|---|---|
  | No pilot | 77.5% | 0.407 | 0.716 | 0.885 |
  | A (pilot last) | **59.4%** | 0.082 | 0.557 | 0.647 |
  | B (pilot + query) | **29.4%** | 0.058 | 0.334 | 0.225 |

- Conclusion: A was chosen at the time.
- Source: DECISIONS.md:399-451; the "OPEN" section at 533-579.

**D18 · ~09-01/02 · Stage 1 prep. L_pil scale**
- Result: ‖v̄‖ ≈ 15.8, so ‖v̄‖² ≈ 250. Per-step L_pil by predictor: z = 0 gives 251.0 (λ·L_pil is 25× L_act); global mean gives 17.9 (1.8×); z = v̄_t gives 8.2 (0.8×); z = v̄_{t+1} (the copy baseline) gives **4.78** (0.5×).
- Conclusion: The gate (b) threshold is held-out L_pil < **4.776**. The bias is seeded with mean v̄.
- Source: DECISIONS.md:495-529.

**D20 · ~09-01/02 · Stage 1. G_psi zero-weight, mean-v̄ bias init**
- Result: The nn.Linear default init gives ‖z‖ 62.9, L_pil 3748.7 and a λ·L_pil : L_act ratio of **167.59×**. Zero weight with mean bias gives ‖z‖ 15.0, L_pil 17.4, ratio **0.42×**. v̄ variance is 18.48.
- Source: DECISIONS.md:584-609.

**D21 · ~09-01/02 · Stage 1. L_pil reference baselines over 15,627 cached frames**
- Result: Constant (mean v̄) = **18.48**; persistence (v̄_t → v̄_{t+2}) = **7.82** (0.423× constant).
- Source: DECISIONS.md:613-627. FLAG: these differ from D18's 17.9 / 8.2 (see Section 9).

**D22 · 09-02 · Stage 1. Fig. 4 layout vs Eq. 12**
- Conclusion: Eq. 12 is implemented literally. Trajectory packing is deferred to Stage 2. Design only, no measurement.
- Source: DECISIONS.md:631-659.

### Stage 0' (09-02)

**E-10 · 09-02 00:31 (aborted) and 00:54-05:42 · Stage 0'. Design A vs B training (NaN-row reproduction)**
- Setup: `scripts/run_ab_overnight.sh`, `train_stage0_prime.py`, 17,400 steps (2 epochs over 69,606 pairs), batch 8, 6 scans / 1,665 episodes, cosine LR (warmup 100, peak 1e-4, floor 5e-6), no L_pil. A: pilot_mode = last. B: pilot_mode = action_query. The first launch at 00:31 was aborted after about 2,650 steps (`logs/ab_overnight_aborted.log`).
- Result: A ran 00:54-03:18 and B ran 03:18-05:42.
  - Final logged step: A loss 0.1555, acc 0.9725, STOP recall 0.769; B loss 0.1190, acc 0.975, STOP recall 0.769.
  - D19's quoted "loss 0.070 vs 0.082, acc 0.984 vs 0.983, STOP recall 0.761 vs 0.770" equals the mean of the last 20 trainlog entries (recomputed: A 0.070 / 0.984 / 0.761 and B 0.082 / 0.982 / 0.770).
- Source: `logs/ab_overnight.log`, `logs/ab_overnight_aborted.log`, `checkpoints/stage0prime_{A,B}/trainlog.json`.

**D19 · 09-02 · Stage 0'. Closed-loop reverses D17: B beats A**
- Setup: Stage 0' A and B final checkpoints, val_unseen **n=150**, diagnostic mode.
- Result:

  | | A | B | Stage 0 |
  |---|---|---|---|
  | SR/OS | **0.0000** | **0.1667** | 0.1667 |
  | NE | 8.79 | 7.95 | not recorded |
  | nDTW | 0.244 | 0.303 | not recorded |
  | Own STOP fired | 39.3% | 30.0% | 0% |

- Conclusion: The zero-shot probe predicted the wrong winner. Use such probes to rule things out, not to pick winners. Design B is adopted.
- Source: DECISIONS.md:454-491. No eval log exists in `logs/` (unverified). The Stage 0 row's n is not stated.

**D23 · 09-02 · Stage 0'/1. 8-way slot-content ablation (Design × slot content)**
- Setup: n=60 each, val_unseen, diagnostic mode. Checkpoints are presumably the Stage 0' A/B finals (not named).
- Result: Design A scored 0.0000 for all four slot contents (z, z_scaled, vbar_current, constant). Design B scored **0.1833** (z), **0.2167** (z_scaled), 0.0000 (vbar_current) and 0.0000 (constant).
- Conclusion: A's failure is architectural. B's Pilot slot is load-bearing. Stage 1 defaults to Design B.
- Source: DECISIONS.md:663-684. No log (unverified).

**D24 · 09-02 · infrastructure. Checkpoints save the adapter only**
- Result: Stage 0' wrote 37 full-model checkpoints (~180 GB). After the fix a checkpoint is **33 MB** (224 LoRA tensors + PilotModule) instead of 4.9 GB. `stage0prime_*/final` still holds a full model.safetensors, and only `final` remains on disk.
- Source: DECISIONS.md:688-694; checkpoint listing.

### Stage 1 (09-02)

**E-11 · 09-02 ~09:50-~11:32 · Stage 1. Learned run #1 (no log in repo)**
- Evidence: Checkpoint directories `stage1_learned/step2000…step10000` were created 10:26-11:32. Their files were overwritten at 12:17-13:24 by run #2.
- Eval of run #1 step10000 (11:57), n=30, diagnostic: SR 0.2667, SPL 0.2667, OS 0.2667, nDTW 0.3751, NE 7.8896; model_stop 0%, within_radius 8 (26.7%), timeout 22 (73.3%); ‖z_t‖ mean 15.19 (min 13.31, max 17.89).
- Recordings: 3 zsNo4HB9uLZ episodes (`record_120550.log`: all timeout, min_dist 7.96 / 6.23 / 3.60 m). 11-scene set (`record_multi_122336.log`): 1/11 within_radius (x8F5xyUWy9e, 2.77 m).
- FLAG: The step10000 checkpoint now on disk is from run #2, not the one evaluated.
- Source: `logs/eval_step10000_115728.log`, `record_120550.log`, `record_multi_122336.log`, checkpoint mtimes.

**E-12 · 09-02 11:51 (aborted at step ~800), 12:00-14:25 and 14:25-16:50 · Stage 1. Learned vs identity G_psi (the decisive gate)**
- Setup: `scripts/run_stage1.sh` → `train_stage1.py`. Design B (action_query), λ = 0.1, 17,400 steps, batch 8, 6 scans / 1,665 episodes. Eq. 14 target coverage 0.952; mean ‖v̄‖ (train) 14.798. G_psi is `learned` in the first run and `identity` (frozen) in the second.
- Training (trainlog.json):
  - Learned: step 50 L_act 2.7858 / L_pil 14.468; step 4000 0.4104 / 4.217; step 10000 0.1000 / 3.213; step 17400 0.0547 / 3.018 (acc 0.988, STOP recall 0.923).
  - Identity: step 50 L_act 4.6275 / L_pil 5070.369; step 17400 L_act 0.7447 / L_pil 6.799 (acc 0.748, STOP recall 0.000).
- Evals, n=30, diagnostic:
  - Learned final (14:31): SR/SPL/OS 0.1000, nDTW 0.3059, NE 8.5797, model_stop 0%.
  - Identity final (16:52): SR/SPL/OS 0.4000, nDTW 0.4042, NE 7.3954, model_stop 0%; ‖z‖ max 23.53.
  - Recordings of learned final on 11 scenes (`record_final_143913.log`): 0/11 success. x8F5xyUWy9e went from 2.77 m (run #1 step10000) to min 9.21 m.
- Source: `logs/stage1_20260902_115156.log` (aborted), `logs/stage1_20260902_120059.log`, `logs/eval_final_143122.log`, `logs/eval_identity_165233.log`, `checkpoints/stage1_{learned,identity}/trainlog.json`.

**D25 · 09-02 ~17:00 · Stage 1. Gate FAILED (both halves)**
- Gate (a): target SR 54.0 / SPL 48.5; measured SR 0.0. This means strict self-stop SR (model_stop was 0%).
- Gate (b), n=30:

  | G_psi | L_pil | OS | NE | nDTW | model_stop |
  |---|---|---|---|---|---|
  | Learned | 3.02 | 0.10 | 8.58 | 0.306 | 0% |
  | Identity | 6.80 | 0.40 | 7.40 | 0.404 | 0% |

  FLAG: The L_pil values equal the last training-log entries, not a held-out evaluation. No held-out L_pil computation exists in the repository.
- Loss-drift mechanism (λ·L_pil / L_act), verified against trainlog.json:

  | Step | L_act | λ·L_pil | Ratio |
  |---|---|---|---|
  | 50 | 2.786 | 1.447 | 0.52× |
  | 4000 | 0.410 | 0.422 | 1.03× |
  | 10000 | 0.100 | 0.321 | 3.21× |
  | 17400 | 0.055 | 0.302 | 5.51× |

- Conclusion: Stop and do not proceed to Stage 2. Rebalancing λ is a STOP-and-ask item.
- Source: DECISIONS.md:699-757.

**E-13 (D26, undocumented in DECISIONS.md) · 09-02 17:05-~18:37 (killed) · Stage 1. Adaptive-λ ("balanced") run**
- Q: Is D25's λ drift what broke Stage 1?
- Setup: `run_lambda_fix.sh`, `--lambda-mode balanced --target-ratio 0.9`. λ_t is recomputed so that λ·L_pil/L_act = 0.9. Everything else matches E-12 (6 scans, seed, 17,400-step schedule).
- Result: The log stops at step 10,750, and the last checkpoint is step10000. λ fell from 0.1737 (step 50) to 0.0407 (step 10000). The second arm (constant λ = 0.02, `checkpoints/stage1_lam002`) was **never run**.
- Evals:
  - step8000, n=30: SR/SPL/OS 0.1333, nDTW 0.2804, NE 8.6863, model_stop 0%.
  - step10000, n=150: SR/SPL/OS 0.0667, nDTW 0.2751, NE 8.3559, model_stop 27 (18.0%).
- Conclusion (from D27 and `scripts/archive/run_step1_then_step2.sh`): "the lambda axis was flat (balanced vs learned, p=0.333)". The p-value computation is not in any source.
- Source: `logs/lambdafix_20260902_170524.log`, `logs/eval_balanced_step8000_181830.log`, `logs/evaln150_stage1_balanced_step10000.log`, `scripts/run_lambda_fix.sh`, `src/train/train_stage1.py:97,239,615`.

**E-14 · 09-02 18:37-19:05 · Stage 1. n=150 re-evaluation ("STEP 1")**
- Why: At n=30 every CI spans about 0.24; n=150 narrows that to about 0.11 (`scripts/archive/run_step1_then_step2.sh` header).
- Result, n=150 (8 scans), diagnostic:

  | Checkpoint | SR | SPL | OS | nDTW | NE | model_stop | ‖z‖ max |
  |---|---|---|---|---|---|---|---|
  | Identity final | 0.1733 | 0.1720 | 0.1733 | 0.2680 | 8.1705 | 0 (0.0%) | 52.17 |
  | Learned final | 0.1067 | 0.1067 | 0.1067 | 0.2768 | 8.4955 | 7 (4.7%) | 21.10 |
  | Balanced step10000 | 0.0667 | 0.0667 | 0.0667 | 0.2751 | 8.3559 | 27 (18.0%) | 19.39 |

- Conclusion: The identity advantage shrank from 4× (n=30) to 0.173 vs 0.107. This led to the pivot (D27).
- Source: `logs/step12_20260902_183744.log`, `logs/evaln150_*.log`.

**E-15 · 09-02 17:13-19:37 · data. Expansion to 16 scans; 16-scan Stage 1 planned and never run**
- Collection of 10 new scans: a first attempt at 17:13 failed (`ModuleNotFoundError: habitat_sim`, wrong env). A second at 17:14 collected 2,538 episodes, taking the corpus to 4,203 episodes / 16 scans. v̄ caching OOMed beside training at 17:24 (9.49 GiB used). The "STEP 2a" cache was interrupted at 400/2538 episodes by re-collection.
- `stage1_16scan` was never trained (`eval16_20260902_184603.log` only says "waiting"; no checkpoint exists). `checkpoints/stage2/` is empty, so Stage 2 (scheduled sampling) was never run, although `src/train/train_stage2.py` exists.
- Source: `logs/collect_20260902_171322.log`, `collect_20260902_171438.log`, `cachevbar_20260902_172725.log`, `step12_20260902_183744.log`, `eval16_20260902_184603.log`.

### Pointing (D27-D29), 09-02 to 09-09

**E-16 · 09-02 19:23-19:37 · pointing data. Re-collect 16 train scans (4,203 episodes) and val_unseen (1,839) with per-step rotations and intrinsics**
- Why: Yaw reconstructed from actions is exact 93.8% of the time, but about 6% are 15° off.
- Result: After re-collection LEFT = +15.000°, RIGHT = −15.000°, FWD = 0.000° exactly. This invalidates the cached vbar.npy.
- Source: `logs/recollect_20260902_192305.log`, `scripts/archive/recollect_with_poses.sh`, DECISIONS.md:827-837.

**D27 · 09-02 ~19:30-20:14 (human-directed pivot) · pointing. Robostral pointing replaces 4-way action classification**
- Q: Do the pointing labels agree with the expert's own actions?
- Label agreement:

  | Label design | Agreement |
  |---|---|
  | Naive | 0.689 |
  | + skip in-place turn steps | 0.927 |
  | + decoupled horizons / first-displaced target | **0.972** |

- Other measurements: one horizon of 24 gives 0.52 coverage and 0.73 agreement; a horizon of 8 gives 0.93 agreement and 0.27 coverage; decoupled horizons give **0.55 coverage AND 0.97 agreement**. With the furthest (rather than first displaced) target, agreement drops 0.93 → 0.75. Visible fraction is 0.55 (vs Robostral's ~0.90).
- Controller: turn if |bearing| > 7.5°, else forward. No diffusion policy.
- Source: DECISIONS.md:762-837.

**E-17 · 09-02 20:14-23:21 · pointing. Pointing training on 16 scans**
- Setup: `run_pointing.sh`, 20,000 steps × batch 8 (0.93 epoch) over 4,203 episodes / 171,281 steps. STOP fraction 0.0245 (pos_weight 39.8); visible fraction 0.6470. Trainable 6,434,822 (LoRA + 12,294 head).
- Training: step 50 act-agree 0.665; step 10000 0.695; step 20000 0.718 (p(stop) 0.002).
- Source: `logs/pointing_20260902_201418.log`, `checkpoints/pointing/trainlog.json`.

**E-18 · 09-02 21:39-23:57 · pointing. Evals of `pointing` checkpoints (STOP threshold and strict sweeps)**
- step10000, n=60, diagnostic, thr 0.5: SR 0.2667, SPL 0.2653, OS 0.2667, nDTW 0.3220, NE 8.6583, model_stop 0%. p(stop) mean 0.0132, p95 0.0440, max 0.2526. This is the README's "pointing step 10,000" row.
- Diagnostic stop-threshold sweep (n=60): see the table in Section 3. SR fell as the threshold dropped (0.1833 → 0.1500 → 0.1000).
- First strict sweep (n=60, 22:05): best strict SR 0.1333 at thr 0.05.
- step16000, n=60, thr 0.05: strict SR 0.2167, diagnostic 0.2167.
- final (step20000), n=150, thr 0.05: strict SR 0.1333 / SPL 0.1304 / OS 0.1400; diagnostic 0.1400.
- Source: `logs/evalpt_213955.log`, `stopsweep_215032.log`, `strictsweep_220514.log`, `eval16k_224747.log`, `evalfinal150_232320.log`.

**E-19 · 09-02 23:21 to 09-03 02:18 (train); evals 09-03 11:50-12:24 · pointing. Data-scaling ablation (6-scan vs 16-scan)**
- Q: Is the ceiling data or architecture? This also un-confounds "pointing beats LatentPilot (p=0.0055)", which compared 4,203 against 1,665 episodes.
- Setup: `run_data_ablation.sh`. Same 20,000 steps × batch 8, same seed and LR schedule, restricted to the 6 LatentPilot scans (1,665 episodes / 69,606 steps, 2.30 epochs).
- Training: final act-agree 0.825 (training loss 0.0756, lower than the 16-scan run's 0.3955).
- In-script evals produced no output (the runner swallowed an error; see the comment in `scripts/archive/run_all_evals.sh`). Re-evaluated at 09-03 12:24 with n=150, thr 0.05: **strict SR 0.0400** / SPL 0.0372 / OS 0.0667; diagnostic 0.0667. For comparison, the 16-scan final gives strict 0.1333.
- Conclusion (implied): more data helps. No explicit conclusion is written anywhere.
- Source: `logs/dataablation_223555.log`, `logs/evalsord_115012.log`, `checkpoints/pointing_6scan/trainlog.json`.

**E-20 · 09-03 00:29-01:19 · data. Full corpus**
- Result: MP3D re-extraction of 72 scans (61 train / 10,819 episodes plus 11 val_unseen). Full train collection of 10,819 episodes (00:49-01:19).
- Source: `logs/download_20260903_002918.log`, `logs/collectfull_004951.log`.

**E-21 (D28, undocumented in DECISIONS.md) · 09-03 02:19-09:26 · pointing. Frame history on the full 61-scan corpus (`pointing_hist`)**
- Setup: `run_overnight_history.sh` intended K=2, stride 8. **The run actually used stride 1** (log: "history frames = 2 (stride 1, spans 2 steps)"; `pointing_hist/final/config.json` history_stride 1). The auto-probed batch was 4. It ran 37,800 steps (7 h at about 1.5 it/s), which is 0.35 epoch over 435,468 steps.
- Training: final act-agree 0.770.
- Evals (09-03 12:24), n=150, thr 0.05: **strict SR 0.1800** / SPL 0.1680 / OS 0.2267 / nDTW 0.3937 / NE 6.5540 (model_stop 68.0%). Diagnostic: 0.2267 / 0.2212 / 0.2267 / 0.4088 / 6.6469.
- Source: `logs/overnight_20260903_005307.log`, `logs/evalsord_115012.log`, `checkpoints/pointing_hist/{trainlog,final/config}.json`. The "wrong stride for 7 hours" incident is noted in `run_full3epoch.sh`.

**E-22 · 09-03 12:49 to 09-04 ~15:10 (killed at about step 131,200 of 326,601) · pointing. `pointing_full3` (3-epoch plan, K=2 stride 8, 61 scans)**
- Setup: batch 4 (13.04 GB peak; batch 6 OOMs), cosine LR over 326,601 steps (so intermediate checkpoints are un-annealed), save every 20k. Checkpoints exist at step20000 through step120000.
- Training act-agree: 0.735 @ 20k; 0.792 @ 40k; 0.772 @ 60k; 0.761 @ 80k; 0.771 @ 100k; 0.806 @ 120k.
- Evals: see Section 3. Key rows:
  - step20000, n=150, thr 0.05: strict SR 0.0867.
  - step20000, n=100, thr 0.35: **strict SR 0.1500 / SPL 0.1307 / OS 0.2800 / nDTW 0.2085 / NE 9.2117**. This is the README's pointing_full3 row.
  - step40000, n=150, thr 0.10: strict SR 0.2467.
  - step60000, n=150, thr 0.20: strict SR 0.2200.
  - step100000, n=150, thr 0.10: strict SR 0.2267.
- Source: `logs/full3_20260903_124907.log`, `evalck20k_193554.log`, `thrsweep_220712.log` (checkpoint unnamed), `sweep40k_002048.log`, `sweep100k_111730.log`, `baseline035_103840.log`, `test80k_125633.log`.

**E-23 · 09-04 11:20-11:29 · pointing. val_seen collection (778 episodes, 53 scans), for choosing the STOP threshold off the test split**
- Why: Every earlier threshold was tuned on val_unseen, which is optimistic.
- Result: No eval on val_seen exists in any log. FLAG: the threshold was never re-selected on val_seen.
- Source: `logs/valseen_112053.log`, `scripts/collect_valseen.sh`.

**E-24 · 09-04 12:51 · pointing. Rollout recordings of `pointing_full3/step100000`**
- Result: 22 episodes; 6 OK (all `model_stop` within 3 m), 16 failures saved (`recordings/step100k_failures/`, `media/rollouts/step100k_failures` per media/README).
- Source: `logs/rec100k_125103.log`.

**E-25 · 09-04 15:11 · pointing. Instruction-dependence test on `pointing_full3/step120000`**
- Setup: 900 held-out steps, 80 distinct instructions, teacher-forced.
- Result:

  | Condition | u MAE | u corr | act agree | bearing median | stop separation |
  |---|---|---|---|---|---|
  | real | 0.0905 | 0.768 | 0.764 | 3.8° | 5.1× |
  | swapped | 0.1248 | 0.517 | 0.686 | 5.5° | 1.0× |
  | empty | 0.1164 | 0.430 | 0.603 | 7.3° | 1.1× |

  Real → swapped: u corr −33%, act −10%. Real → empty: u corr −44%, act −21%.
- Conclusion: The policy uses the instruction. Stop separation collapses without the correct instruction.
- Source: `logs/instrdep_151148.log`, `scripts/test_instruction_dependence.py`.

**E-26 (the D29 evidence) · 09-04 15:48 / 18:14 / 19:17 · pointing. Per-layer linear probe and fusion search**
- Setup: `layer_probe.py` on `pointing_full3/step120000`, K=2 stride 8, 2,000 train frames and 1,000 val_unseen test frames. Stop base rate 0.0420 (train) and 0.0270 (test).
- Result:
  - Final layer 28: u corr 0.317, act 0.407, stop AUC **0.441** (below chance).
  - Best single layers: u at layer 26 (0.357); act at layer 0 (0.646, which equals a majority-like baseline); stop at layer 15 (AUC 0.718); layer 23 stop AUC 0.694.
  - The 570-combination search ranks 5,23,26,28 top (u 0.303, act 0.577, AUC 0.736, score 0.815). The pair 23,26 reaches stop AUC 0.797.
  - FLAG: The exact 23,26,28 combination is not printed. `run_fusion.sh` quotes "~0.30 / ~0.55 / ~0.74".
- Source: `logs/layerprobe_154820.log`, `layersearch_181452.log`, `layersearch2_191721.log`; `logs/layerfeat_step120000_2000_1000.npz`.

**E-27 (D29, undocumented in DECISIONS.md) · 09-04 19:51 to 09-05 ~21:57 (killed at about step 124,000 of 163,300) · pointing. Multi-layer fusion head (layers 23, 26, 28)**
- Setup: `run_fusion.sh`, K=2 stride 8, 61 scans, batch 4, 1.5-epoch cosine schedule. Head has 49,158 params (6,471,686 trainable in total).
- Training act-agree: 0.733 @ 20k; 0.777 @ 40k; 0.790 @ 60k; 0.784 @ 80k; 0.779 @ 100k; 0.809 @ 120k.
- Strict evals, n=150 (full grid in Section 3). The best strict SR is **step120000, thr 0.20: SR 0.3133 / SPL 0.2537 / OS 0.4267 / nDTW 0.3232 / NE 7.2189**. Step60000 at thr 0.20 gives SR 0.3000 / OS 0.4800.
- Matched comparison at thr 0.20: fusion step80000 (0.73 ep) SR 0.2267 vs full3 step60000 (0.55 ep) SR 0.2200. Essentially no difference.
- Source: `logs/fusion_20260904_195116.log`, `fusesweep_234742.log`, `distdiag_004917.log`, `sweep60k_104019.log`, `test80k_125633.log`, `eval100k_185222.log`, `eval120k_220002.log`, `checkpoints/pointing_fusion/*/config.json`.

**E-28 · 09-09 13:56 · pointing. Teacher-forced failure analysis of `pointing_fusion/step120000`**
- Setup: 2,500 non-STOP val_unseen steps. All are on scan 2azQ1b91cZZ, so this is a single-scan analysis.
- Result: Overall agreement **0.771**.
  - By expert action: FWD 0.901, LEFT 0.553, RIGHT 0.557.
  - By distance: 0-3 m 0.615, 3-5 m 0.789, 5-10 m 0.775, >10 m 0.762.
  - By episode position: 0.775 / 0.769 / 0.753 / 0.761 / 0.799.
  - By instruction length: short 0.764, medium 0.775, long 0.772.
  - Confusion: LEFT → FWD 0.388, RIGHT → FWD 0.428, LEFT → RIGHT 0.059, RIGHT → LEFT 0.015.
- Conclusion: Turns are missed by predicting forward, not by choosing the wrong direction.
- Source: `logs/failanalysis_135616.log`.

**E-29 · 09-09 14:01 · pointing. Bearing calibration (regression shrinkage)**
- Setup: 3,000 steps across 3 scans on the same checkpoint.
- Result: pred = **0.424 × true**. Mean |θ| on expert turns: true 41.3°, predicted 27.2°. On straights: true 0.1°, predicted 4.2°.
- Post-hoc decision-threshold sweep (overall / FWD / LEFT / RIGHT):
  - 1.0°: 0.473 / 0.262 / 0.781 / 0.835
  - 2.0°: 0.570 / 0.455 / 0.714 / 0.798
  - 3.0°: 0.633 / 0.586 / 0.670 / 0.750
  - 4.0°: 0.675 / 0.690 / 0.623 / 0.692
  - 5.0°: 0.709 / 0.769 / 0.585 / 0.652
  - 6.0°: 0.722 / 0.816 / 0.548 / 0.611
  - 7.5°: 0.733 / 0.873 / 0.487 / 0.552
  - 10.0°: 0.735 / 0.916 / 0.427 / 0.483
- Source: `logs/bearingcal_140140.log`; raw data in `logs/bearings.npy`.

**E-30 · 09-09 14:10-15:00 · pointing. Closed-loop turn-threshold sweep**
- Setup: `pointing_fusion/step120000`, n=150, **diagnostic**, stop thr 0.20.
- Result:

  | Turn threshold | SR | SPL | OS | nDTW | NE |
  |---|---|---|---|---|---|
  | 3.0° | 0.3533 | 0.3471 | 0.3533 | 0.4072 | 6.8954 |
  | 4.0° | 0.4133 | 0.3974 | 0.4133 | 0.4285 | 6.7753 |
  | 5.0° | 0.4200 | 0.4040 | 0.4200 | 0.4099 | 6.6091 |
  | **7.5°** | **0.4267** | 0.4074 | 0.4267 | 0.3972 | 7.0897 |
  | 10.0° | 0.4000 | 0.3734 | 0.4000 | 0.3672 | 6.6943 |

- Conclusion: 7.5° is kept. This 0.4267 is the "pointing baseline" later quoted in `eval_cosmos_nav.py` / `eval_edge.py`. It is a diagnostic OS-style number; at the same checkpoint and stop threshold, strict SR is 0.3133 (E-27).
- Source: `logs/turnsweep_0909_141023/turn_*.log`. The `summary.txt` in that folder has **no metric lines** (grep mismatch).

### Cosmos3-Edge action / diffusion policy (09-09 to 09-10)

**E-31 · 09-09 15:24-15:25 · Edge. Install diffusers and download nvidia Cosmos3-Edge (46 files) to `checkpoints/Cosmos3-Edge`**
- `edge_load_test.py` is the go/no-go for fitting in 16 GB. Its result is **not recorded**; it is only implied by later runs showing "pipeline on GPU, free 7.60 GB".
- Source: `logs/pip_diffusers.log`, `logs/edge_download.log`.

**E-32 · 09-09 15:58-16:15 · Edge. Zero-shot inverse dynamics (does the pretrained action prior read our video?)**
- Q: Does Edge, with no training, infer our ego-motion from video, and which 9D domain fits better (`av` or `camera_pose`)?
- Runs:
  - 15:58: crash (`image` and `video` both None).
  - 16:00: IndentationError.
  - 16:04: BFloat16 → numpy TypeError.
  - 16:08 `edge_zs_run`: bug, the same episode repeated 3×; 48 frames.
  - 16:12 `edge_zs_dedup` (one scan, 224 frames):
    - av: yaw corr **+0.815**, fwd corr +0.742, turn-direction agreement **0.962** (n=133).
    - camera_pose: yaw corr +0.288, fwd corr +0.694, turn-direction agreement 0.789.
  - 16:15 `edge_zs_multiscan` (av only, 384 frames across 11 scans): yaw corr **+0.837**, fwd corr +0.697, turn-direction agreement **0.964** (n=192).
- Conclusion: The `av` domain prior transfers to our walking video.
- Source: `logs/edge_zeroshot_155833.log`, `edge_zs_160047.log`, `edge_zs_160457.log`, `edge_zs_run.log`, `edge_zs_dedup.log`, `edge_zs_multiscan.log`.

**E-33 · 09-09 16:21 · Edge. Zero-shot policy mode (frame + full R2R instruction → 16 ego-pose deltas)**
- Setup: 24 trajectories, 384 frames, guidance 1.0 (per `eval_edge.py` docstring).
- Result: Correct instruction: fwd corr −0.016, yaw corr +0.134, turn direction **0.594** (n=192). Swapped instruction: fwd +0.090, yaw +0.034, turn direction **0.531**.
- Conclusion: Close to chance, and the model barely uses the instruction.
- Source: `logs/edge_policy_zs.log`.

**E-34 · 09-09 16:37-17:29 · Edge data. Re-render as fixed-rate 15 FPS mp4 plus 9D action labels (resampled)**
- Sizing run: 8 episodes, extrapolated to 13,436 episodes = 1.14 M frames / 6.6 GB / 1.6 GPU-hours.
- Rendered: train **3,602 episodes / 596,980 frames / 4.4 GB**; val_unseen **613 episodes / 99,752 frames / 0.8 GB**.
- Source: `logs/render_size.log`, `render_train.log`, `render_valunseen.log`.

**E-35 · 09-09 18:08 · Edge. Translation-scale probe on resampled video**
- Setup: 20 trajectories / 11 scans, 320 frames.
- Result: Yaw corr **+0.987**, turn direction 1.000 (n=259). Least-squares scale over all translation channels **s = 7.47** (trans_z s = 7.64, corr 0.922).
- Conclusion: Adopted as `translation_scale = 7.47` in all later Edge/reasoner evals.
- Source: `logs/edge_scale.log`.

**E-36 · 09-09 20:56-21:02 · Edge. Flow-matching post-training of the Edge action stream (`edge_r2r`)**
- Setup: LoRA r=16 plus `av` rows of action_proj; 13.05 M trainable params; 10,816 samples; lr 1e-4; chunk 16; accum 8; planned 2,000 steps (config.json).
- Result: One logged line, **step 25 loss 0.0769, 6.31 s/step**. No checkpoint was saved (only config.json). The run was abandoned.
- `verify_flow_convention.py` (the sign-convention check) has **no recorded output**.
- Source: `logs/train_edge.log`, `checkpoints/edge_r2r/config.json`.

**E-37 · 09-09 21:06 and 21:40 · Edge. Controllability with short commands, and the CFG sweep**
- 21:06, 11 frames / 11 scans, guidance 1.0: left vs right separation +0.73°/frame, sign correct 73%.
- CFG sweep, 22 frames / 22 scans, separation and sign correctness by guidance:

  | Guidance | Separation | Sign correct |
  |---|---|---|
  | 1.0 | +0.73° | 82% |
  | 3.0 | +2.58° | 86% |
  | 5.0 | +3.70° | **95%** |
  | 7.5 | +4.50° | **95%** |

- Conclusion: Guidance about 7.5 is required.
- Source: `logs/edge_ctrl.log`, `logs/edge_ctrl_cfg.log`.

**E-38 · 09-09 22:39 · Edge. Full R2R instructions at working guidance (correct vs swapped)**
- Setup: 16 trajectories / 11 scans, n=212.
- Result: At g5.0, correct instruction gives yaw corr +0.380 and turn direction 0.651; swapped gives −0.006 and 0.453. At g7.5, correct gives +0.395 and **0.675**; swapped gives −0.031 and **0.495**.
- Source: `logs/edge_policy_cfg.log`.

**E-39 · 09-09 22:54 to 09-10 00:52 · Edge. Closed-loop Edge policy (chunk replayed as discrete primitives)**
- Common setup: chunk 24, guidance 7.5, 20 steps, scale 7.47, diagnostic.
- Result:
  - Smoke, n=4: SR 0.0000, nDTW 0.2976, NE 7.8860.
  - Zero-shot, n=40 (23:30): **SR/SPL/OS 0.0750**, nDTW 0.2756, NE 8.8134, model_stop 75%. This is the "0.0750 stock-diffusion result".
  - "native", n=40 (00:05): identical per-episode lines and metrics (FLAG: duplicate).
  - Fixed prompt "The camera moves forward.", n=20 (00:22): SR/OS 0.1500, nDTW 0.3300, NE 8.4493. Chunk net forward mean 0.668 m; implied speed 0.42 m/s.
  - `run_edge_ablation.sh` A (R2R instruction) at 00:28-00:52: incomplete, no summary; arm B never started.
- Conclusion (implied): A fixed "forward" prompt does at least as well as the instruction (0.15 on n=20 vs 0.075 on n=40, not paired).
- Source: `logs/eval_edge_smoke.log`, `eval_edge_zeroshot.log`, `eval_edge_native.log`, `eval_edge_fwdonly.log`, `logs/edge_ablation_0910_002859/`.

### Cosmos reasoner (09-10 to 09-11)

**E-40 · 09-10 01:10 · reasoner. First reasoner-tower probe (think on, max 512 tokens, 3 still frames)**
- Setup: 3 trajectories (2azQ1b91cZZ/1039, 8194nk5LbLH/1141, EU6Fwq7SyZv/1111), each with 4 prompts: traj_gripper, traj_nav, next_action, hierarchical.
- Result: Qualitative only. traj_* parsed 5 waypoints each; next_action gave off-task scene descriptions; hierarchical gave "open wooden_door", a truncated answer, and "turn right".
- Source: `results/reasoner_probe.md`, `logs/reasoner_probe.log`, `logs/reasoner_frames/*.png`.

**E-41 · 09-10 00:54-15:42 · reasoner / Edge. `run_ablations.py` prompt/policy arms (n=40 planned each, guidance 7.5, chunk 24, patience 2)**
- Result: `results/ablations.md` was **never produced**. Per-arm outcomes (diagnostic, counted from per-episode lines):
  - `r2r_raw` (Edge, raw R2R prompt): three partial attempts (15/40, 5/40, 9/40). The 15-episode attempt had 2 within_radius, 10 model_stop, 3 timeout.
  - `reasoner_direct` (zero-shot reasoner, think=True, burst fwd 2 / turn 2): **34/40** episodes; 4 within_radius, 12 model_stop, 18 timeout. It was killed at 34/40 by a queue script ("matched a leftover smoke-test JSON"). About 130 s per episode.
  - `reasoner_diffusion_grounded` (reasoner caption → Edge diffusion): OOM at 12:29; 0/40 at 12:32; 4/40 at 12:35; **18/40** at 12:53-15:42, with 5 within_radius, 4 model_stop, 9 timeout (up to 367 s per episode).
  - `reasoner_diffusion_bare`: OOM.
  - `r2r_av_reasoner`: started with no episodes logged.
  - `av_action_only`, `r2r_av_action`, `fixed_forward`: never ran.
- Source: `logs/ablation_driver*.log`, `logs/ablation_0910_*/`, `results/salvaged/*.log`, `scripts/run_ablations.py`, `scripts/archive/{handoff_ablation.sh,queue_after.sh,run_remaining.sh}`.

**E-42 · 09-10 11:13 · reasoner. Nav probe v1 (zero-shot, video)**
- Setup: 36 decision points / 12 trajectories, 16-frame context at 15 FPS, horizon 8 frames. Majority baseline **0.500**.
- Result: hier_full **0.389**, hier_overall **0.417**, hier_overall_nothink **0.167**. Parse rates 0.94 / 0.97 / 0.97.
- Source: `results/reasoner_nav_probe.{md,json}`, `logs/probe_reasoner_nav.log`.

**E-43 · 09-10 11:35 · reasoner. Nav probe v2 (zero-shot)**
- Setup: 96 points / 24 trajectories, horizon 16. Majority baseline **0.573**.
- Result: hier_overall **0.375** (stop recall 0.08); hier_full **0.385**.
- Conclusion: This became the motivation for SFT ("0.375 against a 0.573 constant baseline").
- Source: `results/reasoner_nav_probe_v2.{md,json}`, `logs/probe_reasoner_nav_v2.log`, `scripts/train_reasoner_sft.py` docstring.

**E-44 · 09-10 15:45-18:48 · reasoner. SFT v1 (`reasoner_sft`), onset-balanced sampling**
- Setup: Cosmos3-Edge reasoner tower, LoRA r=16 on LM q/k/v/o (6,422,528 trainable), lr 1e-4, accum 16, 1 epoch. 43,036 samples with labels move forward 50%, turn right 12%, turn left 13%, stop 25%. 16-frame context (VIDEO_SPAN was changed to 120 only at 09-10 20:52).
- Training: loss 0.6421 (step 20) → 0.1977 (step 2680).
- Source: `logs/train_reasoner_sft.log`, `checkpoints/reasoner_sft/config.json`, `src/nav/reasoner_sft_data.py` (uniform=False path).

**E-45 · 09-10 17:07-20:06 · reasoner. Probes and closed loop for SFT v1** (all probes: 160 decision points / 40 trajectories, context 16 frames, horizon 16, majority 0.544)
- Zero-shot, new labels (17:29): hier_overall_nothink **0.325** (parse 0.96); hier_overall (think) **0.331** (parse 0.93).
- step1250 (17:07): **0.656**.
- final (19:59): **0.675** (recall fwd 0.77, left 0.59, right 0.62, stop 0.53).
- Closed loop, direct policy, no think, n=40, diagnostic:
  - step1250: **SR 0.0000**, nDTW 0.2796, NE 8.6351, model_stop 65%. Commands: turn left 42%, turn right 29%.
  - final: **SR/SPL/OS 0.1750**, nDTW 0.3653, NE 7.4113, model_stop 65%. "where it STOPPED within 3m 0.0%".
- Conclusion: Onset over-sampling taught a turn-heavy closed-loop behaviour. Next step: sample uniformly.
- Source: `results/probe_zeroshot_newlabels.*`, `results/probe_sft_step1250.*`, `results/probe_sft_final.*`, `results/runs/sft_*_direct.json`, `logs/eval_sft_*.log`, `logs/probe_sft_final.log`.

**E-46 · 09-10 20:38-~20:52 · reasoner. `reasoner_sft_uniform` (uniform sampling, 45,000 samples: fwd 66%, right 14%, left 16%, stop 5%)**
- Result: Aborted after step 200 (loss 0.1916). No checkpoints.
- Source: `logs/train_reasoner_uniform.log`, `checkpoints/reasoner_sft_uniform/config.json`.

**E-47 · 09-10 20:53 to 09-11 02:38 · reasoner. SFT v3 (`reasoner_sft_v3`)**
- Setup: Uniform sampling (45,000 samples: fwd 66%, right 14%, left 15%, stop 5%). LoRA on LM q/k/v/o plus the projector (6,897,664 trainable). 120 source frames (8 s) sampled at 1 FPS, giving 8 frames. lr 1e-4, accum 16, 0.46 s per sample.
- Training: loss 0.5810 (step 20) → 0.1777 (step 2800). Checkpoints every 250 steps.
- Source: `logs/train_reasoner_v3.log`, `checkpoints/reasoner_sft_v3/config.json`, `src/nav/cosmos_prompts.py:25-34`.

**E-48 · 09-11 02:44-02:58 · reasoner. v3 evals (`eval_v3.sh`)**
- Closed loop, direct, no think, n=40, **diagnostic**, stochastic decoding (top_p 0.8, temp 0.7), single run:
  - step1500: **SR 0.4750 / SPL 0.4745 / OS 0.4750 / nDTW 0.3888 / NE 6.4085**. within_radius 19, timeout 16, model_stop 5. "where it STOPPED within 3m 0.0%". Commands: fwd 55%, stop 1%.
  - final: **SR 0.3250 / SPL 0.3249 / OS 0.3250 / nDTW 0.3044 / NE 7.6240**. model_stop 3; stop commands 0%.
- Probe on v3 final (context 120 frames): **0.631** (stop recall 0.10) against majority 0.544.
- Conclusion (README): the early checkpoint beats final, consistent with over-fitting; n=40.
- Source: `results/runs/v3_*_direct.json`, `logs/eval_v3_*.log`, `results/probe_v3_final.*`.

---

## 3. Consolidated navigation evaluations (R2R-CE val_unseen)

"Mode" is D (diagnostic / proximity break, so SR equals OS) or S (strict, own STOP). "ms" is the model_stop percentage.

### 3a. LatentPilot (Stage 0 / 0' / 1)

| Date | Checkpoint | n | Mode | SR | SPL | OS | nDTW | NE | ms | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 09-01 | stage0/final | 12 (1 scan) | D | not recorded | not recorded | not recorded | not recorded | not recorded | 0/12 | D15 |
| 09-02 | stage0prime_A/final | 150 | D | 0.0000 | not recorded | 0.0000 | 0.244 | 8.79 | 39.3% | D19 (no log) |
| 09-02 | stage0prime_B/final | 150 | D | 0.1667 | not recorded | 0.1667 | 0.303 | 7.95 | 30.0% | D19 (no log) |
| 09-02 | Stage 0 (memoryless) | not recorded | D | 0.1667 | not recorded | 0.1667 | not recorded | not recorded | 0% | D19 |
| 09-02 | A × {z, z_scaled, vbar_current, constant} | 60 each | D | 0.0000 (all) | | | | | | D23 (no log) |
| 09-02 | B × {z, z_scaled, vbar_current, constant} | 60 each | D | 0.1833 / 0.2167 / 0.0000 / 0.0000 | | | | | | D23 (no log) |
| 09-02 11:57 | stage1_learned/step10000 (run #1, since overwritten) | 30 | D | 0.2667 | 0.2667 | 0.2667 | 0.3751 | 7.8896 | 0% | eval_step10000_115728.log |
| 09-02 14:31 | stage1_learned/final | 30 | D | 0.1000 | 0.1000 | 0.1000 | 0.3059 | 8.5797 | 0% | eval_final_143122.log |
| 09-02 16:52 | stage1_identity/final | 30 | D | 0.4000 | 0.4000 | 0.4000 | 0.4042 | 7.3954 | 0% | eval_identity_165233.log |
| 09-02 18:18 | stage1_balanced/step8000 | 30 | D | 0.1333 | 0.1333 | 0.1333 | 0.2804 | 8.6863 | 0% | eval_balanced_step8000_181830.log |
| 09-02 18:46 | stage1_identity/final | 150 | D | 0.1733 | 0.1720 | 0.1733 | 0.2680 | 8.1705 | 0.0% | evaln150_stage1_identity_final.log |
| 09-02 18:56 | stage1_learned/final | 150 | D | 0.1067 | 0.1067 | 0.1067 | 0.2768 | 8.4955 | 4.7% | evaln150_stage1_learned_final.log |
| 09-02 19:05 | stage1_balanced/step10000 | 150 | D | 0.0667 | 0.0667 | 0.0667 | 0.2751 | 8.3559 | 18.0% | evaln150_stage1_balanced_step10000.log |

### 3b. Pointing policy

| Date | Checkpoint (history / head) | n | Mode | stop thr | turn | SR | SPL | OS | nDTW | NE | ms | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09-02 21:39 | pointing/step10000 (none / final layer) | 60 | D | 0.50 | 7.5° | 0.2667 | 0.2653 | 0.2667 | 0.3220 | 8.6583 | 0% | evalpt_213955 |
| 09-02 21:50 | pointing/step10000 | 60 | D | 0.10 | 7.5° | 0.1833 | 0.1833 | 0.1833 | 0.3123 | 8.8767 | 16.7% | stopsweep_215032 |
| " | " | 60 | D | 0.05 | 7.5° | 0.1500 | 0.1500 | 0.1500 | 0.3091 | 8.8979 | 26.7% | " |
| " | " | 60 | D | 0.03 | 7.5° | 0.1000 | 0.1000 | 0.1000 | 0.3040 | 9.0568 | 40.0% | " |
| 09-02 22:05 | pointing/step10000 | 60 | S | 0.10 | 7.5° | 0.1000 | 0.0889 | 0.1833 | 0.2373 | 9.2059 | 20.0% | strictsweep_220514 |
| " | " | 60 | S | 0.05 | 7.5° | 0.1333 | 0.1204 | 0.1500 | 0.2889 | 8.9577 | 36.7% | " |
| " | " | 60 | S | 0.03 | 7.5° | 0.1000 | 0.0967 | 0.1000 | 0.3041 | 8.9228 | 48.3% | " |
| 09-02 22:47 | pointing/step16000 | 60 | S | 0.05 | 7.5° | 0.2167 | 0.2167 | 0.2167 | 0.3997 | 7.7324 | 60.0% | eval16k_224747 |
| " | " | 60 | D | 0.05 | 7.5° | 0.2167 | 0.2167 | 0.2167 | 0.3862 | 7.9597 | 38.3% | " |
| 09-02 23:23 | pointing/final (= step20000) | 150 | S | 0.05 | 7.5° | 0.1333 | 0.1304 | 0.1400 | 0.3499 | 7.3382 | 57.3% | evalfinal150_232320 |
| " | " | 150 | D | 0.05 | 7.5° | 0.1400 | 0.1400 | 0.1400 | 0.3523 | 7.4026 | 44.7% | " |
| 09-03 ~12:0x | pointing_hist/final (K=2 stride 1) | 150 | S | 0.05 | 7.5° | 0.1800 | 0.1680 | 0.2267 | 0.3937 | 6.5540 | 68.0% | evalsord_115012 |
| " | " | 150 | D | 0.05 | 7.5° | 0.2267 | 0.2212 | 0.2267 | 0.4088 | 6.6469 | 46.7% | " |
| 09-03 ~12:2x | pointing_6scan/final (none) | 150 | S | 0.05 | 7.5° | 0.0400 | 0.0372 | 0.0667 | 0.2376 | 8.3687 | 23.3% | " |
| " | " | 150 | D | 0.05 | 7.5° | 0.0667 | 0.0667 | 0.0667 | 0.2527 | 8.3619 | 20.7% | " |
| 09-03 19:35 | pointing_full3/step20000 (K=2 stride 8) | 150 | S | 0.05 | 7.5° | 0.0867 | 0.0853 | 0.1000 | 0.3320 | 7.7184 | 86.7% | evalck20k_193554 |
| " | " | 150 | D | 0.05 | 7.5° | 0.1000 | 0.0961 | 0.1000 | 0.3307 | 7.7247 | 76.7% | " |
| 09-03 22:07 | "step20000" (unnamed; likely full3) | 100 | S | 0.15 | 7.5° | 0.1700 | not printed | 0.1900 | not printed | 8.5131 | 54.0% | thrsweep_220712 |
| " | " | 100 | S | 0.30 | 7.5° | 0.1500 | not printed | 0.2300 | not printed | 9.0805 | 29.0% | " |
| " | " | 100 | S | 0.50 | 7.5° | 0.1400 | not printed | 0.3200 | not printed | 9.3721 | 3.0% | " |
| 09-04 00:20 | pointing_full3/step40000 | 150 | S | 0.10 | 7.5° | 0.2467 | 0.2229 | 0.3133 | 0.3454 | 7.0973 | 62.7% | sweep40k_002048 |
| " | " | 150 | S | 0.20 | 7.5° | 0.1800 | 0.1344 | 0.4133 | 0.2017 | 7.9165 | 17.3% | " |
| " | " | 150 | S | 0.35 | 7.5° | 0.1067 | 0.0581 | 0.4333 | 0.1365 | 8.1312 | 0.0% | " |
| 09-04 11:17 | pointing_full3/step100000 | 150 | S | 0.05 | 7.5° | 0.1600 | 0.1532 | 0.1933 | 0.4368 | 7.0065 | 91.3% | sweep100k_111730 |
| " | " | 150 | S | 0.10 | 7.5° | 0.2267 | 0.2059 | 0.3067 | 0.3947 | 7.0768 | 72.0% | " |
| " | " | 150 | S | 0.20 | 7.5° | 0.2133 | 0.1863 | 0.4333 | 0.2646 | 7.8095 | 30.7% | " |
| 09-04 23:47 | pointing_fusion/step20000 (23, 26, 28) | 150 | S | 0.05 | 7.5° | 0.0400 | 0.0400 | 0.0600 | 0.3280 | 7.8952 | 90.0% | fusesweep_234742 |
| " | " | 150 | S | 0.10 | 7.5° | 0.0867 | 0.0728 | 0.1000 | 0.3146 | 7.9190 | 76.0% | " |
| " | " | 150 | S | 0.20 | 7.5° | 0.1533 | 0.1170 | 0.1933 | 0.2862 | 7.7728 | 56.7% | " |
| 09-05 00:49 | pointing_fusion/step20000 | 100 | S | 0.35 | 7.5° | 0.1900 | not printed | 0.2800 | not printed | 8.5861 | 34.0% | distdiag_004917 |
| 09-05 10:38 | **pointing_full3/step20000** | 100 | S | 0.35 | 7.5° | **0.1500** | **0.1307** | **0.2800** | **0.2085** | **9.2117** | 20.0% | baseline035_103840 (the README row) |
| 09-05 10:40 | pointing_fusion/step60000 | 150 | S | 0.20 | 7.5° | 0.3000 | 0.2427 | 0.4800 | 0.2781 | 7.1283 | 27.3% | sweep60k_104019 |
| " | " | 150 | S | 0.35 | 7.5° | 0.2467 | 0.1734 | 0.5000 | 0.1957 | 7.4983 | 4.0% | " |
| " | " | 150 | S | 0.50 | 7.5° | 0.2333 | 0.1517 | 0.5000 | 0.1775 | 7.5995 | 0.7% | " |
| 09-05 12:56 | pointing_fusion/step80000 | 150 | S | 0.20 | 7.5° | 0.2267 | 0.1872 | 0.4133 | 0.2484 | 7.8985 | 24.0% | test80k_125633 |
| " | " | 150 | S | 0.15 | 7.5° | 0.2067 | 0.1697 | 0.3733 | 0.2694 | 7.8855 | 30.7% | " |
| " | " | 150 | S | 0.30 | 7.5° | 0.1933 | 0.1412 | 0.4533 | 0.1784 | 8.3244 | 4.7% | " |
| " | pointing_full3/step60000 (matched baseline) | 150 | S | 0.20 | 7.5° | 0.2200 | 0.1891 | 0.4133 | 0.2669 | 7.9937 | 31.3% | " |
| 09-05 18:52 | pointing_fusion/step100000 | 150 | S | 0.20 | 7.5° | 0.1867 | 0.1612 | 0.4000 | 0.2576 | 8.4483 | 30.7% | eval100k_185222 |
| " | " | 150 | S | 0.30 | 7.5° | 0.1667 | 0.1340 | 0.4400 | 0.1888 | 8.8546 | 12.7% | " |
| " | " | 150 | S | 0.15 | 7.5° | 0.2000 | 0.1805 | 0.3733 | 0.2952 | 8.2723 | 44.7% | " |
| 09-05 22:00 | pointing_fusion/step120000 | 150 | S | 0.15 | 7.5° | 0.2933 | 0.2444 | 0.4000 | 0.3505 | 7.0527 | 54.7% | eval120k_220002 |
| " | " | 150 | S | **0.20** | 7.5° | **0.3133** | **0.2537** | **0.4267** | 0.3232 | 7.2189 | 46.0% | " |
| " | " | 150 | S | 0.10 | 7.5° | 0.2867 | 0.2425 | 0.3667 | 0.3631 | 7.0260 | 61.3% | " |
| " | " | 150 | S | 0.30 | 7.5° | 0.2933 | 0.2359 | 0.4467 | 0.2867 | 7.5894 | 35.3% | " |
| 09-09 14:10-15:00 | pointing_fusion/step120000 | 150 | D | 0.20 | 3.0° / 4.0° / 5.0° / **7.5°** / 10.0° | 0.3533 / 0.4133 / 0.4200 / **0.4267** / 0.4000 | 0.3471 / 0.3974 / 0.4040 / 0.4074 / 0.3734 | = SR | 0.4072 / 0.4285 / 0.4099 / 0.3972 / 0.3672 | 6.8954 / 6.7753 / 6.6091 / 7.0897 / 6.6943 | 22.7 / 23.3 / 18.7 / 20.7 / 21.3% | turnsweep_0909_141023/ |

All thresholds in this table were chosen on val_unseen itself, which is test-set tuning (acknowledged in `scripts/collect_valseen.sh`).

### 3c. Cosmos3-Edge (action / diffusion) and Cosmos reasoner

All rows below are diagnostic (D), so SR equals OS.

| Date | Policy / checkpoint | n | SR | SPL | OS | nDTW | NE | ms | Source |
|---|---|---|---|---|---|---|---|---|---|
| 09-09 22:54 | Edge zero-shot, smoke | 4 | 0.0000 | 0.0000 | 0.0000 | 0.2976 | 7.8860 | 50% | eval_edge_smoke.log |
| 09-09 23:30 | Edge zero-shot, R2R instruction (g7.5, chunk 24) | 40 | 0.0750 | 0.0750 | 0.0750 | 0.2756 | 8.8134 | 75% | eval_edge_zeroshot.log |
| 09-10 00:05 | Edge "native" (identical to the row above) | 40 | 0.0750 | 0.0750 | 0.0750 | 0.2756 | 8.8134 | 75% | eval_edge_native.log |
| 09-10 00:22 | Edge fixed prompt "The camera moves forward." | 20 | 0.1500 | 0.1500 | 0.1500 | 0.3300 | 8.4493 | 85% | eval_edge_fwdonly.log |
| 09-10 01:06 | Edge r2r_raw (partial) | 15/40 | 2/15 within_radius | | | | | 10/15 | ablation_0910_005453 |
| 09-10 12:28 | Reasoner zero-shot, think, direct burst (partial) | 34/40 | 4/34 within_radius | | | | | 12/34 | results/salvaged/reasoner_direct_partial34.log |
| 09-10 15:42 | Reasoner → Edge diffusion "grounded" (partial) | 18/40 | 5/18 within_radius | | | | | 4/18 | results/salvaged/reasoner_diffusion_grounded_partial.log |
| 09-10 20:06 | reasoner_sft/step1250, direct, no think | 40 | 0.0000 | 0.0000 | 0.0000 | 0.2796 | 8.6351 | 65% | results/runs/sft_step1250_direct.json |
| 09-10 20:03 | reasoner_sft/final | 40 | 0.1750 | 0.1750 | 0.1750 | 0.3653 | 7.4113 | 65% | results/runs/sft_final_direct.json |
| 09-11 02:51 | reasoner_sft_v3/final | 40 | 0.3250 | 0.3249 | 0.3250 | 0.3044 | 7.6240 | 7.5% | results/runs/v3_final_direct.json |
| 09-11 02:57 | **reasoner_sft_v3/step1500** | 40 | **0.4750** | **0.4745** | **0.4750** | **0.3888** | **6.4085** | 12.5% | results/runs/v3_step1500_direct.json |

Reference numbers: expert on val_unseen gives SR 1.0000, SPL 0.9999, nDTW 0.7529, NE 2.87 (E-05). The paper's NaN row is SR 51.7 / SPL 47.1 / NE 5.3 / OS 57.0.

---

## 4. Probes (offline accuracy vs baseline)

| id | Probe | Model | n | Metric | Result | Baseline | Source |
|---|---|---|---|---|---|---|---|
| D5 | Stage -1 forward retrieval v̄_t → v̄_{t+2} | Frozen encoder | 3 clips, 50-frame pool | × chance | fps 4: 2.0 / 2.9 / 8.3×; fps 2: ≈1.0× | 1.0× | DECISIONS.md:66-85 |
| D10 | Zero-shot instruction following, 5 layouts | Cosmos-Reason2-2B, untrained | 160 trials | Accuracy | F 77.5%, C 68.8%, D 41.2%, E 33.1%, A 21.2% | 25% chance | DECISIONS.md:187-196 |
| D17 | Zero-shot slot structure | Untrained pilot | 160 | Accuracy | None 77.5%, A 59.4%, B 29.4% | 25% | DECISIONS.md:422-430 |
| E-07 | 150-step probe run | Stage 0 | training batches | Loss / acc / STOP recall | 1.02 / 0.58 / 0.000 | Marginal 1.029 / majority 0.596 | HANDOFF.md:224-227 |
| D27 | Label-vs-expert agreement | Labels only | not recorded | Agreement | 0.689 → 0.927 → 0.972 | | DECISIONS.md:796-800 |
| E-25 | Instruction dependence (teacher-forced) | pointing_full3/step120000 | 900 steps | act agree | real 0.764; swapped 0.686; empty 0.603 | | instrdep_151148.log |
| E-26 | Per-layer linear probe | pointing_full3/step120000 | 2,000 train / 1,000 test | u corr / act / stop AUC | Layer 28: 0.317 / 0.407 / 0.441; best stop layer 15: 0.718 | Stop AUC 0.5 chance; stop base rate 0.027 (test) | layerprobe_154820.log |
| E-28 | Failure analysis (teacher-forced) | pointing_fusion/step120000 | 2,500 steps (1 scan) | act agree | 0.771 (FWD 0.901, LEFT 0.553, RIGHT 0.557) | not recorded | failanalysis_135616.log |
| E-29 | Bearing calibration | pointing_fusion/step120000 | 3,000 steps / 3 scans | Slope; accuracy at 7.5° | 0.424×; 0.733 | Calibrated slope 1.0 | bearingcal_140140.log |
| E-32 | Edge inverse dynamics, av | Edge zero-shot | 384 frames / 11 scans | yaw corr; turn direction | +0.837; 0.964 (n=192) | 0.5 turn-direction chance | edge_zs_multiscan.log |
| E-32 | Edge inverse dynamics, camera_pose | Edge zero-shot | 224 frames / 1 scan | yaw corr; turn direction | +0.288; 0.789 (n=133) | | edge_zs_dedup.log |
| E-33 | Edge policy, g1.0 | Edge zero-shot | 384 frames | Turn direction | Correct 0.594 vs swapped 0.531 | 0.5 | edge_policy_zs.log |
| E-35 | Edge scale probe (resampled) | Edge zero-shot | 320 frames | yaw corr; s | +0.987; s = 7.47 | | edge_scale.log |
| E-37 | Short-command controllability | Edge zero-shot | 22 frames | Sign correct | g1: 82%, g3: 86%, g5: 95%, g7.5: 95% | 50% | edge_ctrl_cfg.log |
| E-38 | Full instruction at CFG | Edge zero-shot | n=212 | Turn direction | g5: 0.651 vs 0.453 swapped; g7.5: 0.675 vs 0.495 | 0.5 | edge_policy_cfg.log |
| E-42 | Reasoner nav probe v1 | Reasoner zero-shot | 36 points / 12 trajectories | Accuracy | hier_full 0.389; hier_overall 0.417; nothink 0.167 | Majority 0.500 | results/reasoner_nav_probe.md |
| E-43 | Reasoner nav probe v2 | Reasoner zero-shot | 96 / 24 | Accuracy | hier_overall 0.375; hier_full 0.385 | Majority 0.573 | results/reasoner_nav_probe_v2.md |
| E-45 | Next-action probe, new labels | Reasoner zero-shot | 160 / 40 | Accuracy | nothink 0.325; think 0.331 | Majority 0.544 | results/probe_zeroshot_newlabels.md |
| E-45 | Next-action probe | reasoner_sft/step1250 (16-frame context) | 160 / 40 | Accuracy | **0.656** | 0.544 | results/probe_sft_step1250.md |
| E-45 | Next-action probe | reasoner_sft/final (16-frame context) | 160 / 40 | Accuracy | **0.675** | 0.544 | results/probe_sft_final.md |
| E-48 | Next-action probe | reasoner_sft_v3/final (120-frame context) | 160 / 40 | Accuracy | **0.631** (stop recall 0.10) | 0.544 | results/probe_v3_final.md |

Expert label distribution in the 160-point probes: move forward 87, stop 40, turn left 17, turn right 16 (`logs/probe_v3_final.log`).

The key takeaway is that offline probe accuracy did **not** predict closed-loop performance. v3 final (probe 0.631) beats SFT final (0.675) in closed loop, 0.325 vs 0.175. SFT step1250 (probe 0.656) scored SR 0.000.

---

## 5. Ablations

| Ablation | What varied | Held fixed | Result | Source |
|---|---|---|---|---|
| Sequence layout (D10) | Order × markers × chat template | Frames, probes, mRoPE | F 77.5% vs literal Eq. 5 21.2% (160 trials) | DECISIONS.md:180-209 |
| mRoPE vs sequential (D6) | position_ids | Sequence | Hidden rel-L2 0.42-0.50; top-1 agreement 2/5 | DECISIONS.md:88-113 |
| Pilot slot design, zero-shot (D17) | none / A / B | Untrained | 77.5 / 59.4 / 29.4% | DECISIONS.md:422-430 |
| Pilot slot design, closed loop (D19) | A vs B (trained) | Data, seed, 17,400 steps | A 0.0000 vs B 0.1667 (n=150, D) | DECISIONS.md:454-491 |
| Slot content (D23) | {z, z_scaled, vbar_current, constant} × {A, B} | Checkpoints | B: 0.1833 / 0.2167 / 0 / 0; A: all 0 (n=60) | DECISIONS.md:663-684 |
| G_psi init (D20) | Default vs zero-weight / mean-bias | Batch | Ratio 167.59× vs 0.42× | DECISIONS.md:584-609 |
| Gate (b): learned vs identity G_psi (D25) | G_psi trainable vs frozen identity | All else | n=30: 0.10 vs 0.40; n=150: 0.1067 vs 0.1733 | E-12, E-14 |
| λ schedule (D26) | fixed 0.1 vs balanced ratio 0.9 (λ=0.02 arm never run) | 6 scans, seed, schedule | Balanced step10000 n=150: 0.0667 (ms 18%) vs learned final 0.1067 | E-13 |
| Pointing STOP threshold | 0.03-0.50 | Checkpoint | Lower threshold → more self-stops, lower OS; strict SR peaks around 0.10-0.20 for later checkpoints | Section 3b |
| Diagnostic vs strict | Proximity break on/off | Checkpoint, threshold | e.g. pointing/final 0.1400 D vs 0.1333 S; pointing_hist 0.2267 vs 0.1800 | Section 3b |
| Data scale (pointing) | 6 scans (1,665 eps) vs 16 scans (4,203 eps) | 20k steps × batch 8, seed, LR | Strict SR 0.0400 vs 0.1333 (n=150, thr 0.05) | E-19 |
| Frame history (D28) | K=2 stride 1, 61 scans (0.35 ep) vs memoryless, 16 scans | Not a clean ablation (data also changed) | Strict SR 0.1800 vs 0.1333 | E-21 |
| Multi-layer fusion (D29) | Head reads 23/26/28 vs final layer | K=2 stride 8, 61 scans | Matched: fusion step80000 0.2267 vs full3 step60000 0.2200 (thr 0.20, S, n=150); best fusion 0.3133 @120k vs best full3 0.2467 @40k (different thresholds) | E-27, test80k_125633.log |
| Checkpoint over training (fusion, thr 0.20, S, n=150) | Step 20k / 60k / 80k / 100k / 120k | | 0.1533 / 0.3000 / 0.2267 / 0.1867 / 0.3133 (non-monotonic) | Section 3b |
| Turn decision threshold | 3 / 4 / 5 / 7.5 / 10° | fusion step120000, D, thr 0.2 | 0.3533 / 0.4133 / 0.4200 / 0.4267 / 0.4000 | E-30 |
| Instruction condition | real / swapped / empty | full3 step120000 | act 0.764 / 0.686 / 0.603 | E-25 |
| Edge domain | av vs camera_pose | Zero-shot inverse dynamics | yaw corr 0.815 vs 0.288 (224 frames) | E-32 |
| Edge guidance | 1.0 / 3.0 / 5.0 / 7.5 | Short commands | Sign correct 82 / 86 / 95 / 95% | E-37 |
| Edge instruction vs fixed "forward" | Prompt | g7.5, chunk 24 | 0.0750 (n=40) vs 0.1500 (n=20); not paired | E-39 |
| Reasoner think vs no think (zero-shot) | Reasoning on/off | Probe | v1: 0.417 vs 0.167; new labels: 0.331 vs 0.325 | E-42, E-45 |
| Reasoner prompt template | hier_full vs hier_overall | Probe | v1: 0.389 vs 0.417; v2: 0.385 vs 0.375 | E-42, E-43 |
| SFT sampling | Onset-balanced (v1) vs uniform (v3; v3 also changed context 16 → 120 frames and added projector LoRA) | lr, rank, 1 epoch | Closed loop final 0.1750 vs 0.3250; confounded (3 changes at once) | E-44 to E-48 |
| SFT checkpoint | step1250 vs final (v1); step1500 vs final (v3) | | v1: 0.0000 vs 0.1750; v3: 0.4750 vs 0.3250 | E-45, E-48 |
| Prompt / policy arms (`run_ablations.py`) | 8 arms | n=40 planned | Mostly incomplete (OOM or killed); see E-41 | E-41 |

---

## 6. Training runs in `checkpoints/`

| Dir | Base model | Data | Steps / batch | Saved checkpoints | Purpose | Log / config |
|---|---|---|---|---|---|---|
| `stage0/` | Cosmos-Reason2-2B + LoRA r16 | 6 scans / 1,665 eps / 69,606 steps | 8,000 × 8 | step500…step8000, final (PEFT adapter_model.safetensors) | Memoryless Stage 0 (failed, D15) | trainlog.json only |
| `stage0prime_A/` | same + Pilot (Design A) | same | 17,400 × 8 (2 ep) | final only (full model.safetensors, the D24 bug) | NaN-row reproduction, pilot last | logs/ab_overnight.log; trainlog.json |
| `stage0prime_B/` | same + Pilot (Design B, action_query) | same | 17,400 × 8 | final only | same, with action query | logs/ab_overnight.log; trainlog.json |
| `stage1_learned/` | Cosmos-Reason2-2B + LoRA + Pilot B + learned G_psi, λ=0.1 | 6 scans | 17,400 × 8 | step2000…16000, final (adapter.pt + pilot.pt). Step2000-10000 were overwritten by run #2 | Stage 1 claim | stage1_20260902_120059.log; trainlog.json |
| `stage1_identity/` | same, G_psi frozen to identity | 6 scans | 17,400 × 8 | step2000…16000, final | Gate (b) control | same log; trainlog.json |
| `stage1_balanced/` | same, adaptive λ (ratio 0.9) | 6 scans | Planned 17,400 × 8; killed about 10,750 | step2000…10000 | D26 λ-drift fix | lambdafix_20260902_170524.log (no trainlog.json) |
| `stage2/` | (empty) | | | none | Stage 2 scheduled sampling, never run | |
| `pointing/` | Cosmos-Reason2-2B + LoRA r16 + pointing head (12,294 params) | 16 scans / 4,203 eps / 171,281 steps | 20,000 × 8 (0.93 ep) | step2000…20000, final | D27 pointing | pointing_20260902_201418.log; trainlog.json |
| `pointing_6scan/` | same | 6 scans / 1,665 eps | 20,000 × 8 (2.30 ep) | step2000…20000, final | Data-scale ablation | dataablation_223555.log; trainlog.json |
| `pointing_hist/` | same + frame history K=2, **stride 1** | 61 scans / 10,819 eps / 435,468 steps | 37,800 × 4 (0.35 ep) | step2000…36000, final | D28 history | overnight_20260903_005307.log; trainlog.json; config.json |
| `pointing_full3/` | same, K=2 stride 8 | 61 scans | Planned 326,601 × 4 (3 ep); killed about 131,200 | step20000…120000 | Long run | full3_20260903_124907.log; config.json per step (no trainlog.json) |
| `pointing_fusion/` | same, head reads layers 23/26/28 (49,158 params) | 61 scans | Planned 163,300 × 4 (1.5 ep); killed about 124,000 | step20000…120000 | D29 fusion | fusion_20260904_195116.log; config.json per step |
| `edge_r2r/` | nvidia Cosmos3-Edge (diffusion/action), LoRA r16 + av action rows (13.05 M trainable) | 10,816 samples (resampled video) | Planned 2,000 (accum 8); 25 logged | none (config.json only) | Edge flow-matching post-train, abandoned | train_edge.log |
| `reasoner_sft/` | Cosmos3-Edge reasoner tower, LoRA r16 on LM q/k/v/o (6,422,528) | 43,036 samples (onset-balanced; fwd 50 / R 12 / L 13 / stop 25%), 16-frame context | 1 epoch (~2,690 optimizer steps, accum 16) | step250…2500, final | Reasoner SFT v1 | train_reasoner_sft.log; config.json; index_train.json |
| `reasoner_sft_uniform/` | same | 45,000 samples uniform (66 / 14 / 16 / 5%) | Aborted at step 200 | none | Uniform-sampling v2 | train_reasoner_uniform.log; config.json |
| `reasoner_sft_v3/` | same + projector LoRA (6,897,664) | 45,000 uniform (66 / 14 / 15 / 5%), 120 source frames → 8 sampled at 1 FPS | 1 epoch (~2,810 steps, accum 16) | step250…2750, final | Reasoner v3 (best closed loop at step1500) | train_reasoner_v3.log; config.json |

The base Cosmos3-Edge weights sit outside `checkpoints/` subfolders (`checkpoints/Cosmos3-Edge`, per edge_download.log; deleted since, re-downloadable).

---

## 7. Figures plottable from existing logs

| Figure | File(s) | Fields / how to parse |
|---|---|---|
| Stage 0 training curve | `checkpoints/stage0/trainlog.json` | List of {step, loss, acc, stop_recall}; every 50 steps, 160 points. Overlay prior lines: marginal 0.9744, majority 0.639 (HANDOFF.md:213-217) |
| Stage 0' A vs B curves | `checkpoints/stage0prime_{A,B}/trainlog.json` | {step, loss, acc, stop_recall}; 348 points each |
| **Stage 1 loss-balance drift (D25 figure)** | `checkpoints/stage1_{learned,identity}/trainlog.json` | {step, loss, l_act, l_pil, acc, stop_recall, z_norm}. Plot 0.1·l_pil/l_act vs step; L_act and L_pil (log y); z_norm |
| Adaptive-λ run | `logs/lambdafix_20260902_170524.log` | Lines `step N  L x  L_act x  L_pil x  acc x  STOP-rec x(n=k)  \|z\| x  lam x ratio x  lr x`; steps 50-10,750 |
| Pointing training curves | `checkpoints/{pointing,pointing_6scan,pointing_hist}/trainlog.json` | {step, total, point, disp, theta (always 0), stop, visible, act}; 400 / 400 / 756 points |
| Pointing long runs | `logs/full3_20260903_124907.log`, `logs/fusion_20260904_195116.log` | Lines every 200 steps: `step N L x point x disp x theta x stop x act-agree x p(stop) x lr x it/s` |
| Reasoner SFT loss | `logs/train_reasoner_sft.log`, `logs/train_reasoner_v3.log` (`train_reasoner_uniform.log` for 200 steps) | Convert `\r` to newlines, then lines `step N samples a/b loss x s/sample`; every 20 steps (134 and 140 points) |
| Edge train | `logs/train_edge.log` | One point only; not plottable |
| Closed-loop SR/OS vs STOP threshold (per checkpoint) | `logs/stopsweep_215032.log`, `strictsweep_220514.log`, `sweep40k_002048.log`, `sweep100k_111730.log`, `fusesweep_234742.log`, `sweep60k_104019.log`, `test80k_125633.log`, `eval100k_185222.log`, `eval120k_220002.log` | Blocks headed `######## <ckpt> strict thr=X n=N` followed by `SR = / SPL = / OS = / nDTW = / NE =`, model_stop, timeout, p(stop) stats |
| Strict SR vs training step (fusion, thr 0.20) | fusesweep, sweep60k, test80k, eval100k, eval120k | Steps 20k / 60k / 80k / 100k / 120k → 0.1533 / 0.3000 / 0.2267 / 0.1867 / 0.3133 |
| Distance-to-goal summaries | distdiag, baseline035, sweep60k, test80k, eval100k, eval120k, turnsweep logs | `closest approach mean/median/p25/min`, `ever within 3/5/10 m`, `where it STOPPED`. Summary statistics only; no per-episode arrays |
| Turn-threshold sweep | `logs/turnsweep_0909_141023/turn_{3.0,4.0,5.0,7.5,10.0}.log` | Standard eval block |
| Stage 1 comparison bars | `logs/evaln150_*.log`, `eval_*_1xxxxx.log` | Eval blocks |
| Per-layer probe curve | `logs/layerprobe_154820.log` (table: layer, u corr, act, stop AUC, stop sep); `logs/layersearch2_191721.log` (top-12 fusions) | Raw features in `logs/layerfeat_step120000_2000_1000.npz` (Xtr 2000×29×2048, Xte 1000×29×2048, ttr_/tte_ {u, v, dx, dy, stop, vis}) allow re-fitting |
| Bearing shrinkage scatter and threshold sweep | `logs/bearings.npy` (shape 2×3000: row 0 predicted θ, row 1 true θ; from `bearing_calibration.py:76`); `logs/bearingcal_140140.log` | |
| Failure-analysis bars | `logs/failanalysis_135616.log` | Buckets by distance / position / action / length; confusion |
| Instruction dependence bars | `logs/instrdep_151148.log` | 3 conditions × 5 metrics |
| Edge CFG controllability | `logs/edge_ctrl_cfg.log`, `edge_policy_cfg.log` | Per-guidance mean yaw per command, separation, sign % |
| Edge inverse-dynamics per-episode predicted vs true | `logs/edge_zs_multiscan.log` | Per-episode `pred \|trans\|/frame`, `yaw mean/std` vs true; summary corr |
| Reasoner probe confusion matrices | `results/probe_*.json`, `results/reasoner_nav_probe*.json` | `variants[<name>].{accuracy, parse_rate, recall, confusion, mean_seconds}`; `majority_baseline`, `n`; transcripts list {variant, scan, traj, t, label, pred, prompt, answer} |
| Reasoner closed-loop bars and command mix | `results/runs/*.json` | {SR, SPL, OS, nDTW, NE, term_model_stop, term_timeout, term_within_radius, min_dist_mean, within_3m_pct, within_5m_pct, calls_per_episode, commands{…}} |
| Per-episode outcomes (reasoner, Edge) | `logs/eval_*`, ablation logs | Lines `[i/40] scan reason min_d X calls N s/ep` |
| Tables with no raw logs (DECISIONS only) | D5, D6, D10, D17, D19, D23, D27 | Plot from the tables in DECISIONS.md |
| Qualitative | `media/comparisons/*.mp4` (+ index.json), `media/rollouts/*`, `recordings/*` | Videos |

---

## 8. Gaps: things referenced but never run or with no recorded output

* Stage 2 (scheduled sampling): `train_stage2.py` exists, `checkpoints/stage2/` is empty, and there are no logs.
* `stage1_lam002` (λ = 0.02) and `stage1_16scan`: never trained.
* `attention_analysis.py`, `edge_load_test.py`, `verify_flow_convention.py`, `diag_caption_format.py`: no output logs found.
* `run_edge_ablation.sh` arm B (fixed "move forward"): never started. Arm A never finished (`summary.txt` has headers only).
* `results/ablations.md` (from `run_ablations.py`): never created. Arms av_action_only, r2r_av_action and fixed_forward never ran.
* val_seen threshold selection (E-23): data collected, never used.
* The Stage 0' evals behind D19 and D23, and the Stage 0 eval behind D15: no logs in repo.

---

## 9. Contradictions, flags and unverifiable numbers

"README" below means the first public README (commit `9ab89cb`); every item has been corrected in the current README.

1. **The README headline "SR 47.5% / SPL 47.4% vs paper SR 54.0" is oracle-stop success.**
   - Evidence: v3 evals are diagnostic (proximity break; `eval_v3.sh` passes no `--strict`), so SR equals OS. Every success is `within_radius`, and "where it STOPPED within 3m 0.0%".
   - This conflicts with D23's rule "Report as OS, never as SR, when comparing to the paper". The same applies to all reasoner and Edge SRs, the README LatentPilot and pointing-step10000 rows, and the "0.4267 pointing baseline" (a diagnostic turn-sweep number).
   - The best strict (true self-stop) SR anywhere in the repo is **0.3133** (pointing_fusion/step120000, thr 0.20, n=150).
2. **The README pointing table omits the best pointing results.** fusion step120000 (strict 0.3133; diagnostic 0.4267) and pointing_hist (strict 0.1800) are missing. The listed `pointing_full3 step 20,000` row uses n=100, thr 0.35.
3. **The README says the reasoner uses a "16-frame egocentric video context".** Only SFT v1 used 16 frames. v3, the headline model, uses 120 source frames sampled to 8 at 1 FPS (`src/nav/cosmos_prompts.py:25-34`, changed 09-10 20:52).
4. **Reasoner n=40 is small, single-seed and stochastically decoded** (do_sample, top_p 0.8, temp 0.7, `cosmos_navigator.py:93-94`), over 6 scans only. Pointing results use n=150 over 8 scans, so reasoner and pointing numbers are not paired.
5. **"Held-out L_pil 3.02 vs 6.80"** (README, D25) equals the final training-log L_pil (3.018 / 6.799). No held-out L_pil evaluation exists.
6. **"model_stop 0-4.7% in every checkpoint"** (README, D27) is contradicted three ways: stage1_balanced/step10000 had model_stop **18.0%** (n=150); D19 reports own STOP 39.3% (A) and 30.0% (B) for the Stage 0' checkpoints; and D25 gate (a) "SR 0.0" is strict SR, since model_stop was 0 (a diagnostic SR of 0.10 was reported in the same table).
7. **The Stage 1 step10000 checkpoint was overwritten.** The evaluated/recorded step10000 (OS 0.2667, 11:57; recordings 12:05 / 12:23; `media/rollouts/step10000*`) came from an earlier Stage 1 run (~09:50-11:32, no log). Run #2 overwrote its files at 13:24. D25's "OS 0.267 at step10000 → 0.10 at final, same episodes" therefore compares two runs. The step-50 losses of the aborted 11:51 run and the 12:00 run are identical, suggesting determinism, but run #1's code and config are unknown.
8. **pointing_hist was trained with history stride 1, not the intended 8** (`run_overnight_history.sh` sets STRIDE=8; the log and config show stride 1). This matches the "silently trained the wrong stride for 7 hours" note in `run_full3epoch.sh`.
9. **D26, D28 and D29 are referenced in code and scripts but have no entries in DECISIONS.md**, which jumps D25 → D27. D19 also appears before D18 in the file.
10. **The D19 and D23 closed-loop numbers and the D15 Stage 0 eval have no logs in the repo.** The D19 "Stage 0 16.7%" row has no n or other metrics.
11. The p-values ("p=0.333" in D27 and `scripts/archive/run_step1_then_step2.sh`; "p=0.0055" in `run_data_ablation.sh`) have no computation in any source.
12. **D18 vs D21 baselines disagree:** global mean 17.9 vs constant 18.48, and z=v̄_t 8.2 vs persistence 7.82. They were probably measured on different frame sets, but this is not stated. D18 gives ‖v̄‖ ≈ 15.8, while training logs print mean ‖v̄‖ = 14.798 (train split) and eval logs 15.84.
13. **Expert nDTW:** D13 says the expert "now scores nDTW 0.80", but the full-split validation gives 0.7544 (train) and 0.7529 (val_unseen). D14's expert mean NE of 2.91 m vs HANDOFF's 2.88 / 2.87 m.
14. **`eval_edge_native.log` is identical, episode for episode, to `eval_edge_zeroshot.log`** (SR 0.0750). Either a duplicate run or deterministic; its intended difference ("native") is undocumented.
15. **`thrsweep_220712.log` does not name its checkpoint** ("step20000"). By timing it is most likely pointing_full3/step20000, and it omits SPL and nDTW.
16. **The turn-sweep `summary.txt` contains no metrics** (grep anchored at `^SR` failed on indented lines). Use the per-threshold logs.
17. The `run_fusion.sh` header quotes probe numbers for layers 23,26,28 ("~0.30 / ~0.55 / ~0.74"), but that exact combination is not printed in the layersearch logs.
18. The `failure_analysis` run covered only one scan (2azQ1b91cZZ), despite its "BY SCAN" slice.
19. Stop thresholds were all tuned on val_unseen (test-set tuning). val_seen was collected for this purpose but never used.
20. `train_stage2.py`'s docstring cites "STOP-recall ~0.5 at step10000 / 1.000 at final". The trainlog single-batch values are 0.615 and 0.923.
21. The v3-vs-v1 SFT comparison is confounded: sampling (onset → uniform), context (16 → 120 frames) and trainable modules (+ projector) all changed at once.
22. HANDOFF.md was last fully updated at Stage 0 / 0' (its own header says Session 3, 08-31, with a 09-01 addendum). It does not describe any later stage.

---

## 10. Follow-up experiments (2026-09-30)

Narrative in [`EXPERIMENTS.md` §7](EXPERIMENTS.md#7-follow-up-experiments-2026-09-30). All closed-loop runs here are
strict unless marked diagnostic; SR = own STOP within 3 m, end-SR = ended within 3 m.

| id | What | Setup | Result | Source |
|---|---|---|---|---|
| E-49 | Pilot-slot shortcut test | 110 val_unseen expert episodes, 4,306 steps, teacher-forced; slot = next frame / own z / current frame / other episode | Stage 1 learned 84.35 / 30.82 / 45.15 / 27.75 %; identity 63.12 / 61.22 / 63.21 / 62.15; Stage 0′ B 82.00 / 56.64 / 44.50 / 27.78; Stage 0 no slot 66.05 | `results/slot_shortcut.json`, `logs/slotshortcut_20260930_092139.log` |
| E-49b | Shortcut over training | Stage 1 learned, steps 2k–17.4k, 5 episodes/scan | own z 48.5 → 31.9 (4k) → 28–31; next frame 78–87 | `results/slot_shortcut_over_training.json` |
| E-50 | Stage 2, p_final 0.5 / 0.75 | 2,600 steps from Stage 1 learned, 6 scans (1,665 eps), lr ×0.1 | slot test own z 33.63 / 33.88, next frame 84.35 / 83.67; closed-loop OS 16.67 / 18.00 (diagnostic, n=150) | `logs/stage2_followup_*.log`, `logs/followups/cl_stage2_p*.log` |
| E-51 | Pointing strict re-evaluation | n=150, stopping positions logged | SR 12.67 (16 scans, τ .05), 17.33 (history, τ .05), 22.67 (full3 40k, τ .10); end-SR/OS/SPL/NE identical to the original runs | `logs/followups/cl_pointing_*.log` |
| E-52 | Threshold on val_seen | fusion 120k, 8 val_seen scans, 159 eps | SR 39.33 / 35.33 / 31.33 / 24.67 at τ .10/.15/.20/.30 → τ = 0.10 → val_unseen SR 23.33 (logs/eval120k_220002.log) | `logs/followups/cl_valseen_thr*.log` |
| E-53 | Reasoner v3 step 1,500 | n=40, 3 strict + 2 extra diagnostic runs | strict SR 2.5 / 2.5 / 5.0, end-SR 17.5 / 20.0 / 17.5, OS 37.5 / 42.5 / 32.5; diagnostic OS 47.5 / 47.5 / 35.0 | `results/runs/v3_step1500_direct*.json`, `logs/followups/cl_reasoner_*.log` |
| E-54 | Stage 2 long, p_final 0.75 | 7,800 steps (3× E-50) from Stage 1 learned, 6 scans (1,665 eps), lr ×0.1 | slot test next frame 84.16 / own z 35.46 / current 55.88 / other 29.63; closed-loop OS 19.33, nDTW 0.2975, NE 7.86, model_stop 0 (diagnostic, n=150; first eval attempt interrupted at 75/150 and rerun) | `logs/followups/stage2_p75_long.log`, `logs/followups/cl_stage2_p75_long.log` |

Gap closed by these runs: the slot-shortcut test, Stage 2, val_seen threshold selection and the reasoner's strict
evaluation, all listed in §8 above as missing.
