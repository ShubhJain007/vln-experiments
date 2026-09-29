# DECISIONS.md — LatentPilot reimplementation

Every choice that departs from the paper, resolves an ambiguity in it, or fixes
a value the paper does not state. One entry per decision.

**Why this file exists:** the paper's code was never released, so the only
defence against silent divergence is a written record of what we chose and why.
When a metric misses its gate, this is the bisect list.

Status key — **APPROVED**: human signed off · **AGENT**: made by the agent under
AGENTS.md ground rules, reversible · **OPEN**: still undecided.

---

## D1 · Backbone: Cosmos-Reason2-2B instead of LLaVA-Video-7B
**Status:** APPROVED (pre-existing, AGENTS.md Sec. 1.4) · **Affects:** everything

Paper uses LLaVA-Video-7B (Sec. 3.2). We use Cosmos-Reason2-2B (post-trained
from Qwen3-VL-2B-Instruct). Its vision encoder is SigLIP-2 — the same encoder
family the paper uses — so the Pilot Token's target space stays consistent.

**Consequence:** 2B vs 7B, so landing below the paper's absolute numbers is
expected. Gates are judged on relative movement, not absolute parity.

---

## D2 · Vision encoder frozen (paper trains it)
**Status:** APPROVED (pre-existing, AGENTS.md Sec. 3.1) · **Affects:** Eq. 4, 15

Eq. 15 minimises over `θ, φ, ψ` — the paper trains `E_φ`. We freeze it for
Stages 0–3.

**Why:** from Stage 1 the same encoder produces the Pilot Token's regression
target `v̄_{t+2}` (Eq. 11/14). A trainable target invites representation
collapse. VRAM is a second, independent reason. Revisit only with an EMA target
encoder.

---

## D3 · Input resolution 448×448 → N_v = 196
**Status:** AGENT (`# NOT IN PAPER`) · **Affects:** Eq. 4, sequence length

The paper never states an input resolution, and `N_v` is backbone- and
resolution-dependent (AGENTS.md Sec. 8 item 4).

`N_v = (448/16)² / 2² = 196`, verified three independent ways in Session 1:
the formula, the `image_token_id` count, and the encoder's output rows.

**Sanity check:** real R2R-CE instructions give sequence lengths of
min 204 / median 228 / max 329 — matching AGENTS.md's ~300-token design point.

---

## D4 · Standard LoRA, not QLoRA
**Status:** AGENT (measured) · **Affects:** all training

Measured VRAM for bf16 load + forward: **4.64 GB allocated** on a 16 GB RTX 4070.
That leaves >11 GB headroom, so 4-bit quantisation is unnecessary at Stage 0.

**Revisit at Stage 5:** RL/GRPO needs a reference policy resident alongside the
training policy, which likely forces QLoRA. AGENTS.md requires confirming before
switching quantisation schemes.

---

## D5 · Stage -1 gate judged on fps≈4, not the literal cosine threshold
**Status:** APPROVED (human, 2026-08-31) · **Affects:** Stage -1 verdict, data cadence

AGENTS.md Sec. 2 sets thresholds (0.90 marginal / 0.98 fail) on
`cos(v̄_{t+1}, v̄_{t+2})`. Those assume a roughly isotropic embedding space.
**This encoder is strongly anisotropic** — measured: solid black vs solid white
scores **0.907**, unrelated random images average **0.752**. A literal 0.90
threshold would classify black-vs-white as "too smooth to learn".

Judged instead on forward-retrieval (`v̄_t → v̄_{t+2}` against a 50-frame pool),
replicated across 3 independent clips:

| fps | forward-retrieval vs chance |
|---|---|
| 4 | 2.0× / 2.9× / 8.3× — above chance in all 3 |
| 2 | ≈1.0× — exactly chance in all 3 |

fps≈4 also matches the paper's action cadence (FWD = 0.25 m at walking pace
≈ 5 fps). **Decision: fps≈4 is the operating point; Stage -1 passes.**

---

## D6 · mRoPE position ids reconstructed explicitly
**Status:** AGENT (measured) · **Affects:** Eq. 6, every forward pass

Qwen3-VL gives image tokens 3D (temporal, height, width) positions, and an image
advances the position counter by only `max(H,W)/merge = 14`, **not** by
`N_v = 196`. transformers only takes that path when `input_ids`,
`mm_token_type_ids` and `image_grid_thw` are all present. Eq. 5 must build from
`inputs_embeds` (PILOT(z) is a continuous latent with no token id), so the model
**silently falls back to sequential positions**.

Measured cost, identical sequence, only `position_ids` differing:

| Quantity | sequential vs mRoPE |
|---|---|
| hidden-state relative L2 | **0.42 – 0.50** |
| top-1 next token agreed | **2 / 5** |

An order of magnitude worse than the bf16 batch noise (D7).

**Decision:** reconstruct mRoPE explicitly; verified **bitwise** against the
model's own `get_rope_index`. `position_mode="sequential"` kept for comparison.

**Reasoning:** use each backbone *as pretrained*. LLaVA-Video at sequential
positions is in-distribution **for LLaVA-Video**; Cosmos at sequential is
out-of-distribution **for Cosmos**. Fidelity means preserving the relationship,
not copying the mechanism.

---

## D7 · `v̄` cached in fp32; `v_t` left in bf16
**Status:** AGENT (measured) · **Affects:** Stage 1 `v̄` caching, Eq. 11/14

In bf16 the same frame encoded at batch size 3 vs 1 differs by **~7% relative
L2**. Verified as precision, not a bug: preprocessing is bitwise identical, and
in fp32 the same comparison agrees to **8e-6**.

| Tensor | Role | Rule |
|---|---|---|
| `v̄` (2048-d) | Pilot **target** (Eq. 14) | compute and store **fp32** — batch-invariant |
| `v_t` (196×2048) | model **input** (Eq. 5) | bf16, any batch size — too large to cache (160 GB) |

**Rejected alternative:** matching cache batch size to training batch size. That
creates a hidden coupling where changing the training batch size silently
invalidates the whole cache with no error.

**Accepted residual:** `v_t` is computed live, so eval (batch 1) differs slightly
from training (batch N). An *input* perturbation of the same order as ambient
bf16 noise, never compared against a cached reference. Assumed benign, **not
measured**. Revisit if Stage 0 metrics look unstable across batch sizes.

---

## D8 · Actions as existing vocabulary tokens via native `W_LM`
**Status:** AGENT (feasibility confirmed) · **Affects:** Eq. 7, 13

Sec. 3.3 says actions are decoded by the backbone's native LM output
projection. AGENTS.md required STOP-and-ask if infeasible — it is feasible, so
no escalation was needed.

All four action words are **single existing tokens**:
`forward`=13435, `left`=2359, `right`=1291, `stop`=9495.

**No new head, no vocabulary resize.** `W_a` **is** `model.lm_head`, which
resolves AGENTS.md Sec. 8 item 5 ("are `W_a` and `W_LM` the same matrix?" — yes).

Softmax is over the **4 action logits**, not the full 151669-token vocabulary.
Confirmed by AGENTS.md's own criterion: uniform loss = **1.3863 = ln(4)**
(full-vocab uniform would be ln(151669) = 11.93).

---

## D9 · Loss reduction: sum within trajectory, mean across batch
**Status:** AGENT (paper is explicit) · **Affects:** Eq. 13, 14, 15

The paper leaves no room here:
- Eq. 13 `L_act = -Σ_{t=1}^{T} …` — sum, explicit range
- Eq. 14 `L_pil = Σ_{t=1}^{T-2} …` — sum, **different** range
- Eq. 15 `E_{τ~D}[…]`, **λ = 0.1** stated

So: **sum over timesteps within a trajectory, mean across trajectories.**

**The bug this avoids:** an earlier `mean` default divided `L_act` by `T` and
`L_pil` by `T−2` — different denominators — silently turning λ into
`0.1·T/(T−2)`. At T=6 that is 0.15, a **50% error in the one hyperparameter the
paper states**.

Changed at Stage 0 while **provably inert** (no `L_pil` yet, so it folds entirely
into the learning rate). Fixing it later would confound the Pilot Token's effect
with a loss-rescaling change and break one-novelty-per-stage.

---

## D10 · Sequence layout: image-first with vision markers
**Status:** APPROVED (human, 2026-08-31) · **Affects:** Eq. 5, every forward pass

**This deviates from an explicit equation, not a silence.** Eq. 5 writes
`u_t = [Tok(x) ; v_t]` — instruction first — and Fig. 2 shows the same order.
We instead use `[<vision_start> ; v_t ; <vision_end> ; Tok(x)]`.

Measured zero-shot instruction-following, 40 unambiguous probes × 4 real frames
= **160 trials**, chance = 25%, mRoPE correct in every arm:

| Layout | Accuracy | L/R sep | P(correct) |
|---|---|---|---|
| A instr-first, no markers *(literal Eq. 5)* | **21.2%** (below chance) | −0.041 | 0.239 |
| D instr-first + markers | 41.2% | +0.087 | 0.411 |
| E image-first, no markers | 33.1% | −0.023 | 0.328 |
| **F image-first + markers (CHOSEN)** | **77.5%** | **+0.407** | **0.716** |
| C full native + chat template | 68.8% | +0.328 | 0.652 |

The factors **interact**: markers alone +20, ordering alone +12, both +56. Half
the format is one the model has never seen. Adding chat scaffolding on top
(F→C) *hurts*, so **no chat template** is used.

**Method note:** a first pass at 16 trials gave E = 87.5% and would have
supported a different conclusion. Expanding to 160 trials corrected it. Small
probe sets are not adequate for this question.

**Caveat:** zero-shot on an untrained model. LoRA may narrow the gap.
`layout="eq5_literal"` is retained so this can be re-measured after training.

**Invariant:** whichever layout is used must be identical in training and eval.

---

## D11 · SPL denominator is the geodesic start-to-goal distance
**Status:** AGENT (standard convention) · **Affects:** SPL, Stage 0 gate reading

`SPL = S · l / max(l, p)` where **`l` is the geodesic START-TO-GOAL distance**,
not the reference path's own length. This is the standard definition (Anderson
et al.), and what habitat and VLN-CE both use via
`episode["info"]["geodesic_distance"]`.

**Consequence for reading the gate.** Measured on all 1839 val_unseen episodes:
the reference path is on median **1.07× longer** than the stored geodesic, so a
PERFECT agent reproducing the reference path exactly scores **SPL ≈ 0.92**, not
1.0. SPL's practical ceiling is therefore ~0.92 × SR, not SR.

**AGENTS.md Sec. 3.4's "SPL ~1.0" is approximate.** It holds when `l` is the
path's own length (a pure arithmetic check, which we also test), not under the
reporting convention.

**Corroboration the paper uses the same convention:** Table 3's NaN row gives
SR 51.7 / SPL 47.1 — a ratio of **0.911**, essentially this ceiling. So our
numbers are directly comparable to the paper's.

Both readings are pinned by tests:
`test_ground_truth_reference_paths_score_near_perfect` (arithmetic, expects 1.0)
and `test_spl_ceiling_under_the_standard_convention_is_below_one` (convention,
expects ~0.92).

---

## D12 · Expert follows the REFERENCE PATH, not the shortest path
**Status:** AGENT (bug caught by nDTW) · **Affects:** all training data

The obvious rollout is `follower.next_action_along(goal)` — but that walks the
geodesic SHORTEST path and ignores `reference_path` entirely. The instruction
describes the HUMAN's route ("exit the bedroom, turn left, past the gray
couch"); the shortest path is a different route. Training on those pairs would
teach the model that the instruction does not constrain the trajectory — i.e.
**to ignore the language.**

Invisible in SR/SPL, which a shortest path maximises by construction. Only
nDTW measures route fidelity, which is why it was the metric that caught it.

The expert now traverses `reference_path` waypoint by waypoint, then the goal,
emitting STOP exactly once at the end (an intermediate STOP would teach the
policy to stop early).

---

## D13 · nDTW resamples both paths to 0.25 m before comparing
**Status:** AGENT (measured) · **Affects:** nDTW only

Ilharco et al.'s formula `exp(-DTW(Q,R) / (|R|·d_th))` assumes Q and R have
COMPARABLE granularity — in discrete R2R both are viewpoint sequences. In
R2R-CE they are not: the agent path is one point per 0.25 m step, while
`reference_path` is nav-graph waypoints ~1.7 m apart — about 5× coarser.

DTW must match all ~31 agent points to ~6 waypoints, so the cost accumulates
~31 terms while the normaliser counts only 6. Measured on a PERFECT expert
rollout passing within 0.2–1.0 m of every waypoint:

| | raw DTW | normaliser | nDTW |
|---|---|---|---|
| literal formula | 18.76 | 6 × 3.0 = 18.0 | **0.19** |
| both paths resampled to 0.25 m | — | — | **0.80** |

0.19 for a faithful trajectory is not a measurement, it is an artefact.

**Fix:** resample BOTH sequences to 0.25 m (the FWD primitive) before the DTW.
Both, not just the reference — resampling one side reintroduces the asymmetry
in reverse and breaks the identity property (a path vs itself must score 1.0).
`resample_spacing=0` recovers the literal formula.

Sanity: expert now scores nDTW 0.80, and the paper reports 0.62–0.68 for
trained models — a coherent ordering.

---

## D14 · Follower goal radius = 3.0 m (the success threshold)
**Status:** AGENT (`# NOT IN PAPER`) · **Affects:** expert demonstrations

`--goal-radius` defaults to the episode's own `goals[0].radius` = 3.0 m, which
is exactly the SR threshold. Self-consistent, but it leaves the policy **no
margin**: the expert stops at a mean NE of 2.91 m, right on the boundary, so a
model that imitates it faithfully but is a few centimetres worse fails.

A tighter radius buys margin at the cost of longer demonstrations and an expert
that goes closer than the task requires. The paper does not say. The
metric-consistent value is the default and the knob is exposed — **if Stage 0
lands just under the SR gate, try this first.**

---

## D15 · BLOCKER — Stage 0 as specified CANNOT reproduce Table 3's NaN row
**Status:** OPEN — needs a human decision · **Affects:** the whole Stage 0 gate

AGENTS.md Sec. 3 says "STAGE 0 — Skeleton, **no Pilot Token**. Goal: reproduce
the paper's NaN ablation row (Table 3)." Those two clauses are incompatible.

The paper's supplementary Sec. A.1 defines the NaN variant verbatim:

> "In the NaN variant, we still insert `<|placeholder|>` to initialize the
> Pilot Token and **still propagate the Pilot Token across steps exactly as
> usual**. The only difference is that we do not apply any dedicated loss on
> the Pilot Token itself. The model is optimized solely with the action
> cross-entropy."

and on the rollout interface:

> "During rollout, we keep the sequential interface identical to the main
> method: the Pilot Token state produced at step t is fed back as the Pilot
> Token input at step t+1."

So Table 3's NaN row is a **RECURRENT** policy — it carries `z_{t-1}` across
steps and only drops `L_pil`. Our Stage 0 is **memoryless**: Eq. 5 built as
`[<vs>; v_t; <ve>; Tok(x)]`, no Pilot slot, nothing crossing step boundaries.
These are different models. SR 51.7 was never reachable by what we built.

### Empirically confirmed, and it is not subtle

Closed-loop eval of the trained Stage 0 checkpoint (12 episodes, zsNo4HB9uLZ):

| | |
|---|---|
| episodes with EXACTLY zero progress | **10 / 12** |
| rollout action mix | LEFT 44.2%, RIGHT 43.0%, **FWD 12.7%** |
| training-data action mix | FWD **61.7%**, LEFT 19.2%, RIGHT 16.6% |
| `model_stop` fired | **0 / 12** |

Several episodes emitted **zero FWD across 60 steps** (ep 13/14/15:
`{LEFT:32, RIGHT:28}`; ep 38/39: `{RIGHT:30, LEFT:30}`) — the agent spins in
place in a LEFT/RIGHT limit cycle. Training accuracy was 78% on expert states,
so this is pure closed-loop covariate shift, not underfitting.

### Why memorylessness explains BOTH failures at once

The full instruction is re-fed every step. With no state, the policy cannot
know *which clause it is currently executing* — step 3 and step 30 look
identical apart from the image. So it cannot:
- commit to a consistent turn direction -> LEFT/RIGHT oscillation
- recognise it has reached the final clause -> STOP never fires

The STOP collapse (D-earlier) is therefore likely a SYMPTOM of this, not an
independent bug. Reweighting the loss to force STOP would paper over it.

### Options (human decides)

1. **Add the Pilot slot with propagation but no `L_pil`** — this is literally
   what the NaN row is, and would make the gate reachable. It is Stage 1's
   machinery (Eq. 5 full, Eq. 8, PilotCache) minus Eq. 11/12/14. Changes what
   "Stage 0" means, so it needs sign-off.
2. **Keep Stage 0 memoryless, retarget its gate.** Accept it as a genuine
   no-recurrence ablation the paper never ran, and stop comparing it to 51.7.
3. **Some other recurrence** (e.g. feeding the previous action) — a deviation
   the paper does not describe.

Option 1 is the only one that reproduces the paper. Do NOT spend further
effort on the STOP-imbalance fixes until this is settled — they address a
symptom of the missing recurrence.

---

## D16 · `z_0` is a learned `nn.Parameter`, not a tokenizer entry
**Status:** APPROVED (human, 2026-09-01) · **Affects:** Eq. 5's Pilot slot

Sec. 3.2 implements the Pilot slot as "a special placeholder token
`<|placeholder|>` whose embedding provides `z_0`", and AGENTS.md Sec. 4.1 says
to add that token. We use a standalone `nn.Parameter(2048)` instead. Measured:

- `<|placeholder|>` is **not** in this tokenizer — it splits into 5 tokens.
- The embedding matrix is **311,164,928** params, **48×** our whole LoRA
  adapter. `z_0` must be LEARNED, so the embedding-row route means marking all
  311M trainable (abandons LoRA-only, blows the VRAM budget) or hand-masking a
  single row's gradient — which fails silently when wrong.
- The tokenizer **never sees the Pilot slot**: Eq. 5's `PILOT(z)` is a
  continuous latent with no token id, which is why sequences are built from
  `inputs_embeds` (D6). A tokenizer entry would be an id nothing produces.

2,048 trainable params, functionally identical. Only the literal "has an
embedding row" phrasing changes; the substance of AGENTS.md's test (`z_0`
finite, non-zero, learned) is kept.

**Initialisation:** scaled to the REAL token embedding norm (~1.44), not the
untrained spare rows' ~0.36. The slot is out-of-distribution regardless;
entering at the scale the backbone expects is D10's reasoning again.

---

## D17 · Pilot slot goes after Tok(x); Design A over Design B
**Status:** APPROVED (human, 2026-09-01) · **Affects:** Eq. 5, 7, 8

### Placement is forced by causality, not chosen
Attention is causal (verified bitwise), so the Pilot position only sees what
precedes it:

| Pilot position | what `h_pil` sees |
|---|---|
| before `Tok(x)` | image only — **not the instruction** |
| after `Tok(x)` | image + full instruction ✓ |

Eq. 8 makes `z_t = G_psi(h_t^pil)` predict `v_bar_{t+2}` — the view two steps
ahead, which depends on where the agent is heading, which depends on the
instruction. A Pilot before `Tok(x)` makes `z_t` structurally incapable of
encoding intent. Pinned by `test_h_pil_can_see_the_instruction`.

### A vs B — measured
A concern was raised that under Design A the FINAL position (which most
determines next-token prediction) is an out-of-distribution continuous vector,
and that Design B — `[… ; PILOT(z) ; ACTION_QUERY]` — would leave something
more familiar there while also separating `h_act` from `h_pil`.

Zero-shot, 40 probes x 4 real frames = 160 trials, chance 25%, mRoPE correct
in every arm, **untrained** pilot (`z_0` random in-distribution, `G_psi`
untrained), so this isolates the STRUCTURAL cost of the slot:

| condition | accuracy | L/R sep | P(correct) | P(STOP) on STOP |
|---|---|---|---|---|
| no pilot (Stage 0) | 77.5% | 0.407 | 0.716 | 0.885 |
| **A** pilot last | **59.4%** | 0.082 | 0.557 | 0.647 |
| **B** pilot + query | **29.4%** | 0.058 | 0.334 | 0.225 |

The concern is REAL — A costs 18 points. But B costs **48** and sits barely
above chance. Two stacked out-of-distribution vectors compound the damage: in A
the action is read from a state that follows the complete instruction; in B it
is read from a learned query with no pretrained meaning, behind another alien
vector.

**Decision: A.** `[<vs>; v_t; <ve>; Tok(x); PILOT(z)]`, so
`h_act == h_pil == H[-1]` — one hidden state, `W_LM` → action and
`G_psi` → `z_t`. Consistent with Eq. 5's "Pilot last", with Sec. 3.3 decoding
via native `W_LM`, and with the pause-token/Coconut lineage.

**Caveat:** both arms are untrained; B's query could learn to be a good readout
and this measurement cannot see that. `pilot_mode="action_query"` is retained
so B can be re-measured after training. At 1,665 episodes, starting 48 points
down is the worse bet.

**Watch item:** A's 18-point drop is a real cost the slot imposes at init.
Stage 0' must recover it — if trained Stage 0' underperforms the memoryless
Stage 0 on action accuracy, this is the first place to look.

---

## D19 · D17 REVERSED — Design B beats A in closed loop
**Status:** MEASURED, supersedes D17's choice · **Affects:** Eq. 5 layout

D17 chose Design A from a ZERO-SHOT probe (A 59.4% vs B 29.4% on 160 trials).
**That measurement predicted the wrong winner.** Trained to convergence and
evaluated closed-loop on val_unseen (n=150), the ranking inverts:

| | A (pilot last) | B (action_query) | Stage 0 (memoryless) |
|---|---|---|---|
| SR / OS | **0.0000** | **0.1667** | 0.1667 |
| NE | 8.79 | 7.95 | — |
| nDTW | 0.244 | 0.303 | — |
| reached goal region | **0.0%** | **16.7%** | 16.7% |
| own STOP fired | 39.3% | 30.0% | 0% |

Training metrics were near-identical (loss 0.070 vs 0.082, acc 0.984 vs 0.983,
STOP-recall 0.761 vs 0.770), so nothing in the training logs predicted this.

### Mechanism — it is about WHERE the action is decoded from
- **A**: `h_act == h_pil == H[-1]`. The action is read from the position whose
  INPUT EMBEDDING *is* `z_{t-1}`. A corrupted `z` enters the action pathway
  directly.
- **B**: the action is read from `ACTION_QUERY`, whose input embedding is a
  stable learned vector. `z` reaches it only via attention, which can learn to
  downweight it.

So B is structurally robust to a bad `z`; A is maximally exposed. The design
that shields the action head keeps working (16.7%); the one that does not
collapses to zero.

### Lesson for method
A zero-shot structural probe measures the cost of a slot the model has never
been trained to use. It does not predict behaviour after training, and it
cannot see closed-loop robustness at all. Use it to rule things OUT, not to
pick winners.

**B also exactly matches memoryless Stage 0 at 16.7%** — so B preserved
navigation and added stopping, while A destroyed navigation.

---

## D18 · Pilot-loss scale — lambda=0.1 is sound, but the INIT is not
**Status:** AGENT (measured) · **Affects:** Eq. 14, 15; Stage 1 gate

`||.||^2_2` sums over d=2048, so L_pil's magnitude depends entirely on the
scale of `v_bar`. Measured on the fp32 cache: **||v_bar|| ~= 15.8**, hence
`||v_bar||^2 ~= 250`.

L_pil per step for reference predictors, and what lambda=0.1 does to each:

| predictor | L_pil | lambda*L_pil vs L_act(~1.0) |
|---|---|---|
| `z = 0` (default `nn.Linear` bias) | 251.0 | **25x** |
| `z` = global mean `v_bar` | 17.9 | 1.8x |
| `z = v_bar_t` (nothing changes) | 8.2 | 0.8x |
| `z = v_bar_{t+1}` (**copy baseline**) | **4.78** | 0.5x |

**The paper's lambda is well calibrated AT CONVERGENCE** — a competent `G_psi`
lands near 0.5x L_act. But at default initialisation `z ~= 0` gives **25x**, so
Eq. 15's opening steps are dominated ~25:1 by L_pil and can destroy the action
policy before `G_psi` finds the output scale.

**Fix:** `PilotModule.init_bias_from_target_mean()` seeds `G_psi.bias` with the
mean `v_bar`, starting at 1.8x instead of 25x. NOT IN PAPER, but an
initialisation choice only — the objective is untouched.

### Free gift: the Stage 1 gate (b) threshold is now known without training
AGENTS.md Sec. 4.2(b) requires the learned `G_psi` to beat identity-passthrough
on held-out L_pil. That baseline is computable from the cache alone:

> **held-out L_pil must come in under 4.776** to pass gate (b).

Also worth noting the ordering `4.78 < 8.24`: `v_bar_{t+1}` is genuinely closer
to `v_bar_{t+2}` than `v_bar_t` is, so the copy shortcut is the harder baseline
— which is exactly why AGENTS.md names it as the control rather than the
no-op predictor.

---

## ~~OPEN~~ RESOLVED by D17 · Pilot slot position under the D10 layout
**Status:** RESOLVED (see D17) · kept for the reasoning trail

Eq. 5 places `PILOT(z_{t-1})` last: `[Tok(x) ; v_t ; PILOT(z)]`. Under D10 the
instruction is now last, so the Pilot slot's position must be decided —
presumably `[<vs> ; v_t ; <ve> ; Tok(x) ; PILOT(z)]`, keeping "Pilot last".

Related, and genuinely undetermined — where `h_t^act` comes from. In an
autoregressive model `H[i]` is the state that predicts token `i+1`, so two
readings are both coherent:

| | Sequence | `h_act` vs `h_pil` |
|---|---|---|
| **A** | `[… ; PILOT(z)]` | the SAME state `H[N-1]`, with two heads on it: `W_LM·H[N-1]` → action (Eq. 7), `G_ψ·H[N-1]` → `z_t` (Eq. 8) |
| **B** | `[… ; PILOT(z) ; ACTION_QUERY]` | distinct: `h_pil = H[N-2]`, `h_act = H[N-1]` |

**Current lean: A.** Evidence gathered (targeted re-read of p.9 for Fig. 4):

*Fig. 4 shows* axes `I₁I₂ | H₁H₂ | O₁O₂ | F₁ | P₁P₂ | A₁`, so `A` IS a real
token position. But the axes are grouped by token TYPE, not sequence order, and
the repeated indices (`I₁I₂`, `O₁O₂`, `P₁P₂`) suggest a multi-step unrolled
sequence drawn schematically. A token being present does not mean its hidden
state is `h_act`: under teacher forcing the ground-truth action token is part of
the input by construction, which alone explains its row.

*For A — the paper places itself in the autoregressive-VLA family twice:*
- Sec. 3.3 / Eq. 13 decode actions with the **native `W_LM`**, the RT-2 /
  OpenVLA / NaVid pattern. A separate action query with its own head would not
  need `W_LM` at all — and Sec. 3.3 explicitly rules out a new head.
- The Pilot Token's cited lineage is pause tokens [12] and Coconut [13]. Both
  insert latent tokens and then generate the output autoregressively after
  them; neither uses a separate query token.

*For B:* Sec. 3.3's wording, "the action and Pilot-position hidden states",
reads more naturally as two positions than one vector with two heads.

*Field practice for reference:* autoregressive VLM-VLAs (RT-2, OpenVLA, NaVid,
Uni-NaVid) use no separate query; readout/query designs (Octo, ACT, π₀,
Diffusion Policy) do. LatentPilot's `W_LM` decoding places it in the former.

This is the same gap AGENTS.md Sec. 8 item 1 flags ("the token ordering and
grouping must be inferred" from Fig. 4). Still an inference, not a statement in
the text. Resolve when Stage 1 is authorised; **do not scaffold it now**
(AGENTS.md Sec. 6).

Stage 0 is unaffected: with no Pilot Token, `h_act` is simply `H[-1]`, the last
instruction token under the D10 layout.


---

## D20 — `G_psi` is zero-initialised (weight), bias = mean `v_bar`

**Decision.** `G_psi.weight` starts at 0 and `G_psi.bias` at the dataset mean
`v_bar`, so `z` begins at exactly the mean target. `--g-psi-init default`
restores `nn.Linear`'s init.

**Why.** The paper states `lambda = 0.1` (Eq. 15) but never specifies `G_psi`'s
initialisation, and the two interact multiplicatively. Measured on a real batch:

| init | `\|\|z\|\|` | `L_pil` | `lambda*L_pil : L_act` |
|---|---|---|---|
| `nn.Linear` default | 62.9 | 3748.7 | **167.59x** |
| zero weight, mean bias | 15.0 | 17.4 | **0.42x** |

With the default init the action objective is drowned 168:1 and Eq. 15's stated
lambda is meaningless -- the run would look like "the Pilot loss destroys SR"
when the real cause is an unspecified init. Zeroing the weight makes `L_pil`
start at the variance of `v_bar` (18.48 measured), i.e. the loss of the best
CONSTANT predictor, so every later reduction is real learning rather than the
decay of a bad init.

Zero-init is safe here because Sec. 3.2 fixes `G_psi` at a SINGLE linear layer:
`dL/dW = dL/dz . h^T` is nonzero from the first step. It would not be safe for
an MLP.

**Status.** Ours, not the paper's. Recorded as a "value the paper never states".

---

## D21 — `L_pil` reference baselines (what Stage 1 must beat)

Measured over 15,627 cached `v_bar` frames:

| predictor | `L_pil` | note |
|---|---|---|
| constant (`mean v_bar`) | **18.48** | the zero-init starting value |
| persistence (copy `v_bar_t` as `v_bar_{t+2}`) | **7.82** | 0.423x constant |
| learned `G_psi` | must beat 7.82 | |

**Why this matters.** Persistence at 0.423x proves `t+2` is genuinely
predictable from `t`, so a low `L_pil` alone does NOT show the model learned
dynamics -- it could just be copying. Sec. 4.2's gate (b) uses a frozen
identity `G_psi` (`z_t = h_t^pil`); persistence in `v_bar` space is the sharper
test of the same thing. **Report both.**

---

## D22 — Fig. 4's layout diverges from Eq. 12; Eq. 12 implemented literally

Read directly off Fig. 4 (token order `I I | H H | O O | F F | P P | A A`):

1. **`F` (Future) is its own token group**, separate from `P` (Pilot). Each `F`
   attends ONLY to itself; `P_i` attends to `F_i`. Eq. 12 instead puts the
   future *inside* the Pilot slot: `u^tr_t = [Tok(x); v_t; PILOT(v_bar_{t+1})]`.
2. **`P_2` attends to `P_1` and `A_2` to `A_1`** -- a whole trajectory is packed
   into one sequence, so training steps are NOT independent.
3. **`H` and `O` blocks are bidirectional within themselves**, i.e. prefix-LM
   block attention, not the pure causal attention Eq. 6 states.
4. **`A` never attends to `F`** -- the causal guarantee. Information still flows
   `F -> P -> A`.

**Decision.** Implement Eq. 12 literally (per-step, future enters only via the
Pilot slot), which is what AGENTS.md Sec. 4.1 enumerates.

**Why this is defensible.** Collapsing `F` into `P` (as Eq. 12 does) makes our
single-step sequence satisfy every Fig. 4 invariant that is well defined for
one step: `O` sees `I`; `P` sees `I, O, F`; `A` sees `I, O, P` and not `F`
separately. Our causal mask already gives exactly this. We also have no `H`
tokens -- Eq. 5 carries none, and that is the paper's stated efficiency
property.

The genuine divergence is (2), trajectory packing, which is precisely where
`rollout_dataset.py`'s step-level batching assumption stops holding, and which
Sec. 5 (Stage 2, scheduled sampling) exists to address. **Deferred to Stage 2,
not silently adopted.** AGENTS.md Sec. 8 item 1 already flags the mask layout
as unresolved.

---

## D23 — Stage 1 defaults to Design B (`pilot_mode=action_query`)

Two independent lines of evidence:

* **Closed-loop (D19).** The 8-way slot ablation (n=60 each, `val_unseen`):

  | design | `z` | `z_scaled` | `vbar_current` | `constant` |
  |---|---|---|---|---|
  | A | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | B | 0.1833 | **0.2167** | 0.0000 | 0.0000 |

  Design A scores 0.0000 for **every** slot content including a constant, so its
  failure is architectural, not latent-related: under `pilot_mode=last` the
  frozen `lm_head` must decode actions from the injected Pilot position itself.
  Design B is **slot-dependent** -- replacing `z` with a constant or with the
  real current `v_bar` collapses it to 0.0000 -- so its Pilot Token is
  genuinely load-bearing.
* **Fig. 2** draws `h_t^pil` and `h_t^act` as two separate arrows per step.

*(These are oracle-success numbers: the eval harness ends an episode on
proximity by design, to isolate navigation from the very sparse STOP signal.
Report them as OS, never as SR, when comparing to the paper's 51.7.)*

---

## D24 — Checkpoints save the adapter only

`enc.model` is the BASE model (the PEFT wrapper is deliberately discarded), so
`save_pretrained` writes the full 4.9 GB backbone instead of the adapter.
Stage 0' did this across 37 checkpoints (~180 GB). `save_checkpoint` now filters
on `"lora_"` and raises if that set is empty. Verified: **33 MB** (224 LoRA
tensors + `PilotModule`) against 4.9 GB before.


---

## D25 — STAGE 1 GATE FAILED (both halves). Recorded per Sec. 4.2.

Sec. 4.2: "Report this either way -- it is the most important diagnostic in the
project." Both runs: 17,400 steps, batch 8, identical seed/data/schedule.

**Gate (a) -- beats Stage 0: FAIL.** Target SR 54.0 / SPL 48.5; measured SR 0.0.

**Gate (b) -- beats the copy baseline: FAIL.** The gate requires winning on
BOTH held-out `L_pil` AND SR. It splits (val_unseen, n=30):

| | `L_pil` | OS | NE | nDTW | model_stop |
|---|---|---|---|---|---|
| learned `G_psi`  | **3.02** | 0.10 | 8.58 | 0.306 | 0% |
| identity `G_psi` | 6.80 | **0.40** | **7.40** | **0.404** | 0% |

**The learned Pilot Token is a 2.25x better predictor and a 4x worse
navigator.** Freezing `G_psi` to identity -- no learned prediction at all --
navigates dramatically better.

### Mechanism: lambda = 0.1 is balanced only at initialisation

`L_act` converges ~51x (2.79 -> 0.05); `L_pil` only ~4.8x (14.5 -> 3.0). The
EFFECTIVE weighting therefore drifts by an order of magnitude:

| step | `L_act` | `lambda*L_pil` | ratio |
|---|---|---|---|
| 50 | 2.786 | 1.447 | 0.52x |
| 4000 | 0.410 | 0.422 | 1.03x (crossover) |
| 10000 | 0.100 | 0.321 | 3.21x |
| 17400 | 0.055 | 0.302 | **5.51x** |

By the end the model spends 5.5x more gradient on predicting future frames than
on choosing actions. This explains both observations: why longer training made
closed-loop navigation WORSE (OS 0.267 at step10000 -> 0.10 at final, same
episodes, turning a 2.77 m success into a 9.21 m miss), and why the identity
run navigated better -- its `G_psi` receives no gradient and so exerts far less
pull on the shared LoRA weights.

D20 fixed the balance at INITIALISATION. This is the DYNAMIC balance, which
D20 does not address and the paper does not discuss.

### What this is and is not

It is NOT a refutation of the paper. It is a finding about this reproduction at
2B params / 6 scans / 1,665 episodes. The paper uses a 7B backbone and a far
larger corpus, where the convergence rates -- and hence the drift -- may differ.

Note also that `model_stop` is 0% in BOTH runs, so both are scored on oracle
proximity, not self-termination. Report as OS, never SR.

### Consequence

Sec. 4.2: "If the gate fails: STOP. Do not proceed to Stage 2."

Stage 2 (scheduled sampling) targets exposure bias. This result points instead
at loss balance as the more fundamental problem at our scale; fixing exposure
bias while `L_pil` dominates 5.5x is unlikely to help. Any rebalancing of
lambda is a Sec. 0.4 STOP-and-ask trigger ("changing any loss function or its
weighting") and awaits the human.


---

## D27 — PIVOT: pointing supervision (Robostral) replaces action classification

**Human-directed pivot.** LatentPilot's Stage 1 gate failed on both halves
(D25), and the n=150 re-evaluation showed the identity control was not the 4x
winner the n=30 numbers suggested -- it was 0.173 vs 0.107, and the whole
lambda axis was flat (balanced vs learned, p=0.333). Rather than keep
iterating on the latent, we adopt Robostral's pointing formulation
(arXiv:2607.20785 Sec. 2.2).

**This is no longer a LatentPilot reproduction.** Changing the action space is
a Sec. 0.4 STOP-and-ask; the human directed it explicitly.

### Why pointing suits THIS backbone

Cosmos-Reason2 is a grounding VLM -- pointing, counting and object
localisation are in its pretraining. Emitting `"left"`/`"right"`/`"forward"`
through a 4-way softmax over repurposed vocabulary tokens is not. The action
formulation gave ~1 bit of supervision per step and never learned to stop
(`model_stop` 0.0-4.7% across every checkpoint we evaluated).

### What we keep and what we drop

Robostral's stack is VLM -> waypoint -> 121M diffusion policy -> motion
controller -> 100 Hz motors. That tail exists to produce smooth CONTINUOUS
control for real robots. We have four discrete primitives in habitat, so it is
replaced by bearing: turn if |bearing| > 7.5 deg, else forward. **No diffusion
policy, no controller.**

### Measured design decisions

Three bugs were found by checking the labels against the expert's own actions
rather than by inspection. Agreement between "bearing to the chosen waypoint"
and "what the expert actually did":

| label design | agreement |
|---|---|
| naive (next waypoint, single horizon 24) | 0.689 |
| + skip stationary in-place turn steps | 0.927 |
| + decouple point/control horizons, first-displaced control target | **0.972** |

1. **Habitat turns IN PLACE.** During a turn sequence consecutive positions are
   identical, so a "next step" control waypoint gives `dx = dy = 0` and
   `atan2(0,0) = 0` -- a spurious "go straight" label on exactly the steps
   where the expert is turning. Expert-LEFT steps were labelled FWD 28% of the
   time and RIGHT (0.39) more often than LEFT (0.33): worse than chance.
   Fixed with `MIN_WAYPOINT_DISPLACEMENT_M = 0.20`.
2. **Two horizons, not one.** Pointing answers "where am I headed" (Robostral's
   furthest visible waypoint, 24 steps); control answers "what do I do now"
   (8 steps). Sharing one horizon trades them off: 24 gives 0.52 pointing
   coverage but 0.73 agreement, 8 gives 0.93 agreement but 0.27 coverage.
   Decoupled: **0.55 coverage AND 0.97 agreement.**
3. **First displaced, not furthest.** Within the control horizon the FIRST
   genuinely-displaced waypoint, not the furthest: the furthest already aims
   past the turn on a curving path (0.93 -> 0.75).

### Geometric floor on visibility (not a bug)

The camera sits 1.5 m up with a +/-45 deg vertical half-FOV, and expert
waypoints are at floor level, so a waypoint only enters frame beyond
`1.5 / tan(45 deg) = 1.5 m` -- about 6 steps at 0.25 m. Visible fraction is
0.55 against Robostral's ~0.90; they randomise camera pitch 0-25 deg DOWNWARD
(Sec. 2.4), which sees the floor far closer. Ours is pitch 0. The metric
displacement fallback covers the other 45%, which is what Robostral's Eq. 2 is
for.

### Data

Rollouts re-collected with per-step `rotations` and camera intrinsics recorded
in `episode.json`. Yaw reconstructed from the action list alone is exact 93.8%
of the time but lands a full 15 deg off on the rest where collisions
desynchronise it -- and pointing targets are projected through that yaw.
Verified after re-collection: LEFT = +15.000 deg, RIGHT = -15.000 deg,
FWD = 0.000 deg, exactly.

**Note:** `--overwrite` re-collection invalidates the cached `vbar.npy`, so the
LatentPilot Stage 1/2 paths need `cache_vbar.py` re-run before they will load.

---

## Values the paper never states

Recorded for the Stage 0 report (AGENTS.md Sec. 7).

| Value | Setting | Source |
|---|---|---|
| Input resolution | 448×448 | D3, ours |
| `N_v` | 196 | derived from D3 |
| LoRA rank | TBD | AGENTS.md suggests 16–32 |
| Learning rate | TBD | not in paper (Sec. 8 item 3) |
| Batch size | TBD | must come from a VRAM smoke test |
| Epochs / schedule | TBD | not in paper |
| Stage -1 fps | 4 | D5, measured |

Stated by the paper and honoured: **λ = 0.1**, FWD = 0.25 m, turns = 15°,
Pilot horizon t+2, Pilot slot input `v̄_{t+1}`, `G_ψ` a single `nn.Linear(d, d)`.
