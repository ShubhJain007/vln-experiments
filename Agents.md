# AGENTS.md — LatentPilot Reimplementation Build Spec

**Read this file completely before writing any code. Re-read the "Ground Rules"
section at the start of every session.**

Reference paper: `docs/2603_29165v1.pdf` — *LatentPilot: Scene-Aware
Vision-and-Language Navigation by Dreaming Ahead with Latent Visual Reasoning*
(Hao, Chen, Han et al., arXiv:2603.29165, 31 Mar 2026).

**The paper's official code has NOT been released.** Everything here is a
from-scratch reimplementation from the paper text. That is exactly why the
verification gates in this document are mandatory — they are the only signal
that the implementation is correct.

---

## 0. GROUND RULES (non-negotiable)

### 0.1 Fidelity first
- Implement **exactly** what the paper's equations say. Do not "improve",
  "simplify", "modernize", or substitute a component because you believe a
  better option exists.
- If the paper is ambiguous, **STOP and ask the human**. Do not pick an
  interpretation silently.
- If you find yourself writing a comment like "the paper says X but I'll do Y
  because…" — **STOP and ask the human**.

### 0.2 Equation traceability
- Every function that implements a paper equation must carry a docstring citing
  the equation number and restating the equation in ASCII.
- Example:
  ```python
  def build_input_sequence(instr_tokens, visual_tokens, pilot_vec):
      """Eq. 5:  u_t = [ Tok(x) ; v_t ; PILOT(z_{t-1}) ]

      Concatenates instruction tokens, current-frame visual tokens, and the
      Pilot slot. NOTE: only the CURRENT frame appears. No frame history.
      """
  ```

### 0.3 Verify after every equation
- After implementing each equation, write a test in `tests/` that checks the
  *intended behaviour*, not just that the code runs. Shapes, dtypes,
  invariants, and degenerate cases.
- Do not proceed to the next equation until its test passes.
- Report test results to the human before moving on.

### 0.4 STOP-and-ask triggers
Halt and ask the human before doing ANY of the following:
- Changing the backbone model or vision encoder
- Changing any loss function or its weighting (λ)
- Changing the action space
- Changing the Pilot Token's dimensionality, target, or horizon (t+2)
- Adding any component not described in the paper
- Adding any dependency beyond `requirements.txt`
- Skipping a verification gate because it "probably works"
- Beginning a new Stage before the previous Stage's gate has passed

### 0.5 Change one thing per stage
Each Stage adds exactly one novelty. Keep every checkpoint and its config.
If a metric drops three stages later, you must be able to bisect.

### 0.6 No silent defaults
If a hyperparameter is not stated in the paper, do not guess it into the code.
Put it in a config file, mark it `# NOT IN PAPER — chosen by us`, and tell the
human its value and why.

---

### 0.7 DIVISION OF LABOUR — read this before doing anything

The human is on a **Claude Pro subscription**, which has tight usage limits.
Token efficiency is a hard constraint, not a nicety. Work is split as follows.

### THE HUMAN DOES (do not attempt these yourself)
- All installs: conda, CUDA/torch, habitat-sim, habitat-lab
- All downloads: Matterport3D, R2R-CE episodes, model weights
- Running long jobs: pre-rendering frames, `v̄` caching, training runs, eval runs
- Reporting results back by pasting logs into the session

Environment debugging (especially habitat-sim) is the single largest token sink
in this project and produces no reusable artifact. **Do not iterate on installs.**
If the human reports an install error, give ONE diagnosis and ONE suggested fix,
then wait. Do not enter a fix-retry loop.

### YOU DO
- All source code under `src/`
- All tests under `tests/`
- All configs under `configs/`
- All analysis of logs/metrics the human pastes back
- Debugging code errors (not environment errors)

### YOU DO NOT
- Run training. Write the script, hand the human the exact command, stop.
- Watch or poll a running job. This burns tokens for zero output.
- Re-read `docs/2603_29165v1.pdf` after the first session (see 0.8).
- Explore the repo broadly when you need one file.

### 0.8 Token discipline (Pro plan)

**First session only:** read the PDF once, extract every equation and stated
hyperparameter into `docs/EQUATIONS.md` in plain ASCII. Every session after
that reads `docs/EQUATIONS.md`, never the PDF. The PDF is large and re-reading
it each session is the most avoidable cost in the project.

Other rules:
- **One stage per session.** Start a fresh session (`/clear`) at each stage
  boundary. Do not carry Stage 0's context into Stage 1.
- **One equation per exchange** where practical. Implement, write its test, hand
  the human the command to run it, stop.
- Prefer targeted edits over rewriting whole files.
- Never `cat` a large data file, log, or checkpoint. Ask the human for the
  specific lines.
- Keep replies short. Prose costs tokens; the human wants code and a command.

### 0.9 Handoff protocol

Because you do not run jobs, every unit of work ends the same way:

```
1. You write code + test.
2. You output a single fenced block: the exact command(s) to run.
3. You STOP. End the turn. Do not speculate about the result.
4. Human runs it, pastes back output.
5. You interpret, fix if needed, or proceed.
```

Maintain `HANDOFF.md` at the repo root with: current stage, what was just
completed, the exact command pending, and what result would count as a pass.
Update it at the end of every session so the next session can resume without
re-deriving context.

---

## 1. ENVIRONMENT SETUP — HUMAN CHECKLIST

**This entire section is executed by the human, not the agent.** The agent's
only job here is to answer questions if asked, and to verify at the end.

**Target hardware: single NVIDIA RTX 4070, 16 GB VRAM.** Host OS: Linux.
(macOS is for Stage -1 diagnostics only.)

**16 GB is tight from Stage 3 onward. Hard constraints below — do not exceed
them without asking the human:**

- LoRA (not full fine-tune) for the entire project, all stages.
- 8-bit AdamW is mandatory from Stage 3 onward, not optional.
- Vision encoder frozen throughout Stages 0-3 minimum (see Sec. 6 rationale on
  target-collapse; VRAM is a second, independent reason to keep it frozen).
- Never run Habitat rendering and a training job on the GPU at the same time.
  All frames are pre-rendered to disk in advance (Sec. 3.2) — this is not
  optional on this card.
- Stage 3's diffusion action expert starts at **40-60M parameters**, not the
  ~120M Robostral uses. Only scale up if a VRAM check after implementation
  shows headroom. Report the actual parameter count and measured VRAM to the
  human before deciding whether to scale.
- Stage 3's three new heads (pointing, displacement, diffusion) are added and
  gated **one at a time**, not simultaneously — see Sec. "STAGE 3+" for the
  required sub-ordering. If Stage 3 OOMs with all three added at once, you
  cannot tell which one caused it.
- Stage 5 (RL/GRPO) requires a reference policy copy in memory alongside the
  training policy — effectively double the resident model. On 16 GB this will
  likely require **QLoRA (4-bit base)** rather than standard LoRA. Flag this to
  the human when Stage 5 begins; do not silently switch quantization schemes
  without confirming first.
- Prefer gradient accumulation over reducing sequence length if a batch doesn't
  fit — the sequence is already ~300 tokens (Sec. 3.2's design point) and
  shrinking it further changes what's being tested, not just how fast.
- After implementing each stage, write and run a VRAM smoke test (batch of 1,
  forward+backward, report peak allocated) before scaling to the target batch
  size. Report the number.

### 1.1 Repo skeleton — human creates
```
latentpilot/
├── AGENTS.md               <- this file
├── HANDOFF.md              <- session-to-session state (agent maintains)
├── docs/
│   ├── 2603_29165v1.pdf    <- the paper
│   └── EQUATIONS.md        <- agent writes this in session 1, from the PDF
├── configs/
├── src/
│   ├── model/              <- Eq. 4-9
│   ├── losses/             <- Eq. 13-15
│   ├── data/               <- trajectory extraction, v̄ caching
│   ├── train/
│   └── eval/               <- NE, SR, OS, SPL, nDTW
├── tests/
├── scripts/                <- agent writes, human runs
└── checkpoints/
```

### 1.2 Base environment — human runs
```bash
conda create -n latentpilot python=3.10 -y
conda activate latentpilot
# match the local CUDA version — check `nvidia-smi` first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft bitsandbytes datasets einops
pip install flash-attn --no-build-isolation
```

### 1.3 Habitat + VLN-CE — human runs
```bash
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
pip install git+https://github.com/facebookresearch/habitat-lab.git
```
Then obtain **Matterport3D** scenes (requires a signed MP3D agreement — only the
human can do this) and the **R2R-CE** episode files from the VLN-CE repository.

### 1.4 Backbone — human runs

**Do NOT use NVIDIA's Docker container or vLLM for this project.** That path is
built for high-throughput serving (vLLM pre-allocates a large KV-cache block
for batched inference) and is not representative of a LoRA fine-tuning
workload. NVIDIA's own docs state a 24GB minimum for the 2B model **under
that serving stack** — this figure does not necessarily apply to a bare
`transformers` forward/backward pass with LoRA, but it must be **measured, not
assumed**, on the 16GB 4070 (see hardware constraints above). Go through plain
`transformers`, which supports Cosmos-Reason2 natively as of `transformers>=4.57.0` —
no need to clone the `cosmos-reason2` repo at all for this.

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir models/cosmos-reason2-2b
```

**If the VRAM smoke test (Sec. 1.5) shows the bf16 model + LoRA does not fit in
16GB:** the fallback is 4-bit quantized loading via `bitsandbytes` (QLoRA),
already in the dependency list — or NVIDIA's own quantized checkpoints via
`llmcompressor`. **Do not silently switch to a quantized load; report the
smoke-test numbers to the human first and confirm before changing the loading
scheme**, since it affects every stage downstream.

**Deviation from paper — already approved by the human:** the paper uses
LLaVA-Video-7B (Sec. 3.2). We use Cosmos-Reason2-2B, which is post-trained from
Qwen3-VL-2B-Instruct and shares the same architecture. Its vision encoder is
SigLIP-2 (SigLIP2-Large 300M for the 2B variant), which is the **same encoder
family the paper uses**, so the Pilot Token's target space is consistent with
the paper's design. This is the only pre-approved substitution.

### 1.5 Agent's only task in Section 1

Write `scripts/verify_env.py` — a single script that checks everything at once
and prints a report. The human runs it and pastes the output back.

It must print: torch version, `cuda.is_available()`, GPU name and VRAM, whether
habitat-sim imports, result of loading one scene and stepping the agent 10
times (frame shape, start/end position), and for the backbone: hidden size `d`,
visual tokens per frame `N_v` at the chosen resolution, and the vision encoder's
output dimension.

**Do not write this as multiple scripts, and do not ask the human to run
exploratory one-liners.** One script, one run, one report.

**Verify and REPORT to the human before continuing:**
- hidden size `d`
- visual tokens per frame `N_v` at your chosen resolution
- vision encoder output dim (must be `d` after the merger, or note the mismatch)

---

## 2. STAGE -1 — Predictability probe (no training)

**Purpose:** confirm the Pilot Token's target is learnable at all before
building anything. Can run on a MacBook.

**Steps:**
1. Collect ~20 trajectories (Habitat rollouts with the shortest-path follower,
   or real phone footage of corridors subsampled to ~0.25 m / ~15° steps).
2. Encode every frame with the **frozen** vision encoder; mean-pool per Eq. 11:
   `v̄_t = Pool(E_φ(o_t)) ∈ R^d`
3. Compute and report:
   - `cos(v̄_{t+1}, v̄_{t+2})` — the "copy the slot input" shortcut score
   - `cos(v̄_t, v̄_{t+2})` — the "nothing changes" score
   - the same at horizons t+1, +2, +4, +8
   - retrieval@1: given `v̄_{t+2}`, rank 50 frames from the same trajectory

**GATE — report to human, do not proceed unilaterally:**
- `cos(v̄_{t+1}, v̄_{t+2}) > 0.98` → the target is too smooth; the human must
  decide between a longer horizon or delta targets. **STOP and ask.**
- `0.90–0.97` → narrow headroom, proceed but flag it.
- `< 0.90` → proceed.

---

## 3. STAGE 0 — Skeleton, no Pilot Token

**Goal: reproduce the paper's `NaN` ablation row (Table 3).**

### 3.1 Implement, in this order, testing each

**Eq. 4** — `v_t = E_φ(o_t) ∈ R^{N_v × d}`
Freeze the encoder. (The paper trains it, per Eq. 15's `min over θ,φ,ψ`. We
freeze it — **deviation approved by the human** — because the encoder also
produces the Pilot target, and a trainable target invites collapse. Revisit
only with an EMA target encoder.)
*Test:* output shape is `(N_v, d)`; two calls on the same image are identical;
requires_grad is False.

**Eq. 3** — `A = {FWD, LEFT, RIGHT, STOP}`
*Test:* exactly 4 actions; the mapping to Habitat primitives matches the paper's
real-robot params (FWD = 0.25 m, turns = 15°).

**Eq. 5, minus the Pilot slot** — `u_t = [Tok(x); v_t]`
*Test:* sequence length equals `len(instr) + N_v`. Assert **no history frames
are present** — this is the paper's key efficiency property and the most likely
place to accidentally deviate.

**Eq. 6** — `H_t = F_θ(u_t) ∈ R^{N × d}`
*Test:* one hidden vector per input position; causal mask verified by checking
that changing a later token does not alter an earlier position's output.

**Eq. 7 / Eq. 13** — `π_θ = Softmax(W_a h_t^act)`
Per Sec. 3.3, actions are decoded by the backbone's **native LM output
projection** (`W_LM`). Implement the 4 actions as vocabulary tokens, not a new
head. **If this is not feasible with the chosen backbone, STOP and ask.**
*Test:* probabilities sum to 1; loss on a uniform-random model ≈ ln(4) ≈ 1.386.

**L_act** (Eq. 13) — cross-entropy over expert actions.
*Test:* loss → 0 when logits are one-hot on the correct action.

### 3.2 Data
- R2R-CE, start with **1000–2000 training episodes**, not the full set.
- Pre-render frames to disk. Do not run Habitat inside the training loop.
- Cache `v̄` for every frame once (the encoder is frozen).

### 3.3 Training
LoRA (r=16–32) on attention projections, bf16, gradient checkpointing,
8-bit AdamW. Sequence ≈ 300 tokens, comfortable on a 16 GB card at this
stage — run the VRAM smoke test (Sec. 1) to confirm before assuming a batch
size; do not hardcode batch 8 without checking.

### 3.4 Metrics (implement exactly as Sec. 4.1)
- **NE** — geodesic distance from stop point to goal (m)
- **SR** — fraction stopping within 3 m of goal
- **OS** — success if the *closest point along the path* was within 3 m
- **SPL** — success weighted by path length
- **nDTW** — normalized dynamic time warping vs. reference path

*Test:* on ground-truth reference paths, SR must be ~1.0 and SPL ~1.0. If not,
the metric implementation is wrong — fix before evaluating any model.

### 3.5 GATE
Target on **R2R Val-Unseen**, per Table 3 `NaN` row:
`SR ≈ 51.7, SPL ≈ 47.1, NE ≈ 5.3, OS ≈ 57.0`

- Within a few points → proceed.
- SR < 45 → the harness is wrong, not the science. **STOP, report, debug.**
  Do not "fix" it by changing the model.

Note our setup is smaller (2B vs 7B) and offline-only, so landing somewhat
below is expected. Report the actual numbers and let the human judge.

---

## 4. STAGE 1 — Pilot Token (the decisive gate)

### 4.1 Implement

**Special token.** Add `<|placeholder|>` to the tokenizer. Its learned embedding
provides `z_0` at the first step (Sec. 3.2).
*Test:* the token has an embedding row; `z_0` is finite and non-zero.

**Eq. 5, full** — `u_t = [Tok(x); v_t; PILOT(z_{t-1})]`
*Test:* the Pilot slot occupies exactly one position; its index is recoverable.

**Eq. 8** — `z_t = G_ψ(h_t^pil) ∈ R^d`
`G_ψ` is a **single linear layer** (Sec. 3.2: "a simple linear layer, with only
a few parameters"). Not an MLP. Not a transformer. **If you are tempted to make
it deeper, STOP and ask.**
*Test:* `G_ψ` is `nn.Linear(d, d)`; parameter count == `d*d + d`.

**Eq. 11** — `v̄_{t+1} = Pool(E_φ(o_{t+1})) ∈ R^d`
Mean pooling over the `N_v` visual tokens.
*Test:* output is a single `d`-dim vector; equals the arithmetic mean of the
token dimension.

**Eq. 12** — `u^tr_t = [Tok(x); v_t; PILOT(v̄_{t+1})]`
Training only. The Pilot slot is **teacher-forced with the real encoded
one-step-ahead frame**. Note the asymmetry with inference (Eq. 5).
*Test:* assert that in train mode the slot contains `v̄_{t+1}`, and in eval mode
it contains the cached `z_{t-1}`. This is the single easiest thing to get
wrong — test it explicitly.

**Eq. 14** — `L_pil = Σ_{t=1}^{T-2} ||z_t - v̄_{t+2}||²_2`
Note the sum runs to `T-2` (the last two steps have no `t+2`).
*Test:* loss is 0 when `z_t == v̄_{t+2}`; the last two timesteps contribute
nothing; masking is correct at episode boundaries.

**Eq. 15** — `L = L_act + λ·L_pil`, **λ = 0.1** (Sec. 3.3, stated explicitly).
*Test:* with `L_pil` zeroed, total loss equals Stage 0's loss exactly.

**PilotCache** (Sec. 3.3, "Inference with stored pilot").
At inference: read `z_{t-1}` from cache → forward → write `z_t` back. No future
frames, no separate Pilot computation.
*Test:* run 20 inference steps and assert no `v̄_{t+1}` or `v̄_{t+2}` is ever
accessed. Add a hard assertion in the eval path that raises if future-frame
data is touched. **Future leakage at eval invalidates every number.**

**Visibility matrix (Fig. 4).** Token types: Instruction, History, Current,
Future, Pilot, Action. Reproduce the mask from the figure.
*Test:* assert Future tokens are never visible from positions that must remain
causal at inference.

### 4.2 GATE — this is the decisive experiment

Two checks, both required:

**(a) Beats Stage 0.** Target `SR ≈ 54.0, SPL ≈ 48.5` — the paper's
**flywheel round 1** number (Fig. 5), which is the correct offline-equivalent
target. Do NOT target 62.0; that requires 7 rounds of PilotLoop, which we are
not running.

**(b) Beats the copy baseline.** Train a control where `G_ψ` is frozen to the
identity. If the learned model does not beat identity-passthrough on `L_pil`
held-out error and on SR, the model has learned to copy its input, not to
predict dynamics. **Report this either way — it is the most important
diagnostic in the project.**

**If the gate fails:** STOP. Do not proceed to Stage 2. Report to the human.
A negative result here is a real finding, cheaply obtained, and changes the
plan.

---

## 5. STAGE 2 — Scheduled sampling (Pilot-slot mixing)

**Not in the paper. Human-approved addition.** Fixes the train/test mismatch
between Eq. 12 (teacher-forced `v̄_{t+1}`) and Eq. 5 (self-predicted `z_{t-1}`).

### 5.1 The problem being fixed
Per Eq. 12 the Pilot slot always contains a **real encoded future frame** during
training. Per Eq. 5 it always contains the model's **own previous prediction**
at inference. The backbone therefore learns attention weights under an input
distribution it never sees at test time. This is textbook exposure bias, and
because `z_t` is computed from a sequence containing `z_{t-1}`, corruption
compounds across a rollout.

Precedent: π₀.₇ (arXiv:2604.15483) hit the identical problem with subgoal
images and solved it by training on a mix of real future frames and
world-model-generated ones, explicitly "to reduce train-test mismatch."

### 5.2 Three-way slot policy
At each training step, the Pilot slot is filled with one of:

| Content | Symbol | Purpose |
|---|---|---|
| real one-step future | `v̄_{t+1}` | the paper's privileged anchor (Eq. 12) |
| model's own prediction | `z_{t-1}` (detached) | matches inference distribution |
| learned null embedding | `z_null` | **STAGE 4 ONLY — do not add here** |

**Stage 2 uses only the first two.** `z_null` belongs to the masking work in
Stage 4; adding it now would violate the one-novelty-per-stage rule.

Note that in all cases the **target stays `v̄_{t+2}`** (Eq. 14, real encoded
frame). Only the *input* is mixed. The privileged supervision the paper relies
on is not weakened.

### 5.3 Structure: separate fine-tune phase, NOT an in-run ramp

Stage 2 is a **short fine-tune starting from the converged Stage 1
checkpoint**, not a schedule folded into the Stage 1 training run.

```
Stage 1 run:  100% GT (v̄_{t+1}) → convergence → checkpoint_stage1.pt
                                                        │
Stage 2 run:  load checkpoint_stage1.pt, low LR, ramp p: 0 → p_final
```

Why this structure (report to human, confirm before use):

- **Preserves a paper-comparable checkpoint.** Stage 1's gate is a comparison
  against the paper's flywheel-round-1 numbers (SR ≈ 54.0). Folding the ramp
  into the Stage 1 run destroys that comparison — you end up with a hybrid that
  matches nothing in the paper.
- **Makes the `p` sweep cheap.** Fine-tuning from a converged base is short, so
  `p_final ∈ {0.25, 0.5, 0.75}` costs roughly one extra full run in total.
  An in-run ramp requires full retraining per value.
- **Preserves one-novelty-per-stage.** A regression in Stage 2 is attributable
  to self-prediction and nothing else.

### 5.4 Schedule and ratios

**These numbers are NOT from the paper.** Mark them `# NOT IN PAPER` in config.
Starting points to sweep, not ground truth.

```python
# config/stage2.yaml
init_from:         "checkpoints/stage1_final.pt"
lr_scale:          0.1     # NOT IN PAPER — fraction of Stage 1 LR
p_selfpred_start:  0.0     # NOT IN PAPER
p_selfpred_final:  0.5     # NOT IN PAPER — sweep {0.25, 0.5, 0.75}
ramp_frac:         0.3     # NOT IN PAPER — fraction of fine-tune spent ramping
finetune_frac:     0.15    # NOT IN PAPER — length vs Stage 1, in steps
```

- **Ramp inside the fine-tune, do not hard-switch.** The model has specialized
  its Pilot-slot attention to "this slot contains real visual evidence." A step
  change from 0% to 50% is the destabilizing case scheduled sampling exists to
  avoid. Ramp over the first ~30% of the fine-tune.
- **Low LR (≈0.1× Stage 1).** You are adapting a converged model, not training
  one. High LR here risks losing what Stage 1 achieved.
- **Do not exceed p_final = 0.75 without asking the human.** The two-step design
  (t+1 in, t+2 out) exists so the model has a real near-future scaffold. At
  100% self-predicted the scaffold is gone and the task becomes unanchored
  two-step extrapolation — a different problem than the paper posed.

### 5.5 The failure mode to watch for

Starting from a fully-converged GT model creates a specific risk: rather than
learning to use noisier Pilot content, the model may simply learn to **ignore
the Pilot slot entirely.** That solves the distribution shift trivially and
silently discards the paper's entire contribution.

Symptoms: SR does not regress, but `||z_t||` drift does not improve either, and
the Stage 1 → Stage 2 gain is zero.

**Mandatory diagnostic:** measure mean attention mass on the Pilot-slot position
before and after the fine-tune. If it drops substantially, **STOP and report to
the human.** Also re-run the Stage 1 copy-baseline comparison after Stage 2 —
if the model is now no better than identity-passthrough, the slot has been
abandoned.

### 5.6 Implementation requirements

- **CRITICAL: detach `z_{t-1}`.** No gradient through time. The naive version
  will OOM on a 16GB card and silently changes the optimization problem.
  *Test:* assert `z_prev.requires_grad is False` when fed into the slot.
- Sample the choice **per training sample**, not per batch, so each batch
  contains a mix. *Test:* over 1000 samples at p=0.5, the empirical rate is
  within a few percent of 0.5.
- Obtaining `z_{t-1}` requires a forward pass at step `t-1`. Two options:
  (a) sequential unroll within a short window with detach, or
  (b) a stale cache of `z` values from the previous epoch.
  **Option (a) is the faithful one. If (a) is too slow, STOP and ask the human
  before switching to (b)** — (b) changes what distribution the model is
  actually being trained against.
- *Test:* with `p_selfpred = 0.0`, training must be numerically identical to
  Stage 1. Assert this explicitly; it is the cleanest regression check.

### 5.7 Sweep to run
Stage 2 is cheap. Run `p_final ∈ {0.0, 0.25, 0.5}` and report the table. If 0.0
wins, the mismatch is not hurting at this scale and we learn something useful.

### 5.8 Gate
- SR must not regress vs. Stage 1.
- `||z_t||` drift over a 50-step rollout should be **lower** than Stage 1.
  Measure and plot it; this is the quantity scheduled sampling is meant to fix.
- Report both, plus the sweep table.

### 5.9 Explicitly out of scope for Stage 2
π₀.₇ also randomizes the subgoal *horizon* (0.25 probability end-of-segment,
0.75 uniform 0–4 s ahead). Our equivalent would be varying the t+2 target
horizon. This is a **larger deviation from Eq. 14** and is NOT authorized here.
If Stage -1's similarity probe suggests t+2 is too smooth, raise it with the
human as a separate decision.

---

## 6. STAGE 3+ — NOT YET AUTHORIZED

Stages 3 (continuous pointing + diffusion action expert), 4 (confidence head +
masking), 5 (exploration + RL), and 6 (domain transfer) are planned but
**must not be started** until Stages 0–2 have passed their gates and the human
has explicitly authorized the next stage.

Do not scaffold them "to save time later."

---

## 7. REPORTING FORMAT

After each equation: the test name, pass/fail, and any assumption made.

After each stage, report a table:

| Metric | Paper | Ours | Δ |
|---|---|---|---|
| SR | 51.7 | ? | |
| SPL | 47.1 | ? | |
| NE | 5.3 | ? | |
| OS | 57.0 | ? | |

Plus: every hyperparameter marked `# NOT IN PAPER`, and every place the paper
was ambiguous and how it was resolved.

---

## 8. KNOWN AMBIGUITIES IN THE PAPER

These are unresolved in the text. **Ask the human when you reach each one.**

1. **Exact attention-mask layout** — Fig. 4 gives the visibility matrix
   visually; the token ordering and grouping must be inferred.
2. **PilotLoop deviation threshold** — the distance at which the expert takes
   over is not stated. (Not needed until the flywheel is built.)
3. **Learning rate, schedule, batch size, number of epochs** — not stated.
4. **`N_v`** — visual tokens per frame is backbone- and resolution-dependent.
5. **Whether `W_a` (Eq. 7) and `W_LM` (Eq. 13) are the same matrix** — Sec. 3.3
   says actions are decoded by the native LM projection, implying they are.
   Implement as the same; flag if this causes problems.
6. **Episode-boundary handling for the Pilot cache** — presumably reset to
   `z_0` at each episode start; not stated explicitly.