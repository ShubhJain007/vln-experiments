# EQUATIONS.md — LatentPilot Paper Reference
# Source: arXiv:2603.29165 (Hao, Chen, Han et al., 31 Mar 2026)
# Extracted once from the PDF in Session 1. All future sessions read THIS file.
# DO NOT re-read the PDF.

---

## Section 3.1 — Task Definition

### Eq. 1 — Environment dynamics (POMDP transition)
```
s_{t+1} ~ T(· | s_t, a_t)
o_{t+1} ~ Omega(· | s_{t+1})
```
T = state transition model, Omega = observation model.

### Eq. 2 — Trajectory definition
```
tau = (x, o_{1:T}, a_{1:T})
o_{1:t} = (o_1, ..., o_t)
a_{1:t} = (a_1, ..., a_t)
```
Terminates when a_t = STOP or t = T_max.

### Eq. 3 — Discrete action set
```
A = {FWD, LEFT, RIGHT, STOP}
```
Each action maps to a motion primitive in continuous space.
FWD = 0.25 m forward,  LEFT/RIGHT = 15° turn  (from AGENTS.md Sec. 3.1).

---

## Section 3.2 — LatentPilot Architecture

### Eq. 4 — Visual encoding (vision encoder)
```
v_t = E_phi(o_t)  in  R^{N_v x d}
```
- E_phi: SigLIP-initialized vision encoder (SigLIP2-Large 300M for the 2B variant).
- N_v: number of visual tokens per frame (resolution-dependent; see Sec. 1.5 output).
- d: hidden size of the LLM backbone (2048 for Cosmos-Reason2-2B).
- E_phi is FROZEN throughout Stages 0–3 (deviation approved; see AGENTS.md Sec. 3).

### Eq. 5 — Input sequence at inference (full, with Pilot slot)
```
u_t = [ Tok(x) ; v_t ; PILOT(z_{t-1}) ]
```
- Tok(x): tokenized natural-language instruction.
- v_t: current-frame visual tokens (NO history frames — efficiency property).
- PILOT(z_{t-1}): Pilot Token from previous step, placed at a dedicated slot.
- z_0 provided by the embedding of special token <|placeholder|>.

### Eq. 6 — LLM backbone forward pass
```
H_t = F_theta(u_t)  in  R^{N x d}
```
- F_theta: LLM backbone with causal attention.
- N: total sequence length = len(Tok(x)) + N_v + 1 (Pilot slot).
- H_t: hidden states at all positions.

### Eq. 7 — Action distribution (at inference / forward pass)
```
pi_theta(a_t | x, o_t, z_{t-1}) = Softmax(W_a  h_t^act),   a_t in A
```
- h_t^act: hidden state at the action position in H_t.
- W_a = W_LM (the backbone's native LM output projection; same matrix, per Sec. 3.3).
- Actions are implemented as vocabulary tokens, not a separate head.

### Eq. 8 — Pilot module (Pilot Token update)
```
z_t = G_psi(h_t^pil)  in  R^d
```
- h_t^pil: hidden state at the Pilot-slot position in H_t.
- G_psi: a SINGLE linear layer nn.Linear(d, d). NOT an MLP, NOT a transformer.
  (Paper quote: "a simple linear layer, with only a few parameters".)
- Parameter count of G_psi = d*d + d.

### Eq. 9 — One-step transition summary
```
(a_t, z_t) = F_{theta,psi}(x, o_t, z_{t-1})
```
Given instruction, current observation, and previous Pilot Token →
outputs next action and updated Pilot Token.

---

## Section 3.3 — Training with Future-Privileged Supervision

### Eq. 10 — Collected trajectory format (flywheel)
```
tau = (x, o_{1:T}, a^col_{1:T})
```
a^col_t: action collected at step t. Produced by the current model by default;
an expert policy may override it when deviation from the reference becomes large.

### Eq. 11 — Mean-pooled one-step future latent (training target input)
```
v̄_{t+1} = Pool(E_phi(o_{t+1}))  in  R^d
```
Mean-pool the N_v visual tokens of the next observation into a single d-dim vector.
Used as the teacher-forced Pilot slot input during training.

### Eq. 12 — Training input sequence (teacher-forced Pilot slot)
```
u^tr_t = [ Tok(x) ; v_t ; PILOT(v̄_{t+1}) ]
```
At TRAINING ONLY: the Pilot slot is filled with the real encoded one-step-ahead frame v̄_{t+1},
NOT with the model's own previous prediction z_{t-1}.
At INFERENCE: the slot contains z_{t-1} from cache (Eq. 5).
This asymmetry is the most common implementation mistake — test it explicitly.

### Eq. 13 — Action loss (imitation learning, training)
```
p_theta(·) = Softmax(W_LM  h_t^act)

L_act = - sum_{t=1}^{T} log p_theta(a^col_t | x, o_t, v̄_{t+1})
```
Cross-entropy over collected expert actions.
W_LM is the backbone's native LM output projection (same as W_a in Eq. 7).

### Eq. 14 — Pilot loss (future-privileged supervision, training)
```
z_t = G_psi(h_t^pil)  in  R^d

L_pil = sum_{t=1}^{T-2} ||z_t - v̄_{t+2}||^2_2
```
- Sum runs from t=1 to T-2 (last two steps have no t+2 target).
- v̄_{t+2} = Pool(E_phi(o_{t+2})): mean-pooled TWO-step future observation.
- The Pilot slot INPUT is one-step ahead (v̄_{t+1}); the TARGET is two-step ahead (v̄_{t+2}).

### Eq. 15 — Joint training objective
```
min_{theta, phi, psi}  E_{tau ~ D} [ L_act(tau) + lambda * L_pil(tau) ]
```
- lambda = 0.1  (stated explicitly in Sec. 3.3: "we set 0.1 in practice").
- In our implementation phi (vision encoder) is FROZEN → no gradient through E_phi.
- The minimization is over theta (backbone), phi (encoder, frozen), psi (Pilot module).

---

## Visibility Matrix (Fig. 4)

Token types in the input sequence:
- I: Instruction tokens
- H: History tokens (if any — paper's Eq. 5 shows NO history at inference)
- O: Current observation tokens (v_t)
- F: Future tokens (v̄_{t+1} — training only, never at inference)
- P: Pilot token (z_{t-1} or v̄_{t+1} depending on train/eval)
- A: Action token

Attention visibility from Fig. 4 (rows = query position, cols = key position):
- Instruction (I) rows: can attend to I only (or I + H).
- Current observation (O) rows: can attend to I, H, O (causal within current step).
- Future (F) rows: can attend to I, H, O, F (training only; NEVER visible at inference).
- Pilot (P) row: can attend to I, H, O, F, P (sees all non-future at inference; sees F at train).
- Action (A) row: can attend to I, H, O, P, A (full causal; sees Pilot; does NOT see F at inference).

Key invariant: Future tokens (F) are NEVER visible from Action positions at inference.
This is the causal constraint that makes LatentPilot inference-safe.

---

## PilotCache (Inference Protocol, Sec. 3.3)

```
Initialization:  z_0 = embedding(<|placeholder|>)
At step t:
  1. Read z_{t-1} from cache
  2. Forward: H_t = F_theta([Tok(x); v_t; PILOT(z_{t-1})])
  3. Predict action:  a_t = argmax Softmax(W_LM h_t^act)
  4. Predict pilot:   z_t = G_psi(h_t^pil)
  5. Write z_t to cache
  6. Execute a_t in environment -> o_{t+1}
  7. Repeat until a_t = STOP
```
No future frames accessed. No separate world model called. Strictly causal.

---

## Stated Hyperparameters (Sec. 3 / AGENTS.md)

| Parameter | Value | Source |
|---|---|---|
| lambda (loss weight) | 0.1 | Eq. 15, stated explicitly |
| Action set size | 4 | Eq. 3 |
| FWD motion | 0.25 m | AGENTS.md Sec. 3.1 |
| LEFT/RIGHT turn | 15 deg | AGENTS.md Sec. 3.1 |
| Pilot loss horizon | t+2 (two steps ahead) | Eq. 14 |
| Pilot slot input (train) | v̄_{t+1} (one step ahead) | Eq. 12 |
| G_psi architecture | nn.Linear(d, d), single layer | Sec. 3.2 |
| G_psi param count | d*d + d | derived from above |
| Vision encoder | SigLIP2-Large 300M (frozen) | Sec. 3.2 + AGENTS.md |
| Backbone (paper) | LLaVA-Video-7B | Sec. 3.2 |
| Backbone (ours) | Cosmos-Reason2-2B (Qwen3VL-2B) | AGENTS.md Sec. 1.4 |
| d (hidden size, ours) | 2048 | config.json |
| Vision out_hidden_size | 2048 | config.json (matches d) |
| LoRA rank | 16-32 | AGENTS.md Sec. 3.3 |
| LoRA target | attention projections | AGENTS.md Sec. 3.3 |
| Sequence length | ~300 tokens | AGENTS.md Sec. 3.2 |
| Precision | bf16 | AGENTS.md Sec. 3.3 |
| Optimizer | 8-bit AdamW | AGENTS.md Sec. 3.3 |
| Gradient checkpointing | yes | AGENTS.md Sec. 3.3 |

## NOT IN PAPER — chosen by us (must confirm with human)

| Parameter | Value | Reason |
|---|---|---|
| LoRA rank | TBD (16 or 32) | VRAM-dependent |
| LR, schedule, epochs | TBD | NOT stated in paper |
| Batch size | TBD | Must measure via VRAM smoke test |
| Input resolution | 448 x 448 | Verified empirically in Session 1 |
| N_v at 448x448 | 196 | Formula (448/16)^2 / 4 = 196; confirmed empirically |
| VRAM bf16 load+fwd | 4.66 GB alloc / 4.92 GB reserved | Measured Session 1 on RTX 4070 16 GB |
| LoRA strategy | standard LoRA (NOT QLoRA) | 4.66 GB leaves >11 GB headroom; confirmed Session 1 |

---

## Gate Targets (from Table 3 and Fig. 5)

### Stage 0 gate (Table 3, "NaN" row, R2R Val-Unseen)
| Metric | Paper target |
|---|---|
| SR | 51.7 |
| SPL | 47.1 |
| NE | 5.3 |
| OS | 57.0 |

### Stage 1 gate (Fig. 5, flywheel round 1, R2R Val-Unseen)
| Metric | Paper target |
|---|---|
| SR | ~54.0 |
| SPL | ~48.5 |

Note: These are 7B model numbers. With 2B backbone, landing somewhat below is expected.
