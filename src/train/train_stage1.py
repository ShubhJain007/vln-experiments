"""
Stage 1 — the Pilot Token with its dedicated loss (AGENTS.md Sec. 4).

This is the decisive stage. Stage 0' proved nothing about the Pilot Token
because G_psi received no gradient there: z was an untrained random projection,
so its closed-loop "drift" (||z|| 63 vs ||v_bar|| 15.8) measured nothing. Here
G_psi is trained, and the question becomes answerable.

--------------------------------------------------------------------------
WHAT CHANGES FROM STAGE 0' (exactly one thing, per Sec. 0.5)
--------------------------------------------------------------------------
    Stage 0'   L = L_act
    Stage 1    L = L_act + lambda * L_pil          (Eq. 15, lambda = 0.1)

Everything else -- data, LoRA config, schedule, seed, Eq. 12 teacher forcing --
is held fixed so the difference is attributable to L_pil alone.

--------------------------------------------------------------------------
THE TRAIN/EVAL ASYMMETRY (Eq. 12 vs Eq. 5) -- the easiest thing to get wrong
--------------------------------------------------------------------------
    TRAIN (Eq. 12)   u^tr_t = [ Tok(x) ; v_t ; PILOT( v_bar_{t+1} ) ]
    EVAL  (Eq. 5)    u_t    = [ Tok(x) ; v_t ; PILOT( z_{t-1}     ) ]

At training the slot is TEACHER-FORCED with the real one-step-ahead frame.
At inference it holds the model's own cached previous output. The slot input is
one step ahead; the regression TARGET (Eq. 14) is two steps ahead. Sec. 4.1
calls this out as the single easiest error in the project, so it is asserted at
runtime here (`--assert-asymmetry`, on by default) and tested in
tests/test_stage1_asymmetry.py.

--------------------------------------------------------------------------
REDUCTION -- how lambda is kept at exactly 0.1 (D9)
--------------------------------------------------------------------------
Eq. 13 sums to T, Eq. 14 sums to T-2. Reducing them differently silently
rescales lambda. Both terms here are SUMMED over the batch and divided by the
SAME denominator B (the batch's step count):

    l_act = sum_b CE_b / B          (== cross_entropy reduction="mean")
    l_pil = sum_b ||z_b - v_bar_b||^2 * mask_b / B      <- NOT /num_valid

Dividing L_pil by the number of VALID steps instead would inflate it by
T/(T-2), turning lambda into 0.107 at T=30 and 0.15 at T=6. The masked steps
contribute exactly zero to the numerator and still count in B, which is what
Eq. 14's "sum to T-2, average over trajectories" means.

--------------------------------------------------------------------------
GATE (b) -- the copy baseline, Sec. 4.2
--------------------------------------------------------------------------
`--g-psi identity` freezes G_psi to the identity map, so z_t = h_t^pil with no
learned projection. If the trained model does not beat this on held-out L_pil
AND on SR, the model has learned to copy its input rather than predict
dynamics. Sec. 4.2: "Report this either way -- it is the most important
diagnostic in the project."

    conda activate latentpilot
    python src/train/train_stage1.py --smoke-test
    python src/train/train_stage1.py --report-scale          # check lambda first
    python src/train/train_stage1.py --batch-size 8 --steps 17400
    python src/train/train_stage1.py --g-psi identity --steps 17400 \
           --out checkpoints/stage1_identity
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import argparse
import json
import math
import pathlib
import time
from collections import Counter

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))

from data.rollout_dataset import PilotRolloutDataset, load_frame  # noqa: E402
from losses.action_loss import action_loss  # noqa: E402
from losses.pilot_loss import (  # noqa: E402
    LAMBDA_PIL, combined_loss, pilot_loss, pilot_loss_scale_report,
)
from model.action_head import ActionHead  # noqa: E402
from model.action_space import Action  # noqa: E402
from model.backbone import Backbone  # noqa: E402
from model.input_sequence import InputSequenceBuilder, collate_sequences  # noqa: E402
from model.pilot import PilotModule  # noqa: E402
from model.vision_encoder import VisionEncoder  # noqa: E402
from train.schedule import describe, make_scheduler  # noqa: E402

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LR = 1e-4
DEFAULT_WARMUP = 100
# NOT IN PAPER (D26) -- clamps on the adaptive lambda, so a transient tiny
# L_pil cannot produce an enormous weight and destabilise the run.
LAMBDA_MIN = 1e-4
LAMBDA_MAX = 1.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPECTED_ADAPTED_MODULES = 112


# ---------------------------------------------------------------------------
def compute_mean_vbar(dataset: PilotRolloutDataset) -> torch.Tensor:
    """Mean v_bar across the cache, for D18's bias-init trick."""
    return torch.from_numpy(np.concatenate(dataset.vbars).mean(axis=0))


def build_model(lora_r, lora_alpha, lora_dropout, mean_vbar,
                grad_checkpointing=True, pilot_mode="action_query",
                g_psi="learned", g_psi_init="zero"):
    """Backbone + LoRA + PilotModule, with G_psi TRAINABLE (unlike Stage 0').

    pilot_mode defaults to "action_query" (Design B). Two independent reasons:
      * D19 -- Design A collapsed in closed loop, and the slot-content ablation
        showed it fails identically on z, rescaled z, real v_bar, and a
        constant, so its failure is architectural, not latent-related.
      * Fig. 2 of the paper draws h_t^pil and h_t^act as two separate arrows
        per step, i.e. distinct hidden states.

    g_psi="identity" is Sec. 4.2's copy baseline: G_psi is frozen to I so that
    z_t = h_t^pil exactly.
    """
    from peft import LoraConfig, get_peft_model

    enc = VisionEncoder()
    model = enc.model

    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                     bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = get_peft_model(model, cfg)

    adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    visual_adapted = [n for n in adapted if "visual" in n]
    if visual_adapted:
        raise RuntimeError(f"LoRA touched vision modules: {visual_adapted[0]} (D2)")
    if len(adapted) != EXPECTED_ADAPTED_MODULES:
        raise RuntimeError(f"LoRA adapted {len(adapted)}, expected "
                           f"{EXPECTED_ADAPTED_MODULES}")

    if grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    pilot = PilotModule(enc.d, mode=pilot_mode, dtype=torch.bfloat16,
                        device=enc.model.device)
    pilot.assert_shapes()

    if g_psi == "identity":
        # Sec. 4.2 gate (b): z_t = h_t^pil, no learned projection, frozen.
        with torch.no_grad():
            pilot.G_psi.weight.copy_(torch.eye(enc.d, dtype=pilot.G_psi.weight.dtype,
                                               device=pilot.G_psi.weight.device))
            pilot.G_psi.bias.zero_()
        pilot.G_psi.requires_grad_(False)
    elif g_psi == "learned":
        pilot.init_bias_from_target_mean(mean_vbar.to(torch.bfloat16))   # D18
        if g_psi_init == "zero":
            # D20. The paper states lambda=0.1 but never specifies G_psi's
            # initialisation, and the two interact: with nn.Linear's default
            # init, ||z||=62.9 against ||v_bar||=14.9, so L_pil starts at 3749
            # and lambda*L_pil is 168x L_act -- the action objective is drowned
            # and Eq. 15's stated lambda is meaningless.
            #
            # Zeroing the weight (bias already = mean v_bar) makes z start at
            # exactly the mean target, so L_pil begins at the variance of
            # v_bar (18.48 measured) -- the loss of the best CONSTANT
            # predictor. That puts lambda*L_pil at 1.85 against L_act 2.24, a
            # ratio of 0.83, and every subsequent reduction in L_pil is real
            # learning rather than the decay of a bad init.
            #
            # Zero-init is safe for a SINGLE linear layer: dL/dW = dL/dz . h^T
            # is nonzero from the first step. It would not be for an MLP, but
            # Sec. 3.2 fixes G_psi at one layer.
            with torch.no_grad():
                pilot.G_psi.weight.zero_()
    else:
        raise ValueError(f"unknown --g-psi {g_psi!r}")

    return enc, pilot


def make_optimizer(model, pilot, lr):
    import bitsandbytes as bnb

    lora_params = [p for p in model.parameters() if p.requires_grad]
    pilot_params = [p for p in pilot.parameters() if p.requires_grad]
    params = lora_params + pilot_params
    return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999),
                               weight_decay=0.0), params


# ---------------------------------------------------------------------------
def encode_batch(enc, builder, backbone, frames, instructions, pilot_inputs):
    """Eq. 4 -> Eq. 12 (Pilot slot TEACHER-FORCED with v_bar_{t+1}) -> padded."""
    with torch.no_grad():
        v_ts = enc.encode_batch(frames)

    seqs = [
        builder.build(instr, v, pilot_input=pv.to(enc.model.device, torch.bfloat16))
        for instr, v, pv in zip(instructions, v_ts, pilot_inputs)
    ]
    pos = [backbone.build_position_ids(s) for s in seqs]
    if any(p is None for p in pos):
        pos = None
    return collate_sequences(seqs, pos)


def forward_stage1(base_model, head, pilot, batch):
    """Eq. 6 -> (Eq. 7 action logits, Eq. 8 pilot output).

    Returns (logits, z). Under Design B the two hidden states are distinct:
    h_pil sits at the Pilot slot, h_act at the trailing ACTION_QUERY.
    """
    out = base_model.model.language_model(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        use_cache=False,
    )
    H = out.last_hidden_state
    rows = torch.arange(H.shape[0], device=H.device)

    h_act = H[rows, batch.action_index]
    logits = head.logits(h_act)                       # Eq. 7

    h_pil = H[rows, batch.pilot_index]
    z = pilot(h_pil)                                  # Eq. 8
    return logits, z


def stage1_loss(logits, targets, z, vbar_t2, mask, lam,
                lambda_mode="fixed", target_ratio=0.9):
    """Eq. 13 + Eq. 14 -> Eq. 15, with a shared denominator (see module docstring).

    lambda_mode="fixed"    Eq. 15 exactly as written: lambda constant at 0.1.
    lambda_mode="balanced" NOT IN PAPER (D26). lambda_t is recomputed each step
                           to hold lambda*L_pil / L_act at `target_ratio`.

    WHY "balanced" EXISTS (D25). lambda=0.1 is balanced only at
    initialisation. L_act converges ~51x over Stage 1 while L_pil converges
    only ~4.8x, so the EFFECTIVE weighting drifts by an order of magnitude:

        step    50   lambda*L_pil / L_act = 0.52x   (action dominates)
        step 17400   lambda*L_pil / L_act = 5.51x   (Pilot dominates)

    By the end the model spends 5.5x more gradient predicting future frames
    than choosing actions -- and closed-loop navigation degrades accordingly
    (OS 0.267 at step10000 -> 0.10 at final). The identity control, whose ratio
    self-corrected to a stable 0.91x, navigated 4x better. "balanced" removes
    the drift by construction.

    Returns (total, l_act, l_pil, lam_used) -- lam_used is logged because under
    "balanced" it is no longer a constant and the run is uninterpretable
    without it.
    """
    B = logits.shape[0]
    l_act = action_loss(logits, targets, reduction="mean")        # == sum/B
    l_pil = pilot_loss(z, vbar_t2, mask, reduction="sum") / B     # NOT /num_valid

    if lambda_mode == "fixed":
        lam_used = lam
    elif lambda_mode == "balanced":
        # Detached: this sets the WEIGHT, it must not itself be differentiated.
        lam_used = float(
            target_ratio * l_act.detach().item()
            / max(l_pil.detach().item(), 1e-8)
        )
        lam_used = min(max(lam_used, LAMBDA_MIN), LAMBDA_MAX)
    else:
        raise ValueError(f"unknown lambda_mode {lambda_mode!r}")

    return combined_loss(l_act, l_pil, lam_used), l_act, l_pil, lam_used


def make_loader(dataset, batch_size, shuffle=True, workers=4, seed=0):
    from torch.utils.data import DataLoader

    d = dataset.vbars[0].shape[-1]

    def collate(items):
        # Eq. 14 targets: v_bar_{t+2}, MASKED for the final two steps of each
        # episode. Masked rows carry a zero vector and mask=0, so they
        # contribute exactly zero -- they are not dropped (that would change B
        # and hence lambda) and not zero-filled as if they were real targets.
        tgts, masks = [], []
        for s in items:
            tv = dataset.pilot_target(s)
            if tv is None:
                tgts.append(np.zeros(d, dtype=np.float32))
                masks.append(0.0)
            else:
                tgts.append(np.asarray(tv, dtype=np.float32))
                masks.append(1.0)
        return (
            [load_frame(s.frame_path) for s in items],
            [s.instruction for s in items],
            [torch.from_numpy(dataset.pilot_input(s)) for s in items],
            torch.tensor([s.action for s in items], dtype=torch.long),
            torch.from_numpy(np.stack(tgts)),
            torch.tensor(masks, dtype=torch.float32),
        )

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate, generator=g,
                      drop_last=True, persistent_workers=workers > 0)


def save_checkpoint(enc, pilot, path: pathlib.Path):
    """Save the LoRA adapter + PilotModule ONLY.

    `enc.model` is the BASE model -- get_peft_model injected adapters in place
    and its wrapper was discarded -- so `save_pretrained` here writes the full
    4.9 GB backbone instead of the ~25 MB adapter. Stage 0' did exactly that
    and burned ~180 GB across 37 checkpoints. Filtering on "lora_" keeps the
    224 adapter tensors, which is everything that actually changed.
    """
    path.mkdir(parents=True, exist_ok=True)
    lora_sd = {k: v for k, v in enc.model.state_dict().items() if "lora_" in k}
    if not lora_sd:
        raise RuntimeError("no LoRA tensors found -- refusing to save an empty adapter")
    torch.save(lora_sd, path / "adapter.pt")
    torch.save(pilot.state_dict(), path / "pilot.pt")
    return len(lora_sd)


# ---------------------------------------------------------------------------
def assert_asymmetry(builder, pilot, enc, dataset):
    """Sec. 4.1: assert train mode holds v_bar_{t+1} and eval holds z_{t-1}.

    Checked by identity of the injected vector, not by shape -- both are (d,)
    so a shape check would pass even with the two swapped, which is precisely
    the bug this guards against.
    """
    step = dataset[0]
    v = torch.zeros(enc.n_v, enc.d, dtype=torch.bfloat16, device=enc.model.device)

    vbar_t1 = torch.from_numpy(np.asarray(dataset.pilot_input(step), dtype=np.float32))
    vbar_t1 = vbar_t1.to(enc.model.device, torch.bfloat16)
    seq_tr = builder.build(step.instruction, v, pilot_input=vbar_t1)
    got_train = seq_tr.inputs_embeds[0, seq_tr.pilot_index]
    if not torch.equal(got_train.float(), vbar_t1.float()):
        raise AssertionError("Eq. 12 violated: train slot is not v_bar_{t+1}")

    cache = pilot.new_cache()
    z_prev = cache.read()
    seq_ev = builder.build(step.instruction, v, pilot_input=z_prev)
    got_eval = seq_ev.inputs_embeds[0, seq_ev.pilot_index]
    if not torch.equal(got_eval.float(), z_prev.float()):
        raise AssertionError("Eq. 5 violated: eval slot is not z_{t-1}")

    if torch.equal(vbar_t1.float(), z_prev.float()):
        raise AssertionError(
            "train and eval slots are identical -- the Eq.12/Eq.5 asymmetry "
            "is not actually being exercised"
        )
    return True


def report_scale(args):
    """Preflight: is lambda=0.1 balancing the two terms? (D18)"""
    dataset = PilotRolloutDataset(args.split, max_episodes=16)
    mean_vbar = compute_mean_vbar(dataset)
    enc, pilot = build_model(args.lora_r, args.lora_alpha, args.lora_dropout,
                             mean_vbar, grad_checkpointing=False,
                             pilot_mode=args.pilot_mode, g_psi=args.g_psi,
                             g_psi_init=args.g_psi_init)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)

    assert_asymmetry(builder, pilot, enc, dataset)
    print("Eq.12/Eq.5 asymmetry: OK\n")

    frames, instrs, pin, tgt, vb2, mask = next(iter(make_loader(dataset, 8, workers=0)))
    with torch.no_grad():
        batch = encode_batch(enc, builder, backbone, frames, instrs, pin)
        logits, z = forward_stage1(enc.model, head, pilot, batch)
        loss, l_act, l_pil, _ = stage1_loss(logits, tgt.to("cuda"), z,
                                            vb2.to("cuda"), mask.to("cuda"), args.lam,
                                            args.lambda_mode, args.target_ratio)
    rep = pilot_loss_scale_report(z, vb2.to("cuda"), mask.to("cuda"),
                                  l_act=l_act, lam=args.lam)
    print(f"g_psi        = {args.g_psi}")
    print(f"L_act        = {l_act.item():10.4f}")
    print(f"L_pil (/B)   = {l_pil.item():10.4f}")
    print(f"lam*L_pil    = {(args.lam * l_pil).item():10.4f}")
    print(f"TOTAL        = {loss.item():10.4f}")
    print(f"ratio lam*L_pil : L_act = {(args.lam * l_pil / l_act).item():.2f}x")
    print(f"||z||        = {rep['z_norm']:.3f}")
    print(f"||v_bar_t+2||= {rep['target_norm']:.3f}")
    print(f"valid steps  = {int(mask.sum().item())}/{len(mask)}")
    print("\nIf the ratio is >>1 the action objective is drowned; if <<1 the "
          "Pilot Token gets no gradient and Stage 1 reduces to Stage 0'.")
    return 0


# ---------------------------------------------------------------------------
def smoke_test(args):
    print("VRAM SMOKE TEST — Stage 1 (L = L_act + lambda*L_pil)\n")
    dataset = PilotRolloutDataset(args.split, max_episodes=8)
    print(dataset.summary())
    print(f"Eq.14 target coverage: {dataset.target_coverage():.3f}\n")
    mean_vbar = compute_mean_vbar(dataset)

    enc, pilot = build_model(args.lora_r, args.lora_alpha, args.lora_dropout,
                             mean_vbar, grad_checkpointing=not args.no_grad_checkpointing,
                             pilot_mode=args.pilot_mode, g_psi=args.g_psi,
                             g_psi_init=args.g_psi_init)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)
    opt, params = make_optimizer(enc.model, pilot, args.lr)

    assert_asymmetry(builder, pilot, enc, dataset)
    print("Eq.12/Eq.5 asymmetry: OK")
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    print(f"  G_psi trainable: {pilot.G_psi.weight.requires_grad}\n")

    print(f"{'batch':>6} {'peak GB':>9} {'reserved GB':>12} {'s/step':>8}  status")
    print("-" * 52)
    for bs in args.smoke_batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            it = iter(make_loader(dataset, bs, workers=0))
            t0 = None
            for i in range(3):
                frames, instrs, pin, tgt, vb2, mask = next(it)
                if i == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                batch = encode_batch(enc, builder, backbone, frames, instrs, pin)
                logits, z = forward_stage1(enc.model, head, pilot, batch)
                loss, _, _, _ = stage1_loss(logits, tgt.to("cuda"), z, vb2.to("cuda"),
                                            mask.to("cuda"), args.lam,
                                            args.lambda_mode, args.target_ratio)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 2
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            res = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"{bs:>6} {peak:>9.2f} {res:>12.2f} {dt:>8.2f}  OK")
        except torch.OutOfMemoryError:
            print(f"{bs:>6} {'-':>9} {'-':>12} {'-':>8}  OOM")
            torch.cuda.empty_cache()
            break
    print("-" * 52)

    print("\nVerifying G_psi RECEIVES gradient (this is what Stage 0' lacked):")
    frames, instrs, pin, tgt, vb2, mask = next(iter(make_loader(dataset, 4, workers=0)))
    batch = encode_batch(enc, builder, backbone, frames, instrs, pin)
    logits, z = forward_stage1(enc.model, head, pilot, batch)
    loss, _, _, _ = stage1_loss(logits, tgt.to("cuda"), z, vb2.to("cuda"),
                                mask.to("cuda"), args.lam,
                                args.lambda_mode, args.target_ratio)
    loss.backward()
    g = pilot.G_psi.weight.grad
    if args.g_psi == "identity":
        print(f"  G_psi.weight.grad is None: {g is None}  (frozen copy baseline — correct)")
    else:
        ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        print(f"  G_psi.weight.grad nonzero+finite: {bool(ok)} "
              f"({'CORRECT' if ok else 'BROKEN — L_pil is not reaching G_psi'})")
    opt.zero_grad(set_to_none=True)
    return 0


# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PilotRolloutDataset(args.split, max_episodes=args.max_episodes)
    print(dataset.summary())
    print(f"Eq.14 target coverage: {dataset.target_coverage():.3f} "
          f"(steps with a v_bar_{{t+2}}; the rest are masked)\n")
    mean_vbar = compute_mean_vbar(dataset)
    print(f"mean ||v_bar|| = {mean_vbar.norm().item():.3f}\n")

    counts = dataset.action_counts()
    n = sum(counts.values())
    prior = {Action(a).name: c / n for a, c in counts.items()}
    entropy = -sum(p * math.log(p) for p in prior.values())
    print("action prior: " + "  ".join(f"{k}={v:.3f}" for k, v in prior.items()))
    print(f"BASELINES  loss: uniform={math.log(4):.4f}  marginal={entropy:.4f}\n")

    enc, pilot = build_model(args.lora_r, args.lora_alpha, args.lora_dropout,
                             mean_vbar, grad_checkpointing=not args.no_grad_checkpointing,
                             pilot_mode=args.pilot_mode, g_psi=args.g_psi,
                             g_psi_init=args.g_psi_init)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)
    opt, params = make_optimizer(enc.model, pilot, args.lr)
    sched = make_scheduler(opt, args.warmup, args.steps, schedule=args.schedule,
                           min_lr_frac=args.min_lr_frac)

    if args.assert_asymmetry:
        assert_asymmetry(builder, pilot, enc, dataset)
        print("Eq.12/Eq.5 asymmetry: OK  (train=v_bar_{t+1}, eval=z_{t-1})")

    print(f"pilot_mode={args.pilot_mode}   g_psi={args.g_psi}   lambda={args.lam}")
    print(describe(args.warmup, args.steps, args.lr, args.schedule, args.min_lr_frac))
    print(f"epochs over {len(dataset):,} steps at batch {args.batch_size}: "
          f"{args.steps * args.batch_size / len(dataset):.2f}\n")

    loader = make_loader(dataset, args.batch_size, workers=args.workers, seed=args.seed)
    enc.model.train()
    pilot.train()

    step, t0 = 0, time.time()
    run = {"loss": 0.0, "act": 0.0, "pil": 0.0, "lam": 0.0, "ratio": 0.0}
    run_correct, run_n, run_valid = 0, 0, 0
    per_action, per_action_correct = Counter(), Counter()
    log = []

    while step < args.steps:
        for frames, instrs, pin, targets, vb2, mask in loader:
            if step >= args.steps:
                break
            targets = targets.to("cuda")
            vb2, mask = vb2.to("cuda"), mask.to("cuda")

            batch = encode_batch(enc, builder, backbone, frames, instrs, pin)
            logits, z = forward_stage1(enc.model, head, pilot, batch)
            loss, l_act, l_pil, lam_used = stage1_loss(
                logits, targets, z, vb2, mask, args.lam,
                args.lambda_mode, args.target_ratio)

            (loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            pred = logits.argmax(-1)
            run["loss"] += loss.item()
            run["act"] += l_act.item()
            run["pil"] += l_pil.item()
            run["lam"] += lam_used
            run["ratio"] += lam_used * l_pil.item() / max(l_act.item(), 1e-8)
            run_correct += (pred == targets).sum().item()
            run_n += targets.numel()
            run_valid += int(mask.sum().item())
            for t, p in zip(targets.tolist(), pred.tolist()):
                per_action[t] += 1
                per_action_correct[t] += int(t == p)
            step += 1

            if step % args.log_every == 0:
                k = args.log_every
                acc = run_correct / max(run_n, 1)
                stop_n = per_action[int(Action.STOP)]
                stop_rec = (per_action_correct[int(Action.STOP)] / stop_n
                            if stop_n else float("nan"))
                rate = step / (time.time() - t0)
                print(f"step {step:6d}  L {run['loss']/k:8.4f}  "
                      f"L_act {run['act']/k:.4f}  L_pil {run['pil']/k:9.3f}  "
                      f"acc {acc:.3f}  STOP-rec {stop_rec:.3f}(n={stop_n})  "
                      f"|z| {z.float().norm(dim=-1).mean().item():6.2f}  "
                      f"lam {run['lam']/k:.4f} ratio {run['ratio']/k:5.2f}x  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {rate:.2f} it/s")
                log.append({"step": step, "loss": run["loss"]/k,
                            "l_act": run["act"]/k, "l_pil": run["pil"]/k,
                            "acc": acc, "stop_recall": stop_rec,
                            "lam": run["lam"]/k, "ratio": run["ratio"]/k,
                            "z_norm": z.float().norm(dim=-1).mean().item()})
                run = {"loss": 0.0, "act": 0.0, "pil": 0.0, "lam": 0.0, "ratio": 0.0}
                run_correct, run_n, run_valid = 0, 0, 0
                per_action.clear()
                per_action_correct.clear()

            if args.save_every and step % args.save_every == 0:
                n_t = save_checkpoint(enc, pilot, out_dir / f"step{step}")
                print(f"  saved adapter ({n_t} tensors) -> {out_dir / f'step{step}'}")

    n_t = save_checkpoint(enc, pilot, out_dir / "final")
    (out_dir / "trainlog.json").write_text(json.dumps(log, indent=2))
    print(f"\nsaved adapter ({n_t} tensors) + pilot -> {out_dir / 'final'}")
    print(f"log -> {out_dir / 'trainlog.json'}")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 1 — Pilot Token with L_pil (Eq. 15)")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--report-scale", action="store_true",
                    help="Preflight: print L_act vs lambda*L_pil and exit (D18)")
    ap.add_argument("--smoke-batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=pathlib.Path, default=_ROOT / "checkpoints" / "stage1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=17400)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--schedule", default="cosine",
                    choices=["cosine", "linear", "constant"])
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=LAMBDA_PIL,
                    help="Eq. 15 lambda. STATED as 0.1 in Sec. 3.3 — changing "
                         "it is a Sec. 0.4 STOP-and-ask trigger.")
    ap.add_argument("--lambda-mode", default="fixed", choices=["fixed", "balanced"],
                    help="D26, NOT IN PAPER. 'fixed' is Eq. 15 as written. "
                         "'balanced' recomputes lambda each step to hold "
                         "lambda*L_pil/L_act at --target-ratio, removing the "
                         "0.52x -> 5.51x drift measured in D25.")
    ap.add_argument("--target-ratio", type=float, default=0.9,
                    help="NOT IN PAPER. Target lambda*L_pil : L_act under "
                         "--lambda-mode balanced. 0.9 matches the stable ratio "
                         "the identity control settled at while navigating 4x "
                         "better than the learned run.")
    ap.add_argument("--g-psi", default="learned", choices=["learned", "identity"],
                    help="'identity' = Sec. 4.2 gate (b) frozen copy baseline")
    ap.add_argument("--g-psi-init", default="zero", choices=["zero", "default"],
                    help="D20. 'zero' starts z at mean v_bar so L_pil begins at "
                         "the constant-predictor loss and lambda=0.1 is "
                         "meaningful. 'default' reproduces nn.Linear's init, "
                         "which makes lambda*L_pil 168x L_act.")
    ap.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    ap.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot-mode", default="action_query",
                    choices=["last", "action_query"],
                    help="Design B ('action_query') is the default: D19 plus "
                         "Fig. 2's separate h^pil/h^act arrows.")
    ap.add_argument("--no-assert-asymmetry", dest="assert_asymmetry",
                    action="store_false")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1
    if args.report_scale:
        return report_scale(args)
    return smoke_test(args) if args.smoke_test else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
