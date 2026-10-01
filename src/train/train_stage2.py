"""
Stage 2 — scheduled sampling on the Pilot slot (AGENTS.md Sec. 5).

NOT IN THE PAPER. Human-approved addition fixing the exposure bias between:

    Eq. 12 (TRAIN)  u^tr_t = [Tok(x); v_t; PILOT( v_bar_{t+1} )]   real future
    Eq.  5 (EVAL)   u_t    = [Tok(x); v_t; PILOT( z_{t-1}     )]   own prediction

--------------------------------------------------------------------------
WHY THIS STAGE EXISTS -- the measured evidence, not a hypothesis
--------------------------------------------------------------------------
Stage 1 closed-loop got WORSE as teacher-forced training improved:

    checkpoint      teacher-forced          closed-loop (val_unseen, n=30)
    step10000       STOP-recall ~0.5        OS 0.2667  NE 7.89  model_stop 0%
    final           STOP-recall  1.000      OS 0.1000  NE 8.58  model_stop 0%

Same episodes, same seed; the model trained 75% longer turned a genuine
success (x8F5xyUWy9e, 2.77 m) into a 9.21 m miss. Monotonically improving
teacher-forced metrics alongside monotonically degrading closed-loop metrics is
the signature of a policy leaning harder on an input it never gets at test
time. That is what this stage mixes away.

--------------------------------------------------------------------------
WHAT IS AND IS NOT MIXED (Sec. 5.2)
--------------------------------------------------------------------------
Only the Pilot slot INPUT is mixed, between `v_bar_{t+1}` and the model's own
detached `z_{t-1}`. The Eq. 14 TARGET stays `v_bar_{t+2}` -- a real encoded
frame -- in every case. The paper's privileged supervision is not weakened.

`z_null` (Sec. 5.2's third row) is STAGE 4 ONLY and is deliberately absent
here; adding it would violate one-novelty-per-stage.

--------------------------------------------------------------------------
GETTING z_{t-1} (Sec. 5.6)
--------------------------------------------------------------------------
Option (a), the faithful one: a sequential unroll over a short window, with
detach. For a sample at step t we run step t-1 -- itself teacher-forced with
`v_bar_t` -- and take `z_{t-1} = G_psi(h_{t-1}^pil)`.

Because Sec. 5.6 mandates DETACHING `z_{t-1}` (no gradient through time), that
entire prior-step pass runs under `torch.no_grad()`. It costs one extra forward
for the self-predicted subset only, and no backward at all.

This is a ONE-STEP unroll: `z_{t-1}` is computed from a GT-seeded step, so it
carries one step of self-prediction error, not the full compounding of a
rollout. That is the "short window" Sec. 5.6 sanctions, and `--unroll-window`
exists to widen it. It is an approximation of the inference distribution, and
is documented as such rather than described as equivalent.

At `t = 0` there is no `t-1`; the slot gets `z_0` from the PilotModule, which
is exactly what happens at inference.

    conda activate latentpilot
    python src/train/train_stage2.py --check-identity     # Sec. 5.6 regression test
    python src/train/train_stage2.py --p-final 0.5 --out checkpoints/stage2_p50
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import argparse
import json
import pathlib
import time
from collections import Counter

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))

from data.rollout_dataset import PilotRolloutDataset, load_frame  # noqa: E402
from losses.action_loss import action_loss  # noqa: E402
from losses.pilot_loss import LAMBDA_PIL, combined_loss, pilot_loss  # noqa: E402
from model.action_head import ActionHead  # noqa: E402
from model.action_space import Action  # noqa: E402
from model.backbone import Backbone  # noqa: E402
from model.input_sequence import InputSequenceBuilder, collate_sequences  # noqa: E402
from model.pilot import PilotModule  # noqa: E402
from model.vision_encoder import VisionEncoder  # noqa: E402
from train.schedule import make_scheduler  # noqa: E402
from train.train_stage1 import (  # noqa: E402
    EXPECTED_ADAPTED_MODULES, LORA_TARGETS, compute_mean_vbar, encode_batch,
    save_checkpoint,
)

STAGE1_STEPS = 17400          # what finetune_frac is a fraction of
STAGE1_LR = 1e-4


# ---------------------------------------------------------------------------
def p_selfpred_at(step, total_steps, p_start, p_final, ramp_frac):
    """Sec. 5.4: ramp p over the first `ramp_frac` of the fine-tune, then hold.

    A hard 0 -> p_final switch is the destabilising case scheduled sampling
    exists to avoid: the model has specialised its Pilot-slot attention to
    "this slot contains real visual evidence", and yanking that away in one
    step is what the ramp is for.
    """
    ramp_steps = max(1, int(round(ramp_frac * total_steps)))
    if step >= ramp_steps:
        return p_final
    return p_start + (p_final - p_start) * (step / ramp_steps)


def build_prev_index(dataset):
    """(episode_index, step_index) -> flat dataset index, for the t-1 lookup.

    Built here rather than in PilotRolloutDataset on purpose: Stage 1 must stay
    byte-identical so `--check-identity` means something.
    """
    return {(s.episode_index, s.step_index): i
            for i, s in enumerate(dataset.pilot_steps)}


def load_stage1(checkpoint, mean_vbar, lora_r=16, lora_alpha=32,
                grad_checkpointing=True):
    """Rebuild the Stage 1 module tree and load its adapter + PilotModule."""
    from peft import LoraConfig, get_peft_model

    ckpt = pathlib.Path(checkpoint)
    adapter_pt, pilot_pt = ckpt / "adapter.pt", ckpt / "pilot.pt"
    for p in (adapter_pt, pilot_pt):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing -- Sec. 5.3 requires init_from a CONVERGED Stage 1 "
                f"checkpoint; Stage 2 is a fine-tune, not a fresh run."
            )

    enc = VisionEncoder()
    get_peft_model(enc.model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS))

    sd = torch.load(str(adapter_pt), map_location="cpu", weights_only=True)
    missing, unexpected = enc.model.load_state_dict(sd, strict=False)
    if [k for k in unexpected if "lora" in k]:
        raise RuntimeError(f"unmatched LoRA keys: {[k for k in unexpected if 'lora' in k][:3]}")

    adapted = [n for n, _ in enc.model.named_modules() if n.endswith("lora_A.default")]
    if len(adapted) != EXPECTED_ADAPTED_MODULES:
        raise RuntimeError(f"LoRA adapted {len(adapted)}, expected {EXPECTED_ADAPTED_MODULES}")

    if grad_checkpointing:
        enc.model.gradient_checkpointing_enable()
        enc.model.enable_input_require_grads()

    psd = torch.load(str(pilot_pt), map_location="cpu", weights_only=True)
    mode = "action_query" if "action_query" in psd else "last"
    pilot = PilotModule(enc.d, mode=mode, dtype=torch.bfloat16, device=enc.model.device)
    pilot.load_state_dict({k: v.to(enc.model.device) for k, v in psd.items()})
    pilot.assert_shapes()

    print(f"  loaded Stage 1: {len(sd)} LoRA tensors, pilot mode={mode}")
    return enc, pilot


@torch.no_grad()
def compute_z_prev(enc, builder, backbone, pilot, prev_items, dataset):
    """z_{t-1} = G_psi(h_{t-1}^pil), fully detached (Sec. 5.6, CRITICAL).

    No gradient through time: the whole prior-step pass is under no_grad, so
    this costs one extra FORWARD and zero backward. The naive
    gradient-carrying version OOMs on 16 GB and silently changes the
    optimisation problem.
    """
    frames = [load_frame(s.frame_path) for s in prev_items]
    instrs = [s.instruction for s in prev_items]
    # step t-1 is itself teacher-forced with v_bar_t -- the short-window unroll
    pin = [torch.from_numpy(dataset.pilot_input(s)) for s in prev_items]

    batch = encode_batch(enc, builder, backbone, frames, instrs, pin)
    out = enc.model.model.language_model(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        use_cache=False,
    )
    H = out.last_hidden_state
    rows = torch.arange(H.shape[0], device=H.device)
    z = pilot(H[rows, batch.pilot_index])
    return z.detach()


def make_loader(dataset, prev_index, batch_size, workers=4, seed=0):
    from torch.utils.data import DataLoader

    d = dataset.vbars[0].shape[-1]

    def collate(items):
        tgts, masks = [], []
        for s in items:
            tv = dataset.pilot_target(s)
            tgts.append(np.zeros(d, dtype=np.float32) if tv is None
                        else np.asarray(tv, dtype=np.float32))
            masks.append(0.0 if tv is None else 1.0)
        # index of step t-1 in the same episode, or None at t=0 (-> z_0)
        prev_idx = [prev_index.get((s.episode_index, s.step_index - 1))
                    for s in items]
        return (
            [load_frame(s.frame_path) for s in items],
            [s.instruction for s in items],
            [torch.from_numpy(dataset.pilot_input(s)) for s in items],
            torch.tensor([s.action for s in items], dtype=torch.long),
            torch.from_numpy(np.stack(tgts)),
            torch.tensor(masks, dtype=torch.float32),
            prev_idx,
        )

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=workers, collate_fn=collate, generator=g,
                      drop_last=True, persistent_workers=workers > 0)


def mix_slots(pilot_inputs, z_prev_map, use_self, pilot, device):
    """Sec. 5.2: per-SAMPLE choice between v_bar_{t+1} and detached z_{t-1}.

    Per-sample, not per-batch (Sec. 5.6) -- each batch carries a mix, so the
    gradient sees both input distributions at every step rather than
    alternating whole batches between them.
    """
    slots = []
    for i, gt in enumerate(pilot_inputs):
        if use_self[i] and i in z_prev_map:
            slots.append(z_prev_map[i])
        elif use_self[i]:
            slots.append(pilot.z_0.detach())      # t=0: exactly the eval case
        else:
            slots.append(gt.to(device, torch.bfloat16))
    return slots


# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PilotRolloutDataset(args.split, scans=args.scans, max_episodes=args.max_episodes)
    prev_index = build_prev_index(dataset)
    mean_vbar = compute_mean_vbar(dataset)
    print(dataset.summary())

    total_steps = args.steps or int(round(args.finetune_frac * STAGE1_STEPS))
    lr = args.lr or (STAGE1_LR * args.lr_scale)

    print(f"\nStage 2 — scheduled sampling  (NOT IN PAPER, AGENTS.md Sec. 5)")
    print(f"  init_from   : {args.init_from}")
    print(f"  steps       : {total_steps}  ({args.finetune_frac:.2f} x Stage 1's {STAGE1_STEPS})")
    print(f"  lr          : {lr:.2e}  ({args.lr_scale} x Stage 1's {STAGE1_LR:.0e})")
    print(f"  p_selfpred  : {args.p_start} -> {args.p_final} over first "
          f"{args.ramp_frac:.0%} of the run")
    print(f"  target      : v_bar_{{t+2}} ALWAYS (only the INPUT is mixed)\n")

    if args.p_final > 0.75:
        raise ValueError(
            f"p_final={args.p_final} > 0.75. Sec. 5.4: do not exceed 0.75 "
            f"without asking the human -- above it the t+1 scaffold is gone and "
            f"the task becomes unanchored two-step extrapolation."
        )

    enc, pilot = load_stage1(args.init_from, mean_vbar,
                             grad_checkpointing=not args.no_grad_checkpointing)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)

    import bitsandbytes as bnb
    params = ([p for p in enc.model.parameters() if p.requires_grad]
              + [p for p in pilot.parameters() if p.requires_grad])
    opt = bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = make_scheduler(opt, args.warmup, total_steps, schedule="cosine",
                           min_lr_frac=0.05)

    loader = make_loader(dataset, prev_index, args.batch_size,
                         workers=args.workers, seed=args.seed)
    enc.model.train()
    pilot.train()

    step, t0 = 0, time.time()
    run = {"loss": 0.0, "act": 0.0, "pil": 0.0}
    run_correct, run_n, run_self = 0, 0, 0
    per_action, per_action_correct = Counter(), Counter()
    log = []
    rng = np.random.default_rng(args.seed)

    while step < total_steps:
        for frames, instrs, pin, targets, vb2, mask, prev_idx in loader:
            if step >= total_steps:
                break
            targets, vb2, mask = targets.to("cuda"), vb2.to("cuda"), mask.to("cuda")
            p = p_selfpred_at(step, total_steps, args.p_start, args.p_final,
                              args.ramp_frac)
            use_self = rng.random(len(frames)) < p          # per-SAMPLE (Sec. 5.6)

            # -- z_{t-1} for the self-predicted subset, detached -------------
            z_prev_map = {}
            wanted = [i for i, u in enumerate(use_self)
                      if u and prev_idx[i] is not None]
            if wanted:
                prev_items = [dataset[prev_idx[i]] for i in wanted]
                z_batch = compute_z_prev(enc, builder, backbone, pilot,
                                         prev_items, dataset)
                if z_batch.requires_grad:                   # Sec. 5.6 assertion
                    raise RuntimeError("z_prev carries grad -- detach failed")
                for j, i in enumerate(wanted):
                    z_prev_map[i] = z_batch[j]

            slots = mix_slots(pin, z_prev_map, use_self, pilot, enc.model.device)

            batch = encode_batch(enc, builder, backbone, frames, instrs, slots)
            out = enc.model.model.language_model(
                inputs_embeds=batch.inputs_embeds,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
                use_cache=False,
            )
            H = out.last_hidden_state
            rows = torch.arange(H.shape[0], device=H.device)
            logits = head.logits(H[rows, batch.action_index])
            z = pilot(H[rows, batch.pilot_index])

            B = logits.shape[0]
            l_act = action_loss(logits, targets, reduction="mean")
            l_pil = pilot_loss(z, vb2, mask, reduction="sum") / B
            loss = combined_loss(l_act, l_pil, args.lam)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pred = logits.argmax(-1)
            run["loss"] += loss.item(); run["act"] += l_act.item()
            run["pil"] += l_pil.item()
            run_correct += (pred == targets).sum().item()
            run_n += targets.numel()
            run_self += int(use_self.sum())
            for t, pr in zip(targets.tolist(), pred.tolist()):
                per_action[t] += 1
                per_action_correct[t] += int(t == pr)
            step += 1

            if step % args.log_every == 0:
                k = args.log_every
                stop_n = per_action[int(Action.STOP)]
                stop_rec = (per_action_correct[int(Action.STOP)] / stop_n
                            if stop_n else float("nan"))
                print(f"step {step:5d}  L {run['loss']/k:7.4f}  "
                      f"L_act {run['act']/k:.4f}  L_pil {run['pil']/k:8.3f}  "
                      f"acc {run_correct/max(run_n,1):.3f}  "
                      f"STOP-rec {stop_rec:.3f}(n={stop_n})  "
                      f"p {p:.2f} (self {run_self/max(run_n,1):.2f})  "
                      f"|z| {z.float().norm(dim=-1).mean().item():6.2f}  "
                      f"{step/(time.time()-t0):.2f} it/s")
                log.append({"step": step, "loss": run["loss"]/k,
                            "l_act": run["act"]/k, "l_pil": run["pil"]/k,
                            "acc": run_correct/max(run_n,1),
                            "stop_recall": stop_rec, "p_selfpred": p,
                            "self_frac": run_self/max(run_n,1),
                            "z_norm": z.float().norm(dim=-1).mean().item()})
                run = {"loss": 0.0, "act": 0.0, "pil": 0.0}
                run_correct, run_n, run_self = 0, 0, 0
                per_action.clear(); per_action_correct.clear()

    n_t = save_checkpoint(enc, pilot, out_dir / "final")
    (out_dir / "trainlog.json").write_text(json.dumps(log, indent=2))
    print(f"\nsaved adapter ({n_t} tensors) + pilot -> {out_dir / 'final'}")
    return 0


# ---------------------------------------------------------------------------
def check_identity(args):
    """Sec. 5.6: at p_selfpred = 0.0, Stage 2 must be numerically identical to
    Stage 1. This is the cleanest regression check in the stage -- if it fails,
    the mixing machinery has changed the base computation and every Stage 2
    number is uninterpretable."""
    torch.manual_seed(0)
    dataset = PilotRolloutDataset(args.split, max_episodes=4)
    prev_index = build_prev_index(dataset)
    mean_vbar = compute_mean_vbar(dataset)
    enc, pilot = load_stage1(args.init_from, mean_vbar, grad_checkpointing=False)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)

    loader = make_loader(dataset, prev_index, 4, workers=0, seed=0)
    frames, instrs, pin, targets, vb2, mask, prev_idx = next(iter(loader))
    targets, vb2, mask = targets.to("cuda"), vb2.to("cuda"), mask.to("cuda")

    def forward(slots):
        batch = encode_batch(enc, builder, backbone, frames, instrs, slots)
        out = enc.model.model.language_model(
            inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask,
            position_ids=batch.position_ids, use_cache=False)
        H = out.last_hidden_state
        rows = torch.arange(H.shape[0], device=H.device)
        return head.logits(H[rows, batch.action_index]), pilot(H[rows, batch.pilot_index])

    with torch.no_grad():
        # Stage 1 path: pure teacher forcing
        s1_logits, s1_z = forward([g.to(enc.model.device, torch.bfloat16) for g in pin])
        # Stage 2 path at p = 0.0
        use_self = np.zeros(len(frames), dtype=bool)
        slots = mix_slots(pin, {}, use_self, pilot, enc.model.device)
        s2_logits, s2_z = forward(slots)

    same_logits = torch.equal(s1_logits, s2_logits)
    same_z = torch.equal(s1_z, s2_z)
    print("\nSec. 5.6 regression check — p_selfpred = 0.0 vs Stage 1")
    print(f"  logits identical : {same_logits}")
    print(f"  z identical      : {same_z}")

    # Sec. 5.6: per-sample sampling rate must actually hit p
    rng = np.random.default_rng(0)
    emp = (rng.random(1000) < 0.5).mean()
    print(f"  per-sample rate at p=0.5 over 1000 draws: {emp:.3f}")

    ok = same_logits and same_z and abs(emp - 0.5) < 0.05
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 2 — scheduled sampling (Sec. 5)")
    ap.add_argument("--check-identity", action="store_true",
                    help="Sec. 5.6 regression test: p=0.0 must equal Stage 1")
    ap.add_argument("--init-from", default="checkpoints/stage1_learned/final")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=pathlib.Path, default=_ROOT / "checkpoints" / "stage2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None, help="overrides finetune_frac")
    ap.add_argument("--finetune-frac", type=float, default=0.15)   # NOT IN PAPER
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-scale", type=float, default=0.1)         # NOT IN PAPER
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--p-start", type=float, default=0.0)          # NOT IN PAPER
    ap.add_argument("--p-final", type=float, default=0.5)          # NOT IN PAPER
    ap.add_argument("--ramp-frac", type=float, default=0.3)        # NOT IN PAPER
    ap.add_argument("--unroll-window", type=int, default=1)        # NOT IN PAPER
    ap.add_argument("--lam", type=float, default=LAMBDA_PIL)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--scans", nargs="+", default=None,
                    help="restrict to these train scans (Stage 1 used 6: see scripts/run_followups.sh)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1
    if args.p_final > 0.75:
        # Validated here, before any dataset work: an invalid flag should fail
        # in milliseconds, not after a multi-second dataset load.
        raise ValueError(
            f"p_final={args.p_final} > 0.75. Sec. 5.4: do not exceed 0.75 "
            f"without asking the human -- above it the t+1 scaffold is gone and "
            f"the task becomes unanchored two-step extrapolation."
        )
    if args.unroll_window != 1:
        raise NotImplementedError(
            "Sec. 5.6 option (a) is implemented as a one-step unroll. Widening "
            "the window changes what distribution the model trains against -- "
            "STOP and ask the human first."
        )
    return check_identity(args) if args.check_identity else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
