"""
Stage 0 training — reproduce Table 3's "NaN" ablation row.

No Pilot Token. Eq. 4 -> Eq. 5 -> Eq. 6 -> Eq. 7, optimised with Eq. 13's
action cross-entropy alone.

Per AGENTS.md Sec. 3.3: LoRA (r=16-32) on attention projections, bf16,
gradient checkpointing, 8-bit AdamW.

    conda activate latentpilot

    # ALWAYS run this first -- AGENTS.md forbids assuming a batch size
    python src/train/train_stage0.py --smoke-test

    # then train at whatever batch size the smoke test showed fits
    python src/train/train_stage0.py --batch-size N --steps 2000

--------------------------------------------------------------------------
WHAT LORA TOUCHES, AND WHAT IT MUST NOT
--------------------------------------------------------------------------
Adapters go on the LANGUAGE model's q/k/v/o projections only. The vision
encoder stays frozen (DECISION D2): from Stage 1 the same encoder produces the
Pilot target v_bar_{t+2}, and a trainable target invites collapse.

The visual tower's own projections are named `qkv` and `proj`, so the four
explicit names below cannot match them -- but the count is ASSERTED at build
time rather than assumed, because a silent match there would quietly unfreeze
the encoder.

--------------------------------------------------------------------------
LOSS REDUCTION
--------------------------------------------------------------------------
Batches are STEPS, not trajectories, and the loss is the mean over the batch.
For Stage 0 that is equivalent to Eq. 15 up to a scalar that folds into the
learning rate -- see the long note in src/data/rollout_dataset.py. The
equivalence relies on Stage 0 steps being independent and STOPS HOLDING at
Stage 2.

--------------------------------------------------------------------------
READ THE LOSS AGAINST THE ACTION PRIOR, NOT ln(4)
--------------------------------------------------------------------------
The expert action distribution is heavily imbalanced (FWD ~64%, STOP ~2.4%),
so the reference points are:

    LOSS      uniform ln(4)       = 1.386
              marginal H(actions) = 0.974   <- predicting the PRIOR gets here
              perfect             = 0
    ACCURACY  majority class FWD  = ~0.64   <- reachable while input-blind

A loss that falls to ~0.97 and an accuracy near 0.64 mean the model has learned
the action prior and NOTHING about the image or instruction. That is the
plateau to watch for, not a success.

(Note -ln(p_FWD) is NOT an "always-FWD loss": a policy that puts probability 1
on FWD assigns probability 0 to the other 36% of steps, so its cross-entropy is
infinite. Majority class is an ACCURACY baseline only.)

STOP is reported separately because it is 2.4% of steps yet decides SR
outright: predicted late or never, the episode fails however good the other
~39 actions were.
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import argparse
import json
import math
import pathlib
import time
from collections import Counter

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))

from data.rollout_dataset import RolloutDataset, load_frame  # noqa: E402
from losses.action_loss import action_loss  # noqa: E402
from model.action_head import ActionHead  # noqa: E402
from model.action_space import ACTIONS, NUM_ACTIONS, Action  # noqa: E402
from model.backbone import Backbone  # noqa: E402
from model.input_sequence import InputSequenceBuilder, collate_sequences  # noqa: E402
from model.vision_encoder import VisionEncoder  # noqa: E402
from train.schedule import describe, make_scheduler  # noqa: E402

# NOT IN PAPER -- the paper states none of these (AGENTS.md Sec. 8 item 3).
DEFAULT_LORA_R = 16          # AGENTS.md Sec. 3.3 allows 16-32; start low
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LR = 1e-4
DEFAULT_WARMUP = 100
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPECTED_ADAPTED_MODULES = 112   # 28 language layers x 4 projections


# ---------------------------------------------------------------------------
def build_model(lora_r, lora_alpha, lora_dropout, grad_checkpointing=True):
    """Load the backbone, attach LoRA to the LANGUAGE attention projections.

    Returns (encoder, peft_wrapper). NOTE the encoder keeps the ORIGINAL model
    object, not the PeftModel: peft injects adapter layers IN PLACE into the
    base module tree (verified -- `peft.get_base_model() is base` and the base
    object reports the LoRA params as trainable), so the base forward already
    applies them.

    That matters because PeftModel adds a nesting level: `peft.model` is the
    Qwen3VLForConditionalGeneration, so `peft.model.model` is no longer the
    Qwen3VLModel that owns `get_vision_position_ids` and `get_image_features`.
    Routing structural calls through the wrapper raises AttributeError. The
    wrapper is returned solely for `save_pretrained`.
    """
    from peft import LoraConfig, get_peft_model

    enc = VisionEncoder()                       # frozen, eval mode (D2)
    model = enc.model

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, cfg)

    # Assert the adapters landed where intended. A silent match inside the
    # vision tower would unfreeze the encoder and break D2 with no error.
    adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    visual_adapted = [n for n in adapted if "visual" in n]
    if visual_adapted:
        raise RuntimeError(
            f"LoRA attached to {len(visual_adapted)} VISION modules "
            f"(e.g. {visual_adapted[0]}). The encoder must stay frozen (D2)."
        )
    if len(adapted) != EXPECTED_ADAPTED_MODULES:
        raise RuntimeError(
            f"LoRA adapted {len(adapted)} modules, expected "
            f"{EXPECTED_ADAPTED_MODULES}. Target modules or the backbone changed."
        )

    if grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # enc.model deliberately stays the BASE model (adapters are in it already).
    return enc, model


def make_optimizer(model, lr):
    """8-bit AdamW (AGENTS.md Sec. 3.3, mandatory from Stage 3, used here too)."""
    import bitsandbytes as bnb

    params = [p for p in model.parameters() if p.requires_grad]
    return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999),
                               weight_decay=0.0), params


# ---------------------------------------------------------------------------
def encode_batch(enc, builder, backbone, frames, instructions):
    """Eq. 4 -> Eq. 5 -> padded batch, with matching mRoPE position ids."""
    # Eq. 4 -- frozen encoder, no gradient.
    with torch.no_grad():
        v_ts = enc.encode_batch(frames)

    seqs = [builder.build(instr, v) for instr, v in zip(instructions, v_ts)]
    pos = [backbone.build_position_ids(s) for s in seqs]
    if any(p is None for p in pos):
        pos = None
    return collate_sequences(seqs, pos)


def forward_loss(base_model, head, batch, targets):
    """Eq. 6 -> Eq. 7 -> Eq. 13 for one padded batch.

    Takes the BASE model (which carries the injected adapters), not the
    PeftModel wrapper -- see build_model's docstring.
    """
    out = base_model.model.language_model(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        use_cache=False,
    )
    H = out.last_hidden_state                       # (B, L, d)

    # h_t^act per row: the last REAL token, not a shared -1, because the batch
    # is right-padded to different lengths.
    idx = batch.action_index
    h_act = H[torch.arange(H.shape[0], device=H.device), idx]

    logits = head.logits(h_act)                     # (B, 4)
    loss = action_loss(logits, targets, reduction="mean")
    return loss, logits


def make_loader(dataset, batch_size, shuffle=True, workers=4, seed=0):
    from torch.utils.data import DataLoader

    def collate(items):
        return (
            [load_frame(s.frame_path) for s in items],
            [s.instruction for s in items],
            torch.tensor([s.action for s in items], dtype=torch.long),
        )

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate, generator=g,
                      drop_last=True, persistent_workers=workers > 0)


# ---------------------------------------------------------------------------
def smoke_test(args):
    """AGENTS.md Sec. 1: batch of 1, forward+backward, report PEAK allocated,
    before scaling to a target batch size. Sweeps sizes so the number is
    measured rather than assumed."""
    print("VRAM SMOKE TEST — forward + backward, peak allocated\n")
    dataset = RolloutDataset(args.split, max_episodes=8)
    print(dataset.summary(), "\n")

    enc, model = build_model(args.lora_r, args.lora_alpha, args.lora_dropout,
                             grad_checkpointing=not args.no_grad_checkpointing)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)
    opt, params = make_optimizer(model, args.lr)

    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params : {n_train:,} ({100.0 * n_train / n_total:.3f}% of {n_total:,})")
    print(f"grad checkpointing: {not args.no_grad_checkpointing}")
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} on {LORA_TARGETS}\n")

    print(f"{'batch':>6} {'peak GB':>9} {'reserved GB':>12} {'s/step':>8}  status")
    print("-" * 52)
    for bs in args.smoke_batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            loader = make_loader(dataset, bs, workers=0)
            it = iter(loader)
            t0 = None
            for i in range(3):                       # warm up, then time
                frames, instrs, targets = next(it)
                if i == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                batch = encode_batch(enc, builder, backbone, frames, instrs)
                loss, _ = forward_loss(enc.model, head, batch, targets.to("cuda"))
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
    print("\nPick the largest batch that fits with headroom, then:")
    print("  python src/train/train_stage0.py --batch-size <N> --steps 2000")
    return 0


# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RolloutDataset(args.split, max_episodes=args.max_episodes)
    print(dataset.summary())

    counts = dataset.action_counts()
    n = sum(counts.values())
    prior = {Action(a).name: c / n for a, c in counts.items()}
    entropy = -sum(p * math.log(p) for p in prior.values())
    print(f"action prior: " + "  ".join(f"{k}={v:.3f}" for k, v in prior.items()))
    print(f"BASELINES  loss: uniform={math.log(4):.4f}  marginal={entropy:.4f}  "
          f"perfect=0")
    print(f"           acc : majority-class(FWD)={prior['FWD']:.3f}")
    print(f"  -> loss ~{entropy:.2f} with acc ~{prior['FWD']:.2f} means the "
          f"PRIOR was learned and nothing else\n")

    enc, model = build_model(args.lora_r, args.lora_alpha, args.lora_dropout,
                             grad_checkpointing=not args.no_grad_checkpointing)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)
    opt, params = make_optimizer(model, args.lr)

    sched = make_scheduler(opt, args.warmup, args.steps,
                           schedule=args.schedule,
                           min_lr_frac=args.min_lr_frac)
    print(describe(args.warmup, args.steps, args.lr,
                   args.schedule, args.min_lr_frac))
    print(f"epochs over {len(dataset):,} steps at batch {args.batch_size}: "
          f"{args.steps * args.batch_size / len(dataset):.2f}\n")

    loader = make_loader(dataset, args.batch_size, workers=args.workers,
                         seed=args.seed)
    enc.model.train()

    step, t0 = 0, time.time()
    run_loss, run_correct, run_n = 0.0, 0, 0
    per_action = Counter()
    per_action_correct = Counter()
    log = []

    while step < args.steps:
        for frames, instrs, targets in loader:
            if step >= args.steps:
                break
            targets = targets.to("cuda")
            batch = encode_batch(enc, builder, backbone, frames, instrs)
            loss, logits = forward_loss(enc.model, head, batch, targets)

            (loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            pred = logits.argmax(-1)
            run_loss += loss.item()
            run_correct += (pred == targets).sum().item()
            run_n += targets.numel()
            for t, p in zip(targets.tolist(), pred.tolist()):
                per_action[t] += 1
                per_action_correct[t] += int(t == p)
            step += 1

            if step % args.log_every == 0:
                acc = run_correct / max(run_n, 1)
                avg = run_loss / args.log_every
                stop_n = per_action[int(Action.STOP)]
                stop_rec = (per_action_correct[int(Action.STOP)] / stop_n
                            if stop_n else float("nan"))
                rate = step / (time.time() - t0)
                print(f"step {step:6d}  loss {avg:.4f}  acc {acc:.3f}  "
                      f"STOP-recall {stop_rec:.3f} (n={stop_n})  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {rate:.2f} it/s")
                log.append({"step": step, "loss": avg, "acc": acc,
                            "stop_recall": stop_rec})
                run_loss, run_correct, run_n = 0.0, 0, 0
                per_action.clear()
                per_action_correct.clear()

            if args.save_every and step % args.save_every == 0:
                ckpt = out_dir / f"step{step}"
                model.save_pretrained(ckpt)
                print(f"  saved adapter -> {ckpt}")

    final = out_dir / "final"
    model.save_pretrained(final)
    (out_dir / "trainlog.json").write_text(json.dumps(log, indent=2))
    print(f"\nsaved adapter -> {final}")
    print(f"log -> {out_dir / 'trainlog.json'}")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 0 LoRA training")
    ap.add_argument("--smoke-test", action="store_true",
                    help="measure VRAM across batch sizes and exit")
    ap.add_argument("--smoke-batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 12])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=pathlib.Path,
                    default=_ROOT / "checkpoints" / "stage0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation; AGENTS.md prefers this over "
                         "shortening the sequence")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--schedule", default="cosine",
                    choices=["cosine", "linear", "constant"],
                    help="LR decay after warmup. NOT IN PAPER. 'constant' "
                         "reproduces the original warmup-only behaviour.")
    ap.add_argument("--min-lr-frac", type=float, default=0.05,
                    help="LR floor as a fraction of peak. NOT IN PAPER.")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    ap.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1
    return smoke_test(args) if args.smoke_test else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
