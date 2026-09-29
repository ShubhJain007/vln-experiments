"""
Stage 0' training — reproduce Table 3's "NaN" ablation row, CORRECTLY this
time (see D15).

Per the paper's supplementary Sec. A.1, the NaN variant is NOT memoryless:
"we still insert <|placeholder|> to initialize the Pilot Token and still
propagate the Pilot Token across steps exactly as usual. The only difference
is that we do not apply any dedicated loss on the Pilot Token itself."

So Stage 0' = Stage 0's action objective, PLUS:
  - Eq. 5's full form: [<vs>; v_t; <ve>; Tok(x); PILOT(z)]  (Design A, D17)
  - Eq. 12: the Pilot slot is TEACHER-FORCED with cached v_bar_{t+1} during
    training (independent steps, same batching as Stage 0 -- see the long
    note in rollout_dataset.py)
  - Recurrent PilotCache at INFERENCE only (rollout, not training)
  - Eq. 8's G_psi exists and is exercised at inference, but receives NO
    gradient during training: L = L_act only, and L_act does not touch G_psi
    at all under Design A (W_LM and G_psi are two separate heads reading the
    same h_act == h_pil). G_psi therefore sits at its D18-initialised value
    (bias = mean v_bar) for the whole run. This is not a bug -- it is the
    ablation's entire point (Sec. 3.4: "tests whether the propagated latent
    becomes predictive on its own").

    conda activate latentpilot
    python src/train/train_stage0_prime.py --smoke-test
    python src/train/train_stage0_prime.py --batch-size 8 --steps 8000
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
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPECTED_ADAPTED_MODULES = 112


# ---------------------------------------------------------------------------
def compute_mean_vbar(dataset: PilotRolloutDataset) -> torch.Tensor:
    """Mean v_bar across the cache, for D18's bias-init trick."""
    return torch.from_numpy(np.concatenate(dataset.vbars).mean(axis=0))


def build_model(lora_r, lora_alpha, lora_dropout, mean_vbar,
                grad_checkpointing=True, pilot_mode="last"):
    """Backbone + LoRA (identical to Stage 0) + a PilotModule (Design A).

    See train_stage0.py's build_model docstring for why the PEFT wrapper's
    return value is discarded -- adapters land in place on the base model.
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
    pilot.init_bias_from_target_mean(mean_vbar.to(torch.bfloat16))     # D18

    return enc, pilot


def make_optimizer(model, pilot, lr):
    import bitsandbytes as bnb

    lora_params = [p for p in model.parameters() if p.requires_grad]
    pilot_params = list(pilot.parameters())     # z_0 + G_psi (see module docstring:
                                                # G_psi gets NO gradient in Stage 0',
                                                # but including it is harmless and
                                                # keeps this script reusable for Stage 1)
    params = lora_params + pilot_params
    return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999),
                               weight_decay=0.0), params


# ---------------------------------------------------------------------------
def encode_batch(enc, builder, backbone, frames, instructions, pilot_inputs):
    """Eq. 4 -> Eq. 5 (WITH the Pilot slot, teacher-forced) -> padded batch."""
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


def forward_loss(base_model, head, batch, targets):
    """Eq. 6 -> Eq. 7 -> Eq. 13. Design A: h_act is read at action_index, which
    under pilot_mode="last" IS the Pilot position -- see InputSequence."""
    out = base_model.model.language_model(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        use_cache=False,
    )
    H = out.last_hidden_state
    idx = batch.action_index
    h_act = H[torch.arange(H.shape[0], device=H.device), idx]
    logits = head.logits(h_act)
    return action_loss(logits, targets, reduction="mean"), logits


def make_loader(dataset, batch_size, shuffle=True, workers=4, seed=0):
    from torch.utils.data import DataLoader

    def collate(items):
        return (
            [load_frame(s.frame_path) for s in items],
            [s.instruction for s in items],
            [torch.from_numpy(dataset.pilot_input(s)) for s in items],
            torch.tensor([s.action for s in items], dtype=torch.long),
        )

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate, generator=g,
                      drop_last=True, persistent_workers=workers > 0)


# ---------------------------------------------------------------------------
def smoke_test(args):
    print("VRAM SMOKE TEST — Stage 0' (recurrent Pilot, no L_pil)\n")
    dataset = PilotRolloutDataset(args.split, max_episodes=8)
    print(dataset.summary(), "\n")
    mean_vbar = compute_mean_vbar(dataset)

    enc, pilot = build_model(args.lora_r, args.lora_alpha, args.lora_dropout,
                             mean_vbar, grad_checkpointing=not args.no_grad_checkpointing,
                             pilot_mode=args.pilot_mode)
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)
    opt, params = make_optimizer(enc.model, pilot, args.lr)

    n_train = sum(p.numel() for p in params)
    print(f"trainable params : {n_train:,} (LoRA + PilotModule)")
    print(f"  of which z_0 + G_psi: {sum(p.numel() for p in pilot.parameters()):,} "
          f"(G_psi receives NO gradient in Stage 0' -- see module docstring)")
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha}\n")

    print(f"{'batch':>6} {'peak GB':>9} {'reserved GB':>12} {'s/step':>8}  status")
    print("-" * 52)
    for bs in args.smoke_batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            loader = make_loader(dataset, bs, workers=0)
            it = iter(loader)
            t0 = None
            for i in range(3):
                frames, instrs, pinputs, targets = next(it)
                if i == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                batch = encode_batch(enc, builder, backbone, frames, instrs, pinputs)
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

    print("\nVerifying G_psi receives no gradient (Stage 0' ablation property):")
    g_before = pilot.G_psi.weight.clone()
    frames, instrs, pinputs, targets = next(iter(make_loader(dataset, 4, workers=0)))
    batch = encode_batch(enc, builder, backbone, frames, instrs, pinputs)
    loss, _ = forward_loss(enc.model, head, batch, targets.to("cuda"))
    loss.backward()
    g_grad_is_none = pilot.G_psi.weight.grad is None
    print(f"  G_psi.weight.grad is None: {g_grad_is_none} "
          f"({'CORRECT -- matches the NaN ablation' if g_grad_is_none else 'UNEXPECTED'})")
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
          f"(irrelevant here -- Stage 0' has no L_pil)\n")
    mean_vbar = compute_mean_vbar(dataset)
    print(f"mean ||v_bar|| = {mean_vbar.norm().item():.3f} "
          f"(D18 bias-init target)\n")

    counts = dataset.action_counts()
    n = sum(counts.values())
    prior = {Action(a).name: c / n for a, c in counts.items()}
    entropy = -sum(p * math.log(p) for p in prior.values())
    print(f"action prior: " + "  ".join(f"{k}={v:.3f}" for k, v in prior.items()))
    print(f"BASELINES  loss: uniform={math.log(4):.4f}  marginal={entropy:.4f}")
    print(f"           acc : majority-class(FWD)={prior['FWD']:.3f}\n")

    enc, pilot = build_model(args.lora_r, args.lora_alpha, args.lora_dropout,
                             mean_vbar, grad_checkpointing=not args.no_grad_checkpointing,
                             pilot_mode=args.pilot_mode)
    print(f"pilot_mode={args.pilot_mode}  "
          f"({'A: h_act == h_pil == H[-1]' if args.pilot_mode == 'last' else 'B: h_pil = H[-2], h_act = H[-1] via ACTION_QUERY'})")
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot)
    backbone = Backbone(enc.model)
    head = ActionHead(enc.model, enc.processor.tokenizer)
    opt, params = make_optimizer(enc.model, pilot, args.lr)
    sched = make_scheduler(opt, args.warmup, args.steps,
                           schedule=args.schedule,
                           min_lr_frac=args.min_lr_frac)
    print(describe(args.warmup, args.steps, args.lr,
                   args.schedule, args.min_lr_frac))
    print(f"epochs over {len(dataset):,} steps at batch {args.batch_size}: "
          f"{args.steps * args.batch_size / len(dataset):.2f}\n")

    loader = make_loader(dataset, args.batch_size, workers=args.workers, seed=args.seed)
    enc.model.train()
    pilot.train()

    step, t0 = 0, time.time()
    run_loss, run_correct, run_n = 0.0, 0, 0
    per_action, per_action_correct = Counter(), Counter()
    log = []

    while step < args.steps:
        for frames, instrs, pinputs, targets in loader:
            if step >= args.steps:
                break
            targets = targets.to("cuda")
            batch = encode_batch(enc, builder, backbone, frames, instrs, pinputs)
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
                enc.model.save_pretrained(ckpt)
                torch.save(pilot.state_dict(), ckpt / "pilot.pt")
                print(f"  saved -> {ckpt}")

    final = out_dir / "final"
    enc.model.save_pretrained(final)
    torch.save(pilot.state_dict(), final / "pilot.pt")
    (out_dir / "trainlog.json").write_text(json.dumps(log, indent=2))
    print(f"\nsaved adapter + pilot -> {final}")
    print(f"log -> {out_dir / 'trainlog.json'}")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 0' training (recurrent Pilot, no L_pil)")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--smoke-batches", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=pathlib.Path, default=_ROOT / "checkpoints" / "stage0prime")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000)
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
    ap.add_argument("--pilot-mode", default="last", choices=["last", "action_query"],
                    help="D17 Design A ('last': h_act == h_pil) or Design B "
                         "('action_query': distinct states via a trailing query)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1
    return smoke_test(args) if args.smoke_test else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
