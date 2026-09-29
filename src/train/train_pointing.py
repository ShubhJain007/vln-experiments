"""
Pointing training (D27) — Robostral-style waypoint supervision.

Replaces LatentPilot's 4-way action classification. See DECISIONS.md D27 for
why, and for the three label bugs found by validating targets against the
expert's own actions (agreement 0.689 -> 0.972).

WHAT TRAINS
    vision encoder   FROZEN. The 196-token 14x14 grid preserves spatial layout
                     and the backbone is pretrained for grounding, so the
                     location is already representable. D2 forbids LoRA there
                     and it is asserted at build time.
    LLM q/k/v/o      LoRA, 6.4M params. A measured linear probe on the frozen
                     vision grid alone got corr 0.37 / R^2 -0.20 on u -- vision
                     cannot resolve which way to go at a junction, because that
                     depends on the INSTRUCTION. Fusing the two is what this
                     adapts.
    pointing head    14,343 params, reads the action-query hidden state.

    conda activate latentpilot
    python src/train/train_pointing.py --report-scale
    python src/train/train_pointing.py --steps 20000 --batch-size 8
"""

import sys as _sys
_sys.path[:] = [p for p in _sys.path if "/opt/ros/" not in p]

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))

from data.pointing_dataset import PointingDataset, load_frame  # noqa: E402
from losses.pointing_loss import bearing_to_action, pointing_loss  # noqa: E402
from model.action_space import Action  # noqa: E402
from model.backbone import Backbone  # noqa: E402
from model.input_sequence import InputSequenceBuilder, collate_sequences  # noqa: E402
from model.pointing_head import (DEFAULT_FUSION_LAYERS, PointingHead,
                                 split_outputs)  # noqa: E402
from model.vision_encoder import VisionEncoder  # noqa: E402
from train.schedule import describe, make_scheduler  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPECTED_ADAPTED_MODULES = 112
DEFAULT_LR = 1e-4


def build_model(lora_r=16, lora_alpha=32, lora_dropout=0.05,
                grad_checkpointing=True, fusion_layers=DEFAULT_FUSION_LAYERS):
    from peft import LoraConfig, get_peft_model

    enc = VisionEncoder()
    get_peft_model(enc.model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS))

    adapted = [n for n, _ in enc.model.named_modules() if n.endswith("lora_A.default")]
    if [n for n in adapted if "visual" in n]:
        raise RuntimeError("LoRA touched vision modules (D2)")
    if len(adapted) != EXPECTED_ADAPTED_MODULES:
        raise RuntimeError(f"LoRA adapted {len(adapted)}, expected {EXPECTED_ADAPTED_MODULES}")

    if grad_checkpointing:
        enc.model.gradient_checkpointing_enable()
        enc.model.enable_input_require_grads()

    head = PointingHead(enc.d, dtype=torch.float32, device=enc.model.device,
                        layers=fusion_layers)
    return enc, head


def make_loader(ds, batch_size, workers=4, seed=0, shuffle=True):
    """Batches carry (frames, instructions, history_frames, targets)."""
    from torch.utils.data import DataLoader

    def collate(items):
        return (
            [load_frame(s.frame_path) for s in items],
            [s.instruction for s in items],
            [[load_frame(h) for h in s.history_paths] for s in items],
            {k: torch.tensor([getattr(s, k) for s in items], dtype=torch.float32)
             for k in ("u", "v", "dx", "dy", "dtheta", "visible", "is_stop")},
        )

    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate, generator=g,
                      drop_last=True, persistent_workers=workers > 0)


def forward_batch(enc, builder, backbone, head, frames, instrs, history=None):
    with torch.no_grad():
        v_ts = enc.encode_batch(frames)
        # Encode history in ONE batched pass rather than per-sample: K frames
        # per sample would otherwise mean K sequential encoder calls per step.
        h_ts = None
        if history and history[0]:
            flat = [f for per_sample in history for f in per_sample]
            enc_flat = enc.encode_batch(flat)
            K = len(history[0])
            h_ts = [enc_flat[i*K:(i+1)*K] for i in range(len(history))]
    seqs = [builder.build(i, v, history_visual=(h_ts[j] if h_ts else None))
            for j, (i, v) in enumerate(zip(instrs, v_ts))]
    pos = [backbone.build_position_ids(s) for s in seqs]
    if any(p is None for p in pos):
        pos = None
    batch = collate_sequences(seqs, pos)
    need_hidden = head.layers is not None
    out = enc.model.model.language_model(
        inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask,
        position_ids=batch.position_ids, use_cache=False,
        output_hidden_states=need_hidden)
    if need_hidden:
        # D29: read layers 23/26/28, not just the final one. The probe showed
        # the final layer is BELOW chance for stop (AUC 0.441) while layer 23
        # reaches 0.694 and the fusion 0.74-0.80.
        feats = head.fuse(out.hidden_states, batch.action_index, head.layers)
    else:
        H = out.last_hidden_state
        rows = torch.arange(H.shape[0], device=H.device)
        feats = H[rows, batch.action_index].float()
    return split_outputs(head(feats))


def action_metrics(pred, tgt):
    """How often the PREDICTED bearing picks the same action as the TARGET
    bearing. This is the number that actually matters -- the losses are
    proxies for it."""
    agree = 0
    n = len(tgt["dx"])
    for i in range(n):
        a_p = bearing_to_action(pred["dir_x"][i].item(), pred["dir_y"][i].item())
        a_t = bearing_to_action(tgt["dx"][i].item(), tgt["dy"][i].item())
        agree += int(a_p == a_t)
    return agree / max(n, 1)


def report_scale(args):
    ds = PointingDataset(args.split, max_episodes=32,
                         history_frames=args.history_frames,
                         history_stride=args.history_stride)
    print(ds.summary(), "\n")
    enc, head = build_model(grad_checkpointing=False,
                            fusion_layers=None if args.no_fusion
                            else tuple(args.fusion_layers))
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    backbone = Backbone(enc.model)
    frames, instrs, hist, tgt = next(iter(make_loader(ds, 8, workers=0)))
    tgt = {k: v.to("cuda") for k, v in tgt.items()}
    with torch.no_grad():
        pred = forward_batch(enc, builder, backbone, head, frames, instrs, hist)
        total, parts = pointing_loss(pred, tgt, ds.stop_pos_weight())
    print(f"head params : {head.param_count:,}")
    print(f"TOTAL       : {total.item():.4f}")
    for k, v in parts.items():
        print(f"  {k:8s}  : {v:.4f}")
    print("\nIf one term dwarfs the rest it will dominate the gradient -- that is "
          "exactly how Stage 1 failed (D25).")
    return 0


def train(args):
    torch.manual_seed(args.seed)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ds = PointingDataset(args.split, max_episodes=args.max_episodes,
                         scans=args.scans, history_frames=args.history_frames,
                         history_stride=args.history_stride)
    print(ds.summary())
    if args.scans:
        print(f"scans restricted to: {' '.join(args.scans)}")
    pos_w = ds.stop_pos_weight()
    print(f"epochs over {len(ds):,} steps at batch {args.batch_size}: "
          f"{args.steps * args.batch_size / len(ds):.2f}\n")

    fusion = None if args.no_fusion else tuple(args.fusion_layers)
    enc, head = build_model(lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                            grad_checkpointing=not args.no_grad_checkpointing,
                            fusion_layers=fusion)
    print(f"head reads layers: {fusion if fusion else 'final only'}  "
          f"({head.param_count:,} params)")
    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    backbone = Backbone(enc.model)

    import bitsandbytes as bnb
    params = ([p for p in enc.model.parameters() if p.requires_grad]
              + list(head.parameters()))
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = make_scheduler(opt, args.warmup, args.steps, schedule="cosine",
                           min_lr_frac=0.05)
    print(describe(args.warmup, args.steps, args.lr, "cosine", 0.05))
    print(f"trainable: {sum(p.numel() for p in params):,} "
          f"(LoRA + {head.param_count:,} head)\n")

    loader = make_loader(ds, args.batch_size, workers=args.workers, seed=args.seed)
    enc.model.train(); head.train()

    step, t0 = 0, time.time()
    run = {"total": 0.0, "point": 0.0, "disp": 0.0, "theta": 0.0,
           "stop": 0.0, "visible": 0.0, "act": 0.0}
    log = []

    while step < args.steps:
        for frames, instrs, hist, tgt in loader:
            if step >= args.steps:
                break
            tgt = {k: v.to("cuda") for k, v in tgt.items()}
            pred = forward_batch(enc, builder, backbone, head, frames, instrs, hist)
            total, parts = pointing_loss(pred, tgt, pos_w)

            (total / args.accum).backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

            run["total"] += total.item()
            for k, v in parts.items():
                run[k] += v
            with torch.no_grad():
                run["act"] += action_metrics(pred, tgt)
            step += 1

            if step % args.log_every == 0:
                k = args.log_every
                with torch.no_grad():
                    stop_p = torch.sigmoid(pred["stop_logit"]).mean().item()
                print(f"step {step:6d}  L {run['total']/k:7.4f}  "
                      f"point {run['point']/k:.4f}  disp {run['disp']/k:.4f}  "
                      f"theta {run['theta']/k:.4f}  stop {run['stop']/k:.4f}  "
                      f"act-agree {run['act']/k:.3f}  p(stop) {stop_p:.3f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{step/(time.time()-t0):.2f} it/s")
                log.append({"step": step, **{kk: run[kk]/k for kk in run}})
                run = {kk: 0.0 for kk in run}

            if args.save_every and step % args.save_every == 0:
                save(enc, head, out_dir / f"step{step}")
                save_config(out_dir / f"step{step}", args)

    save(enc, head, out_dir / "final")
    save_config(out_dir / "final", args)
    (out_dir / "trainlog.json").write_text(json.dumps(log, indent=2))
    print(f"\nsaved -> {out_dir/'final'}")
    return 0


def save(enc, head, path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)
    lora_sd = {k: v for k, v in enc.model.state_dict().items() if "lora_" in k}
    if not lora_sd:
        raise RuntimeError("no LoRA tensors -- refusing to save an empty adapter")
    torch.save(lora_sd, path / "adapter.pt")
    torch.save(head.state_dict(), path / "head.pt")
    return len(lora_sd)


def save_config(path: pathlib.Path, args):
    """History depth must travel with the checkpoint -- evaluating a
    history-trained model with the wrong K silently changes the input
    distribution."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(
        {"history_frames": args.history_frames,
         "history_stride": args.history_stride,
         "fusion_layers": None if args.no_fusion else list(args.fusion_layers),
         "steps": args.steps,
         "batch_size": args.batch_size, "lr": args.lr,
         "scans": args.scans}, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Pointing training (D27)")
    ap.add_argument("--report-scale", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=pathlib.Path, default=_ROOT/"checkpoints"/"pointing")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--scans", nargs="+", default=None,
                    help="Restrict training to these scans. Used for the data "
                         "scaling ablation: the 6-scan subset reproduces "
                         "LatentPilot's exact corpus (1,665 eps / 69,606 steps).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history-frames", type=int, default=0,
                    help="D28. Past frames prepended to the current one, "
                         "encoded as a VIDEO (mRoPE temporal channel indexes "
                         "the frame). 0 reproduces the memoryless model.")
    ap.add_argument("--fusion-layers", type=int, nargs="+",
                    default=list(DEFAULT_FUSION_LAYERS),
                    help="D29: decoder layers the head reads. Probe found stop "
                         "peaks at 15/23 (AUC .72/.69) while the FINAL layer is "
                         "below chance (.44); fusing gives .74-.80.")
    ap.add_argument("--no-fusion", action="store_true",
                    help="read only the final layer (the old baseline)")
    ap.add_argument("--history-stride", type=int, default=8,
                    help="Steps between history frames. Robostral uses ALL "
                         "past frames (via prefix-tree caching we do not "
                         "have); striding approximates that coverage at fixed "
                         "token cost. stride 8 x K 2 spans 16 steps. Defaults "
                         "to 8: stride 1 is the degenerate case that silently "
                         "trained for 7 hours once, and is now rejected.")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA required"); return 1
    return report_scale(args) if args.report_scale else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
