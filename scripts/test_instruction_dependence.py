#!/usr/bin/env python3
"""
Does the policy actually USE the instruction?

Runs the same held-out frames under three conditions:

    real     the episode's own instruction
    swapped  another episode's instruction (same length distribution, wrong content)
    empty    no instruction at all

If accuracy barely moves between `real` and `swapped`, the model is navigating
on visual cues alone -- following corridors and open space -- and the language
is decoration. That would mean our SR comes from generic traversability, not
instruction following, and no amount of extra supervision on top would fix it.

This is the control that has to pass before adding a progress head: a progress
target predictable from step-count or corridor statistics would be learned
WITHOUT grounding the instruction, giving a good auxiliary number and no better
navigation.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse, json, math, pathlib, random
import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from data.pointing_dataset import PointingDataset, load_frame
    from losses.pointing_loss import bearing_to_action
    from model.backbone import Backbone
    from model.input_sequence import InputSequenceBuilder
    from model.pointing_head import PointingHead, split_outputs
    from model.vision_encoder import VisionEncoder

    ck = pathlib.Path(args.checkpoint)
    cfg = json.loads((ck / "config.json").read_text()) if (ck/"config.json").exists() else {}
    K, S = cfg.get("history_frames", 0), cfg.get("history_stride", 1)

    enc = VisionEncoder()
    sd = torch.load(ck / "adapter.pt", map_location="cpu", weights_only=True)
    r = sd[[k for k in sd if k.endswith("lora_A.default.weight")][0]].shape[0]
    get_peft_model(enc.model, LoraConfig(r=r, lora_alpha=2*r, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    enc.model.load_state_dict(sd, strict=False)
    head = PointingHead(enc.d, dtype=torch.float32, device=enc.model.device)
    head.load_state_dict(torch.load(ck/"head.pt", map_location="cpu", weights_only=True))
    head.to(enc.model.device).eval(); enc.model.eval()
    b = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    bb = Backbone(enc.model)
    print(f"{args.checkpoint}  history K={K} stride={S}")

    ds = PointingDataset(args.split, max_episodes=80, history_frames=K, history_stride=S)
    steps = ds.steps[:args.steps]
    all_instr = sorted({s.instruction for s in ds.steps})
    rng = random.Random(args.seed)
    print(f"held-out steps: {len(steps)}   distinct instructions: {len(all_instr)}\n")

    conds = {"real": None, "swapped": None, "empty": None}
    res = {c: {"u":[], "v":[], "tu":[], "tv":[], "ok":0, "n":0,
               "bear":[], "stop_hi":[], "stop_lo":[]} for c in conds}

    with torch.no_grad():
        for s in steps:
            v_t = enc.encode(load_frame(s.frame_path))
            hist = [enc.encode(load_frame(h)) for h in s.history_paths] if K else None
            for cond in conds:
                if cond == "real":
                    instr = s.instruction
                elif cond == "swapped":
                    # a DIFFERENT episode's instruction -- same distribution,
                    # wrong content
                    other = [i for i in all_instr if i != s.instruction]
                    instr = rng.choice(other) if other else s.instruction
                else:
                    instr = ""
                seq = b.build(instr, v_t, history_visual=hist)
                H = bb.forward(seq, position_ids=bb.build_position_ids(seq))
                o = split_outputs(head(H[seq.action_index].float().unsqueeze(0)))
                R = res[cond]
                if s.visible > 0.5:
                    R["u"].append(o["u"].item()); R["tu"].append(s.u)
                    R["v"].append(o["v"].item()); R["tv"].append(s.v)
                p = torch.sigmoid(o["stop_logit"]).item()
                (R["stop_hi"] if s.is_stop > 0.5 else R["stop_lo"]).append(p)
                if not s.is_stop:
                    ap_ = bearing_to_action(o["dir_x"].item(), o["dir_y"].item())
                    at_ = bearing_to_action(s.dx, s.dy)
                    R["ok"] += int(ap_ == at_); R["n"] += 1
                    thp = math.atan2(o["dir_x"].item(), o["dir_y"].item())
                    tht = math.atan2(s.dx, s.dy)
                    R["bear"].append(abs(math.degrees(
                        math.atan2(math.sin(thp-tht), math.cos(thp-tht)))))

    print(f"{'condition':>9} {'u MAE':>7} {'u corr':>7} {'act agree':>10} "
          f"{'bear med':>9} {'stop sep':>9}")
    print("-"*58)
    base = {}
    for cond in ("real", "swapped", "empty"):
        R = res[cond]
        u, tu = np.array(R["u"]), np.array(R["tu"])
        mae = np.abs(u-tu).mean(); corr = np.corrcoef(u,tu)[0,1]
        agree = R["ok"]/max(R["n"],1); bear = np.median(R["bear"])
        hi, lo = np.mean(R["stop_hi"]), np.mean(R["stop_lo"])
        sep = hi/max(lo,1e-9)
        base[cond] = (mae, corr, agree, bear, sep)
        print(f"{cond:>9} {mae:>7.4f} {corr:>7.3f} {agree:>10.3f} "
              f"{bear:>8.1f}d {sep:>8.1f}x")

    print()
    mr, cr, ar, br, sr_ = base["real"]
    for cond in ("swapped", "empty"):
        m, c, a, bd, sp = base[cond]
        print(f"real -> {cond:8s}:  u corr {cr:.3f}->{c:.3f} ({100*(c-cr)/max(abs(cr),1e-9):+.0f}%)"
              f"   act {ar:.3f}->{a:.3f} ({100*(a-ar)/max(ar,1e-9):+.0f}%)")
    print("\nIf `swapped` is close to `real`, the policy is ignoring the "
          "instruction and\nnavigating on visual cues alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
