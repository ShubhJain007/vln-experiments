#!/usr/bin/env python3
"""
WHEN does the policy fail? Per-step accuracy conditioned on the situation.

Teacher-forced on held-out val_unseen, so this isolates "is the prediction
right here?" from rollout compounding. Slices per-step action agreement by:

  * step index          -- does it degrade over an episode?
  * distance to goal    -- does the final approach differ?
  * expert action       -- are turns harder than straights?
  * instruction length  -- do complex instructions fail?
  * scan                -- is it scene-specific?

Motivation: three separate mechanisms I proposed (lambda drift, layer choice,
pointing visibility) each looked well-argued and each failed when tested. This
measures the failure surface instead of theorising about it.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, json, math, pathlib, collections
import numpy as np, torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/pointing_fusion/step120000")
    ap.add_argument("--steps", type=int, default=2500)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from data.pointing_dataset import PointingDataset, load_frame
    from losses.pointing_loss import bearing_to_action
    from model.backbone import Backbone
    from model.input_sequence import InputSequenceBuilder
    from model.pointing_head import PointingHead, split_outputs
    from model.vision_encoder import VisionEncoder
    from model.action_space import Action

    ck = pathlib.Path(args.checkpoint)
    cfg = json.loads((ck / "config.json").read_text())
    K, S = cfg["history_frames"], cfg["history_stride"]
    FL = tuple(cfg["fusion_layers"]) if cfg.get("fusion_layers") else None

    enc = VisionEncoder()
    sd = torch.load(ck / "adapter.pt", map_location="cpu", weights_only=True)
    r = sd[[k for k in sd if k.endswith("lora_A.default.weight")][0]].shape[0]
    get_peft_model(enc.model, LoraConfig(r=r, lora_alpha=2*r, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    enc.model.load_state_dict(sd, strict=False)
    head = PointingHead(enc.d, dtype=torch.float32, device=enc.model.device, layers=FL)
    head.load_state_dict(torch.load(ck/"head.pt", map_location="cpu", weights_only=True))
    head.to(enc.model.device).eval(); enc.model.eval()
    b = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    bb = Backbone(enc.model)

    ds = PointingDataset("val_unseen", max_episodes=140, history_frames=K, history_stride=S)

    # goal distance per (scan, episode, step)
    goals = {}
    for p in sorted((_ROOT/"data/rollouts/val_unseen").glob("*/*/episode.json")):
        m = json.loads(p.read_text())
        pos = np.array(m["positions"], float); g = np.array(m["goal_position"], float)
        goals[(m["scan"], m["episode_id"])] = (np.linalg.norm(pos-g, axis=1), len(pos))

    steps = [s for s in ds.steps if not s.is_stop][:args.steps]
    print(f"{args.checkpoint}\nheld-out val_unseen, {len(steps)} non-STOP steps\n", flush=True)

    recs = []
    with torch.no_grad():
        for i, s in enumerate(steps):
            v = enc.encode(load_frame(s.frame_path))
            hist = [enc.encode(load_frame(h)) for h in s.history_paths] if K else None
            seq = b.build(s.instruction, v, history_visual=hist)
            if FL:
                lm = enc.model.model.language_model(inputs_embeds=seq.inputs_embeds,
                    attention_mask=seq.attention_mask,
                    position_ids=bb.build_position_ids(seq),
                    use_cache=False, output_hidden_states=True)
                idx = torch.tensor([seq.action_index], device=enc.model.device)
                feats = head.fuse(lm.hidden_states, idx, FL)
            else:
                H = bb.forward(seq, position_ids=bb.build_position_ids(seq))
                feats = H[seq.action_index].float().unsqueeze(0)
            o = split_outputs(head(feats))
            ap_ = bearing_to_action(o["dir_x"].item(), o["dir_y"].item())
            at_ = bearing_to_action(s.dx, s.dy)
            key = (s.scan, s.episode_id)
            dg, T = goals.get(key, (None, 1))
            recs.append({
                "ok": int(ap_ == at_), "t": s.step_index, "T": T,
                "frac": s.step_index / max(T-1, 1),
                "dg": float(dg[s.step_index]) if dg is not None and s.step_index < len(dg) else np.nan,
                "true": Action(int(np.argmax([at_==a for a in Action]))).name if False else at_.name,
                "pred": ap_.name, "instr_len": len(s.instruction.split()),
                "scan": s.scan, "vis": s.visible,
            })
            if (i+1) % 400 == 0: print(f"  {i+1}/{len(steps)}", flush=True)

    R = recs
    def slice_report(title, keyfn, order=None):
        g = collections.defaultdict(list)
        for r_ in R: g[keyfn(r_)].append(r_["ok"])
        print(f"\n{title}")
        print(f"  {'bucket':>16} {'steps':>7} {'agree':>7}")
        keys = order if order else sorted(g)
        for k in keys:
            if k in g and len(g[k]) >= 15:
                print(f"  {str(k):>16} {len(g[k]):>7} {np.mean(g[k]):>7.3f}")

    print(f"\nOVERALL action agreement: {np.mean([r_['ok'] for r_ in R]):.3f}")

    def fb(x):
        if x != x: return "unknown"
        for lo,hi,l in [(0,3,"0-3 m"),(3,5,"3-5 m"),(5,10,"5-10 m"),(10,1e9,">10 m")]:
            if lo <= x < hi: return l
        return "?"
    slice_report("BY DISTANCE TO GOAL", lambda r_: fb(r_["dg"]),
                 ["0-3 m","3-5 m","5-10 m",">10 m"])
    slice_report("BY POSITION IN EPISODE", lambda r_: f"{int(r_['frac']*5)*20}-{int(r_['frac']*5)*20+20}%",
                 ["0-20%","20-40%","40-60%","60-80%","80-100%"])
    slice_report("BY EXPERT ACTION", lambda r_: r_["true"], ["FWD","LEFT","RIGHT"])
    slice_report("BY INSTRUCTION LENGTH", lambda r_: ("short <20w" if r_["instr_len"]<20
                 else "med 20-35w" if r_["instr_len"]<35 else "long >35w"),
                 ["short <20w","med 20-35w","long >35w"])
    slice_report("BY SCAN", lambda r_: r_["scan"])

    conf = collections.Counter((r_["true"], r_["pred"]) for r_ in R)
    print("\nCONFUSION (expert -> predicted)")
    for tr in ("FWD","LEFT","RIGHT"):
        row = {p: conf[(tr,p)] for p in ("FWD","LEFT","RIGHT")}
        n = sum(row.values())
        if n: print(f"  {tr:>6} n={n:>5}: " + "  ".join(f"{k}={v/n:.3f}" for k,v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
