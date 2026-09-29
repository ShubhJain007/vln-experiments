#!/usr/bin/env python3
"""
Closed-loop evaluation for the pointing policy (D27).

    frame -> VLM -> (u, v, dir_x, dir_y, stop, visible)
                      |
                      +-- p(stop) > threshold        -> STOP
                      +-- else atan2(dir_x, dir_y)    -> FWD / LEFT / RIGHT

No PilotCache and no recurrent latent: each step depends only on the current
frame and the instruction, so there is no state to corrupt across a rollout.

--------------------------------------------------------------------------
STOP IS A THRESHOLD, NOT AN ARGMAX
--------------------------------------------------------------------------
The whole point of giving STOP its own binary output is that the operating
point can be chosen AFTER training. The model's mean p(stop) during training
sits near the 2.2% base rate, so a naive 0.5 threshold would never fire --
which is exactly the failure mode the previous 4-way argmax had. Sweep it.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse
import importlib.util
import math
import pathlib
from collections import Counter

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_stage0_prime", _ROOT / "scripts" / "eval_stage0_prime.py")
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

SCENES = _ROOT / "data" / "scene_datasets" / "mp3d"
DEFAULT_MAX_STEPS = 100


def load_policy(checkpoint):
    import torch
    from peft import LoraConfig, get_peft_model

    from model.backbone import Backbone
    from model.input_sequence import InputSequenceBuilder
    from model.pointing_head import PointingHead
    from model.vision_encoder import VisionEncoder

    ck = pathlib.Path(checkpoint)
    for f in ("adapter.pt", "head.pt"):
        if not (ck / f).exists():
            raise FileNotFoundError(f"{ck/f} missing -- not a pointing checkpoint")

    enc = VisionEncoder()
    sd = torch.load(str(ck / "adapter.pt"), map_location="cpu", weights_only=True)
    r = sd[[k for k in sd if k.endswith("lora_A.default.weight")][0]].shape[0]
    get_peft_model(enc.model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    missing, unexpected = enc.model.load_state_dict(sd, strict=False)
    if [k for k in unexpected if "lora" in k]:
        raise RuntimeError("unmatched LoRA keys")

    # History depth MUST come from the checkpoint. Evaluating a K=2 model with
    # K=0 silently feeds a different input distribution -- the sequence is a
    # third of the length the model was trained on -- and produces confident,
    # meaningless numbers.
    import json as _json
    cfg_path = ck / "config.json"
    history_frames, history_stride = 0, 8
    _c = {}
    if cfg_path.exists():
        _c = _json.loads(cfg_path.read_text())
        history_frames = int(_c.get("history_frames", 0))
        # Older checkpoints predate the field; 1 is what they actually used,
        # so read it faithfully rather than assuming the new default.
        history_stride = int(_c.get("history_stride", 1))
    print(f"  history_frames = {history_frames} (stride {history_stride}, spans "
          f"{history_frames*history_stride} steps)"
          f"{' [from checkpoint config]' if cfg_path.exists() else ' [no config; memoryless]'}")

    # Fusion layers must come from the checkpoint: evaluating a fused head as
    # final-only (or vice versa) silently changes the input to the head.
    fusion = _c.get("fusion_layers", None) if cfg_path.exists() else None
    fusion = tuple(fusion) if fusion else None
    print(f"  head reads layers: {fusion if fusion else 'final only'}")
    head = PointingHead(enc.d, dtype=torch.float32, device=enc.model.device,
                        layers=fusion)
    head.load_state_dict(torch.load(str(ck / "head.pt"), map_location="cpu",
                                    weights_only=True))
    head.to(enc.model.device).eval()
    enc.model.eval()
    print(f"  loaded {len(sd)} LoRA tensors (r={r}) + pointing head")

    builder = InputSequenceBuilder(enc.model, enc.processor.tokenizer)
    return enc, builder, Backbone(enc.model), head, history_frames, history_stride


def rollout(sim, enc, builder, backbone, head, episode, max_steps,
            success_radius, stop_threshold, turn_threshold_deg,
            history_frames=0, history_stride=1):
    import torch
    from losses.pointing_loss import bearing_to_action
    from model.action_space import Action
    from model.pointing_head import split_outputs

    goal = episode["goals"][0]["position"]
    instruction = episode["instruction"]["instruction_text"]
    frame, pos = sim.reset(episode["start_position"], episode["start_rotation"])

    positions, reason, min_dist = [], "timeout", float("inf")
    stop_probs, turn_thr = [], math.radians(turn_threshold_deg)

    # Rolling buffer of the K most recent ENCODED frames, oldest -> newest.
    # Seeded with the first frame repeated, matching how the dataset pads the
    # start of an episode, so step 0 at eval looks like step 0 in training.
    # Buffer the last K*stride encoded frames and sample every `stride`-th, so
    # eval reproduces the dataset's strided indexing exactly. A mismatch here
    # feeds the model a different input distribution than it trained on.
    from collections import deque
    span = history_frames * history_stride
    hist_buf = deque(maxlen=span) if history_frames else None

    for _ in range(max_steps):
        positions.append(pos)
        d = sim.geodesic(pos, goal)
        min_dist = min(min_dist, d)
        if d <= success_radius:
            reason = "within_radius"
            break

        with torch.no_grad():
            v_t = enc.encode(frame)
            if hist_buf is not None and not hist_buf:
                for _ in range(span):
                    hist_buf.append(v_t)          # episode start: repeat frame 0
            hist = None
            if hist_buf is not None:
                buf = list(hist_buf)              # oldest -> newest
                hist = [buf[-k * history_stride] for k in
                        range(history_frames, 0, -1)]
            seq = builder.build(instruction, v_t, history_visual=hist)
            if head.layers is not None:
                lm = enc.model.model.language_model(
                    inputs_embeds=seq.inputs_embeds,
                    attention_mask=seq.attention_mask,
                    position_ids=backbone.build_position_ids(seq),
                    use_cache=False, output_hidden_states=True)
                idx = torch.tensor([seq.action_index], device=enc.model.device)
                feats = head.fuse(lm.hidden_states, idx, head.layers)
            else:
                H = backbone.forward(seq,
                                     position_ids=backbone.build_position_ids(seq))
                feats = H[seq.action_index].float().unsqueeze(0)
            out = split_outputs(head(feats))
            p_stop = torch.sigmoid(out["stop_logit"]).item()
            stop_probs.append(p_stop)
            if p_stop > stop_threshold:
                reason = "model_stop"
                break
            action = bearing_to_action(out["dir_x"].item(), out["dir_y"].item(),
                                       turn_threshold_rad=turn_thr)

        if hist_buf is not None:
            hist_buf.append(v_t)                  # current becomes history
        frame, pos = sim.step(action.habitat_name)

    return {"positions": positions, "termination_reason": reason,
            "min_dist_to_goal": min_dist, "stop_probs": stop_probs}


def main():
    ap = argparse.ArgumentParser(description="Closed-loop eval, pointing policy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--scans", nargs="+", default=None)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--success-radius", type=float, default=3.0)
    ap.add_argument("--strict", action="store_true",
                    help="TRUE SR: disable the proximity break so an episode "
                         "can only end by the MODEL's own STOP (or timeout). "
                         "Without this a model_stop is structurally incapable "
                         "of being a success -- the harness ends the episode "
                         "the instant the agent enters the radius, so a stop "
                         "only ever fires when the agent is already >3 m away.")
    ap.add_argument("--stop-threshold", type=float, default=0.5)
    ap.add_argument("--turn-threshold-deg", type=float, default=7.5)
    ap.add_argument("--worker-python", default=None)
    args = ap.parse_args()

    import numpy as np
    from eval.metrics import aggregate, evaluate_episode
    from eval.remote_sim import DEFAULT_WORKER_PY, RemoteSim

    print(f"Loading pointing policy from {args.checkpoint} ...")
    enc, builder, backbone, head, history_frames, history_stride = load_policy(args.checkpoint)
    print(f"stop_threshold={args.stop_threshold}  turn={args.turn_threshold_deg} deg")

    # A negative radius can never be satisfied, so the proximity break never
    # triggers and every episode ends on the model's own STOP or on timeout.
    radius = -1.0 if args.strict else args.success_radius
    if args.strict:
        print("  STRICT: success requires the model's OWN stop within 3 m\n")

    episodes = _eval.load_episodes(args.split, args.scans, args.limit)
    by_scan = {}
    for e in episodes:
        by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
    print(f"split={args.split}  episodes={len(episodes)}  scans={len(by_scan)}\n")

    results, reasons, all_p = [], Counter(), []
    # CLOSEST APPROACH per episode. NE only says where the agent ENDED; this
    # says how close it ever got, which is what separates "reached the goal
    # region then mis-stopped" from "never got near it at all".
    min_dists, final_dists, stop_dists = [], [], []
    with RemoteSim(args.worker_python or DEFAULT_WORKER_PY) as sim:
        geo = sim.geodesic_fn()
        for scan, eps in sorted(by_scan.items()):
            glb = SCENES / scan / f"{scan}.glb"
            if not glb.exists():
                continue
            sim.load_scene(glb)
            for e in eps:
                r = rollout(sim, enc, builder, backbone, head, e, args.max_steps,
                            radius, args.stop_threshold,
                            args.turn_threshold_deg, history_frames, history_stride)
                reasons[r["termination_reason"]] += 1
                all_p.extend(r["stop_probs"])
                min_dists.append(r["min_dist_to_goal"])
                fd = geo(r["positions"][-1], e["goals"][0]["position"])
                final_dists.append(fd)
                if r["termination_reason"] == "model_stop":
                    stop_dists.append(fd)
                results.append(evaluate_episode(
                    path=r["positions"], goal_position=e["goals"][0]["position"],
                    reference_path=e["reference_path"],
                    shortest_path_length=e["info"]["geodesic_distance"],
                    distance_fn=geo))

    agg = aggregate(results)
    n = len(results)
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  POINTING EVAL — {args.checkpoint}")
    print(f"  split={args.split}  n={n}  stop_thr={args.stop_threshold}"
          f"{'  STRICT (own STOP only)' if args.strict else '  diagnostic (proximity break)'}")
    print(sep)
    for k in ("SR", "SPL", "OS", "nDTW", "NE"):
        print(f"    {k:5s} = {agg[k]:.4f}")
    print("\n  TERMINATION:")
    for name in ("model_stop", "within_radius", "timeout"):
        c = reasons.get(name, 0)
        print(f"    {name:15s} {c:4d}  ({100*c/max(n,1):5.1f}%)")
    md, fd_ = np.array(min_dists), np.array(final_dists)
    print(f"\n  DISTANCE TO GOAL (m):")
    print(f"    closest approach : mean {md.mean():5.2f}  median {np.median(md):5.2f}"
          f"  p25 {np.percentile(md,25):5.2f}  min {md.min():5.2f}")
    print(f"    final position   : mean {fd_.mean():5.2f}  median {np.median(fd_):5.2f}")
    for t in (3.0, 5.0, 10.0):
        print(f"    ever within {t:4.1f}m : {100*(md<=t).mean():5.1f}%")
    if stop_dists:
        sdz = np.array(stop_dists)
        print(f"    where it STOPPED : mean {sdz.mean():5.2f}  median "
              f"{np.median(sdz):5.2f}  within 3m {100*(sdz<=3).mean():4.1f}%")
    if all_p:
        a = np.array(all_p)
        print(f"\n  p(stop) over {len(a):,} steps: mean {a.mean():.4f}  "
              f"p95 {np.percentile(a,95):.4f}  max {a.max():.4f}")
        print(f"    (threshold {args.stop_threshold} would fire on "
              f"{(a>args.stop_threshold).mean()*100:.2f}% of steps)")
    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
