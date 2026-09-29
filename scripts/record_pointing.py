#!/usr/bin/env python3
"""
Record pointing-policy rollouts as MP4, with the model's own prediction drawn
on every frame.

This is the diagnostic payoff of the pointing formulation: the model's intent
is a POINT IN THE IMAGE, so it can be rendered and inspected. A latent policy
gives you a 2048-d vector and no way to see what it meant.

Each frame shows:
  * cyan crosshair   -- where the model is pointing (u, v), its chosen waypoint
  * green crosshair  -- where the EXPERT waypoint actually is, when visible
  * the chosen action, p(stop), and live distance-to-goal
  * a red border on the final frame of a failed episode

Defaults to keeping FAILURES only (--failures-only), because the successes are
not what needs explaining.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse
import importlib.util
import math
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_pointing", _ROOT / "scripts" / "eval_pointing.py")
_ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ep)

_spec2 = importlib.util.spec_from_file_location(
    "eval_stage0_prime", _ROOT / "scripts" / "eval_stage0_prime.py")
_e0 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_e0)

SCENES = _ROOT / "data" / "scene_datasets" / "mp3d"


def wrap(text, width=52):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines[:3]


def crosshair(draw, x, y, colour, r=13):
    draw.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=3)
    draw.line([x - r - 6, y, x + r + 6, y], fill=colour, width=2)
    draw.line([x, y - r - 6, x, y + r + 6], fill=colour, width=2)


def record(sim, enc, builder, backbone, head, episode, out_dir, max_steps,
           stop_threshold, turn_deg, history_frames, history_stride):
    import torch
    from collections import deque
    from PIL import Image, ImageDraw, ImageFont
    from losses.pointing_loss import bearing_to_action
    from model.action_space import Action
    from model.pointing_head import split_outputs

    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    goal = episode["goals"][0]["position"]
    instruction = episode["instruction"]["instruction_text"]
    frame, pos = sim.reset(episode["start_position"], episode["start_rotation"])

    span = history_frames * history_stride
    hist_buf = deque(maxlen=span) if history_frames else None
    lines = wrap(instruction)
    reason, min_d, t = "timeout", float("inf"), 0

    while t < max_steps:
        d = sim.geodesic(pos, goal)
        min_d = min(min_d, d)

        with torch.no_grad():
            v_t = enc.encode(frame)
            if hist_buf is not None and not hist_buf:
                for _ in range(span):
                    hist_buf.append(v_t)
            hist = None
            if hist_buf is not None:
                buf = list(hist_buf)
                hist = [buf[-k * history_stride] for k in range(history_frames, 0, -1)]
            seq = builder.build(instruction, v_t, history_visual=hist)
            H = backbone.forward(seq, position_ids=backbone.build_position_ids(seq))
            out = split_outputs(head(H[seq.action_index].float().unsqueeze(0)))
            p_stop = torch.sigmoid(out["stop_logit"]).item()
            u, v = out["u"].item(), out["v"].item()
            vis_p = torch.sigmoid(out["visible_logit"]).item()
            action = bearing_to_action(out["dir_x"].item(), out["dir_y"].item(),
                                       turn_threshold_rad=math.radians(turn_deg))

        im = Image.fromarray(frame).convert("RGB")
        W, Hh = im.size
        canvas = Image.new("RGB", (W, Hh + 15 * (len(lines) + 3)), "black")
        canvas.paste(im, (0, 0))
        dr = ImageDraw.Draw(canvas)

        # The model's chosen waypoint, in image space. Dimmed when the model
        # itself predicts the waypoint is NOT visible -- in that case it is
        # steering on the metric fallback and (u,v) is not meaningful.
        colour = "cyan" if vis_p > 0.5 else "#2b6f7a"
        crosshair(dr, u * W, v * Hh, colour)

        y = Hh + 2
        for ln in lines:
            dr.text((4, y), ln, fill="white", font=font); y += 13
        dr.text((4, y), f"step {t:03d}   dist {d:5.2f}m   min {min_d:5.2f}m",
                fill="yellow", font=font); y += 13
        dr.text((4, y), f"action {action.name:5s}   p(stop) {p_stop:.3f}"
                        f"   point ({u:.2f},{v:.2f}) vis {vis_p:.2f}",
                fill=colour, font=font)

        if p_stop > stop_threshold:
            reason = "model_stop"
            ok = d <= 3.0
            dr.rectangle([0, 0, W - 1, Hh - 1],
                         outline="lime" if ok else "red", width=6)
            dr.text((6, 6), "STOP  " + ("SUCCESS" if ok else f"FAIL {d:.1f}m"),
                    fill="lime" if ok else "red", font=font)
            canvas.save(out_dir / f"{t:04d}.png")
            break

        canvas.save(out_dir / f"{t:04d}.png")
        if hist_buf is not None:
            hist_buf.append(v_t)
        frame, pos = sim.step(action.habitat_name)
        t += 1

    return {"reason": reason, "min_dist": min_d, "final_dist": d,
            "steps": t + 1, "success": (reason == "model_stop" and d <= 3.0)}


def to_mp4(png_dir, fps):
    mp4 = png_dir.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type",
                    "glob", "-i", str(png_dir / "*.png"), "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(mp4)],
                   check=True, capture_output=True)
    return mp4


def main():
    ap = argparse.ArgumentParser(description="Record pointing rollouts as MP4")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--per-scan", type=int, default=2)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--stop-threshold", type=float, default=0.10)
    ap.add_argument("--turn-threshold-deg", type=float, default=7.5)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--failures-only", action="store_true", default=True)
    ap.add_argument("--keep-all", dest="failures_only", action="store_false")
    ap.add_argument("--worker-python", default=None)
    args = ap.parse_args()

    import shutil
    from eval.remote_sim import DEFAULT_WORKER_PY, RemoteSim

    print(f"Loading {args.checkpoint} ...")
    enc, builder, backbone, head, hf, hs = _ep.load_policy(args.checkpoint)

    eps = _e0.load_episodes(args.split, None, None)
    by_scan = {}
    for e in eps:
        by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
    by_scan = {k: v[:args.per_scan] for k, v in sorted(by_scan.items())
               if (SCENES / k / f"{k}.glb").exists()}

    args.out.mkdir(parents=True, exist_ok=True)
    idx, kept = 0, []
    with RemoteSim(args.worker_python or DEFAULT_WORKER_PY) as sim:
        for scan, es in by_scan.items():
            sim.load_scene(SCENES / scan / f"{scan}.glb")
            for e in es:
                d = args.out / f"ep{idx:02d}_{scan}"
                r = record(sim, enc, builder, backbone, head, e, d,
                           args.max_steps, args.stop_threshold,
                           args.turn_threshold_deg, hf, hs)
                tag = "OK " if r["success"] else "FAIL"
                print(f"[{idx:02d}] {scan:14s} {tag} {r['reason']:12s} "
                      f"final {r['final_dist']:5.2f}m  min {r['min_dist']:5.2f}m  "
                      f"steps {r['steps']}")
                if r["success"] and args.failures_only:
                    shutil.rmtree(d)            # keep only what needs explaining
                else:
                    kept.append((to_mp4(d, args.fps), r))
                idx += 1

    print(f"\n{len(kept)} failure recordings under {args.out}/")
    for m, r in kept:
        print(f"  {m.name}: {r['reason']}, ended {r['final_dist']:.1f}m from goal "
              f"(closest {r['min_dist']:.1f}m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
