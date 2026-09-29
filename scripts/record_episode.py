#!/usr/bin/env python3
"""
Record closed-loop rollouts as MP4s -- literally what the eval script measures,
made visible. Same PilotCache loop as eval_stage0_prime.py's rollout_policy,
duplicated here (not imported) purely to add per-frame PNG capture + an info
overlay without touching the eval path that produces our reported numbers.

    conda activate latentpilot
    python scripts/record_episode.py --checkpoint checkpoints/stage1_learned/step10000 \
        --limit 3 --out recordings/step10000
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse
import pathlib
import subprocess

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "eval_stage0_prime", _ROOT / "scripts" / "eval_stage0_prime.py")
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

SCENES = _ROOT / "data" / "scene_datasets" / "mp3d"


def wrap_text(text, width=54):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def record_episode(sim, enc, builder, backbone, head, pilot, episode,
                   max_steps, success_radius, out_dir, slot_mode, mean_vbar,
                   target_norm):
    """rollout_policy's loop, plus a saved+annotated PNG per step."""
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from model.action_space import Action

    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    goal = episode["goals"][0]["position"]
    instruction = episode["instruction"]["instruction_text"]
    frame, pos = sim.reset(episode["start_position"], episode["start_rotation"])

    cache = pilot.new_cache()
    reason, min_dist, t = "timeout", float("inf"), 0
    instr_lines = wrap_text(instruction)

    while t < max_steps:
        d = sim.geodesic(pos, goal)
        min_dist = min(min_dist, d)

        # -- annotate + save the CURRENT frame before acting on it -----------
        im = Image.fromarray(frame).convert("RGB")
        canvas = Image.new("RGB", (im.width, im.height + 15 * (len(instr_lines) + 2)), "black")
        canvas.paste(im, (0, 0))
        draw = ImageDraw.Draw(canvas)
        y = im.height + 2
        for line in instr_lines:
            draw.text((4, y), line, fill="white", font=font)
            y += 13
        draw.text((4, y), f"step {t:03d}  dist-to-goal {d:.2f}m", fill="yellow", font=font)

        if d <= success_radius:
            reason = "within_radius"
            draw.rectangle([0, 0, im.width - 1, im.height - 1], outline="lime", width=4)
            canvas.save(out_dir / f"{t:04d}.png")
            break

        with torch.no_grad():
            v_t = enc.encode(frame)
            if slot_mode == "z":
                slot = cache.read()
            elif slot_mode == "z_scaled":
                z = cache.read().float()
                slot = (z / z.norm().clamp(min=1e-6) * target_norm).to(z.dtype)
            elif slot_mode == "vbar_current":
                slot = v_t.mean(dim=0)
            elif slot_mode == "constant":
                slot = mean_vbar
            else:
                raise ValueError(f"unknown slot_mode {slot_mode!r}")

            seq = builder.build(instruction, v_t, pilot_input=slot)
            H = backbone.forward(seq, position_ids=backbone.build_position_ids(seq))
            action = head.predict(H[seq.action_index])
            z_t = pilot(H[seq.pilot_index])
            cache.write(z_t)

        action_name = Action(action).name
        draw.text((4, y - 13 - 13), f"action: {action_name}", fill="cyan", font=font)
        canvas.save(out_dir / f"{t:04d}.png")

        if action is Action.STOP:
            reason = "model_stop"
            break

        frame, pos = sim.step(action.habitat_name)
        t += 1

    return {"termination_reason": reason, "min_dist_to_goal": min_dist,
            "num_steps": t + 1, "instruction": instruction}


def frames_to_mp4(png_dir: pathlib.Path, fps: int) -> pathlib.Path:
    mp4 = png_dir.with_suffix(".mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-pattern_type", "glob", "-i", str(png_dir / "*.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4),
    ], check=True, capture_output=True)
    return mp4


def main():
    ap = argparse.ArgumentParser(description="Record closed-loop rollouts as MP4")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--per-scan", type=int, default=None,
                    help="Record N episodes from EACH scan instead of the "
                         "first --limit overall. Without this, --limit 3 can "
                         "return three episodes from one building, which says "
                         "nothing about cross-scene generalisation.")
    ap.add_argument("--scans", nargs="+", default=None)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-steps", type=int, default=_eval.DEFAULT_MAX_STEPS)
    ap.add_argument("--success-radius", type=float, default=3.0)
    ap.add_argument("--slot-mode", default="z",
                    choices=["z", "z_scaled", "vbar_current", "constant"])
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--worker-python", default=None)
    args = ap.parse_args()

    from eval.remote_sim import DEFAULT_WORKER_PY, RemoteSim
    import numpy as np
    import torch

    vpaths = sorted((_ROOT / "data" / "rollouts" / "train").rglob("vbar.npy"))[:200]
    _allv = np.concatenate([np.load(v) for v in vpaths])
    mean_vbar = torch.from_numpy(_allv.mean(axis=0))
    target_norm = float(np.linalg.norm(_allv, axis=1).mean())

    print(f"Loading policy from {args.checkpoint} ...")
    enc, builder, backbone, head, pilot = _eval.load_policy(args.checkpoint)
    mean_vbar = mean_vbar.to(enc.model.device, torch.bfloat16)

    if args.per_scan:
        episodes = _eval.load_episodes(args.split, args.scans, None)
        by_scan = {}
        for e in episodes:
            by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
        by_scan = {k: v[:args.per_scan] for k, v in sorted(by_scan.items())
                   if (SCENES / k / f"{k}.glb").exists()}
        episodes = [e for v in by_scan.values() for e in v]
    else:
        episodes = _eval.load_episodes(args.split, args.scans, args.limit)
        by_scan = {}
        for e in episodes:
            by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)

    args.out.mkdir(parents=True, exist_ok=True)
    idx = 0
    with RemoteSim(args.worker_python or DEFAULT_WORKER_PY) as sim:
        for scan, eps in sorted(by_scan.items()):
            glb = SCENES / scan / f"{scan}.glb"
            if not glb.exists():
                continue
            sim.load_scene(glb)
            for e in eps:
                png_dir = args.out / f"ep{idx:02d}_{scan}"
                print(f"[{idx+1}/{len(episodes)}] {scan}  '{e['instruction']['instruction_text'][:60]}...'")
                r = record_episode(sim, enc, builder, backbone, head, pilot, e,
                                   args.max_steps, args.success_radius, png_dir,
                                   args.slot_mode, mean_vbar, target_norm)
                mp4 = frames_to_mp4(png_dir, args.fps)
                print(f"  -> {r['termination_reason']}  "
                      f"min_dist={r['min_dist_to_goal']:.2f}m  "
                      f"steps={r['num_steps']}  saved {mp4}")
                idx += 1

    print(f"\n{idx} recordings written under {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
