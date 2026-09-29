#!/usr/bin/env python3
"""Re-render one or more episodes as fixed-rate mp4 + 9D action labels.

Sizing run first: measures frames, wall-clock and bytes per episode so the full
job can be costed before committing. Deterministic replay of stored poses --
no navigation, no follower, resumable.

Runs under the habitat_render env (py3.9), not latentpilot.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, json, pathlib, time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

FPS, TURN_DPS, WALK_MPS = 15.0, 45.0, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--out", default="data/video")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--fps", type=float, default=FPS)
    args = ap.parse_args()

    import imageio
    from data.collect_rollouts import make_simulator, set_agent_state, SCENES_DIR
    from data.resample_video import resample_poses
    from data.ego_pose_9d import ego_pose_9d, habitat_to_cosmos

    # One video per TRAJECTORY, not per episode. R2R ships ~3 instructions for
    # each path, so the naive loop renders identical footage three times (the
    # sizing run did exactly that for 10355/10356/10357). The instructions ride
    # along in meta.json and are what differ between training samples.
    root = _ROOT / "data" / "rollouts" / args.split
    metas, seen = [], {}
    for f in sorted(root.glob("*/*/episode.json")):
        d = json.loads(f.read_text())
        if len(d["positions"]) < 5:
            continue
        key = (d["scan"], d["trajectory_id"])
        if key in seen:
            seen[key].append(d)          # extra instruction for the same path
            continue
        seen[key] = [d]
        metas.append((f.parent, d))
        if args.limit and len(metas) >= args.limit:
            break

    out_root = _ROOT / args.out / args.split
    by_scan = {}
    for ep_dir, d in metas:
        by_scan.setdefault(d["scan"], []).append((ep_dir, d))

    tot_f = tot_b = 0
    t0 = time.time()
    for scan, items in by_scan.items():
        glb = SCENES_DIR / scan / f"{scan}.glb"
        if not glb.exists():
            print(f"{scan}: MISSING {glb}"); continue
        sim = make_simulator(glb)
        try:
            for ep_dir, d in items:
                ts = time.time()
                P, Q, times = resample_poses(d["positions"], d["rotations"],
                                             fps=args.fps, walk=WALK_MPS, turn=TURN_DPS)
                dst = out_root / scan / f"traj{d['trajectory_id']}"
                dst.mkdir(parents=True, exist_ok=True)
                mp4 = dst / "frames.mp4"
                if mp4.exists() and (dst / "action9d.npy").exists():
                    continue  # resumable: already rendered
                w = imageio.get_writer(str(mp4), fps=args.fps, codec="libx264",
                                       quality=None, ffmpeg_params=["-crf", str(args.crf),
                                                                    "-pix_fmt", "yuv420p"])
                for p, q in zip(P, Q):
                    set_agent_state(sim, p, q)
                    w.append_data(np.asarray(sim.get_sensor_observations()["color_sensor"])[:, :, :3])
                w.close()

                # 9D labels on the SAME grid, already in Cosmos's frame
                a9 = np.stack([habitat_to_cosmos(ego_pose_9d(P[t], Q[t], P[t+1], Q[t+1]))
                               for t in range(len(P) - 1)] or [np.zeros(9, np.float32)])
                np.save(dst / "action9d.npy", a9.astype(np.float32))
                (dst / "meta.json").write_text(json.dumps({
                    "scan": scan, "trajectory_id": d["trajectory_id"],
                    "episode_ids": [e["episode_id"] for e in seen[(scan, d["trajectory_id"])]],
                    "instructions": [e["instruction"] for e in seen[(scan, d["trajectory_id"])]],
                    "goal_position": d["goal_position"],
                    "fps": args.fps, "num_frames": int(len(P)),
                    "recorded_steps": len(d["positions"]),
                    "duration_s": float(times[-1]), "domain_name": "av",
                    "walk_mps": WALK_MPS, "turn_dps": TURN_DPS,
                }, indent=1))
                b = mp4.stat().st_size
                tot_f += len(P); tot_b += b
                print(f"  {scan}/{ep_dir.name}: {len(d['positions'])} steps -> {len(P)} frames "
                      f"({times[-1]:.1f}s)  {b/1024:.0f} KB  {time.time()-ts:.1f}s", flush=True)
        finally:
            sim.close()

    el = time.time() - t0
    n = len(metas)
    print(f"\n{n} episodes  {tot_f:,} frames  {tot_b/1e6:.1f} MB  {el:.1f}s")
    if n:
        print(f"per episode: {tot_f/n:.0f} frames  {tot_b/n/1024:.0f} KB  {el/n:.2f}s")
        print(f"per frame:   {tot_b/max(tot_f,1)/1024:.1f} KB  {el/max(tot_f,1)*1000:.1f} ms")
        TOT = len(metas)
        print(f"\nextrapolated to {TOT:,} episodes: "
              f"{tot_f/n*TOT/1e6:.2f} M frames, {tot_b/n*TOT/1e9:.1f} GB, "
              f"{el/n*TOT/3600:.1f} GPU-hours")


if __name__ == "__main__":
    raise SystemExit(main())
