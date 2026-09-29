#!/usr/bin/env python3
"""What translation UNIT does Edge think our resampled video is in?

`av` actions are raw, so units are ours to choose -- but the choice is not free:
in that domain magnitude is physically tied to optical flow, so a wrong unit
pairs slow indoor footage with car-sized numbers and fights the prior.

Inverse dynamics settles it empirically. The model watches our video and emits
actions; the magnitude it emits IS its answer for our visual flow. Fit a single
scalar s minimising ||pred - s*truth|| and read the unit off directly:
s ~ 1 means metres, s ~ 100 means centimetres.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import json, math, glob, pathlib, argparse
import numpy as np, torch
import imageio.v3 as iio
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
CK = "checkpoints/Cosmos3-Edge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--trajectories", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from data.ego_pose_9d import rot_from_6d

    def yaw(a9):
        R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
        return math.degrees(math.atan2(R[0, 2], R[2, 2]))

    dirs = sorted(glob.glob(str(_ROOT / "data/video/val_unseen/*/traj*")))
    by_scan = {}
    for d in dirs:
        by_scan.setdefault(pathlib.Path(d).parent.name, []).append(d)
    sel = []
    for i in range(max(len(v) for v in by_scan.values())):
        for s in sorted(by_scan):
            if i < len(by_scan[s]):
                sel.append(by_scan[s][i])
        if len(sel) >= args.trajectories:
            break
    sel = sel[:args.trajectories]
    print(f"{len(sel)} trajectories across {len({pathlib.Path(d).parent.name for d in sel})} scans",
          flush=True)

    pipe = Cosmos3OmniPipeline.from_pretrained(CK, dtype=torch.bfloat16).to("cuda")

    P_all, T_all = [], []
    for k, d in enumerate(sel):
        m = json.load(open(f"{d}/meta.json"))
        gt = np.load(f"{d}/action9d.npy")[:args.chunk]
        if len(gt) < args.chunk:
            continue
        frames = iio.imread(f"{d}/frames.mp4", plugin="pyav")[:args.chunk + 1]
        video = [Image.fromarray(f).convert("RGB").resize((480, 480)) for f in frames]
        out = pipe(
            prompt="You are an autonomous vehicle planning system.",
            action=CosmosActionCondition(
                mode="inverse_dynamics", chunk_size=args.chunk, domain_name="av",
                resolution_tier=480, view_point="ego_view", video=video),
            fps=m["fps"], num_inference_steps=args.steps, guidance_scale=1.0,
            use_system_prompt=False,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        a = out.action
        if isinstance(a, (list, tuple)): a = a[0]
        if hasattr(a, "detach"): a = a.detach().float().cpu().numpy()
        P_all.append(np.asarray(a, np.float64).reshape(-1, 9)[:args.chunk])
        T_all.append(gt.astype(np.float64))
        print(f"  [{k+1}/{len(sel)}]", flush=True)

    P, T = np.concatenate(P_all), np.concatenate(T_all)
    print(f"\n=== {len(P)} frames, resampled video @15 FPS ===")
    py = np.array([yaw(x) for x in P]); ty = np.array([yaw(x) for x in T])
    print(f"  yaw      corr {np.corrcoef(py,ty)[0,1]:+.3f}   pred std {py.std():.2f}d  true std {ty.std():.2f}d")
    m = np.abs(ty) > 2
    if m.sum():
        print(f"  turn direction (|true|>2d): {(np.sign(py[m])==np.sign(ty[m])).mean():.3f} (n={m.sum()})")

    print("\n  --- translation scale the model chose ---")
    for i, nm in enumerate(["trans_x", "trans_y", "trans_z"]):
        t, p = T[:, i], P[:, i]
        s = float(t @ p / (t @ t)) if t @ t > 1e-12 else float("nan")
        print(f"  {nm}: least-squares s = {s:9.2f}   corr {np.corrcoef(p,t)[0,1]:+.3f}"
              f"   (pred std {p.std():.4f} / true std {t.std():.4f} = {p.std()/max(t.std(),1e-9):6.2f})")
    t, p = T[:, :3].ravel(), P[:, :3].ravel()
    s = float(t @ p / (t @ t))
    print(f"\n  ALL translation channels: s = {s:.2f}")
    print(f"    s ~ 1   -> the model reads our motion in METRES")
    print(f"    s ~ 100 -> CENTIMETRES")


if __name__ == "__main__":
    raise SystemExit(main())
