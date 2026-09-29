#!/usr/bin/env python3
"""Do FULL R2R instructions steer policy mode once guidance is on?

The first policy test used raw R2R instructions at guidance_scale=1.0 and got
chance. The CFG sweep then showed short commands ("turn left") reach 95% correct
turn direction at guidance 5-7.5 -- so that null was measured at the one setting
where the prompt has no leverage on sampling.

This re-runs the real question at working guidance: one frame + a full R2R
instruction -> does the action chunk match the expert's next 16 steps? The
SWAPPED control (same frame, a different trajectory's instruction) separates
"grounded in the instruction" from "visual prior that happens to look right".

Uses the resampled video, which the model reads at yaw corr 0.987.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import json, math, glob, pathlib, argparse
import numpy as np, torch
import imageio.v3 as iio
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from data.ego_pose_9d import rot_from_6d

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--guidance", type=float, nargs="+", default=[5.0, 7.5])
args = ap.parse_args()

def yaw(a9):
    R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))

dirs = sorted(glob.glob("data/video/val_unseen/*/traj*"))
by_scan = {}
for d in dirs:
    by_scan.setdefault(pathlib.Path(d).parent.name, []).append(d)
sel = []
for i in range(max(len(v) for v in by_scan.values())):
    for s in sorted(by_scan):
        if i < len(by_scan[s]): sel.append(by_scan[s][i])
    if len(sel) >= args.n: break
sel = sel[:args.n]
print(f"{len(sel)} trajectories across {len({pathlib.Path(d).parent.name for d in sel})} scans", flush=True)

# pick a start index where the expert is actually turning -- a chunk of pure
# straight-ahead cannot distinguish a grounded policy from a forward prior
items = []
for d in sel:
    m = json.load(open(f"{d}/meta.json"))
    a = np.load(f"{d}/action9d.npy")
    if len(a) < args.chunk + 1: continue
    ys = np.array([abs(yaw(x)) for x in a])
    win = np.convolve(ys, np.ones(args.chunk), "valid")
    start = int(np.argmax(win))
    items.append((d, m, start, a[start:start + args.chunk]))
print(f"{len(items)} usable; mean |yaw| in chosen window "
      f"{np.mean([np.abs([yaw(x) for x in it[3]]).mean() for it in items]):.2f} deg\n", flush=True)

pipe = Cosmos3OmniPipeline.from_pretrained("checkpoints/Cosmos3-Edge", dtype=torch.bfloat16).to("cuda")

def run(img, prompt, fps, guid):
    o = pipe(prompt=prompt,
        action=CosmosActionCondition(mode="policy", chunk_size=args.chunk, domain_name="av",
            resolution_tier=480, view_point="ego_view", image=img),
        fps=fps, num_inference_steps=args.steps, guidance_scale=guid,
        use_system_prompt=False, generator=torch.Generator(device="cuda").manual_seed(0))
    a = o.action
    if isinstance(a,(list,tuple)): a = a[0]
    if hasattr(a,"detach"): a = a.detach().float().cpu().numpy()
    return np.asarray(a, np.float64).reshape(-1,9)[:args.chunk]

for guid in args.guidance:
    print(f"########  guidance_scale = {guid}  ########", flush=True)
    acc = {"correct": ([],[]), "swapped": ([],[])}
    for k, (d, m, start, gt) in enumerate(items):
        img = Image.fromarray(iio.imread(f"{d}/frames.mp4", plugin="pyav")[start]).convert("RGB").resize((480,480))
        other = items[(k + len(items)//2) % len(items)][1]["instructions"][0]
        for tag, pr in (("correct", m["instructions"][0]), ("swapped", other)):
            acc[tag][0].append(run(img, pr, m["fps"], guid)); acc[tag][1].append(gt)
        print(f"  [{k+1}/{len(items)}]", flush=True)
    for tag in ("correct","swapped"):
        P, T = np.concatenate(acc[tag][0]), np.concatenate(acc[tag][1])
        py = np.array([yaw(x) for x in P]); ty = np.array([yaw(x) for x in T])
        msk = np.abs(ty) > 1.0
        print(f"  {tag:>8}: yaw corr {np.corrcoef(py,ty)[0,1]:+.3f}   "
              f"fwd corr {np.corrcoef(P[:,2],T[:,2])[0,1]:+.3f}   "
              f"turn dir {(np.sign(py[msk])==np.sign(ty[msk])).mean():.3f} (n={msk.sum()})", flush=True)
    print()
