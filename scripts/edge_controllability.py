#!/usr/bin/env python3
"""Does policy mode obey SHORT motion commands natively?

The earlier policy test fed a full R2R instruction into a ~1s action window and
got chance. But the action caption's `description` field was trained on
captions of the motion IN THAT CLIP, not a long-horizon goal -- so that result
does not show the policy head is broken, only that the prompt was out of
distribution.

This asks the fair question: given one frame and "turn left" / "go straight" /
"turn right", does the emitted 9D chunk actually turn that way?

If yes, the policy head works natively and the problem is translating long
instructions into local commands -- a much smaller job than retraining it.
If no, post-training is unavoidable.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import json, math, glob, pathlib
import numpy as np, torch
import imageio.v3 as iio
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from data.ego_pose_9d import rot_from_6d

CHUNK = 16
COMMANDS = {
    "turn left":    "The camera turns left.",
    "go straight":  "The camera moves forward.",
    "turn right":   "The camera turns right.",
    "stop":         "The camera remains stationary.",
}

def yaw(a9):
    R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))

pipe = Cosmos3OmniPipeline.from_pretrained("checkpoints/Cosmos3-Edge", dtype=torch.bfloat16).to("cuda")

dirs = sorted(glob.glob("data/video/val_unseen/*/traj*"))
by_scan = {}
for d in dirs:
    by_scan.setdefault(pathlib.Path(d).parent.name, []).append(d)
sel = [d for v in by_scan.values() for d in v[:2]][:22]
print(f"{len(sel)} conditioning frames from {len(sel)} scans\n", flush=True)

import itertools
for GUID in (1.0, 3.0, 5.0, 7.5):
  print(f"\n########  guidance_scale = {GUID}  ########", flush=True)
  res = {k: [] for k in COMMANDS}
  for d in sel:
      m = json.load(open(f"{d}/meta.json"))
      img = Image.fromarray(iio.imread(f"{d}/frames.mp4", plugin="pyav")[0]).convert("RGB").resize((480,480))
      for name, prompt in COMMANDS.items():
          out = pipe(prompt=prompt,
              action=CosmosActionCondition(mode="policy", chunk_size=CHUNK, domain_name="av",
                  resolution_tier=480, view_point="ego_view", image=img),
              fps=m["fps"], num_inference_steps=30, guidance_scale=GUID,
              use_system_prompt=False,
              generator=torch.Generator(device="cuda").manual_seed(0))
          a = out.action
          if isinstance(a,(list,tuple)): a = a[0]
          if hasattr(a,"detach"): a = a.detach().float().cpu().numpy()
          a = np.asarray(a, np.float64).reshape(-1,9)[:CHUNK]
          res[name].append((np.array([yaw(x) for x in a]).mean(), a[:,2].mean()))

  print(f"  {'command':>14} {'mean yaw/frame':>16} {'mean fwd/frame':>16}")
  for k in COMMANDS:
      y = np.array([r[0] for r in res[k]]); z = np.array([r[1] for r in res[k]])
      print(f"  {k:>14} {y.mean():>+11.2f} deg {z.mean():>+13.4f}")

  L = np.array([r[0] for r in res["turn left"]])
  R = np.array([r[0] for r in res["turn right"]])
  print(f"  left vs right separation: {(R-L).mean():+.2f} deg   "
        f"sign correct on {((R-L)>0).mean()*100:.0f}% (n={len(L)})")
