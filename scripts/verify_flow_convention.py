#!/usr/bin/env python3
"""Is our flow-matching noising/target convention the one the model was trained on?

A flipped sign still yields a finite loss and a falling training curve, then
produces garbage at inference. So validate it against a mode where the
pretrained model is known-accurate: inverse dynamics on our resampled video
reads ego-motion at yaw corr 0.987.

If x_t = (1-s)*x0 + s*eps and the model predicts v = eps - x0, then
    x0_recon = x_t - s*v
must recover the true action. Sweeping s, a correct convention gives high
correlation that degrades gracefully as s -> 1; a wrong one anti-correlates.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import json, glob, pathlib
import numpy as np, torch
import imageio.v3 as iio
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
RAW = 9

from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from data.cosmos_dataset import CosmosChunkDataset
from train.cosmos_pack import _Proxy, _Captured

CHUNK = 16
pipe = Cosmos3OmniPipeline.from_pretrained("checkpoints/Cosmos3-Edge", dtype=torch.bfloat16).to("cuda")
ds = CosmosChunkDataset(split="val_unseen", chunk=CHUNK)

def capture_id(video, fps):
    real = pipe.transformer; proxy = _Proxy(real); pipe.transformer = proxy
    try:
        pipe(prompt="You are an autonomous vehicle planning system.",
             action=CosmosActionCondition(mode="inverse_dynamics", chunk_size=CHUNK,
                 domain_name="av", resolution_tier=480, view_point="ego_view", video=video),
             fps=fps, num_inference_steps=1, guidance_scale=1.0, use_system_prompt=False)
    except _Captured: pass
    finally: pipe.transformer = real
    return proxy.kwargs

dirs = sorted(glob.glob("data/video/val_unseen/*/traj*"))[:8]
print(f"{len(dirs)} trajectories\n")
print(f"  {'sigma':>6} {'corr(x0_recon, truth)':>22} {'corr with FLIPPED sign':>24}")

for sigma in (0.2, 0.4, 0.6, 0.8, 0.95):
    rec, flip, tru = [], [], []
    for d in dirs:
        m = json.load(open(f"{d}/meta.json"))
        a = np.load(f"{d}/action9d.npy")[:CHUNK]
        if len(a) < CHUNK: continue
        a = ds.scale_actions(a)
        frames = iio.imread(f"{d}/frames.mp4", plugin="pyav")[:CHUNK+1]
        video = [Image.fromarray(f).convert("RGB").resize((480,480)) for f in frames]
        kw = capture_id(video, m["fps"])

        x0 = torch.zeros_like(kw["action_tokens"][0]).float()
        x0[:, :RAW] = torch.as_tensor(a, device=x0.device, dtype=x0.dtype)
        eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(0)).to(x0.device)
        eps[:, RAW:] = 0
        xt = (1-sigma)*x0 + sigma*eps; xt[:, RAW:] = 0

        kw = dict(kw)
        kw["action_tokens"] = [xt.to(torch.bfloat16)]
        kw["action_timesteps"] = torch.full_like(kw["action_timesteps"], int(sigma*1000))
        with torch.no_grad():
            _, _, pa = pipe.transformer(**kw)
        v = (pa[0] if isinstance(pa,(list,tuple)) else pa).float()
        rec.append((xt - sigma*v)[:, :RAW].cpu().numpy())
        flip.append((xt + sigma*v)[:, :RAW].cpu().numpy())
        tru.append(a)
    R, F, T = (np.concatenate(x).ravel() for x in (rec, flip, tru))
    print(f"  {sigma:>6.2f} {np.corrcoef(R,T)[0,1]:>22.3f} {np.corrcoef(F,T)[0,1]:>24.3f}")

print("\n  correct convention -> left column high, degrading as sigma -> 1")
