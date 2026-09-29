#!/usr/bin/env python3
"""Zero-shot: does Cosmos3-Edge's pretrained action prior read OUR video?

Inverse dynamics (video -> ego-motion) on habitat frames, against our own
ground-truth 9D. No training. If the predicted trajectory correlates with
truth, the prior transfers to indoor walking and post-training is adaptation.
If it is noise, we are pretraining from scratch either way and Edge buys much
less than hoped.

Two candidate 9D domains: "av" (planar, forward-dominant, like a walker) and
"camera_pose" (free camera, but indoor visual domain). Both are raw_action_dim
9, so both are drop-in.

Our frames are still discrete primitives (instant 15 deg yaws) -- the dynamics
mismatch the resample is meant to fix -- so this is the pessimistic case,
measured before spending a day on the re-render.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import json, math, pathlib, argparse
import numpy as np, torch
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
CK = "checkpoints/Cosmos3-Edge"

PROMPT = {
    "av": "You are an autonomous vehicle planning system.",
    "camera_pose": "Egocentric camera moving through an indoor scene.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--domains", nargs="+", default=["av", "camera_pose"])
    args = ap.parse_args()

    from data.ego_pose_9d import episode_to_9d, habitat_to_cosmos, rot_from_6d
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition

    def yaw(a9):
        R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
        return math.degrees(math.atan2(R[0, 2], R[2, 2]))

    # R2R ships ~3 instructions per PATH, so consecutive episode ids share a
    # trajectory_id and an identical frame sequence. Deduping is essential:
    # without it "3 episodes" is one trajectory measured three times.
    need = args.chunk + 1
    by_scan, seen = {}, set()
    for f in sorted((_ROOT / "data/rollouts/val_unseen").glob("*/*/episode.json")):
        d = json.loads(f.read_text())
        key = (d["scan"], d["trajectory_id"])
        if key in seen or len(d["positions"]) < need + 2:
            continue
        seen.add(key)
        by_scan.setdefault(d["scan"], []).append((f.parent, d))
    # round-robin so the sample spans scans; a sorted glob otherwise returns
    # every trajectory from the alphabetically-first scan.
    eps = []
    for i in range(max(len(v) for v in by_scan.values())):
        for s in sorted(by_scan):
            if i < len(by_scan[s]):
                eps.append(by_scan[s][i])
        if len(eps) >= args.episodes:
            break
    eps = eps[:args.episodes]
    print(f"{len(eps)} distinct trajectories across "
          f"{len({d['scan'] for _, d in eps})} scans, chunk={args.chunk} ({need} frames)", flush=True)

    pipe = Cosmos3OmniPipeline.from_pretrained(CK, dtype=torch.bfloat16).to("cuda")
    print(f"pipeline on GPU, free {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB", flush=True)

    for dom in args.domains:
        print(f"\n############  domain = {dom}  ############", flush=True)
        allp, allt = [], []
        for ep_dir, d in eps:
            frames = sorted((ep_dir / "frames").glob("*.jpg"))[:need]
            video = [Image.open(p).convert("RGB").resize((480, 480)) for p in frames]
            gt = np.stack([habitat_to_cosmos(a) for a in episode_to_9d(d)])[:args.chunk]

            out = pipe(
                prompt=PROMPT[dom],
                action=CosmosActionCondition(
                    mode="inverse_dynamics", chunk_size=args.chunk,
                    domain_name=dom, resolution_tier=480, view_point="ego_view",
                    video=video,
                ),
                fps=10, num_inference_steps=args.steps, guidance_scale=1.0,
                use_system_prompt=False,
                generator=torch.Generator(device="cuda").manual_seed(0),
            )
            act = out.action
            if isinstance(act, (list, tuple)):
                act = act[0]
            if hasattr(act, "detach"):
                act = act.detach().float().cpu().numpy()
            pred = np.asarray(act, dtype=np.float64).reshape(-1, 9)[:args.chunk]
            py = np.array([yaw(a) for a in pred])
            ty = np.array([yaw(a) for a in gt])
            print(f"  {ep_dir.parent.name}/{ep_dir.name}")
            print(f"    pred |trans|/frame {np.abs(pred[:, :3]).mean(0).round(4)}"
                  f"   yaw mean {py.mean():+6.2f} std {py.std():5.2f}")
            print(f"    true |trans|/frame {np.abs(gt[:, :3]).mean(0).round(4)}"
                  f"   yaw mean {ty.mean():+6.2f} std {ty.std():5.2f}", flush=True)
            allp.append(pred)
            allt.append(gt)

        P, T = np.concatenate(allp), np.concatenate(allt)
        print(f"\n  === zero-shot agreement, {dom}, {len(P)} frames ===")
        for i, nm in enumerate(["trans_x", "trans_y(height)", "trans_z(fwd)"]):
            c = np.corrcoef(P[:, i], T[:, i])[0, 1] if T[:, i].std() > 1e-9 else float("nan")
            print(f"    {nm:>16}  corr {c:+.3f}   pred std {P[:, i].std():.4f}"
                  f"   true std {T[:, i].std():.4f}")
        py = np.array([yaw(a) for a in P])
        ty = np.array([yaw(a) for a in T])
        print(f"    {'yaw':>16}  corr {np.corrcoef(py, ty)[0, 1]:+.3f}   "
              f"pred std {py.std():.2f}d   true std {ty.std():.2f}d")
        m = np.abs(ty) > 5
        if m.sum():
            print(f"    turn-direction agreement (|true yaw|>5 deg): "
                  f"{(np.sign(py[m]) == np.sign(ty[m])).mean():.3f}  (n={m.sum()})")


if __name__ == "__main__":
    raise SystemExit(main())
