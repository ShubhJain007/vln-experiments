#!/usr/bin/env python3
"""Zero-shot POLICY mode: can Cosmos3-Edge produce a navigation action chunk?

Inverse dynamics only proved the perception side transfers -- it infers motion
from video that already shows the motion. This is the real task: one frame plus
a language instruction -> the next 16 ego-pose deltas. No training.

Also runs a SWAPPED-INSTRUCTION control (same frame, another trajectory's
instruction). If correct and swapped score the same, the policy is ignoring
language and running on visual priors alone -- the same ablation that exposed
grounding in our own model.

Baseline any finetune must beat.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import json, math, pathlib, argparse
import numpy as np, torch
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
CK = "checkpoints/Cosmos3-Edge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    from data.ego_pose_9d import episode_to_9d, habitat_to_cosmos, rot_from_6d
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition

    def yaw(a9):
        R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
        return math.degrees(math.atan2(R[0, 2], R[2, 2]))

    need = args.chunk + 1
    by_scan, seen = {}, set()
    for f in sorted((_ROOT / "data/rollouts/val_unseen").glob("*/*/episode.json")):
        d = json.loads(f.read_text())
        key = (d["scan"], d["trajectory_id"])
        if key in seen or len(d["positions"]) < need + 2:
            continue
        seen.add(key)
        by_scan.setdefault(d["scan"], []).append((f.parent, d))
    eps = []
    for i in range(max(len(v) for v in by_scan.values())):
        for s in sorted(by_scan):
            if i < len(by_scan[s]):
                eps.append(by_scan[s][i])
        if len(eps) >= args.episodes:
            break
    eps = eps[:args.episodes]
    print(f"{len(eps)} trajectories across {len({d['scan'] for _, d in eps})} scans", flush=True)

    pipe = Cosmos3OmniPipeline.from_pretrained(CK, dtype=torch.bfloat16).to("cuda")
    print(f"free {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB\n", flush=True)

    def run(img, prompt):
        out = pipe(
            prompt=prompt,
            action=CosmosActionCondition(
                mode="policy", chunk_size=args.chunk, domain_name="av",
                resolution_tier=480, view_point="ego_view", image=img,
            ),
            fps=10, num_inference_steps=args.steps, guidance_scale=1.0,
            use_system_prompt=False,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        a = out.action
        if isinstance(a, (list, tuple)): a = a[0]
        if hasattr(a, "detach"): a = a.detach().float().cpu().numpy()
        return np.asarray(a, dtype=np.float64).reshape(-1, 9)[:args.chunk]

    res = {"correct": ([], []), "swapped": ([], [])}
    for k, (ep_dir, d) in enumerate(eps):
        img = Image.open(sorted((ep_dir/"frames").glob("*.jpg"))[0]).convert("RGB").resize((480,480))
        gt = np.stack([habitat_to_cosmos(a) for a in episode_to_9d(d)])[:args.chunk]
        other = eps[(k + len(eps)//2) % len(eps)][1]["instruction"]
        for tag, prompt in (("correct", d["instruction"]), ("swapped", other)):
            res[tag][0].append(run(img, prompt)); res[tag][1].append(gt)
        print(f"  [{k+1}/{len(eps)}] {ep_dir.parent.name}/{ep_dir.name}", flush=True)

    print("\n" + "="*62)
    for tag in ("correct", "swapped"):
        P, T = np.concatenate(res[tag][0]), np.concatenate(res[tag][1])
        py = np.array([yaw(a) for a in P]); ty = np.array([yaw(a) for a in T])
        m = np.abs(ty) > 5
        print(f"\n  {tag.upper()} instruction  ({len(P)} frames)")
        print(f"    trans_z(fwd)  corr {np.corrcoef(P[:,2],T[:,2])[0,1]:+.3f}"
              f"   pred std {P[:,2].std():.4f}  true std {T[:,2].std():.4f}")
        print(f"    yaw           corr {np.corrcoef(py,ty)[0,1]:+.3f}"
              f"   pred std {py.std():.2f}d  true std {ty.std():.2f}d")
        if m.sum():
            print(f"    turn direction     {(np.sign(py[m])==np.sign(ty[m])).mean():.3f}  (n={m.sum()})")
    print("\n  If correct ~= swapped, policy mode is ignoring the instruction.")


if __name__ == "__main__":
    raise SystemExit(main())
