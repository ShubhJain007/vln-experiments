#!/usr/bin/env python3
"""Does LatentPilot's Pilot slot leak the action during training?

Stage 0'/1 train with the Pilot slot TEACHER-FORCED with the true embedding of the NEXT frame, v_bar_{t+1} (Eq. 12).
The next frame is the consequence of the current action, so the action can in principle be read off the slot
(inverse dynamics). At test time the slot holds the model's own previous latent z_{t-1} (Eq. 5) and that shortcut is
gone. This script measures it directly, offline, on held-out expert episodes.

For every step of every selected val_unseen expert episode (observations and instruction teacher-forced along the
expert path), the checkpoint predicts the next action with the slot filled four ways:

    next_frame        v_bar_{t+1}                 the TRAINING condition (privileged future frame)
    own_z             z_{t-1}, carried recurrently the TEST condition (Eq. 5), z_0 at the episode start
    current_frame     v_bar_t                     real, in-distribution, but no future information
    other_next_frame  v_bar_{t'+1} of another     a real next-frame embedding that does not belong to this step
                      episode                     (control: is the slot content actually being read?)

Everything else in the input is identical across conditions, so an accuracy gap is caused by the slot alone.
A memoryless checkpoint (Stage 0, no slot) is scored on the same steps as the reference.

    python scripts/test_slot_shortcut.py --checkpoint checkpoints/stage1_learned/final --name stage1_learned
    python scripts/test_slot_shortcut.py --checkpoint checkpoints/stage0/final --name stage0   # no-slot reference
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

ACTION_NAMES = ["FWD", "LEFT", "RIGHT", "STOP"]


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_episodes(split, per_scan):
    """First `per_scan` episodes (sorted ids) of every scan that have frames and a cached v_bar."""
    root = _ROOT / "data" / "rollouts" / split
    eps = []
    for scan_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        n = 0
        for ep_dir in sorted(scan_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            if n >= per_scan:
                break
            meta_path, vbar_path = ep_dir / "episode.json", ep_dir / "vbar.npy"
            if not (meta_path.exists() and vbar_path.exists()):
                continue
            meta = json.loads(meta_path.read_text())
            vbar = np.load(vbar_path)
            if vbar.shape[0] != len(meta["actions"]):
                continue
            eps.append({"dir": ep_dir, "meta": meta, "vbar": vbar})
            n += 1
    return eps


def confusion_summary(conf):
    conf = np.asarray(conf)
    total = conf.sum()
    out = {"n": int(total), "accuracy": float(np.trace(conf) / max(total, 1))}
    for i, a in enumerate(ACTION_NAMES):
        row = conf[i].sum()
        out[f"recall_{a}"] = float(conf[i, i] / row) if row else None
    out["confusion"] = conf.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--per-scan", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/slot_shortcut.json")
    args = ap.parse_args()

    import torch
    from data.rollout_dataset import load_frame as load_frame_rgb

    rng = np.random.default_rng(args.seed)
    ckpt = pathlib.Path(args.checkpoint)
    has_slot = (ckpt / "pilot.pt").exists()
    print(f"checkpoint {ckpt}  ({'Pilot slot' if has_slot else 'memoryless, no slot'})", flush=True)

    if has_slot:
        enc, builder, backbone, head, pilot = _load_module("eval_stage0_prime").load_policy(ckpt)
        conditions = ["next_frame", "own_z", "current_frame", "other_next_frame"]
    else:
        enc, builder, backbone, head = _load_module("eval_stage0").load_policy(ckpt)
        pilot = None
        conditions = ["no_slot"]
    device = enc.model.device

    episodes = select_episodes(args.split, args.per_scan)
    n_steps = sum(len(e["meta"]["actions"]) for e in episodes)
    print(f"{len(episodes)} episodes, {n_steps} steps, {len({e['meta']['scan'] for e in episodes})} scans", flush=True)

    # Freshness check: the cached v_bar must match what the frozen encoder produces for these frames now.
    e0 = episodes[0]
    with torch.no_grad():
        v0 = enc.encode(load_frame_rgb(e0["dir"] / "frames" / "0000.jpg")).float().mean(dim=0).cpu().numpy()
    cos = float(v0 @ e0["vbar"][0] / (np.linalg.norm(v0) * np.linalg.norm(e0["vbar"][0])))
    print(f"cached v_bar vs fresh encoding, cosine = {cos:.4f}", flush=True)
    if cos < 0.98:
        raise RuntimeError("cached v_bar does not match the frames; re-run src/data/cache_vbar.py first")

    def vb(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device, torch.bfloat16)

    conf = {c: np.zeros((4, 4), dtype=np.int64) for c in conditions}
    per_episode = {c: [] for c in conditions}
    t0 = time.time()
    done = 0
    for ei, ep in enumerate(episodes):
        meta, vbar = ep["meta"], ep["vbar"]
        actions = [int(a) for a in meta["actions"]]
        T = len(actions)
        caches = {"own_z": pilot.new_cache()} if has_slot else {}
        ep_correct = {c: 0 for c in conditions}
        for t in range(T):
            with torch.no_grad():
                v_t = enc.encode(load_frame_rgb(ep["dir"] / "frames" / f"{t:04d}.jpg"))
                for c in conditions:
                    if c == "no_slot":
                        seq = builder.build(meta["instruction"], v_t)
                    else:
                        if c == "next_frame":
                            slot = vb(vbar[min(t + 1, T - 1)])
                        elif c == "own_z":
                            slot = caches["own_z"].read()
                        elif c == "current_frame":
                            slot = vb(vbar[t])
                        else:  # other_next_frame: a real next-frame embedding from a different episode
                            j = int(rng.integers(len(episodes) - 1))
                            j = j + 1 if j >= ei else j
                            other = episodes[j]["vbar"]
                            k = int(rng.integers(other.shape[0]))
                            slot = vb(other[min(k + 1, other.shape[0] - 1)])
                        seq = builder.build(meta["instruction"], v_t, pilot_input=slot)
                    H = backbone.forward(seq, position_ids=backbone.build_position_ids(seq))
                    pred = int(head.logits(H[seq.action_index]).argmax().item())
                    conf[c][actions[t], pred] += 1
                    ep_correct[c] += int(pred == actions[t])
                    if c == "own_z":
                        caches["own_z"].write(pilot(H[seq.pilot_index]))
            done += 1
        for c in conditions:
            per_episode[c].append(ep_correct[c] / T)
        if (ei + 1) % 10 == 0 or ei == len(episodes) - 1:
            accs = "  ".join(f"{c} {np.trace(conf[c]) / conf[c].sum():.3f}" for c in conditions)
            print(f"  [{ei + 1}/{len(episodes)}] {done / (time.time() - t0):.1f} steps/s  {accs}", flush=True)

    result = {"checkpoint": str(ckpt), "split": args.split, "episodes": len(episodes), "steps": n_steps,
              "scans": sorted({e["meta"]["scan"] for e in episodes}), "per_scan": args.per_scan, "seed": args.seed,
              "vbar_cosine_check": cos,
              "conditions": {c: {**confusion_summary(conf[c]),
                                 "episode_accuracy_mean": float(np.mean(per_episode[c])),
                                 "episode_accuracy_std": float(np.std(per_episode[c]))} for c in conditions},
              "expert_action_counts": {a: int(conf[conditions[0]][i].sum()) for i, a in enumerate(ACTION_NAMES)},
              "date": time.strftime("%Y-%m-%d %H:%M")}
    out = pathlib.Path(args.out) if pathlib.Path(args.out).is_absolute() else _ROOT / args.out
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[args.name] = result
    out.write_text(json.dumps(allres, indent=1))
    print(json.dumps({c: round(result["conditions"][c]["accuracy"], 4) for c in conditions}), flush=True)
    print(f"wrote {out} [{args.name}]")


if __name__ == "__main__":
    main()
