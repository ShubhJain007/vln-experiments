#!/usr/bin/env python3
"""
Validate collected expert rollouts before any training runs on them.

AGENTS.md Sec. 3.4 insists the METRICS be verified against ground truth before
evaluating a model. This is the same idea one level earlier: verify the DATA
before training on it. A corrupt demonstration set fails silently -- the model
trains, the loss goes down, and the gate is missed for reasons no amount of
model debugging will find.

Checks, in order of how badly each would hurt:
  1. Expert metrics with REAL geodesic distance. A correct expert must score
     SR ~1.0. Anything less means the demonstrations do not solve the task.
  2. nDTW -- the only metric sensitive to ROUTE, i.e. whether the trajectory
     actually follows the instruction's path rather than a shortcut.
  3. frames == actions per episode, and exactly one STOP, at the end.
  4. Action distribution and episode lengths, for degenerate cases.

RUN IN `habitat_render` (needs habitat for geodesic distance):
    conda activate habitat_render
    python scripts/validate_rollouts.py --split val_unseen
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse
import json
import pathlib
from collections import Counter

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from eval.metrics import aggregate, evaluate_episode  # noqa: E402
from model.action_space import Action  # noqa: E402

SCENES = _ROOT / "data" / "scene_datasets" / "mp3d"


def geodesic_fn(pathfinder):
    import habitat_sim

    def geo(a, b):
        p = habitat_sim.ShortestPath()
        p.requested_start = np.array(a, dtype=np.float32)
        p.requested_end = np.array(b, dtype=np.float32)
        return p.geodesic_distance if pathfinder.find_path(p) else float("inf")

    return geo


def main():
    import habitat_sim

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--rollouts", type=pathlib.Path,
                    default=_ROOT / "data" / "rollouts")
    ap.add_argument("--limit-per-scan", type=int, default=None)
    ap.add_argument("--check-frames", action="store_true",
                    help="also stat every frame file (slower)")
    args = ap.parse_args()

    root = args.rollouts / args.split
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return 1

    results, issues = [], []
    actions = Counter()
    lengths, statuses = [], Counter()
    n_ep = 0

    for scan_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        scan = scan_dir.name
        eps = []
        for d in sorted(scan_dir.iterdir()):
            meta = d / "episode.json"
            if meta.exists():
                eps.append((d, json.loads(meta.read_text())))
        if args.limit_per_scan:
            eps = eps[:args.limit_per_scan]
        if not eps:
            continue

        glb = SCENES / scan / f"{scan}.glb"
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(glb)
        cfg.enable_physics = False
        sim = habitat_sim.Simulator(
            habitat_sim.Configuration(cfg, [habitat_sim.AgentConfiguration()])
        )
        try:
            geo = geodesic_fn(sim.pathfinder)
            for d, e in eps:
                n_ep += 1
                acts = e["actions"]
                statuses[e.get("status", "?")] += 1
                lengths.append(len(acts))
                actions.update(Action(a).name for a in acts)

                # structural integrity
                if args.check_frames:
                    n_frames = len(list((d / "frames").glob("*.jpg")))
                    if n_frames != len(acts):
                        issues.append(f"{scan}/{d.name}: {n_frames} frames "
                                      f"!= {len(acts)} actions")
                if len(e["positions"]) != len(acts):
                    issues.append(f"{scan}/{d.name}: positions "
                                  f"{len(e['positions'])} != actions {len(acts)}")
                if not acts:
                    issues.append(f"{scan}/{d.name}: zero actions")
                    continue
                if acts[-1] != int(Action.STOP):
                    issues.append(f"{scan}/{d.name}: does not end with STOP")
                if acts.count(int(Action.STOP)) != 1:
                    issues.append(f"{scan}/{d.name}: "
                                  f"{acts.count(int(Action.STOP))} STOPs, expected 1")

                results.append(evaluate_episode(
                    path=e["positions"],
                    goal_position=e["goal_position"],
                    reference_path=e["reference_path"],
                    shortest_path_length=e["geodesic_distance"],
                    distance_fn=geo,
                ))
        finally:
            sim.close()
        print(f"  {scan}: {len(eps)} episodes checked")

    if not results:
        print("no episodes found")
        return 1

    agg = aggregate(results)
    sep = "=" * 66
    print(f"\n{sep}")
    print(f"  EXPERT ROLLOUT VALIDATION — split={args.split}")
    print(sep)
    print(f"  episodes            : {n_ep}")
    print(f"  statuses            : {dict(statuses)}")
    print()
    print("  Expert metrics (REAL geodesic distance):")
    print(f"    SR    = {agg['SR']:.4f}   <- must be ~1.0")
    print(f"    SPL   = {agg['SPL']:.4f}")
    print(f"    OS    = {agg['OS']:.4f}")
    print(f"    nDTW  = {agg['nDTW']:.4f}   <- route fidelity")
    print(f"    NE    = {agg['NE']:.4f} m")
    print()
    total = sum(actions.values())
    print("  Action distribution:")
    for name in ("FWD", "LEFT", "RIGHT", "STOP"):
        c = actions[name]
        print(f"    {name:6s} {c:8d}  ({100.0 * c / total:5.2f}%)")
    print()
    ls = sorted(lengths)
    print(f"  Episode length: min={ls[0]}  median={ls[len(ls)//2]}  "
          f"max={ls[-1]}  mean={sum(ls)/len(ls):.1f}")
    print(f"  Total steps (training pairs): {total}")
    print()
    if issues:
        print(f"  ISSUES: {len(issues)}")
        for i in issues[:20]:
            print(f"    {i}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")
    else:
        print("  ISSUES: none")
    print(sep)

    ok = not issues and agg["SR"] > 0.99
    print("  VERDICT:", "PASS" if ok else "FAIL — do not train on this data")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
