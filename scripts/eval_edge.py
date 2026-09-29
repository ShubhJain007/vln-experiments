#!/usr/bin/env python3
"""Closed-loop eval of Cosmos3-Edge policy mode on R2R, via discrete primitives.

Mirrors eval_pointing.py so SR / SPL / nDTW / NE stay directly comparable with
the 0.4267 baseline: same episodes, same metrics, same proximity-break
semantics. The only change is the policy.

Each model call emits a 9D action chunk, which is replayed into habitat
primitives (see eval.chunk_to_primitives -- verified to reproduce the expert's
own 71/15/14 action split from ground-truth chunks). Executing the whole chunk
per call is the point of action chunking, and it is also what makes this
affordable: one ~8s diffusion sample buys several primitives.

Guidance matters enormously here. At guidance_scale=1.0 the instruction barely
reaches the sampler (turn direction 0.594 vs 0.531 swapped); at 7.5 it is
0.675 vs 0.495. Always report the guidance used.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, importlib.util, math, pathlib, time
from collections import Counter

import numpy as np
import torch
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
SCENES = _ROOT / "data" / "scene_datasets" / "mp3d"
CK = "checkpoints/Cosmos3-Edge"


def rollout(sim, pipe, cond, episode, args, rot_from_6d, chunk_to_primitives):
    from diffusers import CosmosActionCondition

    goal = episode["goals"][0]["position"]
    instruction = args.prompt_template.format(
        instruction=episode["instruction"]["instruction_text"]).strip()
    frame, pos = sim.reset(episode["start_position"], episode["start_rotation"])

    positions, reason, min_dist, n_calls = [], "timeout", float("inf"), 0
    executed = 0
    mags = []          # per-call (net_forward_m, net_yaw_deg): the magnitudes
                       # that decide whether a chunk stalls
    empty_run = 0      # consecutive chunks that planned nothing executable
    while executed < args.max_steps:
        positions.append(pos)
        d = sim.geodesic(pos, goal)
        min_dist = min(min_dist, d)
        if d <= args.success_radius:
            reason = "within_radius"
            break

        img = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB").resize((480, 480))
        out = pipe(prompt=instruction,
                   action=CosmosActionCondition(
                       mode="policy", chunk_size=args.chunk, domain_name="av",
                       resolution_tier=480, view_point="ego_view", image=img),
                   fps=15.0, num_inference_steps=args.steps,
                   guidance_scale=args.guidance, use_system_prompt=False,
                   generator=torch.Generator(device="cuda").manual_seed(0))
        n_calls += 1
        a = out.action
        if isinstance(a, (list, tuple)): a = a[0]
        if hasattr(a, "detach"): a = a.detach().float().cpu().numpy()
        a = np.asarray(a, np.float64).reshape(-1, 9)[:args.chunk]

        acts, net_fwd, net_yaw = chunk_to_primitives(a, rot_from_6d,
                                                     translation_scale=args.translation_scale)
        # NATIVE stop only. The `av` domain has no STOP token, so the model's
        # own signal is a chunk carrying less than one primitive's worth of
        # motion (<0.25 m and <15 deg) -- it emits nothing to execute. Any
        # magnitude threshold above that is the harness deciding, not the
        # model, and a hand-set gate ended 75% of episodes in the first run.
        mags.append((net_fwd, net_yaw))
        # A STOP has to be a sustained decision, not one hesitant window.
        # 32% of chunks plan under 0.25 m, so terminating on the first empty
        # chunk ended 85% of episodes on a single 1.6 s pause -- the harness
        # deciding, not the model. Re-plan from the same pose instead, and
        # only stop once the model declines to move `stop_patience` times.
        if not acts:
            empty_run += 1
            if empty_run >= args.stop_patience:
                reason = "model_stop"
                break
            continue
        empty_run = 0

        for name in acts:   # the WHOLE chunk, then re-infer
            frame, pos = sim.step(name)
            executed += 1
            positions.append(pos)
            d = sim.geodesic(pos, goal)
            min_dist = min(min_dist, d)
            if d <= args.success_radius:
                reason = "within_radius"
                break
            if executed >= args.max_steps:
                break
        if reason == "within_radius":
            break

    return {"positions": positions, "termination_reason": reason,
            "min_dist_to_goal": min_dist, "n_calls": n_calls, "mags": mags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--translation-scale", type=float, default=7.47)
    ap.add_argument("--stop-patience", type=int, default=2,
                    help="consecutive empty chunks required to declare STOP")
    ap.add_argument("--success-radius", type=float, default=3.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--worker-python", default=None)
    ap.add_argument("--prompt-template", default="{instruction}",
                    help="prompt sent to the action model; '{instruction}' is "
                         "substituted with the R2R text. A template without the "
                         "placeholder is a fixed-prompt control arm.")
    ap.add_argument("--arm-name", default="unnamed")
    ap.add_argument("--results-json", default=None,
                    help="write this run's metrics as JSON for the ablation table")
    args = ap.parse_args()

    _spec = importlib.util.spec_from_file_location(
        "eval_stage0_prime", _ROOT / "scripts" / "eval_stage0_prime.py")
    _eval = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eval)
    from eval.metrics import aggregate, evaluate_episode
    from eval.remote_sim import DEFAULT_WORKER_PY, RemoteSim
    from eval.chunk_to_primitives import chunk_to_primitives
    from data.ego_pose_9d import rot_from_6d
    from diffusers import Cosmos3OmniPipeline

    if args.strict:
        args.success_radius = -1.0
    print(f"Cosmos3-Edge policy  chunk={args.chunk}  guidance={args.guidance}  "
          f"steps={args.steps}  scale={args.translation_scale}")

    episodes = _eval.load_episodes(args.split, None, args.limit)
    by_scan = {}
    for e in episodes:
        by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
    print(f"split={args.split}  episodes={len(episodes)}  scans={len(by_scan)}\n", flush=True)

    pipe = Cosmos3OmniPipeline.from_pretrained(CK, dtype=torch.bfloat16).to("cuda")

    print(f'  ARM "{args.arm_name}"\n  template: {args.prompt_template[:150]}\n')
    results, reasons = [], Counter()
    min_dists, final_dists, stop_dists, calls, all_mags = [], [], [], [], []
    t0 = time.time()
    with RemoteSim(args.worker_python or DEFAULT_WORKER_PY) as sim:
        geo = sim.geodesic_fn()
        for scan, eps in sorted(by_scan.items()):
            glb = SCENES / scan / f"{scan}.glb"
            if not glb.exists():
                continue
            sim.load_scene(glb)
            for e in eps:
                r = rollout(sim, pipe, None, e, args, rot_from_6d, chunk_to_primitives)
                reasons[r["termination_reason"]] += 1
                min_dists.append(r["min_dist_to_goal"])
                calls.append(r["n_calls"])
                all_mags.extend(r["mags"])
                fd = geo(r["positions"][-1], e["goals"][0]["position"])
                final_dists.append(fd)
                if r["termination_reason"] == "model_stop":
                    stop_dists.append(fd)
                results.append(evaluate_episode(
                    path=r["positions"], goal_position=e["goals"][0]["position"],
                    reference_path=e["reference_path"],
                    shortest_path_length=e["info"]["geodesic_distance"],
                    distance_fn=geo))
                print(f"  [{len(results)}/{len(episodes)}] {scan} "
                      f"{r['termination_reason']:>13}  min_d {r['min_dist_to_goal']:5.2f}  "
                      f"calls {r['n_calls']:3d}  {(time.time()-t0)/len(results):.0f}s/ep", flush=True)

    m = aggregate(results)
    print("\n" + "=" * 62)
    print(f"  COSMOS3-EDGE zero-shot  guidance={args.guidance}  chunk={args.chunk}")
    for k in ("SR", "SPL", "OS", "nDTW", "NE"):
        if k in m: print(f"    {k:5s} = {m[k]:.4f}")
    print("\n  TERMINATION:")
    for k, v in reasons.most_common():
        print(f"    {k:18s} {v:4d}  ({v/len(results)*100:5.1f}%)")
    md = np.array(min_dists); fdv = np.array(final_dists)
    print(f"\n  closest approach : mean {md.mean():5.2f}  median {np.median(md):5.2f}")
    print(f"  final position   : mean {fdv.mean():5.2f}  median {np.median(fdv):5.2f}")
    for r_ in (3.0, 5.0, 10.0):
        print(f"    ever within {r_:4.1f}m : {(md <= r_).mean()*100:5.1f}%")
    if stop_dists:
        sd = np.array(stop_dists)
        print(f"  where it STOPPED : mean {sd.mean():5.2f}  within 3m {(sd<=3).mean()*100:.1f}%")
    print(f"  model calls/episode: mean {np.mean(calls):.1f}  "
          f"(stop_patience={args.stop_patience})")
    if all_mags:
        mf = np.array([m[0] for m in all_mags]); my = np.array([m[1] for m in all_mags])
        print(f"\n  CHUNK MAGNITUDES over {len(mf)} calls (a chunk stalls if "
              f"fwd<0.25m AND |yaw|<15deg):")
        print(f"    net forward : mean {mf.mean():6.3f} m  median {np.median(mf):6.3f}  "
              f"below 0.25m: {(mf<0.25).mean()*100:.0f}%")
        print(f"    net |yaw|   : mean {np.abs(my).mean():6.2f} d  median "
              f"{np.median(np.abs(my)):6.2f}  below 15d: {(np.abs(my)<15).mean()*100:.0f}%")
        print(f"    implied speed: {mf.mean()/(args.chunk/15.0):.2f} m/s "
              f"(our data 0.58, their av prior 6.28)")
    print("=" * 62)

    if args.results_json:
        import json as _json
        mf = np.array([x[0] for x in all_mags]) if all_mags else np.zeros(1)
        my = np.array([x[1] for x in all_mags]) if all_mags else np.zeros(1)
        rec = {
            "arm": args.arm_name, "prompt_template": args.prompt_template,
            "n_episodes": len(results), "split": args.split,
            "chunk": args.chunk, "guidance": args.guidance,
            "inference_steps": args.steps, "stop_patience": args.stop_patience,
            "translation_scale": args.translation_scale,
            "SR": m.get("SR"), "SPL": m.get("SPL"), "OS": m.get("OS"),
            "nDTW": m.get("nDTW"), "NE": m.get("NE"),
            "term_model_stop": reasons.get("model_stop", 0),
            "term_timeout": reasons.get("timeout", 0),
            "term_within_radius": reasons.get("within_radius", 0),
            "min_dist_mean": float(md.mean()),
            "within_3m_pct": float((md <= 3).mean() * 100),
            "within_5m_pct": float((md <= 5).mean() * 100),
            "calls_per_episode": float(np.mean(calls)),
            "chunk_fwd_mean_m": float(mf.mean()),
            "chunk_fwd_below_025_pct": float((mf < 0.25).mean() * 100),
            "chunk_yaw_abs_mean_deg": float(np.abs(my).mean()),
            "implied_speed_mps": float(mf.mean() / (args.chunk / 15.0)),
        }
        pathlib.Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.results_json).write_text(_json.dumps(rec, indent=1))
        print(f"  wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
