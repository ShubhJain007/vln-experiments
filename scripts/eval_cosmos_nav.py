#!/usr/bin/env python3
"""Closed-loop R2R eval of the two-stage Cosmos3-Edge navigator.

Same episodes, metrics and proximity-break semantics as eval_pointing.py /
eval_edge.py, so SR / SPL / nDTW / NE are comparable with the 0.4267 pointing
baseline and the 0.0750 stock-diffusion result.

Loop per episode:
  observe frame -> reasoner decides a command from the last 16 frames + the
  instruction -> execute (direct burst | diffusion chunk) -> re-plan.
  STOP only when the reasoner says "stop" `stop_patience` times in a row.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, importlib.util, json, pathlib, time
from collections import Counter

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
SCENES = _ROOT / "data" / "scene_datasets" / "mp3d"


def rollout(sim, nav, episode, args):
    goal = episode["goals"][0]["position"]
    instruction = episode["instruction"]["instruction_text"]
    frame, pos = sim.reset(episode["start_position"], episode["start_rotation"])
    nav.reset(); nav.observe(frame)

    positions, reason, min_dist = [], "timeout", float("inf")
    executed = calls = stops = unparsed = 0
    cmds, hows = Counter(), Counter()
    while executed < args.max_steps:
        positions.append(pos)
        d = sim.geodesic(pos, goal); min_dist = min(min_dist, d)
        if d <= args.success_radius:
            reason = "within_radius"; break

        if args.policy == "diffusion_grounded":
            cmd, is_stop = nav.decide_caption(instruction)
            cmds[(cmd or "UNPARSED").split("|")[0].strip()[:40]] += 1
            hows[nav.last_how] += 1
            if calls < 2:
                print(f"      caption[{calls}] ({nav.last_how}): {cmd}", flush=True)
            if cmd is None:
                unparsed += 1
                cmd, is_stop = "The camera moves forward.", args.unparsed_default != "forward"
            if is_stop:
                cmd = "stop"
        else:
            cmd = nav.decide(instruction)
            cmds[cmd or "UNPARSED"] += 1
            if cmd is None:
                unparsed += 1
                cmd = "move forward" if args.unparsed_default == "forward" else "stop"
        calls += 1
        if cmd == "stop":
            stops += 1
            if stops >= args.stop_patience:
                reason = "model_stop"; break
            continue
        stops = 0

        for name in nav.plan(cmd, frame):
            frame, pos = sim.step(name); nav.observe(frame)
            executed += 1; positions.append(pos)
            d = sim.geodesic(pos, goal); min_dist = min(min_dist, d)
            if d <= args.success_radius:
                reason = "within_radius"; break
            if executed >= args.max_steps:
                break
        if reason == "within_radius":
            break
    return dict(positions=positions, termination_reason=reason, min_dist_to_goal=min_dist,
                n_calls=calls, unparsed=unparsed, cmds=cmds, hows=hows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["direct", "diffusion", "diffusion_grounded"], default="direct")
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--stop-patience", type=int, default=2)
    ap.add_argument("--unparsed-default", choices=["forward", "stop"], default="forward")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--fwd-steps", type=int, default=2)
    ap.add_argument("--turn-steps", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--success-radius", type=float, default=3.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--worker-python", default=None)
    ap.add_argument("--arm-name", default="cosmos_nav")
    ap.add_argument("--results-json", default=None)
    ap.add_argument("--adapter", default=None, help="SFT'd reasoner adapter dir")
    args = ap.parse_args()
    if args.strict:
        args.success_radius = -1.0

    _spec = importlib.util.spec_from_file_location("eval_stage0_prime", _ROOT / "scripts" / "eval_stage0_prime.py")
    _eval = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_eval)
    from eval.metrics import aggregate, evaluate_episode
    from eval.remote_sim import DEFAULT_WORKER_PY, RemoteSim
    from nav.cosmos_navigator import CosmosNavigator

    print(f'ARM "{args.arm_name}"  policy={args.policy}  think={not args.no_think}  '
          f'burst fwd={args.fwd_steps} turn={args.turn_steps}  patience={args.stop_patience}'
          f'  adapter={args.adapter}')
    episodes = _eval.load_episodes(args.split, None, args.limit)
    by_scan = {}
    for e in episodes:
        by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
    print(f"split={args.split}  episodes={len(episodes)}  scans={len(by_scan)}\n", flush=True)

    nav = CosmosNavigator(policy=args.policy, chunk=args.chunk, steps=args.steps,
                          guidance=args.guidance, think=not args.no_think,
                          fwd_steps=args.fwd_steps, turn_steps=args.turn_steps,
                          adapter=args.adapter)

    results, reasons, cmds, hows = [], Counter(), Counter(), Counter()
    min_d, final_d, stop_d, calls, unparsed = [], [], [], [], 0
    t0 = time.time()
    with RemoteSim(args.worker_python or DEFAULT_WORKER_PY) as sim:
        geo = sim.geodesic_fn()
        for scan, eps in sorted(by_scan.items()):
            glb = SCENES / scan / f"{scan}.glb"
            if not glb.exists():
                continue
            sim.load_scene(glb)
            for e in eps:
                r = rollout(sim, nav, e, args)
                reasons[r["termination_reason"]] += 1; cmds.update(r["cmds"]); hows.update(r["hows"])
                min_d.append(r["min_dist_to_goal"]); calls.append(r["n_calls"]); unparsed += r["unparsed"]
                fd = geo(r["positions"][-1], e["goals"][0]["position"]); final_d.append(fd)
                if r["termination_reason"] == "model_stop":
                    stop_d.append(fd)
                results.append(evaluate_episode(
                    path=r["positions"], goal_position=e["goals"][0]["position"],
                    reference_path=e["reference_path"],
                    shortest_path_length=e["info"]["geodesic_distance"], distance_fn=geo))
                print(f"  [{len(results)}/{len(episodes)}] {scan} {r['termination_reason']:>13}  "
                      f"min_d {r['min_dist_to_goal']:5.2f}  calls {r['n_calls']:3d}  "
                      f"{(time.time()-t0)/len(results):.0f}s/ep", flush=True)

    m = aggregate(results); md = np.array(min_d); fdv = np.array(final_d); n = len(results)
    print("\n" + "=" * 62 + f"\n  COSMOS NAV  {args.arm_name}  policy={args.policy}")
    for k in ("SR", "SPL", "OS", "nDTW", "NE"):
        if k in m: print(f"    {k:5s} = {m[k]:.4f}")
    print("\n  TERMINATION:")
    for k, v in reasons.most_common():
        print(f"    {k:18s} {v:4d}  ({v/n*100:5.1f}%)")
    print(f"\n  closest approach : mean {md.mean():5.2f}  median {np.median(md):5.2f}")
    for r_ in (3.0, 5.0, 10.0):
        print(f"    ever within {r_:4.1f}m : {(md <= r_).mean()*100:5.1f}%")
    if stop_d:
        sd = np.array(stop_d); print(f"  where it STOPPED : mean {sd.mean():5.2f}  within 3m {(sd<=3).mean()*100:.1f}%")
    tot = sum(cmds.values())
    print(f"  reasoner calls/episode: {np.mean(calls):.1f}   unparsed {unparsed}/{tot}")
    print("  commands: " + "  ".join(f"{k} {v/tot*100:.0f}%" for k, v in cmds.most_common()))
    if hows:
        th = sum(hows.values())
        print("  caption parse: " + "  ".join(f"{k} {v/th*100:.0f}%" for k, v in hows.most_common()))
    print("=" * 62)

    if args.results_json:
        rec = dict(arm=args.arm_name, policy=args.policy, think=not args.no_think,
                   prompt_template="reasoner:HIER_OVERALL+EMBODIMENT -> " +
                                   {"direct": "direct burst",
                                    "diffusion": "fixed caption -> diffusion",
                                    "diffusion_grounded": "AgiBot-schema caption -> diffusion"}[args.policy],
                   n_episodes=n, split=args.split, chunk=args.chunk, guidance=args.guidance,
                   inference_steps=args.steps, stop_patience=args.stop_patience,
                   translation_scale=7.47,
                   SR=m.get("SR"), SPL=m.get("SPL"), OS=m.get("OS"), nDTW=m.get("nDTW"), NE=m.get("NE"),
                   term_model_stop=reasons.get("model_stop", 0), term_timeout=reasons.get("timeout", 0),
                   term_within_radius=reasons.get("within_radius", 0),
                   min_dist_mean=float(md.mean()), within_3m_pct=float((md <= 3).mean()*100),
                   within_5m_pct=float((md <= 5).mean()*100), calls_per_episode=float(np.mean(calls)),
                   chunk_fwd_mean_m=float("nan"), chunk_fwd_below_025_pct=float("nan"),
                   chunk_yaw_abs_mean_deg=float("nan"), implied_speed_mps=float("nan"),
                   commands={k: v / tot for k, v in cmds.items()}, unparsed=unparsed,
                   caption_parse={k: v / max(1, sum(hows.values())) for k, v in hows.items()})
        p = pathlib.Path(args.results_json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1)); print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
