#!/usr/bin/env python3
"""
Stage 0 policy evaluation — closed-loop rollout of the TRAINED model.

Runs the LoRA-adapted policy against real habitat, one action at a time:
encode frame (Eq.4) -> build sequence (Eq.5, no Pilot slot) -> backbone (Eq.6)
-> greedy action (Eq.7) -> step the simulator -> repeat.

RUN IN `latentpilot`. Habitat lives in a worker subprocess under
`habitat_render`; this script spawns it automatically.

    conda activate latentpilot
    python scripts/eval_stage0.py --checkpoint checkpoints/stage0/final --limit 200

--------------------------------------------------------------------------
WHY TWO PROCESSES
--------------------------------------------------------------------------
No habitat-sim build supports Python 3.10, while transformers/peft/accelerate
at the versions this project verified all REQUIRE >=3.10. Installing older ML
packages into the 3.9 habitat env would mean evaluating under a different
transformers than training used -- and every API this code leans on
(`get_image_features().pooler_output` as a tuple, `get_vision_position_ids`,
the `compute_3d_position_ids` gating behind the mRoPE fix) was verified against
5.16.1 specifically. Splitting the process keeps both environments exactly as
verified, with zero version drift between training and evaluation.

--------------------------------------------------------------------------
"WITHIN 3M = SUCCESS" -- A DIAGNOSTIC PROTOCOL, NOT THE STAGE 0 GATE
--------------------------------------------------------------------------
The current checkpoint has a confirmed STOP collapse (mean P(STOP) ~0.045 on
true-STOP frames, 0/30 predicted correctly). Under the paper's real protocol --
success requires the model to CHOOSE to stop within 3 m -- it would score SR~0
regardless of how good the navigation is, telling us nothing about whether the
FWD/LEFT/RIGHT behaviour is worth keeping.

So, per human instruction, the rollout treats "came within 3 m of the goal"
as terminating success (Sec. 4.1's OS definition, applied as an early-exit
rollout rule rather than computed post-hoc). This isolates navigation quality
from the broken stopping decision.

THIS IS NOT THE GATE. Table 3's numbers require strict SR. The termination
breakdown below is the honest signal: `model_stop` is real behaviour,
`within_radius` is a forced success, `timeout` is a genuine navigation failure.
Read the headline SR/OS as a CEILING on what fixing STOP could buy.
"""

import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]

import argparse
import gzip
import json
import pathlib
import time
from collections import Counter

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

SCENES = _ROOT / "data" / "scene_datasets" / "mp3d"
R2R_DIR = _ROOT / "data" / "R2R_VLNCE_v1-3"
DEFAULT_OUT = _ROOT / "data" / "eval" / "stage0"
DEFAULT_MAX_STEPS = 100          # ~2.5x the training data's median episode length
EXPECTED_LORA_PARAMS = 6_422_528


def load_episodes(split, scans=None, limit=None):
    with gzip.open(R2R_DIR / split / f"{split}.json.gz", "rt") as fh:
        eps = json.load(fh)["episodes"]
    if scans:
        wanted = set(scans)
        eps = [e for e in eps if e["scene_id"].split("/")[1] in wanted]
    return eps[:limit] if limit else eps


def load_policy(checkpoint, expected_lora_params=EXPECTED_LORA_PARAMS):
    """Backbone with the trained LoRA adapter attached, plus Eq.5/6/7 pieces.

    `PeftModel.from_pretrained` injects the adapter IN PLACE into the base
    module tree, so the return value is deliberately not kept -- everything
    downstream holds the base model, which is what owns `get_vision_position_ids`
    and `lm_head` (the PeftModel wrapper adds a nesting level that breaks those).
    Verified: base-model action probabilities change after the call and the LoRA
    parameter count matches training exactly.

    That is silent-failure-prone -- a wrong path or a no-op load would quietly
    evaluate the UNTRAINED model and read as "training did nothing" -- so the
    load is ASSERTED, not assumed.
    """
    from peft import PeftModel

    from model.action_head import ActionHead
    from model.backbone import Backbone
    from model.input_sequence import InputSequenceBuilder
    from model.vision_encoder import VisionEncoder

    enc = VisionEncoder()
    if checkpoint is not None:
        ckpt = pathlib.Path(checkpoint)
        if not (ckpt / "adapter_model.safetensors").exists():
            raise FileNotFoundError(
                f"no adapter_model.safetensors under {ckpt} -- refusing to run, "
                f"this would silently evaluate the untrained model"
            )
        PeftModel.from_pretrained(enc.model, str(ckpt))

        n_lora = sum(p.numel() for n, p in enc.model.named_parameters()
                     if "lora" in n)
        if n_lora == 0:
            raise RuntimeError(
                "adapter load produced 0 LoRA parameters -- checkpoint did not attach"
            )
        if n_lora != expected_lora_params:
            print(f"  WARNING: {n_lora:,} LoRA params, expected "
                  f"{expected_lora_params:,} (different rank or targets?)")
        print(f"  adapter verified: {n_lora:,} LoRA params active")
    else:
        print("  NO CHECKPOINT -- evaluating the UNTRAINED base model (baseline)")

    enc.model.eval()
    return (enc,
            InputSequenceBuilder(enc.model, enc.processor.tokenizer),
            Backbone(enc.model),
            ActionHead(enc.model, enc.processor.tokenizer))


def rollout_policy(sim, enc, builder, backbone, head, episode,
                   max_steps, success_radius):
    """Closed-loop rollout. Terminates on model STOP, forced success inside
    `success_radius`, or timeout.

    NOTE the success check runs BEFORE the model acts, so an episode already
    inside the radius at t=0 would succeed without the policy moving. Verified
    this cannot happen: min start-to-goal geodesic on val_unseen is 3.85 m
    against a 3.0 m radius, and 0/1839 episodes start inside it.
    """
    import torch

    from model.action_space import Action

    goal = episode["goals"][0]["position"]
    instruction = episode["instruction"]["instruction_text"]
    frame, pos = sim.reset(episode["start_position"], episode["start_rotation"])

    positions, actions_taken = [], []
    reason, min_dist = "timeout", float("inf")

    for _ in range(max_steps):
        positions.append(pos)
        d = sim.geodesic(pos, goal)
        min_dist = min(min_dist, d)
        if d <= success_radius:
            reason = "within_radius"
            break

        with torch.no_grad():
            seq = builder.build(instruction, enc.encode(frame))
            h_act = backbone.forward(
                seq, position_ids=backbone.build_position_ids(seq)
            )[-1]
            action = head.predict(h_act)

        actions_taken.append(int(action))
        if action is Action.STOP:
            reason = "model_stop"
            break

        frame, pos = sim.step(action.habitat_name)
    # loop exhausted without break -> reason stays "timeout"

    return {"positions": positions, "actions": actions_taken,
            "termination_reason": reason, "min_dist_to_goal": min_dist,
            "num_steps": len(positions)}


def main():
    ap = argparse.ArgumentParser(description="Stage 0 closed-loop policy eval")
    ap.add_argument("--checkpoint", default=str(_ROOT / "checkpoints" / "stage0" / "final"))
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="evaluate the untrained base model as a baseline")
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--scans", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=200,
                    help="episode cap (default 200; full val_unseen is 1839)")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--success-radius", type=float, default=3.0)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--save-episodes", action="store_true")
    ap.add_argument("--worker-python", default=None,
                    help="interpreter for the habitat_render env")
    args = ap.parse_args()

    from eval.metrics import aggregate, evaluate_episode
    from eval.remote_sim import DEFAULT_WORKER_PY, RemoteSim

    print(f"Loading policy from "
          f"{'(none)' if args.no_checkpoint else args.checkpoint} ...")
    enc, builder, backbone, head = load_policy(
        None if args.no_checkpoint else args.checkpoint
    )

    episodes = load_episodes(args.split, args.scans, args.limit)
    by_scan = {}
    for e in episodes:
        by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
    print(f"split={args.split}  episodes={len(episodes)}  scans={len(by_scan)}")
    print(f"max_steps={args.max_steps}  success_radius={args.success_radius} m\n")

    results, reasons, never_close = [], Counter(), []
    t0, done = time.time(), 0

    with RemoteSim(args.worker_python or DEFAULT_WORKER_PY) as sim:
        geo = sim.geodesic_fn()
        for scan, eps in sorted(by_scan.items()):
            glb = SCENES / scan / f"{scan}.glb"
            if not glb.exists():
                print(f"{scan}: scene missing, skipping {len(eps)} episodes")
                continue
            sim.load_scene(glb)

            for e in eps:
                r = rollout_policy(sim, enc, builder, backbone, head, e,
                                   args.max_steps, args.success_radius)
                reasons[r["termination_reason"]] += 1
                if r["termination_reason"] == "timeout":
                    never_close.append(r["min_dist_to_goal"])

                results.append(evaluate_episode(
                    path=r["positions"],
                    goal_position=e["goals"][0]["position"],
                    reference_path=e["reference_path"],
                    shortest_path_length=e["info"]["geodesic_distance"],
                    distance_fn=geo,
                ))

                if args.save_episodes:
                    d = args.out / args.split / scan / str(e["episode_id"])
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "episode.json").write_text(json.dumps({
                        **r, "episode_id": e["episode_id"], "scan": scan,
                        "instruction": e["instruction"]["instruction_text"],
                        "goal_position": e["goals"][0]["position"]}))

                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(episodes)}  "
                          f"({done / (time.time() - t0):.2f} ep/s)")

    agg = aggregate(results)
    n = len(results)
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  STAGE 0 POLICY EVAL — split={args.split}  n={n}")
    print(sep)
    print(f"  DIAGNOSTIC METRICS (early exit within {args.success_radius} m "
          f"counts as success):")
    for k in ("SR", "SPL", "OS", "nDTW", "NE"):
        print(f"    {k:5s} = {agg[k]:.4f}")
    print()
    print("  TERMINATION BREAKDOWN — this is the part that matters:")
    for name in ("model_stop", "within_radius", "timeout"):
        c = reasons[name]
        print(f"    {name:14s} {c:5d}  ({100.0 * c / n:5.1f}%)")
    print()
    print("    model_stop    = the policy's OWN STOP fired (real behaviour)")
    print("    within_radius = forced success; STOP never fired but the agent")
    print("                    reached the goal region -- navigation works")
    print("    timeout       = never got close and never stopped -- lost")
    if never_close:
        print(f"\n  Of {len(never_close)} timeouts, closest approach: mean "
              f"{sum(never_close)/len(never_close):.2f} m "
              f"(min {min(never_close):.2f}, max {max(never_close):.2f})")
    print()
    print("  SR/OS above are a CEILING if STOP were fixed, NOT a Stage 0 result.")
    print("  The real gate needs model_stop, not forced early exit.")
    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
