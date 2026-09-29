#!/usr/bin/env python3
"""
Stage 0' evaluation — RECURRENT closed-loop rollout with the PilotCache.

This is Sec. 3.3's "Inference with stored pilot", implemented literally:

    z_0 given by the learned Pilot parameter
    at step t:  read z_{t-1} from cache
                u_t = [<vs>; v_t; <ve>; Tok(x); PILOT(z_{t-1})]
                H_t = F_theta(u_t)
                a_t = argmax Softmax(W_LM h_t^act)     (Eq. 7)
                z_t = G_psi(h_t^pil)                   (Eq. 8)
                write z_t to cache
    reset the cache to z_0 at every episode boundary

WHY THIS EXISTS SEPARATELY FROM eval_stage0.py
----------------------------------------------
eval_stage0.py builds sequences with NO Pilot slot. Pointing it at a Stage 0'
checkpoint would not crash -- it would build a shorter sequence than the model
was trained on and report confident, meaningless numbers. That silent-wrong
failure mode is the one this project keeps hitting, so the two eval paths are
kept separate and this one ASSERTS that a pilot checkpoint is present.

Design A vs B (D17) is inferred from the checkpoint itself: mode B's
state_dict carries an `action_query` parameter, mode A's does not. The rollout
reads h_pil at `seq.pilot_index` and h_act at `seq.action_index`, which are the
same position under A and differ by one under B -- so the same code is correct
for both without branching.

    conda activate latentpilot
    python scripts/eval_stage0_prime.py --checkpoint checkpoints/stage0prime_A/final
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
DEFAULT_MAX_STEPS = 100


def load_episodes(split, scans=None, limit=None):
    with gzip.open(R2R_DIR / split / f"{split}.json.gz", "rt") as fh:
        eps = json.load(fh)["episodes"]
    if scans:
        wanted = set(scans)
        eps = [e for e in eps if e["scene_id"].split("/")[1] in wanted]
    return eps[:limit] if limit else eps


def load_policy(checkpoint):
    """LoRA adapter + PilotModule, with the Pilot slot wired into Eq. 5.

    Both halves are asserted: a missing adapter or a missing pilot.pt aborts
    rather than silently evaluating an untrained or pilot-less model.
    """
    import torch
    from peft import PeftModel

    from model.action_head import ActionHead
    from model.backbone import Backbone
    from model.input_sequence import InputSequenceBuilder
    from model.pilot import PilotModule
    from model.vision_encoder import VisionEncoder

    ckpt = pathlib.Path(checkpoint)
    pilot_pt = ckpt / "pilot.pt"
    if not pilot_pt.exists():
        raise FileNotFoundError(
            f"no pilot.pt under {ckpt} -- this is not a Stage 0'/Stage 1 "
            f"checkpoint. Use scripts/eval_stage0.py for memoryless Stage 0."
        )

    adapter = ckpt / "adapter_model.safetensors"
    adapter_pt = ckpt / "adapter.pt"          # Stage 1 format (D24)
    full = ckpt / "model.safetensors"
    if not any(p.exists() for p in (adapter, adapter_pt, full)):
        raise FileNotFoundError(f"no weights under {ckpt}")

    enc = VisionEncoder()

    if adapter_pt.exists():
        # Stage 1 (D24): only the lora_* tensors were saved, 25 MB rather than
        # the 4.9 GB base model. Rebuild the same module tree PEFT would have
        # built, then load those tensors into it. Key names match because
        # train_stage1.py filtered enc.model.state_dict() directly.
        from peft import LoraConfig, get_peft_model

        sd = torch.load(str(adapter_pt), map_location="cpu", weights_only=True)
        lora_a = [k for k in sd if k.endswith("lora_A.default.weight")]
        if not lora_a:
            raise RuntimeError(f"{adapter_pt} holds no lora_* tensors")
        r = sd[lora_a[0]].shape[0]

        get_peft_model(enc.model, LoraConfig(
            r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
        missing, unexpected = enc.model.load_state_dict(sd, strict=False)
        stray = [k for k in unexpected if "lora" in k]
        if stray:
            raise RuntimeError(f"adapter.pt has unmatched LoRA keys: {stray[:3]}")
        print(f"  loaded adapter.pt ({len(sd)} LoRA tensors, r={r})")
    elif adapter.exists():
        # Proper PEFT adapter directory.
        PeftModel.from_pretrained(enc.model, str(ckpt))
    else:
        # FULL-MODEL checkpoint. train_stage0_prime.py called save_pretrained on
        # the BASE model (the PeftModel wrapper is deliberately discarded there,
        # because its extra nesting breaks get_vision_position_ids), so the file
        # holds base weights AND the injected lora_* tensors under base-model
        # key names -- 4.9 GB rather than an adapter, but nothing is missing.
        #
        # Rebuild the same module tree, then load the state dict into it.
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file

        sd = load_file(str(full))
        lora_a = [k for k in sd if k.endswith("lora_A.default.weight")]
        if not lora_a:
            raise RuntimeError(
                f"{full} contains no lora_* tensors -- it is a plain base "
                f"model, not a trained checkpoint"
            )
        r = sd[lora_a[0]].shape[0]        # lora_A is (r, in_features)

        get_peft_model(enc.model, LoraConfig(
            r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
        missing, unexpected = enc.model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "lora" in k]
        if real_missing:
            raise RuntimeError(
                f"{len(real_missing)} LoRA tensors missing after load, e.g. "
                f"{real_missing[0]}"
            )
        print(f"  loaded full checkpoint (r={r}, {len(lora_a)} adapted modules)")

    n_lora = sum(p.numel() for n, p in enc.model.named_parameters() if "lora" in n)
    if n_lora == 0:
        raise RuntimeError("adapter did not attach (0 LoRA params)")

    state = torch.load(pilot_pt, map_location=enc.model.device, weights_only=True)
    mode = "action_query" if "action_query" in state else "last"
    pilot = PilotModule(enc.d, mode=mode, dtype=torch.bfloat16,
                        device=enc.model.device)
    pilot.load_state_dict(state)
    pilot.eval()
    pilot.assert_shapes()
    enc.model.eval()

    print(f"  adapter: {n_lora:,} LoRA params")
    print(f"  pilot  : mode={mode} "
          f"({'A -- h_act == h_pil' if mode == 'last' else 'B -- distinct h_act/h_pil'})")

    return (enc,
            InputSequenceBuilder(enc.model, enc.processor.tokenizer, pilot=pilot),
            Backbone(enc.model),
            ActionHead(enc.model, enc.processor.tokenizer),
            pilot)


def rollout_policy(sim, enc, builder, backbone, head, pilot, episode,
                   max_steps, success_radius, slot_mode="z", mean_vbar=None,
                   target_norm=None):
    """Recurrent rollout. The Pilot slot carries ONLY z_{t-1} from the cache --
    no v_bar is reachable from here, which is the structural guarantee against
    future leakage at eval (AGENTS.md Sec. 4.1)."""
    import torch

    from model.action_space import Action

    goal = episode["goals"][0]["position"]
    instruction = episode["instruction"]["instruction_text"]
    frame, pos = sim.reset(episode["start_position"], episode["start_rotation"])

    cache = pilot.new_cache()          # z <- z_0 at the episode boundary
    assert cache.is_fresh

    positions, actions_taken, z_norms = [], [], []
    reason, min_dist = "timeout", float("inf")

    for _ in range(max_steps):
        positions.append(pos)
        d = sim.geodesic(pos, goal)
        min_dist = min(min_dist, d)
        if d <= success_radius:
            reason = "within_radius"
            break

        with torch.no_grad():
            v_t = enc.encode(frame)

            # ---- ABLATION: what actually goes in the Pilot slot -------------
            # Diagnostic only. None of these leak future information: v_bar_t
            # is the CURRENT frame, already available at inference.
            if slot_mode == "z":                       # Sec. 3.3, as specified
                slot = cache.read()
            elif slot_mode == "z_scaled":              # z, renormalised to ~||v_bar||
                z = cache.read().float()
                slot = (z / z.norm().clamp(min=1e-6) * target_norm).to(z.dtype)
            elif slot_mode == "vbar_current":          # real v_bar_t, in-distribution
                slot = v_t.mean(dim=0)
            elif slot_mode == "constant":              # in-distribution, zero info
                slot = mean_vbar
            else:
                raise ValueError(f"unknown slot_mode {slot_mode!r}")

            seq = builder.build(instruction, v_t, pilot_input=slot)  # Eq. 5
            H = backbone.forward(seq, position_ids=backbone.build_position_ids(seq))
            action = head.predict(H[seq.action_index])               # Eq. 7
            z_t = pilot(H[seq.pilot_index])                          # Eq. 8
            cache.write(z_t)
            z_norms.append(z_t.float().norm().item())

        actions_taken.append(int(action))
        if action is Action.STOP:
            reason = "model_stop"
            break

        frame, pos = sim.step(action.habitat_name)

    return {"positions": positions, "actions": actions_taken,
            "termination_reason": reason, "min_dist_to_goal": min_dist,
            "num_steps": len(positions), "z_norms": z_norms,
            "cache_steps": cache.episode_steps}


def main():
    ap = argparse.ArgumentParser(description="Stage 0' recurrent closed-loop eval")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--scans", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--success-radius", type=float, default=3.0)
    ap.add_argument("--strict", action="store_true",
                    help="paper protocol: only the model's own STOP counts as "
                         "termination (no forced exit inside the radius)")
    ap.add_argument("--worker-python", default=None)
    ap.add_argument("--slot-mode", default="z",
                    choices=["z", "z_scaled", "vbar_current", "constant"],
                    help="DIAGNOSTIC: what fills the Pilot slot at inference. "
                         "'z' is the specified behaviour; the others isolate "
                         "whether failure is scale, distribution, or information. "
                         "None leak future data.")
    args = ap.parse_args()

    from eval.metrics import aggregate, evaluate_episode
    from eval.remote_sim import DEFAULT_WORKER_PY, RemoteSim

    # v_bar statistics for the ablation modes (from the fp32 cache)
    import numpy as np
    import torch as _torch
    vpaths = sorted((_ROOT / "data" / "rollouts" / "train").rglob("vbar.npy"))[:200]
    _allv = np.concatenate([np.load(v) for v in vpaths])
    _mean = _allv.mean(axis=0)
    TARGET_NORM = float(np.linalg.norm(_allv, axis=1).mean())
    print(f"v_bar reference: mean ||v_bar|| = {TARGET_NORM:.2f}")

    print(f"Loading Stage 0' policy from {args.checkpoint} ...")
    enc, builder, backbone, head, pilot = load_policy(args.checkpoint)
    MEAN_VBAR = _torch.from_numpy(_mean).to(enc.model.device, _torch.bfloat16)
    print(f"slot_mode = {args.slot_mode}")

    radius = -1.0 if args.strict else args.success_radius
    if args.strict:
        print("  STRICT mode: success requires the model's own STOP\n")

    episodes = load_episodes(args.split, args.scans, args.limit)
    by_scan = {}
    for e in episodes:
        by_scan.setdefault(e["scene_id"].split("/")[1], []).append(e)
    print(f"split={args.split}  episodes={len(episodes)}  scans={len(by_scan)}\n")

    results, reasons, all_z = [], Counter(), []
    t0, done = time.time(), 0

    with RemoteSim(args.worker_python or DEFAULT_WORKER_PY) as sim:
        geo = sim.geodesic_fn()
        for scan, eps in sorted(by_scan.items()):
            glb = SCENES / scan / f"{scan}.glb"
            if not glb.exists():
                continue
            sim.load_scene(glb)
            for e in eps:
                r = rollout_policy(sim, enc, builder, backbone, head, pilot, e,
                                   args.max_steps, radius,
                                   slot_mode=args.slot_mode,
                                   mean_vbar=MEAN_VBAR,
                                   target_norm=TARGET_NORM)
                reasons[r["termination_reason"]] += 1
                all_z.extend(r["z_norms"])
                results.append(evaluate_episode(
                    path=r["positions"],
                    goal_position=e["goals"][0]["position"],
                    reference_path=e["reference_path"],
                    shortest_path_length=e["info"]["geodesic_distance"],
                    distance_fn=geo,
                ))
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(episodes)}  "
                          f"({done / (time.time() - t0):.2f} ep/s)")

    agg = aggregate(results)
    n = len(results)
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  STAGE 0' EVAL — {args.checkpoint}")
    print(f"  slot_mode={args.slot_mode}")
    print(f"  split={args.split}  n={n}  "
          f"{'STRICT (own STOP only)' if args.strict else 'diagnostic (3m early exit)'}")
    print(sep)
    for k in ("SR", "SPL", "OS", "nDTW", "NE"):
        print(f"    {k:5s} = {agg[k]:.4f}")
    print()
    print("  TERMINATION:")
    for name in ("model_stop", "within_radius", "timeout"):
        c = reasons[name]
        print(f"    {name:14s} {c:5d}  ({100.0 * c / n:5.1f}%)")
    if all_z:
        import statistics
        print(f"\n  ||z_t|| over {len(all_z):,} steps: "
              f"mean {statistics.mean(all_z):.2f}  "
              f"min {min(all_z):.2f}  max {max(all_z):.2f}")
        print("    (Stage 2 wants this DRIFT to shrink; record it as the "
              "Stage 0' reference)")
    print()
    print("  Paper NaN row (Table 3): SR 51.7  SPL 47.1  NE 5.3  OS 57.0")
    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
