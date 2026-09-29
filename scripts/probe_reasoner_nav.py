#!/usr/bin/env python3
"""Does the Cosmos3-Edge REASONER pick the right navigation action?

Offline, on the rendered 15 FPS video: at decision points along expert
trajectories, feed the last 16 frames (1.07 s of the robot's own view, ending
now) plus the R2R instruction, ask for the next action in habitat's vocabulary,
and score it against what the expert actually does over the next 8 frames.

Fixes every context defect of the first probe:
  * video, not a still (their navigation examples all use video)
  * an explicit embodiment line -- it must know it IS the agent
  * their full three-part hierarchical template (the current-step line was
    dropped last time), alongside the overall-only variant a closed loop can run
  * thinking controlled by the chat template, format block appended per guide
  * a MEASURED score -- accuracy against the expert -- not a read of the prose

Expert label from action9d (Cosmos frame: +yaw = right, translation in metres):
  |net yaw over 16 frames| > 20 deg -> turn; else net fwd > 0.10 m -> move
  forward; else stop. The final frame of a trajectory is always "stop".
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, collections, glob, json, math, pathlib, random, re, time

import numpy as np
import torch
import imageio.v3 as iio

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
CK = "checkpoints/Cosmos3-Edge"
HORIZON, FPS = 16, 15.0   # 1 s horizon: a semantic step, not a heading nudge


def yaw_deg(a9, rot_from_6d):
    R = rot_from_6d(np.asarray(a9[3:], dtype=np.float64))
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))


STOP_TAIL = 6


def expert_label(actions, t, rot_from_6d):
    """Same rule as nav.reasoner_sft_data.label_at: stop = the last STOP_TAIL
    frames of the trajectory, i.e. the instruction is complete. Mid-trajectory
    low motion is 'move forward', not 'stop' (v2 called it stop and swept slow
    manoeuvring into that class)."""
    if t >= len(actions) - STOP_TAIL:
        return "stop"
    seg = actions[t:t + HORIZON]
    yaw = sum(yaw_deg(a, rot_from_6d) for a in seg)
    if yaw > 20.0:
        return "turn right"
    if yaw < -20.0:
        return "turn left"
    return "move forward"


def decision_points(actions, rot_from_6d, k, rng):
    """k points per trajectory: turn onsets first, then spread, plus the end."""
    yaws = np.array([abs(yaw_deg(a, rot_from_6d)) for a in actions])
    onsets = [t for t in range(2, len(actions) - HORIZON)
              if yaws[t:t + 4].sum() > 15 and yaws[max(0, t - 4):t].sum() < 4]
    pts = set(rng.sample(onsets, min(len(onsets), k - 2))) if onsets else set()   # turns first
    while len(pts) < k - 1:
        pts.add(rng.randint(1, max(1, len(actions) - HORIZON - 1)))
    pts.add(len(actions) - 1)                      # the STOP case (end of trajectory)
    return sorted(pts)


def step_proxy(instruction, frac):
    """R2R has no progress annotation: use the sentence at this fraction of the path."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", instruction.strip()) if s.strip()]
    if not sents:
        return instruction
    return sents[min(len(sents) - 1, int(frac * len(sents)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=int, default=24)
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--variants", nargs="+", default=["hier_overall", "hier_full"])
    ap.add_argument("--out", default="results/reasoner_nav_probe")
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (SFT'd reasoner)")
    ap.add_argument("--stop-tail", type=int, default=6,
                    help="frames from the end labelled stop; mid-trajectory low motion is NOT stop")
    args = ap.parse_args()
    global STOP_TAIL; STOP_TAIL = args.stop_tail

    from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration
    from data.ego_pose_9d import rot_from_6d
    from nav.cosmos_prompts import SYSTEM, reasoner_prompt, parse_command, COMMANDS, VIDEO_SPAN, video_kwargs
    CTX = VIDEO_SPAN

    rng = random.Random(0)
    dirs = sorted(glob.glob(str(_ROOT / "data/video/val_unseen/*/traj*")))
    by_scan = collections.defaultdict(list)
    for d in dirs:
        by_scan[pathlib.Path(d).parent.name].append(d)
    sel = []
    for i in range(50):
        for s in sorted(by_scan):
            if i < len(by_scan[s]):
                sel.append(by_scan[s][i])
        if len(sel) >= args.trajectories:
            break
    sel = sel[:args.trajectories]

    processor = AutoProcessor.from_pretrained(CK)
    model = Cosmos3EdgeForConditionalGeneration.from_pretrained(CK, dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        print(f"loaded adapter {args.adapter}", flush=True)
    print(f"reasoner on GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    samples = []
    for d in sel:
        meta = json.loads((pathlib.Path(d) / "meta.json").read_text())
        acts = np.load(f"{d}/action9d.npy")
        frames = iio.imread(f"{d}/frames.mp4", plugin="pyav")
        for t in decision_points(acts, rot_from_6d, args.points, rng):
            lo = max(0, t - CTX + 1)
            clip = frames[lo:t + 1]
            if len(clip) < CTX:                         # pad the start by repeating frame 0
                clip = np.concatenate([np.repeat(clip[:1], CTX - len(clip), 0), clip])
            samples.append(dict(scan=meta["scan"], traj=meta["trajectory_id"], t=t,
                                frac=t / max(1, len(acts) - 1),
                                instruction=meta["instructions"][0],
                                label=expert_label(acts, t, rot_from_6d), clip=np.asarray(clip)))
    print(f"{len(samples)} decision points from {len(sel)} trajectories; expert labels: "
          f"{dict(collections.Counter(s['label'] for s in samples))}\n", flush=True)

    def gen(clip, text, think):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "video", "video": clip},
                                             {"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=think, **video_kwargs(len(clip)),
        ).to(model.device)
        kw = dict(do_sample=True, top_p=0.95, top_k=20, temperature=0.6) if think else \
             dict(do_sample=True, top_p=0.8, top_k=20, temperature=0.7)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **kw)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]

    report = {}
    transcripts = []
    for var in args.variants:
        think = not var.endswith("_nothink")
        use_step = var.startswith("hier_full")
        conf = collections.Counter(); n_ok = n_parse = 0; times = []
        for i, s in enumerate(samples):
            step = step_proxy(s["instruction"], s["frac"]) if use_step else None
            text = reasoner_prompt(s["instruction"], step=step, think=think)
            t0 = time.time()
            ans = gen(s["clip"], text, think)
            times.append(time.time() - t0)
            pred = parse_command(ans)
            n_parse += pred is not None
            n_ok += pred == s["label"]
            conf[(s["label"], pred or "UNPARSED")] += 1
            if i < 6:
                transcripts.append(dict(variant=var, scan=s["scan"], traj=s["traj"], t=s["t"],
                                        label=s["label"], pred=pred, prompt=text, answer=ans))
            print(f"  [{var}] {i+1}/{len(samples)} expert={s['label']:<12} pred={str(pred):<12} "
                  f"{'OK' if pred == s['label'] else '--'}  {times[-1]:.1f}s", flush=True)
        acc = n_ok / len(samples)
        rec = {c: (sum(v for (l, p), v in conf.items() if l == c and p == c) /
                   max(1, sum(v for (l, p), v in conf.items() if l == c))) for c in COMMANDS}
        report[var] = dict(accuracy=acc, parse_rate=n_parse / len(samples), recall=rec,
                           confusion={f"{l}->{p}": v for (l, p), v in sorted(conf.items())},
                           mean_seconds=float(np.mean(times)))
        print(f"\n=== {var}: accuracy {acc:.3f}  parse {n_parse/len(samples):.2f}  "
              f"recall {{{', '.join(f'{k}: {v:.2f}' for k, v in rec.items())}}}  "
              f"{np.mean(times):.1f}s/call ===\n", flush=True)

    maj = max(collections.Counter(s["label"] for s in samples).values()) / len(samples)
    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(dict(
        n=len(samples), majority_baseline=maj, variants=report, transcripts=transcripts), indent=1))
    md = [f"# Reasoner navigation probe\n\n{len(samples)} decision points, {len(sel)} trajectories, "
          f"val_unseen, video ctx {CTX} frames @ {FPS:.0f} FPS, horizon {HORIZON} frames\n\n"
          f"majority-class baseline: **{maj:.3f}**\n\n"
          "| variant | accuracy | parse | fwd | left | right | stop | s/call |\n|---|---|---|---|---|---|---|---|\n"]
    for v, r in report.items():
        md.append(f"| {v} | **{r['accuracy']:.3f}** | {r['parse_rate']:.2f} | "
                  + " | ".join(f"{r['recall'][c]:.2f}" for c in COMMANDS) + f" | {r['mean_seconds']:.1f} |\n")
    md.append("\n## Confusion (expert -> predicted)\n")
    for v, r in report.items():
        md.append(f"\n**{v}**\n\n```\n" + "\n".join(f"{k:32s} {c}" for k, c in r["confusion"].items()) + "\n```\n")
    md.append("\n## Sample transcripts\n")
    for tr in transcripts:
        md.append(f"\n### {tr['variant']} — {tr['scan']}/traj{tr['traj']} t={tr['t']} "
                  f"expert={tr['label']} pred={tr['pred']}\n\n**prompt**\n```\n{tr['prompt']}\n```\n"
                  f"**answer**\n```\n{tr['answer'].strip()[:1500]}\n```\n")
    out.with_suffix(".md").write_text("".join(md))
    print(f"wrote {out}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
