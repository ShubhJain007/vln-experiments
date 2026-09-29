#!/usr/bin/env python3
"""Invoke Cosmos3-Edge's REASONER tower -- the path that actually emits tokens.

Everything else we have run today used Cosmos3OmniPipeline (diffusers), which
decodes latents and never produces a token: the model has literally never
reasoned. This calls Cosmos3EdgeForConditionalGeneration (transformers), which
their docs describe as loading "only the Reasoner tower from the unified
checkpoint", dropping generator/audio/action parameters.

Procedure is verbatim from NVIDIA's Reasoner Prompt Guide:
  * system prompt is literally "You are a helpful assistant."
  * reasoning is toggled by APPENDING the <think> format block to the user
    text -- there is no enable_thinking flag
  * with reasoning: top_p 0.95, top_k 20, temperature 0.6, presence_penalty 0
  * media goes BEFORE text in the content list

Their Action CoT template says "the 2D trajectory your END EFFECTOR should
follow" with label "gripper trajectory" -- manipulation phrasing. They ship no
navigation trajectory example, so the navigation variants below are OUR
adaptation and are untested by definition; --mode runs them side by side rather
than picking one on my judgement.

Trajectory convention (theirs): normalized 0-1000 per axis, origin top-left,
X right, Y down. That is the same image-space convention our own pointing head
used, so eval.pointing geometry can consume it directly.
"""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import argparse, glob, json, pathlib, re

import torch
import imageio.v3 as iio
from PIL import Image

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
CK = "checkpoints/Cosmos3-Edge"

SYSTEM = "You are a helpful assistant."
THINK = ("\nAnswer the question using the following format:\n\n<think>\n"
         "Your reasoning.\n</think>\n\nWrite your final answer immediately "
         "after the </think> tag.")

# theirs, verbatim (manipulation phrasing)
TRAJ_GRIPPER = ('You are given the task "{instruction}". Specify the 2D trajectory '
                'your end effector should follow in pixel space. Return the '
                'trajectory coordinates in JSON format like this: '
                '{{"point_2d": [x, y], "label": "gripper trajectory"}}.')
# our navigation adaptation -- NOT from their guide, untested
TRAJ_NAV = ('You are given the task "{instruction}". Specify the 2D trajectory '
            'you should follow in pixel space to navigate. Return the '
            'trajectory coordinates in JSON format like this: '
            '{{"point_2d": [x, y], "label": "navigation trajectory"}}.')
# theirs, verbatim (Robotics Next Action)
NEXT_ACTION = "What can be the next immediate action?"
# theirs, verbatim (Assisted Task Next Action) -- hierarchical, closest to R2R
HIERARCHICAL = ('This is the overall task that the agent is trying to complete: '
                '"{instruction}"\nWhat should be the next action of the agent?')

MODES = {"traj_gripper": TRAJ_GRIPPER, "traj_nav": TRAJ_NAV,
         "next_action": NEXT_ACTION, "hierarchical": HIERARCHICAL}


def parse_points(text):
    """Pull [{'point_2d': [x, y], ...}] out of the post-</think> answer."""
    tail = text.split("</think>")[-1]
    pts = [(int(a), int(b)) for a, b in
           re.findall(r'"point_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', tail)]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--out", default="results/reasoner_probe.md")
    args = ap.parse_args()

    from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

    processor = AutoProcessor.from_pretrained(CK)
    model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
        CK, dtype=torch.bfloat16, device_map="auto")
    print(f"reasoner loaded; GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    dirs = sorted(glob.glob(str(_ROOT / "data/video/val_unseen/*/traj*")))
    by_scan = {}
    for d in dirs:
        by_scan.setdefault(pathlib.Path(d).parent.name, []).append(d)
    sel = [v[0] for v in by_scan.values()][:args.n]

    tmp = _ROOT / "logs" / "reasoner_frames"; tmp.mkdir(parents=True, exist_ok=True)
    out = [f"# Cosmos3-Edge reasoner probe\n\nthink={'off' if args.no_think else 'on'}"
           f"  max_new_tokens={args.max_new_tokens}\n\n"]

    for d in sel:
        meta = json.loads((pathlib.Path(d) / "meta.json").read_text())
        instr = meta["instructions"][0].strip()
        fp = tmp / f"{meta['scan']}_{meta['trajectory_id']}.png"
        Image.fromarray(iio.imread(f"{d}/frames.mp4", plugin="pyav")[0]).save(fp)
        out.append(f"\n## {meta['scan']} / traj{meta['trajectory_id']}\n\n"
                   f"**instruction:** {instr}\n")
        print(f"\n=== {meta['scan']}/traj{meta['trajectory_id']} ===\n{instr}", flush=True)

        for mode in args.modes:
            task = MODES[mode].format(instruction=instr)
            text = task if args.no_think else task + THINK
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image", "path": str(fp)},     # media BEFORE text
                    {"type": "text", "text": text},
                ]},
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device, torch.bfloat16)
            gen = model.generate(**inputs, do_sample=True, max_new_tokens=args.max_new_tokens,
                                 top_p=0.95, top_k=20, temperature=0.6)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
            ans = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            pts = parse_points(ans)
            print(f"\n--- {mode} --- ({len(pts)} points parsed)\n{ans[:700]}", flush=True)
            out.append(f"\n### {mode}\n\n_{len(pts)} waypoints parsed_\n\n"
                       f"```\n{ans.strip()}\n```\n")

    p = _ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(out))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
