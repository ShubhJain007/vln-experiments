#!/usr/bin/env python3
"""Why does the reasoner miss the `<task> | <scene>` schema? Show raw answers."""
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
import glob, json, pathlib, random, collections
import numpy as np, torch, imageio.v3 as iio
_ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_ROOT / "src"))
from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration
from nav.cosmos_prompts import SYSTEM, caption_prompt, parse_caption
CK, CTX, FPS = "checkpoints/Cosmos3-Edge", 16, 15.0
N, MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 10, int(sys.argv[2]) if len(sys.argv) > 2 else 768
proc = AutoProcessor.from_pretrained(CK)
model = Cosmos3EdgeForConditionalGeneration.from_pretrained(CK, dtype=torch.bfloat16, device_map="cuda")
rng = random.Random(1)
dirs = sorted(glob.glob("data/video/val_unseen/*/traj*")); rng.shuffle(dirs)
tally = collections.Counter()
for d in dirs[:N]:
    m = json.load(open(f"{d}/meta.json")); fr = iio.imread(f"{d}/frames.mp4", plugin="pyav")
    t = rng.randint(CTX, len(fr) - 1); clip = np.asarray(fr[t - CTX + 1:t + 1])
    text = caption_prompt(m["instructions"][0], think=True)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "video", "video": clip}, {"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=True,
            video_metadata=[{"fps": FPS, "frames_indices": list(range(CTX)), "total_num_frames": CTX, "duration": CTX / FPS}]).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAXTOK, do_sample=True, top_p=0.95, top_k=20, temperature=0.6)
    ntok = out.shape[1] - inp["input_ids"].shape[1]
    ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    closed = "</think>" in ans; tail = ans.split("</think>")[-1].strip()
    cap, _, how = parse_caption(ans)
    kind = f"{how}" + ("" if closed else " [truncated]")
    tally[kind] += 1
    print(f"\n##### {m['scan']}/traj{m['trajectory_id']} t={t}  tokens={ntok}  -> {kind}")
    print("AFTER </think>:", repr(tail[:300]) if closed else "(none)")
    print("CAPTION ->", cap)
    if not closed: print("THINK TAIL:", repr(ans[-250:]))
print("\n=== tally ===", dict(tally))
